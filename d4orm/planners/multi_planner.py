# Standard library imports
import os
import time
import random
import queue
from dataclasses import dataclass

# Third-party imports
import tyro
import jax
from jax import numpy as jnp, config
from scipy.interpolate import interp1d
from typing import Callable, List, Any

# Local application/library-specific imports
import d4orm
from d4orm.envs.multibase import State
from d4orm.envs.multibase import MultiBase

# NOTE: enable this if you want higher precision
# config.update("jax_enable_x64", True)

# from jax import config
config.update("jax_default_matmul_precision", "float32")


## load config
@dataclass
class Args:
    # exp
    seed: int = random.randint(0, 2**32 - 1)
    disable_recommended_params: bool = False
    not_render: bool = False
    # env
    env_name: str = "multi2dholo"
    Nagent: int = 8  # number of agents
    # diffusion
    Nsample: int = 2048  # number of samples
    Hsample: int = 100  # horizon
    Ndiffuse: int = 100  # number of denoising steps
    Niteration: int = 30  # number of iterations
    temp_sample: float = 0.3  # temperature for sampling
    beta1: float = 1e-4  # initial beta
    betaT: float = 2e-2  # final beta
    dt_factor: int = 1
    # result
    save_images: bool = False
    save_data: bool = False
    print_info: bool = False


@dataclass
class ReverseOnceFn:
    fn: Callable
    horizon: int
    joint_u_dim: int


@dataclass
class CarryConfig:
    penalty_weight: float = 1.0
    use_mask: bool = True
    margin_factor: int = 1
    dt: float = 0.1


def make_reverse_once_naive(Nsample: int,
                            Hsample: int,
                            Nu: int,
                            args: Args,
                            alphas_bar: jax.Array,
                            sigmas: jax.Array,
                            rollout_env_jit: Callable):
    """
    Make one denoising step function
    """
    @jax.jit
    def reverse_once(carry, unused):
        i, rng, U_i, const = carry
        U_base, state_init, xg, penalty_weight, use_mask, margin_factor, dt = const

        # --- Step 1: compute Ubar_i
        Ubar_i = U_i / jnp.sqrt(alphas_bar[i])

        # --- Step 2: sample from q_i
        rng, rng_sample = jax.random.split(rng)
        eps_u = jax.random.normal(rng_sample, (Nsample, Hsample, Nu))
        U_is = eps_u * sigmas[i] + Ubar_i # shape (Nsample, Hsample, Nu)

        # --- Step 3: rollout trajectories
        rollout_fn = jax.vmap(rollout_env_jit, in_axes=(None, None, 0, None, None, None, None))
        rewss, _, _, _ = rollout_fn(state_init, xg, U_is + U_base, penalty_weight, use_mask, margin_factor, dt)

        rews = rewss.mean(axis=-1)
        rew_std = rews.std()
        rew_std = jnp.where(rew_std < 1e-4, 1.0, rew_std)
        rew_mean = rews.mean()
        logp0 = (rews - rew_mean) / rew_std / args.temp_sample
        weights = jax.nn.softmax(logp0)
        Ubar_0 = jnp.einsum("n,nij->ij", weights, U_is)

        U_im1 = jnp.sqrt(alphas_bar[i-1]) * Ubar_0

        rews_mean = rews.mean()

        return (i - 1, rng, U_im1, const), rews_mean
    
    return ReverseOnceFn(fn=reverse_once, horizon=Hsample, joint_u_dim=Nu)


def plan_joint_naive(args: Args,
                     rng: jax.Array,
                     max_itr: int,
                     state_init: State,
                     xg: jax.Array,
                     rollout_env_jit: Callable,
                     reverse_once_obj: ReverseOnceFn,
                     carry_cfg: CarryConfig,
                     warm_up: bool=False,
                     print_info: bool=False,
                     U_base: jax.Array=None):
    """
    D4orm base version
    """
    if warm_up:
        max_itr = 3

    # Decode the parameters
    reverse_once_fn, horizon, Nu = reverse_once_obj.fn, reverse_once_obj.horizon, reverse_once_obj.joint_u_dim
    penalty_weight, use_mask, margin_factor, dt = carry_cfg.penalty_weight, carry_cfg.use_mask, carry_cfg.margin_factor, carry_cfg.dt

    U_N = jnp.zeros([horizon, Nu])
    rew_lst = []
    num_deforms = 0
    success = False

    if U_base is None:
        U_base = jnp.zeros([horizon, Nu])

    for itr in range(max_itr):
        const = (U_base, state_init, xg, penalty_weight, use_mask, margin_factor, dt)
        (_, rng, U_deform, _), _ = jax.lax.scan(f=reverse_once_fn,
                                                init=(args.Ndiffuse, rng, U_N, const),
                                                xs=None,
                                                length=args.Ndiffuse)
            
        U_base = U_base + U_deform
        num_deforms += 1

        rews, state_pipelines, goal_masks, collisions = rollout_env_jit(state=state_init,
                                                                        xg=xg,
                                                                        us=U_base,
                                                                        penalty_weight=penalty_weight,
                                                                        use_mask=use_mask,
                                                                        margin_factor=margin_factor,
                                                                        dt=dt)
        rew_lst.append(round(float(rews.mean()), 4))

        # Mask out actions after reach and stop at the goal
        if use_mask:
            idx_first_1 = jnp.argmax(goal_masks, axis=0)
            cols = jnp.arange(goal_masks.shape[-1])
            goal_masks_action = goal_masks.at[idx_first_1, cols].set(0)
            goal_masks_action = jnp.repeat(goal_masks_action, repeats=Nu//args.Nagent, axis=-1)
            U_base = U_base * (1 - goal_masks_action)

        if (jnp.all(goal_masks[-1, :]) & jnp.all(collisions == 0)):
            if print_info: print(f"Find solution at iteration {itr+1}")
            success = True
            break

    return (U_base, state_pipelines, goal_masks, rew_lst, success, num_deforms)


def run_diffusion(args: Args):

    rng = jax.random.PRNGKey(seed=args.seed)

    ## setup env

    args.Hsample = args.dt_factor * (args.Hsample // args.dt_factor)
    env = d4orm.envs.get_env(args.env_name, args.Nagent)
    Nx = env.observation_size
    Nu = env.action_size
    step_env_jit = jax.jit(env.step)
    reset_env_jit = jax.jit(env.reset)
    rollout_env_jit = jax.jit(env.rollout)
    clip_actions_jit = jax.jit(env.clip_actions)

    rng, rng_reset = jax.random.split(rng)  # NOTE: rng_reset should never be changed.
    state_init = reset_env_jit(rng_reset)

    ## run diffusion

    Ndiffuse_ref = 100
    betas_ref = jnp.linspace(args.beta1, args.betaT, Ndiffuse_ref)
    betas_ref = jnp.concatenate([jnp.array([0.0]), betas_ref])
    alphas_ref = 1.0 - betas_ref
    alphas_bar_ref = jnp.cumprod(alphas_ref)

    # Interpolate alphas_bar to maintain the same noise schedule
    interp_func = interp1d(jnp.linspace(0, 1, Ndiffuse_ref + 1), alphas_bar_ref, kind='linear', fill_value="extrapolate")
    alphas_bar = jnp.array(interp_func(jnp.linspace(0, 1, args.Ndiffuse + 1)))
    alphas = jnp.concatenate([alphas_bar[:1], alphas_bar[1:] / alphas_bar[:-1]])

    sigmas = jnp.sqrt(1 / alphas_bar - 1)
    Sigmas_cond = (
        (1 - alphas) * (1 - jnp.sqrt(jnp.roll(alphas_bar, 1))) / (1 - alphas_bar)
    )
    sigmas_cond = jnp.sqrt(Sigmas_cond)
    sigmas_cond = sigmas_cond.at[0].set(0.0)
    print(f"init sigma = {sigmas[-1]:.2e}")
    print(f"random seed: {args.seed}")
    print("Hsample:", args.Hsample)

    path = f"{d4orm.__path__[0]}/../results/d4orm/{args.env_name}"
    os.makedirs(path, exist_ok=True)

    """For warm-up"""
    print("\n========== Start Warm Up ==========")
    rng, rng_warmup = jax.random.split(rng)

    reverse_once_obj = make_reverse_once_naive(Nsample=args.Nsample,
                                               Hsample=args.Hsample,
                                               Nu=Nu,
                                               args=args,
                                               alphas_bar=alphas_bar,
                                               sigmas=sigmas,
                                               rollout_env_jit=rollout_env_jit)
    carry_cfg = CarryConfig(penalty_weight=1.0, use_mask=True, margin_factor=1, dt=0.1*args.dt_factor)

    _ = plan_joint_naive(args=args,
                         rng=rng_warmup,
                         max_itr=1,
                         state_init=state_init,
                         xg=env.xg,
                         rollout_env_jit=rollout_env_jit,
                         reverse_once_obj=reverse_once_obj,
                         carry_cfg=carry_cfg,
                         warm_up=True)

    for seed in range(1):
        rng = jax.random.PRNGKey(seed=seed+args.seed)
        rng, _ = jax.random.split(rng)

        start_time = time.time()  ###################################################################################
        print("\n========== Start Main Computation ==========")
        plan_joint_outcome = plan_joint_naive(args=args,
                                              rng=rng,
                                              max_itr=args.Niteration,
                                              state_init=state_init,
                                              xg=env.xg,
                                              rollout_env_jit=rollout_env_jit,
                                              reverse_once_obj=reverse_once_obj,
                                              carry_cfg=carry_cfg,
                                              print_info=args.print_info)
        U_base, state_pipelines, goal_masks, rew_lst, success, num_deforms = plan_joint_outcome

        end_time = time.time()  ###################################################################################
        elapsed_time = end_time - start_time
        print(f"\nElapsed time: {elapsed_time:.6f} seconds")

        U_base = clip_actions_jit(U_base)
        xs = jnp.array([state_init.pipeline_state])
        collision, result = 0, 0
        state = state_init
        max_distances = jnp.linalg.norm(state.pipeline_state.reshape(args.Nagent, -1) - env.xg.reshape(args.Nagent, -1), axis=1)

        for t in range(U_base.shape[0]):
            state = step_env_jit(state=state,
                                 xg=env.xg,
                                 action=U_base[t],
                                 max_distances=max_distances,
                                 penalty_weight=carry_cfg.penalty_weight,
                                 use_mask=carry_cfg.use_mask,
                                 margin_factor=carry_cfg.margin_factor,
                                 dt=carry_cfg.dt)
            xs = jnp.concatenate([xs, state.pipeline_state[None]], axis=0)

            if not jnp.all(state.collision == 0):
                collision = 1
            result = int(jnp.count_nonzero(state.mask)) / int(state.mask.size)

        if collision: result = -1.0
        print("Total number of deformations:", num_deforms)
        print("Success rate:", result)

        if args.save_images:
            filename = os.path.join(path, f"d4orm_{args.env_name}_movement.gif")
            filename2 = os.path.join(path, f"d4orm_{args.env_name}_trajectory.png")
            env.render_gif(xs, filename, filename2)
            if args.env_name in {"multi3dholo"}:
                env.render_gif_interactive(xs)

        rewss_final, _ , masks, _ = rollout_env_jit(state=state_init,
                                                    xg=env.xg,
                                                    us=U_base,
                                                    penalty_weight=carry_cfg.penalty_weight,
                                                    use_mask=carry_cfg.use_mask,
                                                    margin_factor=carry_cfg.margin_factor,
                                                    dt=carry_cfg.dt)
        if result == 1.0:
            avg_steps_to_goal = jnp.sum(masks == 0) // args.Nagent
            max_steps_to_goal = jnp.max(jnp.argmax(masks, axis=0))
        else:
            avg_steps_to_goal = -1
            max_steps_to_goal = -1
        rew_final = rewss_final[:args.Hsample].mean()

        print("Avergae flow time:", avg_steps_to_goal)
        print("Max flow time:", max_steps_to_goal)
        print("Reward:", rew_final)

        def encode_float(val):
            if isinstance(val, float):
                s = f"{val:.0e}"  # always scientific notation, e.g., 4.0e-01
                s = s.replace('e-', 'em').replace('e+', 'ep')
                return s
            return str(val)

        def encode_args_for_filename(args: Args):
            keys = ['Nagent', 'Nsample', 'Hsample', 'Ndiffuse', 'temp_sample', 'beta1', 'betaT']
            return "_".join(encode_float(getattr(args, k)) for k in keys)

        if args.save_data:
            filename = os.path.join(path, f"runtime_data/{encode_args_for_filename(args)}.txt")
            os.makedirs(os.path.dirname(filename), exist_ok=True)  # Ensure the directory exists
            with open(filename, 'a') as f:
                f.write(f"{result}, {elapsed_time:.6f}, {num_deforms}, {rew_final}, {avg_steps_to_goal}, {max_steps_to_goal}\n")

    return rew_final


if __name__ == "__main__":
    run_diffusion(args=tyro.cli(Args))
