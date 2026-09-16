"""Shared command-line runner for all trajectory planners."""

import argparse
import dataclasses
import json
import pathlib
import time

import jax
import jax.numpy as jnp
import numpy as np

from d4orm import envs
from d4orm.planners import core, decoupled, diffusion, path_integral_api

PLANNERS = {
    "d4orm": diffusion.D4ORMPlanner,
    "d4orm-d": decoupled.D4ORMDPlanner,
    "mppi": path_integral_api.MPPIPlanner,
    "cem": path_integral_api.CEMPlanner,
}


def make_parser(default_method: str = "d4orm") -> argparse.ArgumentParser:
    """Builds command-line options for planner selection and execution."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--method", choices=(*PLANNERS, "decouple"), default=default_method
    )
    parser.add_argument(
        "--env-name",
        "--env_name",
        default="multi2dholo",
        help=(
            "multi2d, multi2dheter, multi2dholo, multi3dholo, "
            "multi2dholo_obsX, or multi2dholo_random"
        ),
    )
    parser.add_argument("--num-agents", "--Nagent", type=int, default=16)
    add_planner_arguments(parser)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save-images", "--save_images", action="store_true")
    parser.add_argument("--save-data", "--save_data", action="store_true")
    parser.add_argument("--print-info", "--print_info", action="store_true")
    parser.add_argument(
        "--output-dir", type=pathlib.Path, default=pathlib.Path("results")
    )

    return parser


def add_planner_arguments(parser: argparse.ArgumentParser) -> None:
    """Adds shared optimization and rollout settings to a parser."""
    parser.add_argument("--num-samples", "--Nsample", type=int, default=1024)
    parser.add_argument("--horizon", "--Hsample", type=int, default=100)
    parser.add_argument("--num-steps", "--Ndiffuse", type=int, default=100)
    parser.add_argument(
        "--max-iterations", "--Niteration", type=int, default=50
    )
    parser.add_argument(
        "--temperature", "--temp_sample", type=float, default=0.3
    )
    parser.add_argument("--beta-start", "--beta1", type=float, default=1e-4)
    parser.add_argument("--beta-end", "--betaT", type=float, default=2e-2)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--dt-factor", "--dt_factor", type=float, default=1.0)
    parser.add_argument(
        "--direct-path-init",
        "--direct_path_init",
        action="store_true",
        help="Initialize controls with the environment's direct-path seed.",
    )


def configuration_from_args(
    args: argparse.Namespace,
) -> tuple[core.PlannerConfig, core.RolloutConfig]:
    """Builds validated planner and rollout configurations from CLI settings."""
    config = core.PlannerConfig(
        direct_path_init=args.direct_path_init,
        horizon=args.horizon,
        num_samples=args.num_samples,
        num_steps=args.num_steps,
        max_iterations=args.max_iterations,
        temperature=args.temperature,
        beta_start=args.beta_start,
        beta_end=args.beta_end,
    )

    rollout_config = core.RolloutConfig(
        dt=args.dt * args.dt_factor,
    )

    return config, rollout_config


def run(args: argparse.Namespace) -> core.PlanResult:
    """Runs a planner and optionally saves data and visualization."""

    # Build the environment, rollout, and selected planner.
    method = "d4orm-d" if args.method == "decouple" else args.method
    environment = envs.get_env(args.env_name, args.num_agents, seed=args.seed)
    config, rollout_config = configuration_from_args(args)
    rollout = core.Rollout(environment, rollout_config)
    planner = PLANNERS[method](rollout, config)

    # Reset the environment with a separate random key from planning.
    random_key, reset_key = jax.random.split(jax.random.PRNGKey(args.seed))
    initial_state = environment.reset(reset_key)

    # Compile planning kernels with a dedicated warm-up random key.
    warm_up_key = jax.random.fold_in(random_key, 1)
    print("Warming up JAX kernels...")
    warm_up_start = time.perf_counter()
    planner.warm_up(warm_up_key, initial_state, environment.xg)
    warm_up_elapsed = time.perf_counter() - warm_up_start
    print(f"Warm-up time: {warm_up_elapsed:.3f} seconds")

    # Time planning through device completion, excluding compilation warm-up.
    start_time = time.perf_counter()
    result = planner.plan(
        random_key, initial_state, environment.xg, print_info=args.print_info
    )
    jax.block_until_ready(
        (
            result.actions,
            result.states,
            result.goal_masks,
            result.collisions,
            result.rewards,
            result.random_key,
        )
    )
    elapsed = time.perf_counter() - start_time

    print(
        f"{method}: success={result.success}, iterations={result.iterations}, "
        f"reward={float(result.rewards.mean()):.4f}"
    )
    print(f"Planning time (excluding warm-up): {elapsed:.3f} seconds")
    if args.print_info:
        print(f"Reward history: {result.reward_history}")

    # Export trajectory data, run metadata, and requested visualizations.
    if args.save_images or args.save_data:
        output = args.output_dir / method / args.env_name
        output.mkdir(parents=True, exist_ok=True)
        stem = f"agents_{args.num_agents}_seed_{args.seed}"

        if args.save_data:
            np.savez_compressed(
                output / f"{stem}.npz",
                actions=np.asarray(result.actions),
                states=np.asarray(result.states),
                initial_state=np.asarray(environment.x0),
                goals=np.asarray(environment.xg),
                goal_masks=np.asarray(result.goal_masks),
                collisions=np.asarray(result.collisions),
                rewards=np.asarray(result.rewards),
                penalty_weights=np.asarray(result.penalty_weights),
                reward_history=result.reward_history,
                obstacle_centers=np.asarray(environment.obstacle_centers),
                obstacle_radii=np.asarray(environment.obstacle_radii),
            )

            metadata = {
                "method": method,
                "environment": args.env_name,
                "seed": args.seed,
                "num_agents": args.num_agents,
                "planner": dataclasses.asdict(config),
                "rollout": dataclasses.asdict(rollout_config),
                "success": result.success,
                "iterations": result.iterations,
                "elapsed_seconds": elapsed,
                "includes_compilation": False,
                "warm_up_seconds": warm_up_elapsed,
            }
            (output / f"{stem}.json").write_text(
                json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
            )

        if args.save_images:
            # Render the trajectory with a noninteractive plotting backend.
            import matplotlib  # pylint: disable=import-outside-toplevel

            matplotlib.use("Agg", force=True)
            trajectory = jnp.concatenate(
                (initial_state.pipeline_state[None], result.states), axis=0
            )
            environment.render_gif(
                trajectory,
                str(output / f"{stem}.gif"),
                str(output / f"{stem}.png"),
            )

    return result


def main(default_method: str = "d4orm") -> None:
    """Parses arguments and runs the selected planner."""
    parser = make_parser(default_method)
    args = parser.parse_args()

    try:
        run(args)
    except ValueError as error:
        parser.error(str(error))
