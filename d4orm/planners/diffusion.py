"""Vanilla D4ORM using compiled, reward-weighted diffusion deformations."""

import jax
import jax.numpy as jnp

from d4orm.planners import core


def diffusion_schedule(
    config: core.PlannerConfig,
) -> tuple[jax.Array, jax.Array]:
    """Interpolates a 100-step noise schedule to the requested step count."""
    reference_betas = jnp.linspace(config.beta_start, config.beta_end, 100)
    reference_alphas = jnp.concatenate(
        (jnp.ones(1), jnp.cumprod(1.0 - reference_betas))
    )

    cumulative_alphas = jnp.interp(
        jnp.linspace(0.0, 1.0, config.num_steps + 1),
        jnp.linspace(0.0, 1.0, 101),
        reference_alphas,
    )

    return cumulative_alphas, jnp.sqrt(1.0 / cumulative_alphas - 1.0)


class D4ORMPlanner(core.Planner):
    """Joint denoising with a fresh deformation around each previous plan."""

    def __init__(
        self,
        rollout: core.Rollout,
        config: core.PlannerConfig = core.PlannerConfig(),
    ):
        super().__init__(rollout, config)
        self._cumulative_alphas, self._noise_scales = diffusion_schedule(config)
        self._alphas = jnp.concatenate(
            (
                jnp.ones_like(self._cumulative_alphas[:1]),
                self._cumulative_alphas[1:] / self._cumulative_alphas[:-1],
            )
        )
        self._compiled_iteration = jax.jit(self._optimize_iteration)

    def _run_iteration(
        self,
        iteration,
        random_key,
        controls,
        optimizer_state,
        initial_state,
        goals,
        penalty_weights,
    ):
        del iteration
        return self._compiled_iteration(
            random_key,
            controls,
            optimizer_state,
            initial_state,
            goals,
            None,
            None,
            penalty_weights,
        )

    def _optimize_iteration(
        self,
        random_key,
        controls,
        optimizer_state,
        initial_state,
        goals,
        group_membership,
        active_agents,
        penalty_weights,
        include_robot_collisions=True,
    ):
        # Restrict deformations to agents selected for replanning.
        control_mask = (
            jnp.ones(self.rollout.control_size, dtype=bool)
            if active_agents is None
            else jnp.repeat(
                active_agents,
                self.rollout.control_size // self.rollout.num_agents,
            )
        )

        def denoise(carry, denoising_step):
            sample_key, deformation = carry

            # Read the diffusion coefficients for this denoising step.
            alpha_bar = self._cumulative_alphas[denoising_step]
            alpha = self._alphas[denoising_step]
            noise_variance = 1.0 - alpha_bar

            # Sample deformations at the current diffusion noise level.
            sample_key, noise_key = jax.random.split(sample_key)
            candidate_shape = (self.config.num_samples,) + controls.shape
            noise = jax.random.normal(noise_key, candidate_shape)
            mean = deformation / jnp.sqrt(alpha_bar)
            candidates = (
                mean + noise * self._noise_scales[denoising_step]
            ) * control_mask

            # Evaluate the perturbed plans using rollout rewards.
            rewards = self.rollout.compute_rewards(
                initial_state,
                goals,
                controls + candidates,
                penalty_weights,
                include_robot_collisions,
            )

            # Perform score-ascent
            mean_deformation = self._weighted_deformation(
                rewards, candidates, group_membership
            )
            score = (
                -deformation / noise_variance
                + jnp.sqrt(alpha_bar) / noise_variance * mean_deformation
            )
            deformation = (deformation + noise_variance * score) / jnp.sqrt(alpha)

            return (sample_key, deformation), None

        # Denoise from high to low noise and apply the final deformation.
        (random_key, deformation), _ = jax.lax.scan(
            denoise,
            (random_key, jnp.zeros_like(controls)),
            jnp.arange(self.config.num_steps, 0, -1),
        )

        return random_key, controls + deformation, optimizer_state

    def _weighted_deformation(self, rewards, candidates, group_membership):
        del group_membership  # Vanilla D4ORM uses one joint reward.
        weights = core.reward_weights(
            rewards.mean(axis=-1), self.config.temperature
        )
        return jnp.einsum("s,sth->th", weights, candidates)
