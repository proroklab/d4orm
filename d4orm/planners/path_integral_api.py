"""MPPI and CEM planners using Gaussian trajectory sampling."""

import math

import jax
import jax.numpy as jnp

from d4orm.planners import core


class PathIntegralPlanner(core.Planner):
    """Shared Gaussian sampling loop for MPPI and CEM."""

    def _initial_sampling_deviation(self, actions):
        """Initializes MPPI and CEM sampling with unit standard deviation."""
        return jnp.full_like(actions, 1.0)

    def _update_distribution(self, rewards, candidates, sample_std):
        raise NotImplementedError

    def _optimize_cycle(
        self,
        random_key,
        actions,
        sample_std,
        initial_state,
        goals,
        reward_groups,
        active_agents,
        penalty_weights,
    ):
        del (
            reward_groups,
            active_agents,
        )  # Optimize all agents jointly.

        def update(carry, unused_step):
            del unused_step  # Fixed-length scan needs no step index.
            sample_key, mean, deviation = carry

            # Sample candidate control sequences from the current distribution.
            sample_key, noise_key = jax.random.split(sample_key)
            candidates = mean + deviation * jax.random.normal(
                noise_key, (self.config.num_samples,) + actions.shape
            )

            # Compute each candidate's mean reward across agents.
            rewards = self.rollout.compute_rewards(
                initial_state, goals, candidates, penalty_weights
            ).mean(-1)

            # Update the sampling distribution from candidate rewards.
            mean, deviation = self._update_distribution(
                rewards, candidates, deviation
            )
            return (sample_key, mean, deviation), None

        result, _ = jax.lax.scan(
            update,
            (random_key, actions, sample_std),
            xs=None,
            length=self.config.num_steps,
        )
        return result


class MPPIPlanner(PathIntegralPlanner):
    """MPPI with standardized reward softmax and fixed sampling variance."""

    def _update_distribution(self, rewards, candidates, sample_std):
        weights = core.reward_weights(rewards, self.config.temperature)
        return jnp.einsum("s,sth->th", weights, candidates), sample_std


class CEMPlanner(PathIntegralPlanner):
    """CEM with elite mean and per-control variance."""

    def _update_distribution(self, rewards, candidates, sample_std):
        del sample_std
        num_elites = min(
            self.config.num_samples,
            max(
                2,
                math.ceil(0.05 * self.config.num_samples),
            ),
        )
        _, elite_indices = jax.lax.top_k(rewards, num_elites)
        elites = candidates[elite_indices]

        return elites.mean(axis=0), jnp.maximum(elites.std(axis=0), 0.1)
