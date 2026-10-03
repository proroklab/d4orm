"""Top-k distributed D4ORM with independent, vectorized denoising nodes."""

import dataclasses

import jax
import jax.numpy as jnp

from d4orm.planners import core, diffusion


def select_top_k(random_key, controls, rewards, goal_masks, collisions, k):
    """Resamples node base controls from top-k plans and selects the best plan.

    Candidates are ranked by mean robot reward. Each node independently chooses
    a top-k candidate uniformly with replacement. A successful candidate takes
    precedence for the returned plan, even if unsuccessful rewards are higher.
    """
    scores = rewards.mean(axis=-1)
    successful = jnp.all(goal_masks[:, -1], axis=-1) & ~jnp.any(
        collisions, axis=(1, 2)
    )
    eligible_scores = jnp.where(
        jnp.any(successful), jnp.where(successful, scores, -jnp.inf), scores
    )
    best_index = jnp.argmax(eligible_scores)
    _, top_k_indices = jax.lax.top_k(scores, k)
    selected_candidate_indices = jax.random.randint(
        random_key, (controls.shape[0],), minval=0, maxval=k
    )
    return controls[top_k_indices[selected_candidate_indices]], controls[best_index]


class DistributedD4ORMPlanner(core.Planner):
    """Exchanges top-k plans between full joint denoising iterations.

    Computational nodes use JAX vmap, without explicit device sharding or
    network communication. num_samples is the total per-step budget
    and must divide evenly among nodes. Each planning call starts a fresh
    population, initialized from the same supplied or default controls.
    """

    def __init__(
        self,
        rollout: core.Rollout,
        config: core.PlannerConfig = core.PlannerConfig(),
    ):
        num_nodes = (
            rollout.num_agents
            if config.num_workers is None
            else config.num_workers
        )
        top_k = max(1, num_nodes // 4) if config.top_k is None else config.top_k
        if top_k > num_nodes:
            raise ValueError("top_k must not exceed num_workers.")
        if config.num_samples % num_nodes or config.num_samples < num_nodes:
            raise ValueError(
                "num_samples must be at least num_workers and divide evenly "
                "among nodes."
            )
        config = dataclasses.replace(config, num_workers=num_nodes, top_k=top_k)
        super().__init__(rollout, config)
        self._node_planner = diffusion.D4ORMPlanner(
            rollout,
            dataclasses.replace(
                config, num_samples=config.num_samples // num_nodes
            ),
        )
        self._compiled_iteration = jax.jit(self._optimize_iteration)

    def _initial_optimizer_state(self, controls, *, warm_start):
        del warm_start
        return jnp.broadcast_to(
            controls, (self.config.num_workers,) + controls.shape
        )

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
            penalty_weights,
        )

    def _optimize_iteration(
        self,
        random_key,
        controls,
        node_base_controls,
        initial_state,
        goals,
        penalty_weights,
    ):
        # Node base controls carry the full population between iterations.
        del controls
        random_key, nodes_key, selection_key = jax.random.split(random_key, 3)
        node_keys = jax.random.split(nodes_key, self.config.num_workers)

        def denoise_node(node_key, base_controls):
            # Reuse the internal diffusion kernel within the node vmap.
            # pylint: disable-next=protected-access
            _, refined_controls, _ = self._node_planner._optimize_iteration(
                node_key,
                base_controls,
                None,
                initial_state,
                goals,
                None,
                None,
                penalty_weights,
            )
            return refined_controls

        node_candidate_controls = jax.vmap(denoise_node)(
            node_keys, node_base_controls
        )
        rewards, _, goal_masks, collisions = jax.vmap(
            lambda candidate: self.rollout.evaluate(
                initial_state, goals, candidate, penalty_weights
            )
        )(node_candidate_controls)
        node_candidate_controls = jax.vmap(
            core.mask_goal_controls, in_axes=(0, None, 0)
        )(node_candidate_controls, initial_state.mask, goal_masks)
        next_node_base_controls, best = select_top_k(
            selection_key,
            node_candidate_controls,
            rewards,
            goal_masks,
            collisions,
            self.config.top_k,
        )
        return random_key, best, next_node_base_controls
