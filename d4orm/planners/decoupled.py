"""D4ORM-D: decoupled denoising with collision-connected reward groups."""

import functools

import jax
import jax.numpy as jnp
import numpy as np

from d4orm.planners import core, diffusion


@jax.jit
def connected_groups(adjacency: jax.Array) -> jax.Array:
    """Returns component membership, including singleton diagonal entries."""

    # Include each agent in its own group, then propagate connectivity.
    reachable = adjacency.astype(bool) | jnp.eye(adjacency.shape[0], dtype=bool)

    def connect(index, groups):
        return groups | (groups[:, index, None] & groups[None, index, :])

    return jax.lax.fori_loop(0, adjacency.shape[0], connect, reachable)


@jax.jit
def augment_collision_graph(
    current_collisions: jax.Array,
    previous_groups: jax.Array,
    previous_collisions: jax.Array,
) -> jax.Array:
    """Reconnects boundary robots to collision-free previous direct neighbors.

    Args:
        current_collisions: Current symmetric raw collision adjacency matrix.
        previous_groups: Previous augmented component membership matrix.
        previous_collisions: Previous raw collision adjacency matrix.

    Returns:
        Symmetric augmented adjacency. Reconnection uses only current raw
        collisions, so restored edges cannot trigger further reconnections.
    """

    # Find agents colliding outside their previous group.
    same_previous_group = previous_groups.astype(bool)
    outside_collision = jnp.any(
        current_collisions & ~same_previous_group, axis=1
    )
    collision_free = ~jnp.any(current_collisions, axis=1)

    # Reconnect their previous direct neighbors that are now collision-free.
    restored_edges = (
        outside_collision[:, None]
        & previous_collisions
        & same_previous_group
        & collision_free[None, :]
    )
    return current_collisions | restored_edges | restored_edges.T


@jax.jit
def failure_identifiers(
    groups: jax.Array, active_agents: jax.Array
) -> jax.Array:
    """Encodes group IDs, -1 for isolated failures, and 0 for success."""
    representatives = jnp.argmax(groups.astype(bool), axis=1)
    multiple_members = groups.sum(axis=1) > 1
    roots = (representatives == jnp.arange(groups.shape[0])) & multiple_members
    component_ids = jnp.cumsum(roots)[representatives]
    return jnp.where(
        multiple_members, component_ids, jnp.where(active_agents, -1, 0)
    )


class D4ORMDPlanner(diffusion.D4ORMPlanner):
    """Replans failed agents with sample weights shared within collision groups.

    Robots colliding outside their previous group reconnect to their previous
    direct neighbors that are currently collision-free. Components of this
    augmented graph share weights. Successful agents' actions stay fixed while
    their trajectories still participate in collision costs.
    """

    def __init__(
        self,
        rollout: core.Rollout,
        config: core.PlannerConfig = core.PlannerConfig(),
    ):
        super().__init__(rollout, config)

        self._compiled_probe = jax.jit(
            functools.partial(
                self._optimize_cycle, include_robot_collisions=False
            )
        )
        self._compiled_groups = jax.jit(self._find_groups)

    def _uses_independent_probe(self, initial_actions):
        return initial_actions is None and not self.config.direct_path_init

    def _initial_adaptation_state(self):
        zeros = jnp.zeros(self.rollout.num_agents, dtype=jnp.int32)
        return zeros, zeros

    @functools.partial(jax.jit, static_argnums=(0,))
    def _prepare_cycle(
        self,
        iteration,
        actions,
        collisions,
        reward_groups,
        active_agents,
        penalty_weights,
        adaptation_state,
        initialization_probe,
    ):
        """Resets probes, adapts penalties, and clips persistent failures."""
        previous_ids, failure_counts = adaptation_state

        # Restart controls from zero after the initialization probe.
        actions = jnp.where(
            initialization_probe, jnp.zeros_like(actions), actions
        )

        # Increase penalties for colliding agents up to 3; reset others to 1.
        collided = jnp.any(collisions, axis=0)
        updated_penalties = jnp.where(
            collided,
            jnp.minimum(penalty_weights + 0.5, 3.0),
            jnp.ones_like(penalty_weights),
        )

        # The first evaluation establishes history without escalating costs.
        penalty_weights = jnp.where(
            iteration > 0, updated_penalties, penalty_weights
        )

        # Count consecutive failures with the same group identifier.
        failure_ids = failure_identifiers(reward_groups, active_agents)
        failure_counts = jnp.where(
            (failure_ids != 0) & (failure_ids == previous_ids),
            failure_counts + 1,
            0,
        )

        # Clip persistently failing agents' controls and reset their counters.
        needs_clipping = failure_counts > 5
        action_mask = jnp.repeat(
            needs_clipping, self.rollout.action_size // self.rollout.num_agents
        )
        actions = jnp.where(
            action_mask[None, :],
            self.rollout.environment.clip_actions(actions),
            actions,
        )
        failure_counts = jnp.where(needs_clipping, 0, failure_counts)
        return actions, penalty_weights, (failure_ids, failure_counts)

    def _update_groups(
        self,
        states,
        goal_masks,
        collisions,
        previous_groups,
        previous_collisions,
    ):
        return self._compiled_groups(
            states, goal_masks, collisions, previous_groups, previous_collisions
        )

    def _find_groups(
        self,
        states,
        goal_masks,
        collisions,
        previous_groups,
        previous_collisions,
    ):
        # Combine robot-to-robot collisions across the trajectory into edges.
        environment = self.rollout.environment
        robot_states = states.reshape(
            states.shape[0], self.rollout.num_agents, -1
        )
        adjacency = jnp.any(
            environment.collision_matrix(robot_states),
            axis=0,
        )

        # Replan agents with collisions or an unreached goal.
        active_agents = jnp.any(collisions, axis=0) | ~goal_masks[-1].astype(
            bool
        )

        # Restore neighbor links and form connected reward groups.
        augmented_graph = augment_collision_graph(
            adjacency, previous_groups, previous_collisions
        )
        groups = connected_groups(augmented_graph)

        return groups.astype(jnp.float32), active_agents, adjacency

    def _print_iteration(
        self,
        iteration,
        rewards,
        goal_masks,
        collisions,
        reward_groups,
        active_agents,
    ):
        """Prints robot status and groups for the evaluated trajectory."""
        groups, masks, collision_flags, active = jax.device_get(
            (reward_groups, goal_masks, collisions, active_agents)
        )

        # The smallest one-based robot ID represents each component.
        group_ids = np.argmax(groups.astype(bool), axis=1) + 1
        collision_steps = np.count_nonzero(collision_flags, axis=0)

        label = "Initial plan" if iteration == 0 else f"Iteration {iteration}"
        print(f"\nD4ORM-D | {label} | reward={float(rewards.mean()):.4f}")
        print("IDs are 1-based; Group is - for robots needing no replanning.")
        print(
            f"{'Robot':>7} {'Group':>7} {'Collision':>11} "
            f"{'Collision steps':>17} {'At goal':>9} {'Needs replanning':>18}"
        )

        for robot_index in range(self.rollout.num_agents):
            collided = "yes" if collision_steps[robot_index] else "no"
            at_goal = "yes" if masks[-1, robot_index] else "no"
            replan = "yes" if active[robot_index] else "no"
            group_label = (
                str(group_ids[robot_index]) if active[robot_index] else "-"
            )
            print(
                f"{robot_index + 1:>7} {group_label:>7} "
                f"{collided:>11} {collision_steps[robot_index]:>17} "
                f"{at_goal:>9} {replan:>18}"
            )

    def _weighted_deformation(self, rewards, candidates, reward_groups):
        # Sum rewards within each group to obtain shared sample weights.
        group_rewards = rewards @ reward_groups.T
        weights = core.reward_weights(group_rewards, self.config.temperature)
        agent_candidates = candidates.reshape(
            self.config.num_samples,
            self.config.horizon,
            self.rollout.num_agents,
            -1,
        )

        # Average each agent's candidate deformations using its group's weights.
        deformation = jnp.einsum("sa,stad->tad", weights, agent_candidates)
        return deformation.reshape(
            self.config.horizon, self.rollout.action_size
        )
