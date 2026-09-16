"""Shared acceleration-controlled holonomic robot dynamics."""

import functools

import jax
import jax.numpy as jnp

from d4orm.envs import multibase


def limit_norm(vectors: jax.Array, maximum_norm: float) -> jax.Array:
    """Scales vectors along the last axis without dividing by zero."""
    norms = jnp.linalg.norm(vectors, axis=-1, keepdims=True)
    return vectors * jnp.minimum(1.0, maximum_norm / jnp.maximum(norms, 1e-8))


class HolonomicEnvironment(multibase.MultiBase):
    """Common double-integrator dynamics for two or three spatial dimensions."""

    def __init__(self, num_agents: int, position_dim: int):
        super().__init__(num_agents)
        self.action_dim_agent = position_dim
        self.obsv_dim_agent = 2 * position_dim
        self.pos_dim_agent = position_dim
        self.diameter = 5.0
        self.agent_radius = 0.15
        self.safe_margin = 0.05
        self.stop_distance = self.agent_radius / 2
        self.stop_velocity = float("inf")
        self.max_speed = 1.0
        self.max_acceleration = 1.0
        initial_states, goals = self.generate_positions(
            self.diameter, num_agents
        )
        self.x0 = initial_states.flatten()
        self.xg = goals.flatten()
        self.lim = self.diameter / 2 + 1
        self.max_distance = self.diameter

    def direct_path_actions(
        self, initial_state: multibase.State, goals: jax.Array, horizon: int
    ) -> jax.Array:
        """Repeats maximum acceleration toward each robot's goal position."""
        positions = initial_state.pipeline_state.reshape(self.num_agents, -1)
        goal_states = goals.reshape(self.num_agents, -1)
        directions = (
            goal_states[:, : self.pos_dim_agent]
            - positions[:, : self.pos_dim_agent]
        )
        distances = jnp.linalg.norm(directions, axis=-1, keepdims=True)
        accelerations = self.max_acceleration * directions / (distances + 1e-8)
        return jnp.tile(accelerations.flatten(), (horizon, 1))

    @functools.partial(jax.jit, static_argnums=(0,))
    def agent_dynamics(self, x: jax.Array, u: jax.Array) -> jax.Array:
        """Returns velocity and bounded control derivatives."""
        return jnp.concatenate(
            (x[self.pos_dim_agent :], limit_norm(u, self.max_acceleration))
        )

    @functools.partial(jax.jit, static_argnums=(0,))
    def clip_actions(self, traj: jax.Array, factor: float = 1) -> jax.Array:
        """Clips joint actions to the scaled actuation limits."""
        actions = traj.reshape(-1, self.num_agents, self.action_dim_agent)
        return limit_norm(actions, self.max_acceleration * factor).reshape(
            -1, self.action_size
        )

    def clip_velocity(self, x: jax.Array) -> jax.Array:
        """Clips a robot velocity to the speed limit."""
        return x.at[self.pos_dim_agent :].set(
            limit_norm(x[self.pos_dim_agent :], self.max_speed)
        )

    def get_current_velocity(self, q: jax.Array) -> jax.Array:
        """Returns scalar speeds for the robot state matrix."""
        return jnp.linalg.norm(q[:, self.pos_dim_agent :], axis=-1)

    @property
    def action_size(self) -> int:
        """Number of components in the joint action vector."""
        return self.action_dim_agent * self.num_agents
