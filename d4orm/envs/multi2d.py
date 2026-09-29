"""Differential-drive robots with bounded angular velocity and acceleration."""

import functools

import jax
import jax.numpy as jnp

from d4orm.envs import multibase


class Multi2d(multibase.MultiBase):
    """Robots with state (x, y, heading, speed) and two control inputs."""

    def __init__(self, num_agents: int):
        super().__init__(num_agents)
        self.action_dim_agent = 2
        self.obsv_dim_agent = 4
        self.pos_dim_agent = 2
        self.diameter = 5.0
        self.agent_radius = 0.15
        self.safe_margin = 0.05
        self.stop_distance = self.agent_radius / 2
        self.stop_velocity = float("inf")
        self.max_angular_speed = jnp.pi / 2
        self.max_speed = 1.0
        self.max_acceleration = 1.0
        initial_states, goals = self.generate_positions(
            self.diameter, num_agents
        )
        directions = goals[:, :2] - initial_states[:, :2]
        headings = jnp.arctan2(directions[:, 1], directions[:, 0])
        self.x0 = initial_states.at[:, 2].set(headings).flatten()
        self.xg = goals.flatten()
        self.lim = self.diameter / 2 + 1
        self.max_distance = self.diameter

    def direct_path_actions(
        self, initial_state: multibase.State, goals: jax.Array, horizon: int
    ) -> jax.Array:
        """Repeats forward acceleration with zero steering, as in the reference.

        Default reset headings already face the goals. Custom initial headings
        are preserved, so the seed follows those headings without steering.
        """
        del initial_state, goals  # The reference seed assumes aligned headings.
        robot_actions = jnp.array([0.0, self.max_acceleration])
        return jnp.tile(robot_actions, (horizon, self.num_agents))

    @functools.partial(jax.jit, static_argnums=(0,))
    def agent_dynamics(self, x: jax.Array, u: jax.Array) -> jax.Array:
        """Returns velocity and bounded control derivatives."""
        heading, speed = x[2], x[3]
        angular_speed = jnp.clip(
            u[0], -self.max_angular_speed, self.max_angular_speed
        )
        acceleration = jnp.clip(
            u[1], -self.max_acceleration, self.max_acceleration
        )
        return jnp.array(
            [
                speed * jnp.cos(heading),
                speed * jnp.sin(heading),
                angular_speed,
                acceleration,
            ]
        )

    @functools.partial(jax.jit, static_argnums=(0,))
    def clip_actions(self, traj: jax.Array, factor: float = 1) -> jax.Array:
        """Clips joint actions to the scaled actuation limits."""
        actions = traj.reshape(-1, self.num_agents, self.action_dim_agent)
        limits = (
            jnp.array([self.max_angular_speed, self.max_acceleration]) * factor
        )
        return jnp.clip(actions, -limits, limits).reshape(-1, self.action_size)

    def clip_velocity(self, x: jax.Array) -> jax.Array:
        """Clips a robot velocity to the speed limit."""
        return x.at[3].set(jnp.clip(x[3], -self.max_speed, self.max_speed))

    def get_current_velocity(self, q: jax.Array) -> jax.Array:
        """Returns scalar speeds for the robot state matrix."""
        return jnp.abs(q[:, 3])

    @property
    def action_size(self) -> int:
        """Number of components in the joint action vector."""
        return self.action_dim_agent * self.num_agents

    def get_heading_line(self, state, position, agent_idx):
        """Returns a short line indicating robot heading."""
        del agent_idx  # All robots use the same footprint.
        heading = state[2]
        end_x = position[0] + self.agent_radius * jnp.cos(heading)
        end_y = position[1] + self.agent_radius * jnp.sin(heading)
        return [position[0], end_x], [position[1], end_y]
