"""Alternating holonomic and differential-drive robots, as in mbd-multi."""

import functools

import jax
import jax.numpy as jnp

from d4orm.envs import holonomic, multi2d


class Multi2DHeter(multi2d.Multi2d):
    """Mixed robots sharing four state entries and two control entries.

    Even robots use (x, y, vx, vy) and Cartesian acceleration. Odd robots
    use (x, y, heading, speed) and angular velocity / forward acceleration.
    Collision geometry and reward semantics use the shared MultiBase rollout.
    """

    def __init__(self, num_agents: int):
        super().__init__(num_agents)
        self.agent_types = jnp.arange(num_agents) % 2
        self.holo_mv = 1.0
        self.holo_ma = 1.0
        self.diff_mav = jnp.pi / 2
        self.diff_mlv = 1.0
        self.diff_mla = 2.0
        states = self.x0.reshape(num_agents, 4)
        self.x0 = (
            states.at[:, 2]
            .set(jnp.where(self.agent_types == 1, states[:, 2], 0.0))
            .flatten()
        )

    def agent_dynamics(self, x, u, agent_type):
        """Evaluates the appropriate dynamics for a single robot."""
        return jax.lax.cond(
            agent_type == 0,
            lambda: jnp.concatenate(
                (x[2:], holonomic.limit_norm(u, self.holo_ma))
            ),
            lambda: jnp.array(
                [
                    x[3] * jnp.cos(x[2]),
                    x[3] * jnp.sin(x[2]),
                    jnp.clip(u[0], -self.diff_mav, self.diff_mav),
                    jnp.clip(u[1], -self.diff_mla, self.diff_mla),
                ]
            ),
        )

    def clip_velocity(self, x, agent_type):
        """Enforces the type-specific speed limit."""
        return jax.lax.cond(
            agent_type == 0,
            lambda: x.at[2:].set(holonomic.limit_norm(x[2:], self.holo_mv)),
            lambda: x.at[3].set(jnp.clip(x[3], -self.diff_mlv, self.diff_mlv)),
        )

    @functools.partial(jax.jit, static_argnums=(0,))
    def rk4(self, x, u, dt, agent_id):
        """Integrates one robot with its own dynamics and velocity limits."""
        agent_type = self.agent_types[agent_id]
        k1 = self.agent_dynamics(x, u, agent_type)
        k2 = self.agent_dynamics(x + dt / 2 * k1, u, agent_type)
        k3 = self.agent_dynamics(x + dt / 2 * k2, u, agent_type)
        k4 = self.agent_dynamics(x + dt * k3, u, agent_type)
        return self.clip_velocity(
            x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4), agent_type
        )

    def integrate_states(self, states, actions, dt):
        """Dispatches integration by robot index within the shared rollout."""
        return jax.vmap(self.rk4, in_axes=(0, 0, None, 0))(
            states, actions, dt, jnp.arange(self.num_agents)
        )

    def get_current_velocity(self, q):
        """Returns speed magnitudes under each robot's state convention."""
        return jnp.where(
            self.agent_types == 0,
            jnp.linalg.norm(q[:, 2:], axis=-1),
            jnp.abs(q[:, 3]),
        )

    @functools.partial(jax.jit, static_argnums=(0,))
    def clip_actions(self, traj, factor=1):
        """Clips Cartesian accelerations or steering / acceleration controls."""
        actions = traj.reshape(-1, self.num_agents, 2)
        holo = holonomic.limit_norm(actions, self.holo_ma * factor)
        limits = jnp.array([self.diff_mav, self.diff_mla]) * factor
        differential = jnp.clip(actions, -limits, limits)
        return jnp.where(
            self.agent_types[None, :, None] == 0, holo, differential
        ).reshape(-1, self.action_size)

    def direct_path_actions(self, initial_state, goals, horizon):
        """Builds type-specific direct controls using the supplied start/goals."""
        starts = initial_state.pipeline_state.reshape(self.num_agents, 4)
        directions = goals.reshape(self.num_agents, 4)[:, :2] - starts[:, :2]
        holo = (
            directions
            / (jnp.linalg.norm(directions, axis=-1, keepdims=True) + 1e-8)
            * self.holo_ma
        )
        actions = jnp.where(
            self.agent_types[:, None] == 0,
            holo,
            jnp.array([0.0, self.diff_mla]),
        )
        return jnp.tile(actions.flatten(), (horizon, 1))

    def get_heading_line(self, state, position, agent_idx):
        """Draws headings only for differential-drive robots."""
        if int(self.agent_types[agent_idx]) == 0:
            return [], []
        return super().get_heading_line(state, position, agent_idx)
