"""Spatial holonomic robots with starts distributed over a sphere."""

import jax.numpy as jnp

from d4orm.envs import holonomic


class Multi3dHolo(holonomic.HolonomicEnvironment):
    """Spatial robots with three position and three velocity components."""

    def __init__(self, num_agents: int):
        super().__init__(num_agents, position_dim=3)
        self.agent_radius_original = 0.15
        self.agent_radius = self.agent_radius_original * 2
        self.safe_margin = 0.05
        self.stop_distance = self.agent_radius_original / 2
        self.obstacle_centers = jnp.empty((0, 3))

    def generate_positions(self, diameter: float, num_agents: int):
        """Creates antipodal starts and goals using a Fibonacci sphere."""
        indices = jnp.arange(num_agents)
        polar_angles = jnp.arccos(1 - 2 * (indices + 0.5) / num_agents)
        azimuths = jnp.pi * (1 + 5**0.5) * indices
        radius = diameter / 2
        positions = jnp.stack(
            [
                radius * jnp.sin(polar_angles) * jnp.cos(azimuths),
                radius * jnp.sin(polar_angles) * jnp.sin(azimuths),
                radius * jnp.cos(polar_angles),
            ],
            axis=-1,
        )
        initial_states = jnp.concatenate(
            (positions, jnp.zeros_like(positions)), axis=-1
        )
        return initial_states, -initial_states
