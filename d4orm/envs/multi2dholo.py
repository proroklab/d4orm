"""Planar holonomic robots with optional circular static obstacles."""

import jax
import jax.numpy as jnp

from d4orm.envs import holonomic


def obstacle_positions(num_obstacles: int, seed: int = 0) -> jax.Array:
    """Creates the reference layouts inside a unit-radius disk."""
    if num_obstacles == 0:
        return jnp.zeros((0, 2))
    if num_obstacles == 1:
        return jnp.zeros((1, 2))
    if num_obstacles <= 4:
        angles = jnp.arange(num_obstacles) * (2 * jnp.pi / num_obstacles)
        return jnp.stack((jnp.sin(angles), jnp.cos(angles)), axis=-1)
    radius_key, angle_key = jax.random.split(jax.random.PRNGKey(seed))
    radii = jnp.sqrt(jax.random.uniform(radius_key, (num_obstacles,)))
    angles = 2 * jnp.pi * jax.random.uniform(angle_key, (num_obstacles,))
    return radii[:, None] * jnp.stack(
        (jnp.cos(angles), jnp.sin(angles)), axis=-1
    )


class Multi2dHolo(holonomic.HolonomicEnvironment):
    """Planar robots with position and velocity states."""

    def __init__(self, num_agents: int, num_obstacles: int = 0, seed: int = 0):
        if not isinstance(num_obstacles, int) or num_obstacles < 0:
            raise ValueError("num_obstacles must be a nonnegative integer.")
        super().__init__(num_agents, position_dim=2)
        self.num_obstacles = num_obstacles
        self.obstacle_centers = obstacle_positions(num_obstacles, seed)
        radius = 0.5 if num_obstacles == 1 else 0.15
        self.obstacle_radii = jnp.full((num_obstacles,), radius)
