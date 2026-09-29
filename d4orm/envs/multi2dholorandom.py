"""Seeded random starts and grid goals, following the reference's mode 3."""

import math

import jax.numpy as jnp
import numpy as np

from d4orm.envs import multi2dholo


def sample_starts(
    goals: np.ndarray,
    seed: int,
    side: float = 10.0,
    separation: float = 0.9,
    min_distance: float = 5.0,
    max_distance: float = 6.0,
) -> np.ndarray:
    """Samples separated starts with bounded distances to corresponding goals.

    Uses up to 20 restarts and 1,000 candidate positions per robot per restart.
    Initialization runs on the host; planning and rollouts remain JAX compiled.

    Args:
        goals: Goal positions shaped (num_agents, 2).
        seed: NumPy random seed for reproducibility.
        side: Square workspace side length.
        separation: Minimum distance between starting positions.
        min_distance: Minimum start-to-goal distance.
        max_distance: Maximum start-to-goal distance.

    Returns:
        Starting positions shaped (num_agents, 2).

    Raises:
        ValueError: If sampling exceeds its retry budget.
    """
    random = np.random.default_rng(seed)
    for _ in range(20):
        starts = []
        for goal in goals:
            candidates = random.uniform(-side / 2, side / 2, (1000, 2))
            distances = np.linalg.norm(candidates - goal, axis=-1)
            valid = (distances >= min_distance) & (distances <= max_distance)
            if starts:
                pairwise = candidates[:, None, :] - np.asarray(starts)[None]
                valid &= np.all(
                    np.sum(pairwise**2, axis=-1) >= separation**2, axis=1
                )
            indices = np.flatnonzero(valid)
            if indices.size == 0:
                break
            starts.append(candidates[indices[0]])
        if len(starts) == len(goals):
            return np.asarray(starts)
    raise ValueError(
        f"Could not place {len(goals)} robots with these constraints "
        "after 20 attempts; use fewer robots or a different seed."
    )


class Multi2dHoloRandom(multi2dholo.Multi2dHolo):
    """Random starts in a 10 m square and deterministic grid goals."""

    def __init__(self, num_agents: int, seed: int = 0):
        super().__init__(num_agents)
        self.diameter = 10.0
        grid_size = math.ceil(math.sqrt(num_agents))
        coordinates = np.linspace(
            -self.diameter / 2, self.diameter / 2, grid_size
        )
        grid_x, grid_y = np.meshgrid(coordinates, coordinates)
        goals = np.stack((grid_x.ravel(), grid_y.ravel()), axis=-1)[:num_agents]
        if grid_size > 1 and self.diameter / (grid_size - 1) <= (
            2 * self.agent_radius + self.safe_margin
        ):
            raise ValueError("Too many robots for collision-free grid goals.")
        starts = sample_starts(goals, seed, separation=6 * self.agent_radius)
        velocities = np.zeros((num_agents, 2))
        self.x0 = jnp.asarray(
            np.concatenate((starts, velocities), axis=1), dtype=jnp.float32
        ).flatten()
        self.xg = jnp.asarray(
            np.concatenate((goals, velocities), axis=1), dtype=jnp.float32
        ).flatten()
        self.lim = self.diameter / 2 + 1
        self.max_distance = jnp.linalg.norm(
            jnp.asarray(starts - goals), axis=-1
        )
