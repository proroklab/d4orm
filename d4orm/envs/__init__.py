"""Multi-robot environments with a common rollout interface."""

import re

from d4orm.envs import (
    multi2d,
    multi2dheter,
    multi2dholo,
    multi2dholorandom,
    multi3dholo,
    multibase,
)

# Preserve the existing environment class imports.
Multi2DHeter = multi2dheter.Multi2DHeter
Multi2d = multi2d.Multi2d
Multi2dHolo = multi2dholo.Multi2dHolo
Multi3dHolo = multi3dholo.Multi3dHolo
Multi2dHoloRandom = multi2dholorandom.Multi2dHoloRandom


def get_env(
    env_name: str, num_agents: int, seed: int = 0
) -> multibase.MultiBase:
    """Creates an environment by name; raises ValueError for unknown names."""
    if env_name in ("multi2dholo_random", "multi2dholorandom"):
        return Multi2dHoloRandom(num_agents, seed)
    obstacle_match = re.fullmatch(r"multi2dholo_obs(\d+)", env_name)
    if obstacle_match:
        return Multi2dHolo(num_agents, int(obstacle_match.group(1)), seed)
    environments = {
        "multi2d": Multi2d,
        "multi2dheter": Multi2DHeter,
        "multi2dholo": Multi2dHolo,
        "multi3dholo": Multi3dHolo,
    }
    if env_name not in environments:
        raise ValueError(f"Unknown environment: {env_name}")
    return environments[env_name](num_agents)
