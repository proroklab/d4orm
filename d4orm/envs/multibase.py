"""Shared multi-robot dynamics, collision costs, rollouts, and rendering."""

import functools

import jax
import jax.numpy as jnp
from flax import struct


@struct.dataclass
class State:
    """Joint state, per-agent rewards, goal masks, and collision flags."""

    pipeline_state: jax.Array
    reward: jax.Array
    mask: jax.Array
    collision: jax.Array


class MultiBase:
    """Shared simulation for subclasses defining geometry and robot dynamics."""

    # Subclasses define geometry before using the shared dynamics.
    action_dim_agent: int
    obsv_dim_agent: int
    pos_dim_agent: int
    x0: jax.Array
    xg: jax.Array
    agent_radius: float
    safe_margin: float
    stop_distance: float
    stop_velocity: float
    lim: float

    def __init__(self, num_agents: int):
        if not isinstance(num_agents, int) or num_agents < 1:
            raise ValueError("num_agents must be a positive integer.")
        self.num_agents = num_agents
        self.offset = 1
        self.num_obstacles = 0
        self.obstacle_centers = jnp.zeros((0, 2))
        self.obstacle_radii = jnp.zeros((0,))

    def generate_positions(self, diameter: float, num_agents: int):
        """Creates antipodal starts and goals on a circle."""
        angles = jnp.linspace(0, 2 * jnp.pi, num_agents, endpoint=False)
        positions = [
            diameter / 2 * jnp.cos(angles),
            diameter / 2 * jnp.sin(angles),
        ]
        padding = [
            jnp.zeros_like(angles)
            for _ in range(self.obsv_dim_agent - self.pos_dim_agent)
        ]
        initial_states = jnp.stack(positions + padding, axis=-1)
        return initial_states, -initial_states

    def reset(self, rng: jax.Array) -> State:
        """Returns the deterministic initial state."""
        return self.reset_conditioned(self.x0, rng)

    def reset_conditioned(self, x0: jax.Array, rng: jax.Array) -> State:
        """Returns a state with supplied joint coordinates and cleared flags."""
        del rng  # These environments have deterministic resets.
        return State(
            pipeline_state=jnp.asarray(x0, dtype=jnp.float32),
            reward=jnp.zeros(self.num_agents, dtype=jnp.float32),
            mask=jnp.zeros(self.num_agents, dtype=jnp.float32),
            collision=jnp.zeros(self.num_agents, dtype=jnp.float32),
        )

    def direct_path_actions(
        self, initial_state: State, goals: jax.Array, horizon: int
    ) -> jax.Array:
        """Builds a constant-control seed for a direct path.

        Args:
            initial_state: State from which planning starts.
            goals: Flattened joint goal state.
            horizon: Number of control timesteps.

        Returns:
            Controls shaped (horizon, action_size). Collision avoidance and
            goal stopping are handled by the normal rollout and optimizer.
        """
        raise NotImplementedError

    def clip_actions(self, traj: jax.Array, factor: float = 1):
        """Clips joint actions to the environment's actuation limits."""
        raise NotImplementedError

    def agent_dynamics(self, x: jax.Array, u: jax.Array):
        """Returns the time derivative of a single robot state."""
        raise NotImplementedError

    def clip_velocity(self, x: jax.Array):
        """Clips the velocity components of a single robot state."""
        raise NotImplementedError

    def get_current_velocity(self, q: jax.Array):
        """Returns each robot's scalar speed from a matrix of robot states."""
        raise NotImplementedError

    @functools.partial(jax.jit, static_argnums=(0,))
    def rk4(self, x: jax.Array, u: jax.Array, dt: float):
        """Integrates one control timestep with fourth-order Runge-Kutta."""
        first_slope = self.agent_dynamics(x, u)
        second_slope = self.agent_dynamics(x + dt / 2 * first_slope, u)
        third_slope = self.agent_dynamics(x + dt / 2 * second_slope, u)
        fourth_slope = self.agent_dynamics(x + dt * third_slope, u)
        next_state = x + dt / 6 * (
            first_slope + 2 * second_slope + 2 * third_slope + fourth_slope
        )
        return self.clip_velocity(next_state)

    def integrate_states(self, states, actions, dt):
        """Integrates the robot batch; subclasses may dispatch by robot type."""
        return jax.vmap(self.rk4, in_axes=(0, 0, None))(states, actions, dt)

    def rollout(
        self,
        state: State,
        xg: jax.Array,
        us: jax.Array,
        penalty_weight: float = 1.0,
        dt: float = 0.1,
    ):
        """Returns rewards, states, goal masks, and collisions for controls.

        Args:
            state: Initial environment state.
            xg: Flattened joint goal state.
            us: Controls of shape (horizon, joint_action_dim).
            penalty_weight: Cost per colliding neighbor.
            dt: Integration timestep in seconds.

        Returns:
            Mean reward per robot and post-step state, goal, and collision
            histories. The initial state is not included in these histories.
        """
        return self._rollout(state, xg, us, penalty_weight, dt, True)

    def score_actions(
        self,
        state: State,
        xg: jax.Array,
        us: jax.Array,
        penalty_weight: float = 1.0,
        dt: float = 0.1,
        include_robot_collisions: bool = True,
    ) -> jax.Array:
        """Returns rollout rewards without allocating trajectory histories.

        Setting include_robot_collisions=False ignores robot-pair costs for
        initialization. Obstacle costs and goal progress remain unchanged.
        """
        return self._rollout(
            state, xg, us, penalty_weight, dt, False, include_robot_collisions
        )

    @functools.partial(jax.jit, static_argnums=(0, 6, 7))
    def _rollout(
        self,
        state,
        goals,
        actions,
        penalty_weight,
        dt,
        record_history,
        include_robot_collisions=True,
    ):
        initial_positions = state.pipeline_state.reshape(self.num_agents, -1)
        goal_positions = goals.reshape(self.num_agents, -1)
        initial_distances = jnp.linalg.norm(
            initial_positions[:, : self.pos_dim_agent]
            - goal_positions[:, : self.pos_dim_agent],
            axis=-1,
        )

        def advance(carry, action):
            current_state, reward_sum = carry
            next_state = self.step(
                current_state,
                goals,
                action,
                initial_distances,
                penalty_weight,
                dt,
                include_robot_collisions,
            )
            history = (
                (
                    next_state.pipeline_state,
                    next_state.mask,
                    next_state.collision,
                )
                if record_history
                else None
            )
            return (next_state, reward_sum + next_state.reward), history

        (_, reward_sum), history = jax.lax.scan(
            advance, (state, jnp.zeros_like(state.reward)), actions
        )
        rewards = reward_sum / actions.shape[0]
        if record_history:
            return (rewards, *history)
        return rewards

    @functools.partial(jax.jit, static_argnums=(0, 7))
    def step(
        self,
        state: State,
        xg: jax.Array,
        action: jax.Array,
        max_distances: jax.Array,
        penalty_weight: float = 1.0,
        dt: float = 0.1,
        include_robot_collisions: bool = True,
    ) -> State:
        """Advances robots and evaluates goal stopping and collision costs."""
        robot_states = state.pipeline_state.reshape(self.num_agents, -1)
        robot_actions = action.reshape(self.num_agents, -1)
        goals = xg.reshape(self.num_agents, -1)
        next_states = self.integrate_states(robot_states, robot_actions, dt)
        stopped = state.mask.astype(bool)
        next_states = jnp.where(stopped[:, None], robot_states, next_states)
        goal_distances = jnp.linalg.norm(
            next_states[:, : self.pos_dim_agent]
            - goals[:, : self.pos_dim_agent],
            axis=-1,
        )
        reached_goals = (goal_distances < self.stop_distance) & (
            self.get_current_velocity(next_states) <= self.stop_velocity
        )
        # Once reached, goals remain marked and robots stay stopped.
        goal_mask = reached_goals | stopped
        rewards, collisions = self.get_reward(
            next_states,
            goal_distances,
            max_distances,
            penalty_weight,
            include_robot_collisions,
        )
        return state.replace(
            pipeline_state=next_states.flatten(),
            reward=rewards,
            mask=goal_mask.astype(state.mask.dtype),
            collision=collisions.astype(state.collision.dtype),
        )

    @functools.partial(jax.jit, static_argnums=(0, 5))
    def get_reward(
        self,
        q: jax.Array,
        distances_to_goals: jax.Array,
        max_distances: jax.Array,
        penalty_weight: float = 1.0,
        include_robot_collisions: bool = True,
    ):
        """Returns normalized progress minus collision penalties."""
        rewards = 1.0 - distances_to_goals / jnp.maximum(max_distances, 1e-6)
        colliding_pairs = self.collision_matrix(q)
        obstacle_collisions = self.obstacle_collision_matrix(q)
        if include_robot_collisions:
            penalties = colliding_pairs.sum(axis=1) + obstacle_collisions.sum(axis=1)
        else:
            penalties = obstacle_collisions.sum(axis=1)
        return (
            rewards - penalties * penalty_weight,
            jnp.any(colliding_pairs, axis=1)
            | jnp.any(obstacle_collisions, axis=1),
        )

    @functools.partial(jax.jit, static_argnums=(0,))
    def collision_matrix(self, robot_states: jax.Array) -> jax.Array:
        """Returns pairwise collision flags with optional leading batch axes.

        Args:
            robot_states: States shaped (..., num_agents, state_dim).

        Returns:
            Boolean array shaped (..., num_agents, num_agents), with a false
            diagonal. Exact overlap and safety-boundary contact are collisions.
        """
        positions = robot_states[..., : self.pos_dim_agent]
        differences = positions[..., :, None, :] - positions[..., None, :, :]
        squared_distances = jnp.sum(differences**2, axis=-1)
        threshold = 2 * self.agent_radius + self.safe_margin
        return (squared_distances <= threshold**2) & ~jnp.eye(
            self.num_agents, dtype=bool
        )

    @functools.partial(jax.jit, static_argnums=(0,))
    def obstacle_collision_matrix(self, robot_states: jax.Array) -> jax.Array:
        """Returns (..., num_agents, num_obstacles) collision flags.

        Static obstacles contribute to failure and cost without connecting
        unrelated robots in the robot-to-robot collision graph.
        """
        if self.num_obstacles == 0:
            return jnp.zeros(robot_states.shape[:-1] + (0,), dtype=bool)
        positions = robot_states[..., : self.pos_dim_agent]
        differences = positions[..., :, None, :] - self.obstacle_centers
        thresholds = self.agent_radius + self.obstacle_radii + self.safe_margin
        return jnp.sum(differences**2, axis=-1) <= thresholds**2

    @property
    def action_size(self) -> int:
        """Number of components in the joint action vector."""
        return self.action_dim_agent * self.num_agents

    @property
    def observation_size(self) -> int:
        """Number of components in the joint state vector."""
        return self.obsv_dim_agent * self.num_agents

    def get_heading_line(self, state, position, agent_idx):
        """Returns empty heading coordinates for rotation-invariant robots."""
        del state, position, agent_idx  # Holonomic robots have no heading.
        return [], []

    def render_gif(self, xs, gif_output_path, trajectory_image_path, ids=None):
        """Saves an animation and static plot; imports rendering on demand."""
        # Keep simulation independent of Matplotlib.
        # pylint: disable-next=import-outside-toplevel
        from d4orm.envs import rendering

        rendering.render_trajectory(
            self, xs, gif_output_path, trajectory_image_path, ids
        )

    def render_gif_interactive(self, xs):
        """Opens a trajectory plot in an interactive Matplotlib window."""
        # Keep simulation independent of Matplotlib.
        # pylint: disable-next=import-outside-toplevel
        from d4orm.envs import rendering

        rendering.show_trajectory(self, xs)
