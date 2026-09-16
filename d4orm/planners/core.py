"""Shared configuration, rollout interface, and planning lifecycle."""

import abc
import dataclasses
import math

import jax
import jax.numpy as jnp

from d4orm.envs import multibase


@dataclasses.dataclass(frozen=True)
class PlannerConfig:
    """Sampling settings shared by all planners.

    Attributes:
        direct_path_init: Use direct controls when no warm start is supplied.
        horizon: Number of control timesteps.
        num_samples: Candidate trajectories per optimization step.
        num_steps: Optimization steps per iteration (one diffusion schedule).
        max_iterations: Maximum iterations; zero evaluates the initial actions.
        temperature: Temperature of standardized reward weights.
        beta_start: First beta in the base 100-step diffusion schedule.
        beta_end: Last beta in the base diffusion schedule.
    """

    horizon: int = 100
    num_samples: int = 1024
    num_steps: int = 100
    max_iterations: int = 50
    temperature: float = 0.3
    beta_start: float = 1e-4
    beta_end: float = 2e-2
    direct_path_init: bool = False

    def __post_init__(self):
        for name in ("horizon", "num_samples", "num_steps"):
            value = getattr(self, name)
            if (
                not isinstance(value, int)
                or isinstance(value, bool)
                or value < 1
            ):
                raise ValueError(f"{name} must be a positive integer.")

        if not isinstance(self.max_iterations, int) or self.max_iterations < 0:
            raise ValueError("max_iterations must be a nonnegative integer.")

        if not math.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("temperature must be finite and positive.")

        if not 0 < self.beta_start <= self.beta_end < 1:
            raise ValueError("Require 0 < beta_start <= beta_end < 1.")


@dataclasses.dataclass(frozen=True)
class RolloutConfig:
    """Settings for rollout integration."""

    dt: float = 0.1

    def __post_init__(self):
        if not math.isfinite(self.dt) or self.dt <= 0:
            raise ValueError("dt must be finite and positive.")


@dataclasses.dataclass(frozen=True)
class PlanResult:
    """A plan and its evaluation using the same rollout as optimization.

    Actions have shape (horizon, joint_action_dim), with controls after goal
    arrival zeroed for each robot. States exclude the initial
    state and have shape (horizon, joint_state_dim). Goal masks and collisions
    have shape (horizon, num_agents). Rewards are time means per agent.
    Penalty weights record the per-robot costs used for this evaluation.
    Reward history includes the initial evaluation. Iterations counts completed
    optimization cycles, including unsuccessful ones. Random key can be reused
    to continue planning without repeating samples.
    """

    actions: jax.Array
    states: jax.Array
    goal_masks: jax.Array
    collisions: jax.Array
    rewards: jax.Array
    reward_history: tuple[float, ...]
    success: bool
    iterations: int
    random_key: jax.Array
    penalty_weights: jax.Array


class Rollout:
    """Compiled candidate reward computation and trajectory evaluation.

    All planners accept the same instance. Candidate rewards are accumulated
    without storing state histories; evaluation uses the same dynamics and cost.
    Treat the environment as immutable after creating this object.
    """

    def __init__(
        self,
        environment: multibase.MultiBase,
        config: RolloutConfig = RolloutConfig(),
    ):
        self.environment = environment
        self.config = config
        self.num_agents = environment.num_agents
        self.action_size = environment.action_size
        self.observation_size = environment.observation_size

        self.evaluate = jax.jit(self._evaluate)
        self.compute_rewards = jax.jit(
            self._compute_rewards_batch,
            static_argnames=("include_robot_collisions",),
        )

    def _evaluate(self, initial_state, goals, actions, penalty_weights=None):
        return self.environment.rollout(
            initial_state,
            goals,
            actions,
            penalty_weight=1.0 if penalty_weights is None else penalty_weights,
            dt=self.config.dt,
        )

    def _compute_rewards_batch(
        self,
        initial_state,
        goals,
        actions,
        penalty_weights=None,
        include_robot_collisions=True,
    ):
        def compute_candidate_rewards(candidate_actions):
            return self._compute_rewards(
                initial_state,
                goals,
                candidate_actions,
                penalty_weights,
                include_robot_collisions,
            )

        return jax.vmap(compute_candidate_rewards)(actions)

    def _compute_rewards(
        self,
        initial_state,
        goals,
        actions,
        penalty_weights=None,
        include_robot_collisions=True,
    ):
        return self.environment.score_actions(
            initial_state,
            goals,
            actions,
            penalty_weight=1.0 if penalty_weights is None else penalty_weights,
            dt=self.config.dt,
            include_robot_collisions=include_robot_collisions,
        )


def reward_weights(rewards: jax.Array, temperature: float) -> jax.Array:
    """Returns sample-axis softmax weights, including for constant rewards."""
    reward_std = rewards.std(axis=0, keepdims=True)
    reward_std = jnp.where(reward_std < 1e-4, 1.0, reward_std)
    logits = (rewards - rewards.mean(axis=0, keepdims=True)) / reward_std
    return jax.nn.softmax(logits / temperature, axis=0)


class Planner(abc.ABC):
    """Shared lifecycle for planners with compiled optimization cycles."""

    def __init__(
        self, rollout: Rollout, config: PlannerConfig = PlannerConfig()
    ):
        self.rollout = rollout
        self.config = config
        self._compiled_cycle = jax.jit(self._optimize_cycle)
        self._compiled_probe = self._compiled_cycle

    @abc.abstractmethod
    def _optimize_cycle(
        self,
        random_key,
        actions,
        sample_std,
        initial_state,
        goals,
        reward_groups,
        active_agents,
        penalty_weights,
    ):
        """Returns an updated random key, actions, and sampling deviation."""

    def _uses_independent_probe(self, initial_actions):
        """Whether the first optimization cycle is an independent probe."""
        del initial_actions
        return False

    def _initial_sampling_deviation(self, actions):
        """Returns initial sampling deviation, unused by diffusion planners."""
        del actions
        return None

    def _initial_adaptation_state(self):
        """Returns per-call history for method-specific control adaptation."""
        return ()

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
        """Adapts base controls and costs before the next optimization cycle."""
        del (
            iteration,
            collisions,
            reward_groups,
            active_agents,
            initialization_probe,
        )
        return actions, penalty_weights, adaptation_state

    def _update_groups(
        self,
        states,
        goal_masks,
        collisions,
        previous_groups,
        previous_collisions,
    ):
        del states, goal_masks, collisions
        return (
            previous_groups,
            jnp.ones(self.rollout.num_agents, dtype=bool),
            previous_collisions,
        )

    def _print_iteration(
        self,
        iteration,
        rewards,
        goal_masks,
        collisions,
        reward_groups,
        active_agents,
    ):
        """Prints progress outside compiled optimization kernels."""
        del goal_masks, collisions, reward_groups, active_agents
        print(f"Iteration {iteration}: reward={float(rewards.mean()):.4f}")

    def plan(
        self,
        random_key: jax.Array,
        initial_state: multibase.State,
        goals: jax.Array,
        initial_actions: jax.Array | None = None,
        *,
        print_info: bool = False,
    ) -> PlanResult:
        """Optimizes a joint control sequence, stopping on a valid solution.

        Args:
            random_key: JAX random key; never mutated or stored on the planner.
            initial_state: Environment state from reset or reset_conditioned.
            goals: Flattened joint goal state.
            initial_actions: Warm start with shape (horizon, action_size).
            print_info: Print the initial evaluation and each iteration status.

        Returns:
            Actions, full evaluation, history, success, and the next random key.

        Raises:
            ValueError: If inputs do not match the rollout dimensions.
        """
        return self._plan(
            random_key,
            initial_state,
            goals,
            initial_actions,
            max_iterations=self.config.max_iterations,
            stop_on_success=True,
            print_info=print_info,
        )

    def warm_up(
        self,
        random_key: jax.Array,
        initial_state: multibase.State,
        goals: jax.Array,
        initial_actions: jax.Array | None = None,
    ) -> None:
        """Warms every planning phase and waits for device completion.

        Uses this planner's configured shapes and compiled functions. The cycle
        runs even if the initial plan already succeeds, so early stopping cannot
        skip compilation. Independent initialization uses two cycles to cover
        both the probe and full collision reward paths. The resulting plan is
        discarded; planning starts from its own inputs and random key.

        Args:
            random_key: A key dedicated to warm-up sampling.
            initial_state: State with the same shapes and dtypes as planning.
            goals: Joint goals with the same shape and dtype as planning.
            initial_actions: Optional controls matching the intended warm start.
        """
        result = self._plan(
            random_key,
            initial_state,
            goals,
            initial_actions,
            max_iterations=2
            if self._uses_independent_probe(initial_actions)
            else 1,
            stop_on_success=False,
        )
        jax.block_until_ready(
            (
                result.actions,
                result.states,
                result.goal_masks,
                result.collisions,
                result.rewards,
                result.random_key,
            )
        )

    def _plan(
        self,
        random_key,
        initial_state,
        goals,
        initial_actions,
        *,
        max_iterations,
        stop_on_success,
        print_info=False,
    ) -> PlanResult:
        """Runs the shared lifecycle for optimization or compilation warm-up."""

        # Initialize controls and validate the joint action and state shapes.
        action_shape = (self.config.horizon, self.rollout.action_size)
        actions = (
            jnp.zeros(action_shape)
            if initial_actions is None
            else jnp.asarray(initial_actions, dtype=jnp.float32)
        )
        goals = jnp.asarray(goals)

        if actions.shape != action_shape:
            raise ValueError(f"initial_actions must have shape {action_shape}.")
        state_shape = (self.rollout.observation_size,)
        if initial_state.pipeline_state.shape != state_shape:
            raise ValueError(f"initial_state must have shape {state_shape}.")
        if goals.shape != state_shape:
            raise ValueError(f"goals must have shape {state_shape}.")

        if initial_actions is None and self.config.direct_path_init:
            actions = self.rollout.environment.direct_path_actions(
                initial_state, goals, self.config.horizon
            )
        if not bool(jnp.all(jnp.isfinite(actions))):
            raise ValueError("initial_actions must be finite.")

        # Initialize sampling, reward groups, and collision adaptation history.
        sample_std = self._initial_sampling_deviation(actions)
        reward_groups = jnp.eye(self.rollout.num_agents)
        previous_collisions = jnp.zeros(
            (self.rollout.num_agents, self.rollout.num_agents), dtype=bool
        )
        penalty_weights = jnp.full(
            (self.rollout.num_agents,),
            1.0,
            dtype=actions.dtype,
        )
        adaptation_state = self._initial_adaptation_state()
        independent_probe = self._uses_independent_probe(initial_actions)
        history = []

        for iteration in range(max_iterations + 1):
            # Evaluate the current plan's rewards, goals, and collisions.
            rewards, states, goal_masks, collisions = self.rollout.evaluate(
                initial_state, goals, actions, penalty_weights
            )

            # Zero controls for agents that reached their goal before each step.
            stopped_before_step = jnp.concatenate(
                (initial_state.mask[None, :], goal_masks[:-1]), axis=0
            ).astype(bool)
            action_mask = jnp.repeat(
                stopped_before_step,
                self.rollout.action_size // self.rollout.num_agents,
                axis=1,
            )
            actions = jnp.where(action_mask, 0.0, actions)

            # Record progress and stop on success or the iteration limit.
            history.append(float(rewards.mean()))
            success = bool(jnp.all(goal_masks[-1]) & ~jnp.any(collisions))
            finished = (
                success and stop_on_success
            ) or iteration == max_iterations
            if finished and not print_info:
                break

            if independent_probe and iteration == 0:
                # Start the independent probe with one reward group per agent.
                reward_groups = jnp.eye(self.rollout.num_agents)
                active_agents = jnp.ones(self.rollout.num_agents, dtype=bool)
            else:
                # Update collision groups and select agents for replanning.
                reward_groups, active_agents, previous_collisions = (
                    self._update_groups(
                        states,
                        goal_masks,
                        collisions,
                        reward_groups,
                        previous_collisions,
                    )
                )

            if print_info:
                self._print_iteration(
                    iteration,
                    rewards,
                    goal_masks,
                    collisions,
                    reward_groups,
                    active_agents,
                )

            if finished:
                break

            # Adapt controls and collision penalties for the next cycle.
            actions, penalty_weights, adaptation_state = self._prepare_cycle(
                iteration,
                actions,
                collisions,
                reward_groups,
                active_agents,
                penalty_weights,
                adaptation_state,
                initialization_probe=(
                    (
                        iteration == 0
                        and self.config.direct_path_init
                        and initial_actions is None
                    )
                    or (independent_probe and iteration == 1)
                ),
            )

            if print_info:
                print(f"Next-cycle collision penalties: {penalty_weights}")

            # Run denoising optimization
            cycle = (
                self._compiled_probe
                if independent_probe and iteration == 0
                else self._compiled_cycle
            )
            random_key, actions, sample_std = cycle(
                random_key,
                actions,
                sample_std,
                initial_state,
                goals,
                reward_groups,
                active_agents,
                penalty_weights,
            )

        # Return the evaluated plan and the state needed to continue sampling.
        return PlanResult(
            actions=actions,
            states=states,
            goal_masks=goal_masks,
            collisions=collisions,
            rewards=rewards,
            reward_history=tuple(history),
            success=success,
            iterations=iteration,
            random_key=random_key,
            penalty_weights=penalty_weights,
        )
