"""Shared configuration, rollout interface, and planning lifecycle."""

import abc
import dataclasses
import math
from typing import NamedTuple

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
        max_iterations: Maximum iterations; zero evaluates the initial controls.
        temperature: Temperature of standardized reward weights.
        beta_start: First beta in the base 100-step diffusion schedule.
        beta_end: Last beta in the base diffusion schedule.
        num_workers: D-D4ORM computational nodes; None uses the robot count.
        top_k: D-D4ORM elite count; None uses max(1, num_workers // 4).
    """

    horizon: int = 100
    num_samples: int = 1024
    num_steps: int = 100
    max_iterations: int = 50
    temperature: float = 0.3
    beta_start: float = 1e-4
    beta_end: float = 2e-2
    direct_path_init: bool = False
    num_workers: int | None = None
    top_k: int | None = None

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

        for name in ("num_workers", "top_k"):
            value = getattr(self, name)
            if value is not None and (
                not isinstance(value, int)
                or isinstance(value, bool)
                or value < 1
            ):
                raise ValueError(f"{name} must be a positive integer or None.")


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

    Controls have shape (horizon, joint_control_dim), with controls after goal
    arrival zeroed for each robot. States exclude the initial
    state and have shape (horizon, joint_state_dim). Goal masks and collisions
    have shape (horizon, num_agents). Rewards are time means per agent.
    Penalty weights record the per-robot costs used for this evaluation.
    Reward history includes the initial evaluation. `iterations` counts completed
    optimization iterations, including unsuccessful ones. Random key can be
    reused to continue planning without repeating samples.
    """

    controls: jax.Array
    states: jax.Array
    goal_masks: jax.Array
    collisions: jax.Array
    rewards: jax.Array
    reward_history: tuple[float, ...]
    success: bool
    iterations: int
    random_key: jax.Array
    penalty_weights: jax.Array


class TrajectoryEvaluation(NamedTuple):
    """Rollout observations passed to planner lifecycle hooks."""

    rewards: jax.Array
    states: jax.Array
    goal_masks: jax.Array
    collisions: jax.Array


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
        self.control_size = environment.control_size
        self.observation_size = environment.observation_size

        self.evaluate = jax.jit(self._evaluate)
        self.compute_rewards = jax.jit(
            self._compute_rewards_batch,
            static_argnames=("include_robot_collisions",),
        )

    def _evaluate(self, initial_state, goals, controls, penalty_weights=None):
        return self.environment.rollout(
            initial_state,
            goals,
            controls,
            penalty_weight=1.0 if penalty_weights is None else penalty_weights,
            dt=self.config.dt,
        )

    def _compute_rewards_batch(
        self,
        initial_state,
        goals,
        controls,
        penalty_weights=None,
        include_robot_collisions=True,
    ):
        def compute_candidate_rewards(candidate_controls):
            return self._compute_rewards(
                initial_state,
                goals,
                candidate_controls,
                penalty_weights,
                include_robot_collisions,
            )

        return jax.vmap(compute_candidate_rewards)(controls)

    def _compute_rewards(
        self,
        initial_state,
        goals,
        controls,
        penalty_weights=None,
        include_robot_collisions=True,
    ):
        return self.environment.score_controls(
            initial_state,
            goals,
            controls,
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


def mask_goal_controls(controls, initial_mask, goal_masks):
    """Zeros controls after arrival, preserving the step reaching the goal."""
    stopped_before_step = jnp.concatenate(
        (initial_mask[None, :], goal_masks[:-1]), axis=0
    ).astype(bool)
    control_mask = jnp.repeat(
        stopped_before_step, controls.shape[-1] // initial_mask.shape[0], axis=1
    )
    return jnp.where(control_mask, 0.0, controls)


class Planner(abc.ABC):
    """Shared validation, evaluation, stopping, and result construction.

    Subclasses own their numerical kernels and per-call optimizer state.
    Lifecycle hooks run outside JIT; their defaults require no optimizer state
    and leave controls and penalties unchanged. State must remain local to a
    planning call so warm-up and repeated trials cannot affect later calls.
    """

    def __init__(
        self, rollout: Rollout, config: PlannerConfig = PlannerConfig()
    ):
        self.rollout = rollout
        self.config = config

    def plan(
        self,
        random_key: jax.Array,
        initial_state: multibase.State,
        goals: jax.Array,
        initial_controls: jax.Array | None = None,
        *,
        print_info: bool = False,
    ) -> PlanResult:
        """Optimizes a joint control sequence, stopping on a valid solution.

        Args:
            random_key: JAX random key; never mutated or stored on the planner.
            initial_state: Environment state from reset or reset_conditioned.
            goals: Flattened joint goal state.
            initial_controls: Warm start with shape (horizon, control_size).
            print_info: Print the initial evaluation and each iteration status.

        Returns:
            Controls, evaluation, history, success, and the next random key.

        Raises:
            ValueError: If inputs do not match the rollout dimensions.
        """
        return self._plan(
            random_key,
            initial_state,
            goals,
            initial_controls,
            max_iterations=self.config.max_iterations,
            stop_on_success=True,
            print_info=print_info,
        )

    def warm_up(
        self,
        random_key: jax.Array,
        initial_state: multibase.State,
        goals: jax.Array,
        initial_controls: jax.Array | None = None,
    ) -> None:
        """Warms every planning phase and waits for device completion.

        Uses this planner's configured shapes and compiled functions. Runs an
        iteration even if the initial plan succeeds, so early stopping cannot
        skip compilation. Subclasses specify how many iterations cover their
        planning phases through _warm_up_iterations. The resulting plan is
        discarded; planning starts from its own inputs and random key.

        Args:
            random_key: A key dedicated to warm-up sampling.
            initial_state: State with the same shapes and dtypes as planning.
            goals: Joint goals with the same shape and dtype as planning.
            initial_controls: Optional controls for the intended warm start.
        """
        result = self._plan(
            random_key,
            initial_state,
            goals,
            initial_controls,
            max_iterations=self._warm_up_iterations(initial_controls),
            stop_on_success=False,
        )
        jax.block_until_ready(
            (
                result.controls,
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
        initial_controls,
        *,
        max_iterations,
        stop_on_success,
        print_info=False,
    ) -> PlanResult:
        """Runs the shared lifecycle for optimization or compilation warm-up."""
        # Initialize controls and validate the joint control and state shapes.
        control_shape = (self.config.horizon, self.rollout.control_size)
        controls = (
            jnp.zeros(control_shape)
            if initial_controls is None
            else jnp.asarray(initial_controls, dtype=jnp.float32)
        )
        goals = jnp.asarray(goals)

        if controls.shape != control_shape:
            raise ValueError(
                f"initial_controls must have shape {control_shape}."
            )
        state_shape = (self.rollout.observation_size,)
        if initial_state.pipeline_state.shape != state_shape:
            raise ValueError(f"initial_state must have shape {state_shape}.")
        if goals.shape != state_shape:
            raise ValueError(f"goals must have shape {state_shape}.")

        if initial_controls is None and self.config.direct_path_init:
            controls = self.rollout.environment.direct_path_controls(
                initial_state, goals, self.config.horizon
            )
        if not bool(jnp.all(jnp.isfinite(controls))):
            raise ValueError("initial_controls must be finite.")

        # Initialize fresh optimizer state for this planning call.
        # pylint: disable-next=assignment-from-none
        optimizer_state = self._initial_optimizer_state(
            controls, warm_start=initial_controls is not None
        )
        penalty_weights = jnp.full(
            (self.rollout.num_agents,),
            1.0,
            dtype=controls.dtype,
        )
        history = []

        for iteration in range(max_iterations + 1):
            # Evaluate the current plan's rewards, goals, and collisions.
            evaluation = TrajectoryEvaluation(
                *self.rollout.evaluate(
                    initial_state, goals, controls, penalty_weights
                )
            )
            rewards, states, goal_masks, collisions = evaluation

            # Zero controls for agents that reached their goal before each step.
            controls = mask_goal_controls(
                controls, initial_state.mask, goal_masks
            )

            # Record progress and stop on success or the iteration limit.
            history.append(float(rewards.mean()))
            success = bool(jnp.all(goal_masks[-1]) & ~jnp.any(collisions))
            finished = (
                success and stop_on_success
            ) or iteration == max_iterations
            if finished and not print_info:
                break

            optimizer_state = self._observe_iteration(
                iteration, evaluation, optimizer_state
            )

            if print_info:
                self._print_iteration(
                    iteration,
                    evaluation,
                    optimizer_state,
                )

            if finished:
                break

            # Adapt controls and collision penalties for the next iteration.
            controls, penalty_weights, optimizer_state = (
                self._prepare_iteration(
                    iteration,
                    controls,
                    penalty_weights,
                    evaluation,
                    optimizer_state,
                )
            )

            if print_info:
                print(f"Next-iteration collision penalties: {penalty_weights}")

            # Run one iteration using the subclass's numerical kernel.
            random_key, controls, optimizer_state = self._run_iteration(
                iteration,
                random_key,
                controls,
                optimizer_state,
                initial_state,
                goals,
                penalty_weights,
            )

        # Return the evaluated plan and the state needed to continue sampling.
        return PlanResult(
            controls=controls,
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

    # pylint: disable-next=useless-return
    def _initial_optimizer_state(self, controls, *, warm_start):
        """Creates fresh per-call state; stateless planners return None.

        warm_start indicates explicitly supplied controls, independently of
        whether the configuration requests a direct-path initialization.
        """
        del controls, warm_start
        return None

    def _observe_iteration(self, iteration, evaluation, optimizer_state):
        """Updates state before preparation or requested final diagnostics.

        Called after rollout evaluation and goal masking. This hook must not
        change evaluated controls or costs; use _prepare_iteration for those.
        The default preserves state without processing the observation.
        """
        del iteration, evaluation
        return optimizer_state

    def _prepare_iteration(
        self,
        iteration,
        controls,
        penalty_weights,
        evaluation,
        optimizer_state,
    ):
        """Preserves controls, penalties, and optimizer state by default."""
        del iteration, evaluation
        return controls, penalty_weights, optimizer_state

    @abc.abstractmethod
    def _run_iteration(
        self,
        iteration,
        random_key,
        controls,
        optimizer_state,
        initial_state,
        goals,
        penalty_weights,
    ):
        """Runs one iteration, returning a key, controls, and optimizer state.

        Subclasses dispatch their compiled kernels here. optimizer_state is
        opaque to the base planner and need not be a JAX-compatible object.
        Returned controls must retain the input shape; per-call state belongs
        in the return value, not on the planner instance.
        """

    def _warm_up_iterations(self, initial_controls):
        """Returns the iterations needed to compile all planning phases."""
        del initial_controls
        return 1

    def _print_iteration(
        self,
        iteration,
        evaluation,
        optimizer_state,
    ):
        """Prints progress outside compiled optimization kernels."""
        del optimizer_state
        print(
            f"Iteration {iteration}: "
            f"reward={float(evaluation.rewards.mean()):.4f}"
        )
