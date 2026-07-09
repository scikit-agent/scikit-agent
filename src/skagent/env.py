"""
Environment adapters for :class:`~skagent.bellman.BellmanPeriod` models.

Two interfaces are provided:

- :class:`Environment` — plain Python environment that steps a BellmanPeriod
  by drawing shocks and applying a decision rule. Returns the full transition
  ``(state, action, reward, next_state, discount, obs)`` as dicts keyed by
  symbol, suitable for off-policy RL algorithms that consume full
  transitions.

- :class:`GymEnv` — :mod:`gymnasium` adapter wrapping a BellmanPeriod so
  Stable Baselines3 algorithms (PPO, SAC, TD3, …) can drive a BellmanPeriod
  directly. The action space is normalised to ``[-1, 1]``; the env
  unscales each action to the control's per-state bounds (taken from
  ``Control.lower_bound`` / ``Control.upper_bound``) before applying it.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces

from skagent.bellman import BellmanPeriod
from skagent.simulation.monte_carlo import draw_shocks


def _to_tensor_dict(d: dict[str, Any]) -> dict[str, torch.Tensor]:
    return {k: torch.as_tensor(np.asarray(v)).float() for k, v in d.items()}


def _to_scalar(x: Any) -> float:
    if isinstance(x, torch.Tensor):
        return float(x.detach().cpu().reshape(-1)[0].item())
    return float(np.asarray(x).reshape(-1)[0])


def _draw_period_shocks(bp: BellmanPeriod, rng) -> dict[str, torch.Tensor]:
    shocks = bp.get_shocks()
    if not shocks:
        return {}
    return _to_tensor_dict(draw_shocks(shocks, [], n=1, rng=rng))


class Environment:
    """Step a :class:`BellmanPeriod` one transition at a time.

    Single agent. Returns torch tensors keyed by symbol so downstream code
    can index by name.

    Parameters
    ----------
    bp : BellmanPeriod
        Model definition (block dynamics, calibration, discount variable).
    initial : dict[str, Distribution]
        Maps arrival-state symbols to ``skagent`` ``Distribution`` objects,
        used to sample fresh initial states each :meth:`reset`.
    rng : numpy.random.Generator, optional
        RNG used for shock and initial-state draws.
    """

    def __init__(
        self,
        bp: BellmanPeriod,
        initial: dict,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.bp = bp
        self.initial = initial
        self.rng = rng
        # Construct shock distributions from their (cls, kwargs) tuples if they
        # haven't been yet. Idempotent: already-constructed distributions are
        # left alone.
        bp.block.construct_shocks(bp.calibration, rng=rng)
        self.state: dict[str, torch.Tensor] | None = None
        self.reset()

    def reset(self) -> dict[str, torch.Tensor]:
        initial_vals = draw_shocks(self.initial, [0], rng=self.rng)
        self.state = _to_tensor_dict(initial_vals)
        return self.state

    def step(self, decision_rule: dict):
        """Advance one period.

        Parameters
        ----------
        decision_rule : dict[str, Callable]
            Maps control symbol to a callable on the control's information set.

        Returns
        -------
        tuple
            ``(state_t, action, reward, state_t_plus_1, discount, obs)``.
            ``obs`` is the information set seen by the policy.
        """
        shocks = _draw_period_shocks(self.bp, self.rng)
        post = self.bp.post_function(
            self.state, controls={}, shocks=shocks, decision_rules=decision_rule
        )

        state_t = self.state
        action = {csym: post[csym] for csym in decision_rule}
        reward = {rsym: post[rsym] for rsym in self.bp.get_reward_syms()}
        discount = self.bp.resolve_discount_factor(post)

        def _detach(d):
            return {
                k: v.detach() if isinstance(v, torch.Tensor) else v
                for k, v in d.items()
            }

        state_t_plus = _detach({sym: post[sym] for sym in self.bp.get_arrival_states()})

        control_sym = next(iter(decision_rule))
        iset = self.bp.block.dynamics[control_sym].iset
        obs = _detach({sym: post[sym] for sym in iset})

        self.state = state_t_plus
        return state_t, action, reward, state_t_plus, discount, obs


def discounted_rollout_reward(
    bp: BellmanPeriod,
    decision_rule: dict,
    initial: dict,
    steps: int,
    rng: np.random.Generator | None = None,
) -> float:
    """Realized discounted reward of a single rollout under ``decision_rule``.

    Simulates one episode of ``steps`` periods through an :class:`Environment`,
    accumulating per-period rewards weighted by the running product of the
    model's (possibly per-period) discount factor.

    Parameters
    ----------
    bp : BellmanPeriod
        Model definition.
    decision_rule : dict[str, Callable]
        Maps control symbol to a callable on the control's information set, as
        consumed by :meth:`Environment.step`.
    initial : dict[str, Distribution]
        Maps arrival-state symbols to skagent ``Distribution`` objects used to
        sample the initial state.
    steps : int
        Number of periods to simulate.
    rng : numpy.random.Generator, optional
        RNG used for the initial-state and shock draws.

    Returns
    -------
    float
        ``sum_t (prod_{s<t} discount_s) * reward_t``.
    """
    env = Environment(bp, initial, rng=rng)
    env.reset()
    total = 0.0
    discount_acc = 1.0
    for _ in range(steps):
        _, _, reward, _, discount, _ = env.step(decision_rule)
        total += discount_acc * sum(_to_scalar(v) for v in reward.values())
        discount_acc *= _to_scalar(discount)
    return total


# ---------------------------------------------------------------------------
# Gymnasium adapter
# ---------------------------------------------------------------------------


class GymEnv(gym.Env):
    """:mod:`gymnasium` adapter for a single-agent, single-control BellmanPeriod.

    Designed to be driven by Stable Baselines3 (PPO, SAC, TD3, …).

    Action space
        ``Box(-1, 1, shape=(1,))`` — *normalised*. Each action is unscaled to
        the control's per-state bounds via ::

            a_real = lo + (a_norm + 1) / 2 * (hi - lo)

        where ``lo``, ``hi`` are evaluated at the current pre-decision state.
        Missing bounds fall back to ``default_lower`` / ``default_upper``.

    Observation space
        ``Box(-inf, inf, shape=(|iset|,))`` over the control's information
        set (``Control.iset``), in the iset's declared order.

    Episode timing
        ``terminated`` is always ``False`` (no native end-of-life signal yet);
        ``truncated`` fires when ``max_episode_steps`` is reached. PPO
        bootstraps correctly across truncations as long as the wrapper sets
        the flag — which we do.

    Parameters
    ----------
    bp : BellmanPeriod
        Model definition.
    initial : dict[str, Distribution]
        Maps arrival-state symbols to skagent ``Distribution`` objects used
        to sample initial states on :meth:`reset`.
    max_episode_steps : int, optional
        Episode horizon. Default 200.
    control_sym : str, optional
        Which control to drive. If omitted and the block has exactly one
        control, that control is used.
    default_lower, default_upper : float, optional
        Fallback bounds when a control omits ``lower_bound`` or
        ``upper_bound``. Defaults ``0.0`` / ``1.0`` only make sense for
        action spaces that happen to lie in ``[0, 1]``; for other ranges
        pass explicit values.
    bound_clearance : float, optional
        Fraction of the ``(hi - lo)`` span pulled in from each bound when
        unscaling, to avoid degenerate values at the edge (e.g. ``c=0``
        under log utility). Default ``1e-3``.
    seed : int, optional
        Seed for the env's internal numpy RNG.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        bp: BellmanPeriod,
        initial: dict,
        max_episode_steps: int = 200,
        *,
        control_sym: str | None = None,
        default_lower: float = 0.0,
        default_upper: float = 1.0,
        bound_clearance: float = 1e-3,
        seed: int | None = None,
    ) -> None:
        super().__init__()

        controls = list(bp.get_controls())
        if control_sym is None:
            if len(controls) != 1:
                raise ValueError(
                    "GymEnv currently supports a single control unless "
                    f"`control_sym` is given; block has controls={controls}"
                )
            control_sym = controls[0]
        elif control_sym not in controls:
            raise ValueError(
                f"control_sym {control_sym!r} is not a control of the block "
                f"(controls={controls})"
            )

        self.bp = bp
        self.initial = initial
        self.max_episode_steps = max_episode_steps
        self.control_sym = control_sym
        self.control = bp.block.dynamics[control_sym]
        self.iset: list[str] = list(self.control.iset)
        self.reward_syms = bp.get_reward_syms()
        self.default_lower = float(default_lower)
        self.default_upper = float(default_upper)
        self.bound_clearance = float(bound_clearance)

        self._rng = np.random.default_rng(seed)

        # Construct shock distributions from their (cls, kwargs) tuples if
        # they haven't been yet. Idempotent.
        bp.block.construct_shocks(bp.calibration, rng=self._rng)

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(len(self.iset),),
            dtype=np.float32,
        )

        self._state: dict[str, torch.Tensor] | None = None
        self._pending_shocks: dict[str, torch.Tensor] = {}
        self._steps = 0

    # -- gymnasium API ----------------------------------------------------

    def reset(self, *, seed: int | None = None, options=None):
        """Sample a fresh initial state and return ``(observation, info)``.

        Draws arrival states from ``initial`` and the first period's shocks,
        then returns the observation over the control's information set. ``info``
        is an empty dict. Reseeds the internal RNG when ``seed`` is given.
        """
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self._steps = 0
        initial_vals = draw_shocks(self.initial, [0], rng=self._rng)
        self._state = _to_tensor_dict(initial_vals)
        self._pending_shocks = _draw_period_shocks(self.bp, self._rng)
        obs = self._compute_obs(self._state, self._pending_shocks)
        return obs, {}

    def step(self, action):
        """Apply a normalised ``action`` and return the gymnasium 5-tuple.

        ``action`` is a 1-element array in ``[-1, 1]``; it is unscaled to the
        control's per-state bounds before being applied. Returns
        ``(observation, reward, terminated, truncated, info)``, where
        ``terminated`` is always ``False`` and ``truncated`` fires at
        ``max_episode_steps``. ``info`` carries the resolved ``discount``, the
        ``action_unscaled`` value, and the ``bounds`` used for unscaling.
        """
        if self._state is None:
            raise RuntimeError("Call reset() before step().")

        action_arr = np.asarray(action, dtype=np.float32).reshape(-1)
        if action_arr.size != 1:
            raise ValueError(
                "GymEnv expects a 1-dim action for a single control; "
                f"got shape={action_arr.shape}"
            )

        shocks = self._pending_shocks

        # Evaluate state-dependent bounds at the pre-decision state.
        pre = self.bp.compute_pre_state(self.control_sym, self._state, shocks=shocks)
        lo, hi = self._bounds_at(pre)
        a_real = self._unscale_one(action_arr[0], lo, hi)
        ctrl_tensor = torch.as_tensor([a_real], dtype=torch.float32)

        post = self.bp.post_function(
            self._state,
            controls={self.control_sym: ctrl_tensor},
            shocks=shocks,
        )

        reward = sum(_to_scalar(post[sym]) for sym in self.reward_syms)

        next_state = {
            sym: post[sym].detach()
            if isinstance(post[sym], torch.Tensor)
            else post[sym]
            for sym in self.bp.get_arrival_states()
        }
        self._state = next_state

        self._steps += 1
        truncated = self._steps >= self.max_episode_steps
        terminated = False

        # Draw next-period shocks now so the observation the agent sees
        # next is consistent with the (state, shock) that will drive the
        # next transition.
        self._pending_shocks = _draw_period_shocks(self.bp, self._rng)
        next_obs = self._compute_obs(next_state, self._pending_shocks)

        info = {
            "discount": _to_scalar(self.bp.resolve_discount_factor(post)),
            "action_unscaled": a_real,
            "bounds": (lo, hi),
        }
        return next_obs, float(reward), terminated, truncated, info

    # -- public helpers ---------------------------------------------------

    def unscale_action(self, action_norm, obs) -> np.ndarray:
        """Map normalised action(s) in ``[-1, 1]`` to the control's real value.

        For each row of ``obs``, evaluates the control's per-state bounds
        ``lo``/``hi`` (from ``Control.lower_bound`` / ``Control.upper_bound``
        applied to the iset values) and unscales the corresponding action via ::

            a_real = (lo + ε·span) + ½ (a_norm + 1) (span − 2 ε·span)

        where ``span = hi − lo`` and ``ε = bound_clearance``. This is the same
        transform :meth:`step` applies internally, so callers who run a trained
        SB3 model outside the gym loop (e.g. for diagnostics or to build a
        skagent decision rule) can use it directly instead of re-deriving the
        unscaling.

        Parameters
        ----------
        action_norm : array-like
            Normalised action(s). Scalar, shape ``(N,)``, or shape ``(N, 1)``.
            Values outside ``[-1, 1]`` are clipped.
        obs : array-like
            Observation(s). Shape ``(|iset|,)`` for a single state, or
            ``(N, |iset|)`` for a batch. Must broadcast with ``action_norm``
            along the leading axis.

        Returns
        -------
        np.ndarray, shape ``(N,)``
            Unscaled action values. Always 1-D; scalar inputs return shape ``(1,)``.
        """
        a = np.asarray(action_norm, dtype=np.float32).reshape(-1)
        obs_arr = np.atleast_2d(np.asarray(obs, dtype=np.float32))
        if obs_arr.shape[1] != len(self.iset):
            raise ValueError(
                f"obs has {obs_arr.shape[1]} columns but iset has "
                f"{len(self.iset)} variables ({self.iset})"
            )
        if obs_arr.shape[0] != a.shape[0]:
            raise ValueError(
                f"action_norm has {a.shape[0]} entries but obs has "
                f"{obs_arr.shape[0]} rows; shapes must match along leading axis"
            )

        out = np.empty(a.shape[0], dtype=np.float32)
        for i in range(a.shape[0]):
            lo, hi = self._bounds_at_iset(obs_arr[i])
            out[i] = self._unscale_one(a[i], lo, hi)
        return out

    # -- internal helpers -------------------------------------------------

    def _unscale_one(self, a_norm: float, lo: float, hi: float) -> float:
        span = max(hi - lo, 1e-12)
        eps = self.bound_clearance * span
        a = float(np.clip(a_norm, -1.0, 1.0))
        return (lo + eps) + 0.5 * (a + 1.0) * (span - 2.0 * eps)

    def _bounds_at_iset(self, iset_values) -> tuple[float, float]:
        return self._bounds_at({sym: iset_values[i] for i, sym in enumerate(self.iset)})

    def _bounds_at(self, pre: dict) -> tuple[float, float]:
        iset_vals = [pre[s] for s in self.iset]
        lo = (
            _to_scalar(self.control.lower_bound(*iset_vals))
            if self.control.lower_bound is not None
            else self.default_lower
        )
        hi = (
            _to_scalar(self.control.upper_bound(*iset_vals))
            if self.control.upper_bound is not None
            else self.default_upper
        )
        # lo == hi is a valid single-point feasible set (the natural borrowing
        # limit collapses the choice to one action); only a truly inverted
        # bound (hi < lo) is an error.
        if hi < lo:
            raise ValueError(
                f"Control {self.control_sym!r} has inverted bounds at "
                f"pre-state {pre}: lo={lo}, hi={hi}"
            )
        return lo, hi

    def _compute_obs(self, state: dict, shocks: dict) -> np.ndarray:
        pre = self.bp.compute_pre_state(self.control_sym, state, shocks=shocks)
        return np.array([_to_scalar(pre[s]) for s in self.iset], dtype=np.float32)
