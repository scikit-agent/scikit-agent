"""
Stable Baselines3 wrappers for :class:`~skagent.bellman.BellmanPeriod` models.

Provides :class:`PPOAgent`, a thin wrapper around SB3's PPO that:

* builds a :class:`skagent.env.GymEnv` from a ``BellmanPeriod`` + initial
  state distribution,
* delegates training to ``stable_baselines3.PPO.learn``,
* exposes a :meth:`PPOAgent.decision_rule` that returns the trained policy
  as a skagent-style ``{control_sym: callable}`` dict — i.e. the same shape
  consumed by :class:`skagent.env.Environment` and the rest of the skagent
  decision-rule API. Actions are unscaled back to real units via
  :meth:`GymEnv.unscale_action`, so downstream code does not see the
  ``[-1, 1]`` SB3 representation.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList

from skagent.bellman import BellmanPeriod
from skagent.env import GymEnv


class _EpisodeRewardLogger(BaseCallback):
    """Append each completed episode's undiscounted reward to a list.

    SB3 auto-wraps single envs in ``Monitor``, which populates
    ``info["episode"]`` at episode boundaries — we just collect those.
    """

    def __init__(self, sink: list[float]) -> None:
        super().__init__()
        self._sink = sink

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            ep = info.get("episode")
            if ep is not None:
                self._sink.append(float(ep["r"]))
        return True


class PPOAgent:
    """Train SB3's PPO on a ``BellmanPeriod`` and emit a skagent decision rule.

    Parameters
    ----------
    bp : BellmanPeriod
        Model definition.
    initial : dict
        Maps arrival-state symbols to ``skagent`` ``Distribution`` objects,
        used by :class:`GymEnv` to sample fresh initial states on
        ``reset``.
    max_episode_steps : int, optional
        Episode horizon for the underlying ``GymEnv``. Default 200.
    seed : int, optional
        Seed for both the environment and the PPO algorithm.
    gym_kwargs : dict, optional
        Extra keyword arguments forwarded to :class:`GymEnv` (e.g.
        ``default_lower``, ``default_upper``, ``bound_clearance``,
        ``control_sym``).
    ppo_kwargs : dict, optional
        Extra keyword arguments forwarded to ``stable_baselines3.PPO``
        (e.g. ``n_steps``, ``batch_size``, ``learning_rate``,
        ``n_epochs``, ``policy_kwargs``). ``gamma`` defaults to
        ``bp.calibration[bp.discount_variable]`` if it is a finite scalar;
        callers can override by passing ``gamma`` here.
    policy : str, optional
        SB3 policy class string. Default ``"MlpPolicy"``.
    device : str, optional
        Torch device for PPO. Default ``"cpu"`` (SB3's recommended default
        for ``MlpPolicy``).
    verbose : int, optional
        Verbosity passed to PPO. Default 0.

    Attributes
    ----------
    env : GymEnv
        The constructed gymnasium environment.
    model : stable_baselines3.PPO
        The SB3 model. ``None`` until :meth:`learn` is called the first
        time (constructed lazily so callers can inspect ``env`` without
        paying PPO's setup cost).
    """

    def __init__(
        self,
        bp: BellmanPeriod,
        initial: dict,
        *,
        max_episode_steps: int = 200,
        seed: Optional[int] = None,
        gym_kwargs: Optional[dict] = None,
        ppo_kwargs: Optional[dict] = None,
        policy: str = "MlpPolicy",
        device: str = "cpu",
        verbose: int = 0,
    ) -> None:
        self.bp = bp
        self.initial = initial
        self.seed = seed

        self.env = GymEnv(
            bp,
            initial,
            max_episode_steps=max_episode_steps,
            seed=seed,
            **(gym_kwargs or {}),
        )

        ppo_kwargs = dict(ppo_kwargs or {})
        ppo_kwargs.setdefault("gamma", self._default_gamma())
        if seed is not None:
            ppo_kwargs.setdefault("seed", seed)
        ppo_kwargs.setdefault("device", device)
        ppo_kwargs.setdefault("verbose", verbose)

        self.model: PPO = PPO(policy, self.env, **ppo_kwargs)
        self.episode_rewards: list[float] = []

    # ---- training -------------------------------------------------------

    def learn(
        self,
        total_timesteps: int,
        callback: Any = None,
        **kwargs: Any,
    ) -> "PPOAgent":
        """Run ``PPO.learn``. Returns ``self``.

        Every completed episode's undiscounted reward is appended to
        ``self.episode_rewards`` via an internal SB3 callback; repeated
        ``learn`` calls accumulate. A user-supplied ``callback`` is merged
        with the internal one via ``CallbackList``. Extra ``**kwargs``
        forward to ``model.learn``.
        """
        ep_cb = _EpisodeRewardLogger(self.episode_rewards)
        if callback is None:
            cb: Any = ep_cb
        elif isinstance(callback, list):
            cb = CallbackList([ep_cb, *callback])
        else:
            cb = CallbackList([ep_cb, callback])
        self.model.learn(total_timesteps=total_timesteps, callback=cb, **kwargs)
        return self

    # ---- inference ------------------------------------------------------

    def predict_unscaled(self, obs, deterministic: bool = True) -> np.ndarray:
        """Predict an unscaled action for ``obs``.

        ``obs`` may be a single observation (shape ``(|iset|,)``) or a batch
        (``(N, |iset|)``). Returns a 1-D array of shape ``(N,)``.
        """
        obs_arr = np.atleast_2d(np.asarray(obs, dtype=np.float32))
        action, _ = self.model.predict(obs_arr, deterministic=deterministic)
        return self.env.unscale_action(action, obs_arr)

    def decision_rule(self, deterministic: bool = True) -> dict[str, Callable]:
        """Return a skagent decision rule that uses the trained policy.

        The returned dict has the form ``{control_sym: callable}`` where the
        callable accepts positional arguments matching the control's iset
        order (i.e. the same signature skagent ``Environment.step`` and
        ``BellmanPeriod.decision_function`` call). The callable's output is
        a ``torch.Tensor`` of unscaled action values; the inputs may be
        scalars, numpy arrays, or torch tensors of compatible length.

        Parameters
        ----------
        deterministic : bool, optional
            Whether to use a deterministic (mean) policy. Default ``True`` —
            matches typical skagent decision-rule semantics.
        """
        env = self.env
        model = self.model
        iset_len = len(env.iset)

        def rule(*iset_values):
            if len(iset_values) != iset_len:
                raise TypeError(
                    f"decision rule for {env.control_sym!r} expects "
                    f"{iset_len} iset arguments {env.iset}, got "
                    f"{len(iset_values)}"
                )
            cols = [
                np.asarray(
                    v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else v,
                    dtype=np.float32,
                ).reshape(-1)
                for v in iset_values
            ]
            n = max(c.size for c in cols)
            cols = [np.broadcast_to(c, (n,)) for c in cols]
            obs = np.stack(cols, axis=1)
            action, _ = model.predict(obs, deterministic=deterministic)
            unscaled = env.unscale_action(action, obs)
            return torch.as_tensor(unscaled, dtype=torch.float32)

        return {env.control_sym: rule}

    # ---- internals ------------------------------------------------------

    def _default_gamma(self) -> float:
        """Use the model's discount factor for PPO's ``gamma`` when scalar.

        Falls back to PPO's default (0.99) when the discount variable is not
        a simple numeric scalar in the calibration (e.g. when the period's
        discount factor is itself state-dependent and computed inside the
        dynamics).
        """
        dv = self.bp.discount_variable
        val = self.bp.calibration.get(dv)
        try:
            f = float(val)
        except (TypeError, ValueError):
            return 0.99
        if not np.isfinite(f) or not (0.0 < f < 1.0):
            return 0.99
        return f
