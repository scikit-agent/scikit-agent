"""Tests for skagent.algos.sb3 (PPOAgent)."""

import numpy as np
import pytest
import torch
from stable_baselines3.common.callbacks import BaseCallback

from skagent.algos.sb3 import PPOAgent
from skagent.bellman import BellmanPeriod
from skagent.distributions import Uniform
from skagent.env import Environment
from skagent.models.benchmarks import (
    d2_analytical_policy,
    d2_block,
    d2_calibration,
    d4_block,
    d4_calibration,
    d4_vfi_reference_policy,
    u1_analytical_policy,
    u1_block,
    u1_calibration,
    u2_analytical_policy,
    u2_block,
    u2_calibration,
)

from tests.conftest import case_0, case_2, case_5


# Short PPO settings so tests stay fast.
PPO_KWARGS = {"n_steps": 32, "batch_size": 16, "n_epochs": 2}


@pytest.fixture
def case0_agent():
    bp = case_0["bp"]
    return PPOAgent(
        bp,
        {"a": Uniform(low=0.0, high=1.0)},
        max_episode_steps=16,
        seed=0,
        ppo_kwargs=PPO_KWARGS,
    )


@pytest.fixture
def d2_agent():
    bp = BellmanPeriod(d2_block, "DiscFac", d2_calibration)
    return PPOAgent(
        bp,
        {"a": Uniform(low=0.5, high=2.0)},
        max_episode_steps=16,
        seed=0,
        ppo_kwargs=PPO_KWARGS,
    )


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_gamma_defaults_to_model_discount_factor(d2_agent):
    assert d2_agent.model.gamma == pytest.approx(d2_calibration["DiscFac"])


def test_gamma_falls_back_when_discount_not_scalar():
    # BellmanPeriod with a discount_variable that is not a numeric scalar in
    # calibration → _default_gamma should fall back to PPO's 0.99.
    bp = BellmanPeriod(case_0["block"], "no_such_var", case_0["calibration"])
    agent = PPOAgent(
        bp,
        {"a": Uniform(low=0.0, high=1.0)},
        max_episode_steps=8,
        seed=0,
        ppo_kwargs=PPO_KWARGS,
    )
    assert agent.model.gamma == pytest.approx(0.99)


def test_gym_and_ppo_kwargs_forwarded():
    agent = PPOAgent(
        case_0["bp"],
        {"a": Uniform(low=0.0, high=1.0)},
        max_episode_steps=8,
        seed=0,
        gym_kwargs={"default_upper": 7.0},
        ppo_kwargs={**PPO_KWARGS, "gamma": 0.5},
    )
    assert agent.env.default_upper == 7.0
    assert agent.model.gamma == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Training callback wiring
# ---------------------------------------------------------------------------


def test_learn_records_episode_rewards(case0_agent):
    assert case0_agent.episode_rewards == []
    case0_agent.learn(total_timesteps=64)
    assert len(case0_agent.episode_rewards) > 0
    # max_episode_steps=16 → 64 timesteps gives at least 4 episodes
    assert len(case0_agent.episode_rewards) >= 4


def test_learn_accumulates_across_calls(case0_agent):
    case0_agent.learn(total_timesteps=64)
    n_after_first = len(case0_agent.episode_rewards)
    case0_agent.learn(total_timesteps=64)
    assert len(case0_agent.episode_rewards) > n_after_first


def test_user_callback_runs_alongside_internal_logger(case0_agent):
    class CountSteps(BaseCallback):
        def __init__(self):
            super().__init__()
            self.n = 0

        def _on_step(self) -> bool:
            self.n += 1
            return True

    cb = CountSteps()
    case0_agent.learn(total_timesteps=64, callback=cb)
    assert cb.n == 64
    assert len(case0_agent.episode_rewards) > 0


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------


def test_predict_unscaled_respects_per_state_bounds(d2_agent):
    d2_agent.learn(total_timesteps=64)
    obs = np.array([[1.0], [2.5], [5.0]], dtype=np.float32)
    c = d2_agent.predict_unscaled(obs)
    # D-2 is unconstrained: lo=0 (default), hi=m+H (natural borrowing limit).
    ub = d2_block.dynamics["c"].upper_bound(obs.reshape(-1))
    assert c.shape == (3,)
    assert np.all(c >= 0.0)
    assert np.all(c <= ub)


def test_decision_rule_drives_environment(d2_agent):
    d2_agent.learn(total_timesteps=64)
    dr = d2_agent.decision_rule()
    assert list(dr.keys()) == ["c"]

    env = Environment(
        BellmanPeriod(d2_block, "DiscFac", d2_calibration),
        {"a": Uniform(low=0.5, high=2.0)},
        rng=np.random.default_rng(0),
    )
    env.reset()
    for _ in range(5):
        state, action, reward, _, _, _ = env.step(dr)
        c = float(action["c"][0])
        m = float(state["a"][0]) * d2_calibration["R"] + d2_calibration["y"]
        ub = d2_block.dynamics["c"].upper_bound(m)  # m + H, natural borrowing limit
        assert 0.0 <= c <= ub  # respects D-2 natural borrowing limit
        assert torch.is_tensor(reward["u"])


def test_decision_rule_rejects_wrong_iset_arity(d2_agent):
    dr = d2_agent.decision_rule()
    with pytest.raises(TypeError, match="iset arguments"):
        dr["c"](torch.tensor([1.0]), torch.tensor([2.0]))  # iset=['m'] → 1 arg


# ---------------------------------------------------------------------------
# Policy snapshots
# ---------------------------------------------------------------------------

SNAPSHOT_OBS = np.array([[1.0], [2.5], [5.0]], dtype=np.float32)


def test_snapshot_matches_live_policy_at_capture(d2_agent):
    d2_agent.learn(total_timesteps=64)
    snap = d2_agent.snapshot()
    assert np.allclose(
        snap.predict_unscaled(SNAPSHOT_OBS), d2_agent.predict_unscaled(SNAPSHOT_OBS)
    )


def test_snapshot_is_frozen_across_further_training(d2_agent):
    d2_agent.learn(total_timesteps=64)
    snap = d2_agent.snapshot()
    before = snap.predict_unscaled(SNAPSHOT_OBS)

    # Further training mutates the live agent but must not touch the snapshot.
    d2_agent.learn(total_timesteps=256, reset_num_timesteps=False)
    after = snap.predict_unscaled(SNAPSHOT_OBS)

    assert np.array_equal(before, after)  # snapshot exactly frozen
    assert not np.array_equal(before, d2_agent.predict_unscaled(SNAPSHOT_OBS))


def test_snapshot_decision_rule_drives_environment(d2_agent):
    d2_agent.learn(total_timesteps=64)
    dr = d2_agent.snapshot().decision_rule()
    assert list(dr.keys()) == ["c"]

    env = Environment(
        BellmanPeriod(d2_block, "DiscFac", d2_calibration),
        {"a": Uniform(low=0.5, high=2.0)},
        rng=np.random.default_rng(0),
    )
    env.reset()
    for _ in range(5):
        state, action, reward, _, _, _ = env.step(dr)
        c = float(action["c"][0])
        m = float(state["a"][0]) * d2_calibration["R"] + d2_calibration["y"]
        ub = d2_block.dynamics["c"].upper_bound(m)  # m + H, natural borrowing limit
        assert 0.0 <= c <= ub  # respects D-2 natural borrowing limit


def test_snapshot_decision_rule_rejects_wrong_iset_arity(d2_agent):
    d2_agent.learn(total_timesteps=32)
    dr = d2_agent.snapshot().decision_rule()
    with pytest.raises(TypeError, match="iset arguments"):
        dr["c"](torch.tensor([1.0]), torch.tensor([2.0]))


# ---------------------------------------------------------------------------
# Convergence tests — exercise the same training setup as
# diagnostic_ppo_allbench.py; tolerances are derived from that diagnostic's
# observed errors (with generous headroom for SB3 non-determinism).
# ---------------------------------------------------------------------------

CONVERGENCE_PPO_KWARGS = {"n_steps": 256, "batch_size": 64, "n_epochs": 4}
CONVERGENCE_TRAIN_STEPS = 10_000
CONVERGENCE_MAX_EPISODE_STEPS = 64
CONVERGENCE_OBS_GRID = np.linspace(0.1, 1.0, 10, dtype=np.float32).reshape(-1, 1)


@pytest.mark.parametrize(
    "case,gym_kwargs,mae_tol,max_err_tol",
    [
        # case_0: c*=0, no bounds → use wide default range. Diagnostic: MAE=0.012.
        (case_0, {"default_lower": -1.0, "default_upper": 1.0}, 0.05, 0.10),
        # case_2: c*=0 (θ hidden), no bounds. Diagnostic: MAE=0.014.
        (case_2, {"default_lower": -1.0, "default_upper": 1.0}, 0.05, 0.10),
        # case_5: c*=a, Control(upper=a, lower=0). Diagnostic: MAE=0.107.
        (case_5, {}, 0.20, 0.30),
    ],
    ids=["case_0_c=0", "case_2_c=0_hidden_shock", "case_5_c=a_constrained"],
)
def test_ppo_converges_to_optimal(case, gym_kwargs, mae_tol, max_err_tol):
    agent = PPOAgent(
        case["bp"],
        {"a": Uniform(low=0.1, high=1.0)},
        max_episode_steps=CONVERGENCE_MAX_EPISODE_STEPS,
        seed=0,
        gym_kwargs=gym_kwargs,
        ppo_kwargs=CONVERGENCE_PPO_KWARGS,
    )
    agent.learn(total_timesteps=CONVERGENCE_TRAIN_STEPS)

    obs = CONVERGENCE_OBS_GRID
    c_learned = agent.predict_unscaled(obs)

    # Match the diagnostic: evaluate optimal_dr at each obs row and clip to
    # the per-state bounds the env actually enforces.
    optimal_c = case["optimal_dr"]["c"]
    c_opt = np.empty(obs.shape[0], dtype=np.float32)
    for i in range(obs.shape[0]):
        lo, hi = agent.env._bounds_at_iset(obs[i])
        c_opt[i] = float(np.clip(float(optimal_c(*obs[i])), lo, hi))

    err = c_learned - c_opt
    mae = float(np.mean(np.abs(err)))
    max_err = float(np.max(np.abs(err)))
    assert mae <= mae_tol, f"MAE={mae:.4f} exceeds tolerance {mae_tol}"
    assert max_err <= max_err_tol, (
        f"MaxErr={max_err:.4f} exceeds tolerance {max_err_tol}"
    )


# ---------------------------------------------------------------------------
# Convergence tests — analytically solvable benchmark models.
#
# Same idea as the conftest convergence factory above, but the targets are the
# closed-form consumption-savings benchmarks in skagent.models.benchmarks.
# These are genuine infinite-horizon savings problems, so (unlike the conftest
# cases, whose myopic optimum already equals the analytical one) PPO only
# converges with a much larger budget: the longer episodes, bigger rollout
# buffer, and ~130k timesteps documented in
# examples/algorithms/plot_sb3_ppo.py. Each run takes ~1 minute, so the whole
# group is marked ``slow`` and is deselected unless pytest is given --runslow.
#
# Settings below mirror that example, and tolerances are derived from the
# policy's observed MAE/MaxErr against the closed form on this grid (seed 0,
# 130k steps) with generous headroom for SB3 non-determinism.
#
# D-4 is included but compares against a numerical VFI reference rather than a
# closed form: its binding ``c <= m`` constraint precludes an analytical policy.
# Its ``[0, m]`` action range is O(1) and well-scaled, so PPO converges as
# tightly as U-2 with no special handling.
#
# Excluded / xfailed benchmarks:
#   D-1  finite horizon — time-as-state ``t`` ticks toward ``T``; not a single
#        steady-state truncation this PPO loop can target.
#   D-2  xfailed below — see the mark reason.
#   D-3  i.i.d. mortality — under truncated-horizon PPO the agent overconsumes
#        at high wealth and does not converge to the κ_s analytical rule.
#   U-3  buffer stock — no closed-form policy to compare against.
# ---------------------------------------------------------------------------

BENCH_CONVERGENCE_PPO_KWARGS = {
    "n_steps": 2048,
    "batch_size": 64,
    "n_epochs": 10,
    "learning_rate": 3e-4,
}
BENCH_CONVERGENCE_TRAIN_STEPS = 130_000
BENCH_CONVERGENCE_MAX_EPISODE_STEPS = 200
# Wide cash-on-hand grid matching the worked example (m ∈ [0.5, 10]).
BENCH_CONVERGENCE_OBS_GRID = np.linspace(0.5, 10.0, 41, dtype=np.float32).reshape(-1, 1)


def _scalar(x):
    if hasattr(x, "item"):
        return float(x.item())
    return float(np.asarray(x).reshape(-1)[0])


# Each benchmark control has iset ``[m]`` (cash-on-hand). The analytical
# policies are keyed on the arrival state, so we invert ``m = a*R + y`` (or the
# normalized ``m = R*a + 1`` for U-2) to recover it, then evaluate the closed
# form. The test clips the result to the env's per-state bounds afterwards.


def _d2_optimal_c(obs_row):
    a = (float(obs_row[0]) - d2_calibration["y"]) / d2_calibration["R"]
    return _scalar(d2_analytical_policy({"a": a}, {}, d2_calibration)["c"])


def _d4_optimal_c(obs_row):
    # D-4 has no closed form; the oracle is the validated VFI reference policy.
    a = (float(obs_row[0]) - d4_calibration["y"]) / d4_calibration["R"]
    return _scalar(d4_vfi_reference_policy({"a": a}, {}, d4_calibration)["c"])


def _u1_optimal_c(obs_row):
    y = u1_calibration["y_mean"]
    A = (float(obs_row[0]) - y) / u1_calibration["R"]
    return _scalar(u1_analytical_policy({"A": A, "y": y}, {}, u1_calibration)["c"])


def _u2_optimal_c(obs_row):
    a = (float(obs_row[0]) - 1.0) / u2_calibration["R"]
    return _scalar(
        u2_analytical_policy({"a": torch.tensor([a])}, {}, u2_calibration)["c"]
    )


@pytest.mark.slow
@pytest.mark.parametrize(
    "block,calibration,initial,optimal_fn,mae_tol,max_err_tol",
    [
        # D-2: xfailed. Its borrowing bound was relaxed to the natural limit
        # c <= m + H (H = y/(R-1) ~= 33), so GymEnv's action range [0, m+H] is
        # ~30x wider than the O(1) optimal consumption. The optimum then sits at
        # the action-space edge (a_norm ~= -0.93) while PPO inits at the midpoint
        # and the per-step CRRA gradient pushes the wrong way, so PPO barely
        # moves (MAE ~= 18 vs tol 1.2). The fix is an artificial action-scale
        # override -- the RL analog of VFI's artificial_borrowing_constraint --
        # which is deferred; D-4 covers the timely convergence demo meanwhile.
        pytest.param(
            d2_block,
            d2_calibration,
            {"a": Uniform(low=0.01, high=5.0)},
            _d2_optimal_c,
            1.2,
            2.5,
            id="D-2",
            marks=pytest.mark.xfail(
                reason=(
                    "natural borrowing limit c <= m+H makes the action range "
                    "~30x the optimal c; optimum sits at the action-space edge "
                    "and PPO does not move (MAE ~= 18). Needs an artificial "
                    "action-scale override (deferred)."
                ),
                strict=False,
            ),
        ),
        # D-4: binding c <= m constraint, no closed form -> VFI reference oracle.
        # Well-scaled O(1) action range; converges as tightly as U-2 with no
        # special handling. Observed MAE=0.21, MaxErr=0.57.
        pytest.param(
            d4_block,
            d4_calibration,
            {"a": Uniform(low=0.01, high=5.0)},
            _d4_optimal_c,
            0.5,
            1.2,
            id="D-4",
        ),
        # U-1: Hall PIH with a Normal income shock (noisier). Observed
        # MAE=0.98, MaxErr=1.99.
        pytest.param(
            u1_block,
            u1_calibration,
            {"A": Uniform(low=0.01, high=5.0), "y": Uniform(low=0.9, high=1.1)},
            _u1_optimal_c,
            1.5,
            2.8,
            id="U-1",
        ),
        # U-2: normalized log-utility PIH; converges tightest. Observed
        # MAE=0.23, MaxErr=0.80.
        pytest.param(
            u2_block,
            u2_calibration,
            {"a": Uniform(low=0.01, high=5.0)},
            _u2_optimal_c,
            0.5,
            1.2,
            id="U-2",
        ),
    ],
)
def test_benchmark_ppo_converges_to_analytical(
    block, calibration, initial, optimal_fn, mae_tol, max_err_tol
):
    bp = BellmanPeriod(block, "DiscFac", calibration)
    agent = PPOAgent(
        bp,
        initial,
        max_episode_steps=BENCH_CONVERGENCE_MAX_EPISODE_STEPS,
        seed=0,
        ppo_kwargs=BENCH_CONVERGENCE_PPO_KWARGS,
    )
    agent.learn(total_timesteps=BENCH_CONVERGENCE_TRAIN_STEPS)

    obs = BENCH_CONVERGENCE_OBS_GRID
    c_learned = agent.predict_unscaled(obs)

    # Closed-form action per row, clipped to the per-state bounds the env
    # enforces — the policy PPO can actually realize in this env.
    c_opt = np.empty(obs.shape[0], dtype=np.float32)
    for i in range(obs.shape[0]):
        lo, hi = agent.env._bounds_at_iset(obs[i])
        c_opt[i] = float(np.clip(optimal_fn(obs[i]), lo, hi))

    err = c_learned - c_opt
    mae = float(np.mean(np.abs(err)))
    max_err = float(np.max(np.abs(err)))
    assert mae <= mae_tol, f"MAE={mae:.4f} exceeds tolerance {mae_tol}"
    assert max_err <= max_err_tol, (
        f"MaxErr={max_err:.4f} exceeds tolerance {max_err_tol}"
    )
