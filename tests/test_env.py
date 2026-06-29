"""Tests for skagent.env (Environment and GymEnv)."""

import numpy as np
import pytest
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

from skagent.bellman import BellmanPeriod
from skagent.block import Control, DBlock
from skagent.distributions import Uniform
from skagent.env import Environment, GymEnv, discounted_rollout_reward
from skagent.models.benchmarks import d2_block, d2_calibration

from tests.conftest import case_0, case_1, case_5, case_7, case_10


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


def test_environment_step_returns_full_transition():
    env = Environment(case_0["bp"], {"a": Uniform(low=0.0, high=1.0)})
    state, action, reward, next_state, discount, obs = env.step({"c": lambda a: a})
    assert set(state) == {"a"}
    assert set(action) == {"c"}
    assert set(reward) == {"u"}
    assert set(next_state) == {"a"}
    assert set(obs) == {"a"}
    assert float(discount) == pytest.approx(0.9)


def test_environment_seeded_reset_is_reproducible():
    bp = case_1["bp"]
    initial = {"a": Uniform(low=0.0, high=1.0)}
    s1 = Environment(bp, initial, rng=np.random.default_rng(123)).state
    s2 = Environment(bp, initial, rng=np.random.default_rng(123)).state
    assert torch.allclose(s1["a"], s2["a"])


# ---------------------------------------------------------------------------
# GymEnv: spaces, episode lifecycle, bounds
# ---------------------------------------------------------------------------


def test_gymenv_spaces_match_iset_and_normalized_action():
    env = GymEnv(case_5["bp"], {"a": Uniform(low=0.1, high=1.0)}, seed=0)
    assert env.action_space.shape == (1,)
    assert env.action_space.low.tolist() == [-1.0]
    assert env.action_space.high.tolist() == [1.0]
    # case_5: Control(["a"]) → obs is 1-D
    assert env.observation_space.shape == (1,)


def test_gymenv_truncation_at_max_steps():
    env = GymEnv(
        case_0["bp"], {"a": Uniform(low=0.0, high=1.0)}, max_episode_steps=3, seed=0
    )
    env.reset()
    truncs = []
    for _ in range(3):
        _, _, term, trunc, _ = env.step(env.action_space.sample())
        truncs.append((term, trunc))
    assert truncs == [(False, False), (False, False), (False, True)]


def test_gymenv_unscaled_action_respects_state_dependent_bounds():
    # case_5: lower=0, upper=a — unscaled action must lie in [0, a].
    env = GymEnv(case_5["bp"], {"a": Uniform(low=0.2, high=1.0)}, seed=0)
    env.reset()
    rng = np.random.default_rng(7)
    for _ in range(20):
        a_state = float(env._state["a"].item())
        action = rng.uniform(-1.0, 1.0, size=(1,)).astype(np.float32)
        _, _, _, _, info = env.step(action)
        lo, hi = info["bounds"]
        assert lo == 0.0
        assert hi == pytest.approx(a_state)
        assert lo <= info["action_unscaled"] <= hi


def test_gymenv_partial_bounds_uses_default_upper():
    # case_7: lower=1 (set), upper=None → falls back to default_upper.
    env = GymEnv(
        case_7["bp"], {"a": Uniform(low=0.0, high=1.0)}, default_upper=5.0, seed=0
    )
    env.reset()
    _, _, _, _, info = env.step(np.array([0.0], dtype=np.float32))
    assert info["bounds"] == (1.0, 5.0)


def test_gymenv_multi_control_raises():
    with pytest.raises(ValueError, match="single control"):
        GymEnv(case_10["bp"], {"a": Uniform(low=0.0, high=1.0)})


# ---------------------------------------------------------------------------
# SB3 integration
# ---------------------------------------------------------------------------


@pytest.fixture
def d2_env():
    bp = BellmanPeriod(d2_block, "DiscFac", d2_calibration)
    return GymEnv(bp, {"a": Uniform(low=0.5, high=2.0)}, max_episode_steps=20, seed=0)


def test_check_env_passes_on_d2_benchmark(d2_env):
    check_env(d2_env)


def test_ppo_learn_smoke(d2_env):
    model = PPO("MlpPolicy", d2_env, verbose=0, n_steps=32, batch_size=16, device="cpu")
    model.learn(total_timesteps=64)
    obs, _ = d2_env.reset()
    action, _ = model.predict(obs, deterministic=True)
    assert action.shape == (1,)


# ---------------------------------------------------------------------------
# discounted_rollout_reward
# ---------------------------------------------------------------------------


def test_discounted_rollout_reward_matches_geometric_sum():
    # Constant reward (c = 1 each period) under a constant discount β = 0.5
    # gives a closed-form discounted return: Σ_{t=0}^{T-1} β^t.
    block = DBlock(
        **{
            "name": "rollout_const",
            "dynamics": {
                "c": Control(["a"]),
                "a": lambda a: a,  # state persists
                "u": lambda c: c,  # reward equals the control
            },
            "reward": {"u": "consumer"},
        }
    )
    bp = BellmanPeriod(block, "beta", {"beta": 0.5})
    dr = {"c": lambda a: a * 0.0 + 1.0}  # constant 1, shape-matched to a

    T = 5
    expected = sum(0.5**t for t in range(T))
    got = discounted_rollout_reward(
        bp, dr, {"a": Uniform(low=0.0, high=1.0)}, T, np.random.default_rng(0)
    )
    assert got == pytest.approx(expected)


def test_discounted_rollout_reward_zero_for_optimal_policy():
    # case_1: u = -(θ - c)². The optimal rule c = θ yields zero reward every
    # period regardless of the realized shocks, so the discounted return is 0.
    bp = case_1["bp"]
    got = discounted_rollout_reward(
        bp,
        case_1["optimal_dr"],
        {"a": Uniform(low=0.0, high=1.0)},
        10,
        np.random.default_rng(0),
    )
    assert got == pytest.approx(0.0, abs=1e-6)


def test_discounted_rollout_reward_is_seed_reproducible():
    bp = case_1["bp"]
    dr = {"c": lambda a, theta: a}  # suboptimal → shock-dependent, nonzero reward
    initial = {"a": Uniform(low=0.0, high=1.0)}

    r1 = discounted_rollout_reward(bp, dr, initial, 8, np.random.default_rng(7))
    r2 = discounted_rollout_reward(bp, dr, initial, 8, np.random.default_rng(7))
    r3 = discounted_rollout_reward(bp, dr, initial, 8, np.random.default_rng(99))

    assert r1 == pytest.approx(r2)  # same seed → identical rollout
    assert r1 != pytest.approx(r3)  # different seed → different rollout
