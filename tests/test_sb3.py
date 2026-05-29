"""Tests for skagent.algos.sb3 (PPOAgent)."""

import numpy as np
import pytest
import torch
from stable_baselines3.common.callbacks import BaseCallback

from skagent.algos.sb3 import PPOAgent
from skagent.bellman import BellmanPeriod
from skagent.distributions import Uniform
from skagent.env import Environment
from skagent.models.benchmarks import d2_block, d2_calibration

from tests.conftest import case_0


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
    # D-2: lo=0 (default), hi=m (Control.upper_bound).
    assert c.shape == (3,)
    assert np.all(c >= 0.0)
    assert np.all(c <= obs.reshape(-1))


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
        assert 0.0 <= c <= m  # respects D-2 borrowing constraint
        assert torch.is_tensor(reward["u"])


def test_decision_rule_rejects_wrong_iset_arity(d2_agent):
    dr = d2_agent.decision_rule()
    with pytest.raises(TypeError, match="iset arguments"):
        dr["c"](torch.tensor([1.0]), torch.tensor([2.0]))  # iset=['m'] → 1 arg
