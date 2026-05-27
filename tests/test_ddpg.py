import torch
from skagent.algos.ddpg import ddpg_training_loop
from skagent.bellman import BellmanPeriod
from skagent.distributions import Bernoulli, MeanOneLogNormal
from skagent.models.benchmarks import (
    d2_block,
    d2_calibration,
    get_analytical_policy,
    get_benchmark_calibration,
    get_benchmark_model,
    get_test_states,
)

TEST_SEED = 42
NUM_EPISODES = 5
MAX_STEPS = 20
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class TestDDPGTraining:
    def test_ddpg_d2_training_loop(self):
        """DDPG training loop runs without error on the D-2 benchmark (infinite-horizon CRRA)."""
        bp = BellmanPeriod(d2_block, "DiscFac", d2_calibration)
        initial = {"a": MeanOneLogNormal(sigma=1)}

        agent, episode_rewards = ddpg_training_loop(
            bp,
            initial,
            num_episodes=NUM_EPISODES,
            max_steps_per_episode=MAX_STEPS,
            batch_size=64,
            device=DEVICE,
            random_seed=TEST_SEED,
        )

        assert len(episode_rewards) == NUM_EPISODES
        assert len(agent.replay_buffer) == NUM_EPISODES * MAX_STEPS

    def test_ddpg_d2_convergence(self):
        """DDPG on D-2 converges toward the analytical solution c_t = κ*(m_t + H)."""
        bp = BellmanPeriod(d2_block, "DiscFac", d2_calibration)
        initial = {"a": MeanOneLogNormal(sigma=1)}

        agent, _ = ddpg_training_loop(
            bp,
            initial,
            num_episodes=200,
            max_steps_per_episode=100,
            batch_size=64,
            hidden_dim=64,
            device=DEVICE,
            random_seed=TEST_SEED,
            warmup_episodes=20,
        )

        # Analytical solution over test states
        test_states = get_test_states("D-2")
        c_analytical = get_analytical_policy("D-2")(test_states, {}, d2_calibration)[
            "c"
        ]

        # Actor input is the info-set m = a*R + y (not the arrival state a)
        m = test_states["a"].float() * d2_calibration["R"] + d2_calibration["y"]
        m_in = m.unsqueeze(1).to(DEVICE)  # [n_test, 1]

        agent.actor.eval()
        with torch.no_grad():
            c_learned = agent.actor(m_in).flatten().cpu()
        agent.actor.train()

        mape = (torch.abs(c_learned - c_analytical) / c_analytical.abs()).mean().item()
        assert mape < 0.25, (
            f"MAPE {mape:.3f} exceeds 25% — DDPG may not be learning D-2 correctly"
        )

    def test_ddpg_d3_training_loop(self):
        """DDPG training loop runs without error on D-3 benchmark (Blanchard mortality shock)."""
        block = get_benchmark_model("D-3")
        calibration = get_benchmark_calibration("D-3")
        block.construct_shocks(calibration)
        bp = BellmanPeriod(block, "DiscFac", calibration)

        initial = {"a": MeanOneLogNormal(sigma=0.5), "liv": Bernoulli(p=1.0)}

        agent, episode_rewards = ddpg_training_loop(
            bp,
            initial,
            num_episodes=NUM_EPISODES,
            max_steps_per_episode=MAX_STEPS,
            batch_size=64,
            device=DEVICE,
            random_seed=TEST_SEED,
        )

        assert len(episode_rewards) == NUM_EPISODES
        assert len(agent.replay_buffer) == NUM_EPISODES * MAX_STEPS

    def test_ddpg_u1_training_loop(self):
        """DDPG training loop runs without error on U-1 benchmark (Hall random walk, income shock)."""
        block = get_benchmark_model("U-1")
        calibration = get_benchmark_calibration("U-1")
        block.construct_shocks(calibration)
        bp = BellmanPeriod(block, "DiscFac", calibration)

        initial = {"A": MeanOneLogNormal(sigma=0.5)}

        agent, episode_rewards = ddpg_training_loop(
            bp,
            initial,
            num_episodes=NUM_EPISODES,
            max_steps_per_episode=MAX_STEPS,
            batch_size=64,
            device=DEVICE,
            random_seed=TEST_SEED,
        )

        assert len(episode_rewards) == NUM_EPISODES
        assert len(agent.replay_buffer) == NUM_EPISODES * MAX_STEPS
