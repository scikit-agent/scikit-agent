import numpy as np
import torch
from skagent.algos.ddpg import DDPG, Environment
from skagent.bellman import BellmanPeriod
from skagent.distributions import Bernoulli, MeanOneLogNormal
from skagent.models.benchmarks import (
    d2_block,
    d2_calibration,
    get_benchmark_calibration,
    get_benchmark_model,
)

TEST_SEED = 42


class TestDDPGTraining:
    def test_ddpg_d2_training_loop(self):
        """DDPG training loop runs without error on the D-2 benchmark (infinite-horizon CRRA)."""
        rng = np.random.default_rng(TEST_SEED)

        bp = BellmanPeriod(d2_block, "DiscFac", d2_calibration)
        initial = {"a": MeanOneLogNormal(sigma=1)}
        env = Environment(bp, initial, rng=rng)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        agent = DDPG(bp, max_action=1.0, device=device)

        num_episodes = 5
        max_steps = 20
        episode_rewards = []

        for episode in range(num_episodes):
            state = env.reset()
            agent.noise.reset()
            episode_reward = 0

            for step in range(max_steps):
                state, action, reward, next_state, discount, obs = env.step(
                    agent.get_decision_rule()
                )

                done = step == max_steps - 1
                agent.replay_buffer.push(
                    state, action, reward, next_state, discount, done, obs
                )
                agent.train(batch_size=64)

                state = next_state
                episode_reward += sum(reward[rsym] for rsym in reward)

                if done:
                    break

            episode_rewards.append(episode_reward)

        assert len(agent.replay_buffer) == num_episodes * max_steps
        assert len(episode_rewards) == num_episodes

    def test_ddpg_d3_training_loop(self):
        """DDPG training loop runs without error on D-3 benchmark (Blanchard mortality shock)."""
        rng = np.random.default_rng(TEST_SEED)

        block = get_benchmark_model("D-3")
        calibration = get_benchmark_calibration("D-3")
        block.construct_shocks(calibration, rng=rng)
        bp = BellmanPeriod(block, "DiscFac", calibration)

        # D-3 arrival states: a (assets) and liv (living indicator, starts at 1)
        initial = {"a": MeanOneLogNormal(sigma=0.5), "liv": Bernoulli(p=1.0)}
        env = Environment(bp, initial, rng=rng)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        agent = DDPG(bp, device=device)

        num_episodes = 5
        max_steps = 20
        episode_rewards = []

        for episode in range(num_episodes):
            state = env.reset()
            agent.noise.reset()
            episode_reward = 0

            for step in range(max_steps):
                state, action, reward, next_state, discount, obs = env.step(
                    agent.get_decision_rule()
                )

                done = step == max_steps - 1
                agent.replay_buffer.push(
                    state, action, reward, next_state, discount, done, obs
                )
                agent.train(batch_size=64)

                state = next_state
                episode_reward += sum(reward[rsym] for rsym in reward)

                if done:
                    break

            episode_rewards.append(episode_reward)

        assert len(agent.replay_buffer) == num_episodes * max_steps
        assert len(episode_rewards) == num_episodes

    def test_ddpg_u1_training_loop(self):
        """DDPG training loop runs without error on U-1 benchmark (Hall random walk, income shock)."""
        rng = np.random.default_rng(TEST_SEED)

        block = get_benchmark_model("U-1")
        calibration = get_benchmark_calibration("U-1")
        block.construct_shocks(calibration, rng=rng)
        bp = BellmanPeriod(block, "DiscFac", calibration)

        # U-1 arrival state: A (financial assets)
        initial = {"A": MeanOneLogNormal(sigma=0.5)}
        env = Environment(bp, initial, rng=rng)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        agent = DDPG(bp, device=device)

        num_episodes = 5
        max_steps = 20
        episode_rewards = []

        for episode in range(num_episodes):
            state = env.reset()
            agent.noise.reset()
            episode_reward = 0

            for step in range(max_steps):
                state, action, reward, next_state, discount, obs = env.step(
                    agent.get_decision_rule()
                )

                done = step == max_steps - 1
                agent.replay_buffer.push(
                    state, action, reward, next_state, discount, done, obs
                )
                agent.train(batch_size=64)

                state = next_state
                episode_reward += sum(reward[rsym] for rsym in reward)

                if done:
                    break

            episode_rewards.append(episode_reward)

        assert len(agent.replay_buffer) == num_episodes * max_steps
        assert len(episode_rewards) == num_episodes
