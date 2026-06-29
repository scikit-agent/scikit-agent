from tests.conftest import case_1
from skagent.bellman import BellmanPeriod
from skagent.algos.ddpg import ddpg_training_loop
from skagent.distributions import MeanOneLogNormal
import torch

import matplotlib.pyplot as plt
import numpy as np

TEST_SEED = 51
NUM_EPISODES = 5
MAX_STEPS = 20
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

case_1["block"].construct_shocks(
    case_1["calibration"], rng=np.random.default_rng(TEST_SEED)
)
bp = BellmanPeriod(case_1["block"], "beta", case_1["calibration"])
initial = {"a": MeanOneLogNormal(sigma=1)}

agent, episode_rewards = ddpg_training_loop(
    bp,
    initial,
    num_episodes=NUM_EPISODES * 5,
    max_steps_per_episode=MAX_STEPS * 5,
    batch_size=64,
    device=DEVICE,
    random_seed=TEST_SEED,
    warmup_episodes=5,
    random_rollout_every=3,
)

a_test_states = torch.linspace(-1.0, 1.0, 11).to(DEVICE)
theta_test_states = torch.linspace(-1.0, 1.0, 11).to(DEVICE)

test_states = torch.cartesian_prod(a_test_states, theta_test_states)

print(test_states)

agent.actor.eval()
with torch.no_grad():
    c_learned = agent.actor(test_states).flatten()

    c_optimal = torch.vmap(case_1["optimal_dr"]["c"])(
        test_states[:, 0], test_states[:, 1]
    )

    print(c_learned - c_optimal)
    print(c_learned - test_states[:, 1])

print(episode_rewards)

print(f"Mean Absolute Error: {torch.abs(c_learned - c_optimal).mean()}")


plt.plot(c_learned.cpu(), label="learned")
plt.plot(c_optimal.cpu(), label="optimal")
plt.plot((c_learned - c_optimal).cpu(), label="diff")
plt.legend()
plt.title("Error across test cases")
plt.show()

plt.plot(episode_rewards)
plt.title("Episode rewards")
plt.show()
