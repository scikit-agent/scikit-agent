"""
For reference: https://arxiv.org/pdf/1509.02971
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
from skagent.ann import BlockPolicyNet, BlockQNet
from skagent.bellman import BellmanPeriod
from skagent.simulation.monte_carlo import draw_shocks
from skagent.distributions import MeanOneLogNormal


class ReplayBuffer:
    """Experience replay buffer for storing and sampling transitions. [made by Claude]"""

    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones),
        )

    def __len__(self):
        return len(self.buffer)


class OUNoise:
    """Ornstein-Uhlenbeck process for exploration. [Made by Claude]"""

    def __init__(self, action_dim, mu=0.0, theta=0.15, sigma=0.2):
        self.action_dim = action_dim
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_dim) * self.mu
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu

    def sample(self):
        dx = self.theta * (self.mu - self.state) + self.sigma * np.random.randn(
            self.action_dim
        )
        self.state += dx
        return self.state


class DDPG:
    """Deep Deterministic Policy Gradient agent"""

    def __init__(
        self,
        bp: BellmanPeriod,
        max_action=1.0,
        hidden_dim=256,
        lr_actor=1e-4,
        lr_critic=1e-3,
        gamma=0.99,
        tau=0.005,
        device="cpu",
    ):
        self.device = device
        # TODO: this is assuming a fixed discount factor, not taking the discount factor from the BP
        self.gamma = gamma
        self.tau = tau
        self.max_action = max_action

        # state_dim = bp.get_states_dim()
        action_dim = bp.get_action_dim()

        # Actor networks
        self.actor = BlockPolicyNet(bp, width=hidden_dim).to(device)
        self.actor_target = BlockPolicyNet(bp, width=hidden_dim).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)

        # Critic networks
        self.critic = BlockQNet(
            bp,
            width=hidden_dim,
        ).to(device)
        self.critic_target = BlockQNet(
            bp,
            width=hidden_dim,
        ).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)

        # Replay buffer and noise
        self.replay_buffer = ReplayBuffer()
        self.noise = OUNoise(action_dim)

    def get_decision_rule(self, add_noise=True):
        """Produce the decision rule using current policy with optional exploration noise"""
        dr_basic = self.actor.get_decision_rule()

        def decision_rule(*information):
            action = dr_basic[self.actor.control_sym](*information)

            if add_noise:
                for csym in action:
                    action[csym] = action[csym] + self.noise.sample()

                    ## TODO: need to deal with the bounds as provided by the BP
                    action[csym] = np.clip(
                        action[csym], -self.max_action, self.max_action
                    )

            return action

        return {self.actor.control_sym: decision_rule}

    def train(self, batch_size=64):
        """Train the agent on a batch of experiences"""
        if len(self.replay_buffer) < batch_size:
            return None, None

        # Sample batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            batch_size
        )

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        # Update Critic
        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            target_q = self.critic_target(next_states, next_actions)
            target_q = rewards + (1 - dones) * self.gamma * target_q

        current_q = self.critic(states, actions)
        critic_loss = nn.MSELoss()(current_q, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Update Actor
        actor_loss = -self.critic(states, self.actor(states)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Soft update target networks
        self._soft_update(self.actor, self.actor_target)
        self._soft_update(self.critic, self.critic_target)

        return actor_loss.item(), critic_loss.item()

    def _soft_update(self, source, target):
        """Soft update target network parameters"""
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.tau) + param.data * self.tau
            )

    def save(self, filename):
        """Save model parameters"""
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
                "actor_target": self.actor_target.state_dict(),
                "critic_target": self.critic_target.state_dict(),
            },
            filename,
        )

    def load(self, filename):
        """Load model parameters"""
        checkpoint = torch.load(filename)
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])
        self.actor_target.load_state_dict(checkpoint["actor_target"])
        self.critic_target.load_state_dict(checkpoint["critic_target"])


class Environment:
    def __init__(self, bp, initial, rng: np.random.Generator | None = None):
        self.bp = bp
        self.initial = initial
        self.rng = rng

        # this is the saved state of the environment
        # it corresponds to arrival states of the BP
        self.state = None
        self.reset()

    def reset(self):
        # TODO fix
        initial_vals = draw_shocks(
            self.initial,
            [0],  # only one agent
            rng=self.rng,
        )

        self.state = initial_vals

        return initial_vals

    def step(self, decision_rule):
        # TODO implement
        shocks = bp.draw_shocks()

        post = bp.forward_function(self.state, shocks, {}, decision_rules=decision_rule)

        state_t = self.state
        action = {csym: post[csym] for csym in decision_rule}
        reward = {
            rsym: post[rsym]
            for rsym in self.bp.block.reward
            # if agent is None or self.block.reward[rsym] == agent # TODO deal with multiple cagents
        }
        state_t_plus = {a_sym: post[a_sym] for a_sym in self.bp.get_arrival_states()}

        self.state = state_t_plus

        return state_t, action, reward, state_t_plus


# Example usage
if __name__ == "__main__":
    from skagent.models.benchmarks import d2_calibration, d2_block

    bp = BellmanPeriod(d2_block, d2_calibration)

    print("arrival states:", bp.get_arrival_states())

    initial = {
        "a": MeanOneLogNormal(sigma=1)
    }  # this is a dictionary of distributions for the initial values of the state variables.
    # this gets passed in to something...
    # combined with the BellmanPeriod it is the environment.

    env = Environment(bp, initial)

    state_dim = bp.get_states_dim()
    action_dim = bp.get_action_dim()

    print("dimensions: ", state_dim, action_dim)

    max_action = 1.0  ## todo: use BP limits

    # Initialize agent
    agent = DDPG(
        bp,
        max_action=max_action,  ## TODO: replace this with conditions from BP
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    print(agent.device)

    # Training loop example
    num_episodes = 100
    max_steps = 500

    for episode in range(num_episodes):
        state = env.reset()

        agent.noise.reset()
        episode_reward = 0

        for step in range(max_steps):
            # Select action
            print("state:", state)

            state, action, reward, next_state = env.step(agent.get_decision_rule())

            print("action:", action)
            print("reward:", reward)
            print("next_state:", next_state)

            done = step == max_steps - 1

            # Store transition
            agent.replay_buffer.push(state, action, reward, next_state, done)

            # Train agent
            actor_loss, critic_loss = agent.train(batch_size=64)

            state = next_state
            episode_reward += reward

            if done:
                break

        if episode % 10 == 0:
            print(f"Episode {episode}, Reward: {episode_reward:.2f}")

    # Save model
    agent.save("ddpg_model.pth")
