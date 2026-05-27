"""
For reference: https://arxiv.org/pdf/1509.02971
"""

import logging
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
from skagent.ann import BlockPolicyNet, BlockQNet
from skagent.bellman import BellmanPeriod
from skagent.simulation.monte_carlo import draw_shocks

logger = logging.getLogger(__name__)


class ReplayBuffer:
    """Experience replay buffer for storing and sampling transitions. [made by Claude]"""

    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, discount, done, obs):
        def _detach(d):
            return {
                k: v.detach() if isinstance(v, torch.Tensor) else v
                for k, v in d.items()
            }

        self.buffer.append(
            (
                _detach(state),
                _detach(action),
                _detach(reward),
                _detach(next_state),
                discount,
                done,
                _detach(obs),
            )
        )

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, discounts, dones, obss = zip(*batch)

        states = {key: torch.cat([d[key] for d in states]) for key in states[0]}
        actions = {key: torch.cat([d[key] for d in actions]) for key in actions[0]}
        rewards = {key: torch.cat([d[key] for d in rewards]) for key in rewards[0]}
        next_states = {
            key: torch.cat([d[key] for d in next_states]) for key in next_states[0]
        }
        obss = {key: torch.cat([d[key] for d in obss]) for key in obss[0]}

        return states, actions, rewards, next_states, discounts, dones, obss

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
        self.state = torch.Tensor(np.ones(self.action_dim) * self.mu)

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
        tau=0.005,
        device="cpu",
    ):
        self.device = device
        self.bp = bp
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
        control = self.bp.block.dynamics[self.actor.control_sym]

        def decision_rule(*information):
            # Ensure float32 — shocks and dynamics can produce float64 numpy/tensors
            info_f32 = tuple(torch.as_tensor(x).float() for x in information)
            env_device = info_f32[0].device if info_f32 else torch.device("cpu")

            # Call actor directly (bypassing ann.py's module-level device closure)
            # and return to the environment's device
            input_tensor = torch.stack(info_f32).T.to(self.device)
            action = self.actor(input_tensor).flatten().to(env_device)

            if add_noise:
                noise = self.noise.sample().to(env_device)
                action = action + noise

            # Clip to control bounds. The network enforces these via sigmoid/softplus
            # in its forward pass, but noise can push the action outside them.
            lo = (
                torch.as_tensor(control.lower_bound(*info_f32)).float()
                if control.lower_bound is not None
                else None
            )
            hi = (
                torch.as_tensor(control.upper_bound(*info_f32)).float()
                if control.upper_bound is not None
                else None
            )
            if lo is not None or hi is not None:
                action = torch.clamp(action, min=lo, max=hi)

            return action

        return {self.actor.control_sym: decision_rule}

    def train(self, batch_size=64):
        """Train the agent on a batch of experiences"""
        if len(self.replay_buffer) < batch_size:
            return None, None

        # Sample batch
        states, actions, rewards, next_states, discounts, dones, obss = (
            self.replay_buffer.sample(batch_size)
        )

        states_t = torch.stack(list(states.values()), dim=1).float().to(self.device)
        actions_t = torch.stack(list(actions.values()), dim=1).float().to(self.device)
        rewards_t = torch.stack(list(rewards.values()), dim=1).float().to(self.device)
        next_states_t = (
            torch.stack(list(next_states.values()), dim=1).float().to(self.device)
        )
        obs_t = torch.stack(list(obss.values()), dim=1).float().to(self.device)
        discounts_t = torch.FloatTensor(discounts).unsqueeze(1).to(self.device)
        dones_t = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        # Update Critic
        with torch.no_grad():
            # Compute next info-set by running forward_function with dummy controls
            next_shocks_raw = self.bp.draw_shocks(n=batch_size)
            next_shocks = {k: torch.FloatTensor(v) for k, v in next_shocks_raw.items()}
            dummy_controls = {
                csym: torch.zeros(batch_size) for csym in self.bp.get_controls()
            }
            next_post = self.bp.forward_function(
                next_states, next_shocks, dummy_controls
            )
            next_obs_t = torch.stack(
                [
                    torch.as_tensor(next_post[sym]).float().flatten()
                    for sym in self.actor.iset
                ],
                dim=1,
            ).to(self.device)

            next_actions = self.actor_target(next_obs_t)
            target_q = self.critic_target(next_states_t, next_actions)
            target_q = rewards_t + (1 - dones_t) * discounts_t * target_q

        current_q = self.critic(states_t, actions_t)
        critic_loss = nn.MSELoss()(current_q, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Update Actor
        actor_loss = -self.critic(states_t, self.actor(obs_t)).mean()

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

        # ok this is pretty annoying
        initial_vals = {sym: torch.Tensor(initial_vals[sym]) for sym in initial_vals}

        self.state = initial_vals

        return initial_vals

    def step(self, decision_rule):
        shocks = self.bp.draw_shocks(n=1)

        post = self.bp.forward_function(
            self.state, shocks, {}, decision_rules=decision_rule
        )

        state_t = self.state
        action = {csym: post[csym] for csym in decision_rule}
        reward = {
            rsym: post[rsym]
            for rsym in self.bp.block.reward
            # if agent is None or self.block.reward[rsym] == agent # TODO deal with multiple agents
        }
        discount = self.bp.resolve_discount_factor(post)
        state_t_plus = {
            a_sym: post[a_sym].detach()
            if isinstance(post[a_sym], torch.Tensor)
            else post[a_sym]
            for a_sym in self.bp.get_arrival_states()
        }

        # info-set for each control (what the actor sees)
        control_sym = next(iter(decision_rule))
        iset = self.bp.block.dynamics[control_sym].iset
        obs = {
            sym: post[sym].detach()
            if isinstance(post[sym], torch.Tensor)
            else post[sym]
            for sym in iset
        }

        self.state = state_t_plus

        return state_t, action, reward, state_t_plus, discount, obs


def ddpg_training_loop(
    bellman_period: BellmanPeriod,
    initial: dict,
    num_episodes: int = 1000,
    max_steps_per_episode: int = 200,
    batch_size: int = 64,
    hidden_dim: int = 256,
    lr_actor: float = 1e-4,
    lr_critic: float = 1e-3,
    tau: float = 0.005,
    device: str = "cpu",
    random_seed: Optional[int] = None,
    tolerance: Optional[float] = None,
    convergence_window: int = 50,
    log_every: int = 10,
) -> tuple:
    """
    Run the DDPG training loop for a BellmanPeriod model.

    Parameters
    ----------
    bellman_period : BellmanPeriod
        Model definition containing block dynamics and transitions.
    initial : dict
        Initial state distribution; maps arrival-state symbols to
        Distribution objects used to reset the environment each episode.
    num_episodes : int, optional
        Maximum number of episodes to train for. Default is 1000.
    max_steps_per_episode : int, optional
        Steps per episode before forced termination. Default is 200.
    batch_size : int, optional
        Replay buffer sample size for each training step. Default is 64.
    hidden_dim : int, optional
        Width of hidden layers in actor and critic networks. Default is 256.
    lr_actor : float, optional
        Adam learning rate for the actor. Default is 1e-4.
    lr_critic : float, optional
        Adam learning rate for the critic. Default is 1e-3.
    tau : float, optional
        Polyak averaging coefficient for target network updates. Default is 0.005.
    device : str, optional
        Torch device for the networks. Default is "cpu".
    random_seed : int, optional
        Seed for reproducibility. Default is None.
    tolerance : float, optional
        If set, stop early when the absolute difference in mean episode reward
        between two consecutive windows of ``convergence_window`` episodes is
        below this threshold. Default is None (run for all episodes).
    convergence_window : int, optional
        Number of episodes in each window used for the convergence check.
        Default is 50.
    log_every : int, optional
        Log a summary every this many episodes. Default is 10.

    Returns
    -------
    tuple
        ``(agent, episode_rewards)`` where ``agent`` is the trained
        :class:`DDPG` instance and ``episode_rewards`` is a list of total
        undiscounted reward per episode.
    """
    if bellman_period is None:
        raise TypeError("bellman_period cannot be None")
    if not initial:
        raise ValueError("initial cannot be empty")
    if num_episodes < 1:
        raise ValueError(f"num_episodes must be >= 1, got {num_episodes}")
    if max_steps_per_episode < 1:
        raise ValueError(
            f"max_steps_per_episode must be >= 1, got {max_steps_per_episode}"
        )
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")
    if tolerance is not None and tolerance <= 0:
        raise ValueError(f"tolerance must be > 0, got {tolerance}")
    if convergence_window < 1:
        raise ValueError(f"convergence_window must be >= 1, got {convergence_window}")

    rng = np.random.default_rng(random_seed)
    if random_seed is not None:
        torch.manual_seed(random_seed)

    agent = DDPG(
        bellman_period,
        hidden_dim=hidden_dim,
        lr_actor=lr_actor,
        lr_critic=lr_critic,
        tau=tau,
        device=device,
    )
    env = Environment(bellman_period, initial, rng=rng)

    episode_rewards: list[float] = []
    converged = False

    for episode in range(num_episodes):
        env.reset()
        agent.noise.reset()
        episode_reward = 0.0
        actor_losses: list[float] = []
        critic_losses: list[float] = []

        for step in range(max_steps_per_episode):
            state, action, reward, next_state, discount, obs = env.step(
                agent.get_decision_rule()
            )

            done = step == max_steps_per_episode - 1
            agent.replay_buffer.push(
                state, action, reward, next_state, discount, done, obs
            )

            actor_loss, critic_loss = agent.train(batch_size=batch_size)
            if actor_loss is not None:
                actor_losses.append(actor_loss)
                critic_losses.append(critic_loss)

            episode_reward += sum(
                v.detach().item() if isinstance(v, torch.Tensor) else float(v)
                for v in reward.values()
            )

            if done:
                break

        episode_rewards.append(episode_reward)

        if (episode + 1) % log_every == 0:
            mean_actor = float(np.mean(actor_losses)) if actor_losses else float("nan")
            mean_critic = (
                float(np.mean(critic_losses)) if critic_losses else float("nan")
            )
            logger.info(
                "Episode %d/%d: reward=%.4f, actor_loss=%.4e, critic_loss=%.4e",
                episode + 1,
                num_episodes,
                episode_reward,
                mean_actor,
                mean_critic,
            )

        if tolerance is not None and len(episode_rewards) >= 2 * convergence_window:
            prev_mean = float(
                np.mean(episode_rewards[-2 * convergence_window : -convergence_window])
            )
            curr_mean = float(np.mean(episode_rewards[-convergence_window:]))
            if abs(curr_mean - prev_mean) < tolerance:
                converged = True
                logger.info(
                    "Converged after %d episodes: mean reward change %.2e < tolerance %.2e",
                    episode + 1,
                    abs(curr_mean - prev_mean),
                    tolerance,
                )
                break

    if not converged and tolerance is not None:
        logger.warning(
            "Training completed without convergence after %d episodes.", num_episodes
        )

    return agent, episode_rewards
