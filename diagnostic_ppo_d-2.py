"""
Diagnostic script: PPO on the D-2 benchmark (infinite-horizon CRRA, no shocks).

Trains PPO via :class:`skagent.algos.sb3.PPOAgent` and compares the learned
consumption policy against the closed-form solution
``c_t = κ · (m_t + y/(R-1))``.

Reports
-------
1. MAE of c_learned vs c_optimal over a grid of m values.
2. Mean total *discounted* reward over N rollouts under each policy.
3. Two plots:
   * learned vs optimal consumption function;
   * episode rewards over training (from ``agent.episode_rewards``).
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import torch

from skagent.algos.sb3 import PPOAgent
from skagent.bellman import BellmanPeriod
from skagent.distributions import Uniform
from skagent.env import Environment
from skagent.models.benchmarks import (
    d2_analytical_policy,
    d2_block,
    d2_calibration,
)

SEED = 0
# Snapshot the learned consumption function at each of these cumulative
# training-timestep counts to show whether/how PPO converges to the optimum.
CHECKPOINTS = [50_000, 100_000, 200_000]
MAX_EPISODE_STEPS = 200
N_EVAL_ROLLOUTS = 50
EVAL_ROLLOUT_STEPS = 200
INITIAL_A_LOW = 0.01
INITIAL_A_HIGH = 5.0


def discounted_rollout_reward(
    bp: BellmanPeriod, dr: dict, rng: np.random.Generator, steps: int, initial: dict
) -> float:
    env = Environment(bp, initial, rng=rng)
    env.reset()
    total = 0.0
    discount_acc = 1.0
    for _ in range(steps):
        _, _, reward, _, discount, _ = env.step(dr)
        r = sum(
            float(v.detach().item()) if isinstance(v, torch.Tensor) else float(v)
            for v in reward.values()
        )
        total += discount_acc * r
        discount_acc *= float(
            discount.detach().item() if isinstance(discount, torch.Tensor) else discount
        )
    return total


def optimal_decision_rule() -> dict:
    """Constrained closed-form policy: ``c = min(κ(m+H), m)``."""

    def c_rule(m_val):
        m_arr = np.asarray(
            m_val.detach().cpu().numpy() if isinstance(m_val, torch.Tensor) else m_val,
            dtype=np.float32,
        ).reshape(-1)
        a = (m_arr - d2_calibration["y"]) / d2_calibration["R"]
        c = d2_analytical_policy({"a": a}, {}, d2_calibration)["c"]
        c = np.minimum(np.asarray(c, dtype=np.float32), m_arr)
        return torch.as_tensor(c, dtype=torch.float32)

    return {"c": c_rule}


def main() -> None:
    bp = BellmanPeriod(d2_block, "DiscFac", d2_calibration)
    initial = {"a": Uniform(low=INITIAL_A_LOW, high=INITIAL_A_HIGH)}

    # -- training ----------------------------------------------------------
    agent = PPOAgent(
        bp,
        initial,
        max_episode_steps=MAX_EPISODE_STEPS,
        seed=SEED,
        ppo_kwargs={
            "n_steps": 2048,
            "batch_size": 64,
            "n_epochs": 10,
            "learning_rate": 3e-4,
        },
    )
    # Train incrementally, snapshotting the learned consumption function at
    # each checkpoint. ``reset_num_timesteps=False`` keeps PPO's internal step
    # counter (and schedules) continuous across the successive ``learn`` calls.
    m_grid = np.linspace(0.5, 10.0, 41, dtype=np.float32)
    obs_grid = m_grid.reshape(-1, 1)
    c_learned_by_checkpoint: dict[int, np.ndarray] = {}
    prev = 0
    for i, checkpoint in enumerate(CHECKPOINTS):
        agent.learn(total_timesteps=checkpoint - prev, reset_num_timesteps=(i == 0))
        c_learned_by_checkpoint[checkpoint] = agent.predict_unscaled(obs_grid)
        prev = checkpoint
    total_timesteps = CHECKPOINTS[-1]
    episode_rewards = np.asarray(agent.episode_rewards, dtype=np.float32)

    # The final-checkpoint policy is used for the rollouts.
    # c_learned = c_learned_by_checkpoint[CHECKPOINTS[-1]]

    # -- policy comparison on a grid of m values ---------------------------
    a_grid = (m_grid - d2_calibration["y"]) / d2_calibration["R"]
    c_optimal_unc = np.asarray(
        d2_analytical_policy({"a": a_grid}, {}, d2_calibration)["c"], dtype=np.float32
    )
    # The unconstrained closed form ``c = κ(m + H)`` ignores the borrowing
    # constraint ``c ≤ m`` baked into ``Control(upper_bound=lambda m: m)``.
    # At small m, ``κ(m + H) > m``, so the true constrained optimum is
    # ``c = min(κ(m + H), m)`` — the constraint binds and assets go to zero.
    c_optimal = np.minimum(c_optimal_unc, m_grid)

    # Per-checkpoint error vs the closed-form optimum, to show convergence.
    print(f"Policy error vs closed-form (over m ∈ [{m_grid[0]}, {m_grid[-1]}]):")
    print(f"  {'checkpoint':>12}  {'MAE':>10}  {'MaxErr':>10}")
    mae_by_checkpoint: dict[int, float] = {}
    for checkpoint in CHECKPOINTS:
        err = np.abs(c_learned_by_checkpoint[checkpoint] - c_optimal)
        mae_ckpt = float(np.mean(err))
        max_err_ckpt = float(np.max(err))
        mae_by_checkpoint[checkpoint] = mae_ckpt
        print(f"  {checkpoint:>12,}  {mae_ckpt:>10.4f}  {max_err_ckpt:>10.4f}")
    mae = mae_by_checkpoint[CHECKPOINTS[-1]]

    # -- discounted-reward Monte-Carlo comparison --------------------------
    rng = np.random.default_rng(SEED + 1)
    learned_dr = agent.decision_rule()
    optimal_dr = optimal_decision_rule()

    learned_returns = [
        discounted_rollout_reward(bp, learned_dr, rng, EVAL_ROLLOUT_STEPS, initial)
        for _ in range(N_EVAL_ROLLOUTS)
    ]
    optimal_returns = [
        discounted_rollout_reward(bp, optimal_dr, rng, EVAL_ROLLOUT_STEPS, initial)
        for _ in range(N_EVAL_ROLLOUTS)
    ]

    print(
        f"\nDiscounted return over {EVAL_ROLLOUT_STEPS} steps "
        f"({N_EVAL_ROLLOUTS} rollouts each):"
    )
    print(
        f"  learned: mean = {np.mean(learned_returns):.4f}  "
        f"std = {np.std(learned_returns):.4f}"
    )
    print(
        f"  optimal: mean = {np.mean(optimal_returns):.4f}  "
        f"std = {np.std(optimal_returns):.4f}"
    )
    gap = float(np.mean(optimal_returns) - np.mean(learned_returns))
    print(f"  gap (optimal − learned) = {gap:.4f}")

    # -- plots -------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for checkpoint in CHECKPOINTS:
        axes[0].plot(
            m_grid,
            c_learned_by_checkpoint[checkpoint],
            label=f"PPO @ {checkpoint:,} steps",
        )
    axes[0].plot(m_grid, c_optimal, label="closed form (constrained)", linestyle="--")
    axes[0].plot(
        m_grid,
        c_optimal_unc,
        label="closed form (unconstrained)",
        linestyle="--",
        alpha=0.4,
    )
    axes[0].plot(m_grid, m_grid, label="c = m (upper bound)", linestyle=":", alpha=0.5)
    axes[0].set_xlabel("m (cash-on-hand)")
    axes[0].set_ylabel("c (consumption)")
    axes[0].set_title(f"D-2 policy: MAE={mae:.3f}")
    axes[0].legend()

    if len(episode_rewards) > 0:
        window = max(1, len(episode_rewards) // 20)
        rolling = np.convolve(episode_rewards, np.ones(window) / window, mode="valid")
        axes[1].plot(episode_rewards, alpha=0.3, label="episode reward")
        axes[1].plot(
            np.arange(window - 1, len(episode_rewards)),
            rolling,
            label=f"rolling mean (w={window})",
        )
        axes[1].set_xlabel("episode")
        axes[1].set_ylabel("undiscounted episode reward")
        axes[1].set_title(f"Training curve ({total_timesteps:,} timesteps)")
        axes[1].legend()
    else:
        axes[1].set_title("No episodes completed during training")

    fig.tight_layout()
    out_path = "diagnostic_ppo_d-2.png"
    fig.savefig(out_path, dpi=120)
    print(f"\nSaved plot to {out_path}")
    plt.show()


if __name__ == "__main__":
    main()
