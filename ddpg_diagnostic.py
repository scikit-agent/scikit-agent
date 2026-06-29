"""
Diagnostic tools for DDPG training on the D-2 benchmark.

Runs a training loop and reports / plots:
  1. Reward curve
  2. Learned vs analytical policy c(m)
  3. Savings behaviour (is a_next > 0?)
  4. Critic Q(a, c) vs analytical Q*(a, c) — the key convergence diagnostic

Analytical Q* for D-2:
    Q*(a, c) = u(c) + β · V*(a')
    where  a' = m − c,  m = a·R + y,
    and V*(a') is the optimal value function from benchmarks.d2_analytical_lifetime_reward.

Usage:
    python ddpg_diagnostic.py
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from skagent.algos.ddpg import ddpg_training_loop, Environment
from skagent.bellman import BellmanPeriod
from skagent.distributions import MeanOneLogNormal
from skagent.models.benchmarks import (
    d2_block,
    d2_calibration,
    d2_analytical_lifetime_reward,
    get_analytical_policy,
    get_test_states,
)

SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# Analytical Q* for D-2
# ---------------------------------------------------------------------------


def analytical_q_d2(a, c, cal=d2_calibration):
    """
    Q*(a, c) = u(c) + β · V*(a')  for D-2 (infinite-horizon CRRA, perfect foresight).

    Parameters
    ----------
    a : float or array-like  — arrival state (assets)
    c : float or array-like  — consumption action
    cal : dict               — D-2 calibration

    Returns
    -------
    np.ndarray  — Q*(a, c) for each (a, c) pair
    """
    a = np.asarray(a, dtype=float)
    c = np.asarray(c, dtype=float)

    beta = cal["DiscFac"]
    R = cal["R"]
    sigma = cal["CRRA"]
    y = cal["y"]

    m = a * R + y  # cash-on-hand
    a_next = m - c  # next-period arrival state
    m_next = a_next * R + y  # next-period cash-on-hand

    # Immediate utility  u(c) = c^(1-σ) / (1-σ)
    u_c = c ** (1 - sigma) / (1 - sigma)

    # Optimal continuation value V*(m_next)
    # d2_analytical_lifetime_reward returns -∞ if m_next <= 0
    v_next = np.where(
        m_next > 0,
        np.vectorize(
            lambda m_n: d2_analytical_lifetime_reward(m_n, beta, R, sigma, income=y)
        )(m_next),
        -np.inf,
    )

    return u_c + beta * v_next


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def run_training(
    num_episodes=50,
    max_steps=100,
    hidden_dim=64,
    warmup_episodes=20,
    random_rollout_every=20,
    random_rollout_episodes=1,
    **kwargs,
):
    bp = BellmanPeriod(d2_block, "DiscFac", d2_calibration)
    initial = {"a": MeanOneLogNormal(sigma=1)}
    agent, rewards = ddpg_training_loop(
        bp,
        initial,
        num_episodes=num_episodes,
        max_steps_per_episode=max_steps,
        hidden_dim=hidden_dim,
        device=DEVICE,
        random_seed=SEED,
        log_every=num_episodes,
        warmup_episodes=warmup_episodes,
        random_rollout_every=random_rollout_every,
        random_rollout_episodes=random_rollout_episodes,
        **kwargs,
    )
    return agent, rewards, bp


# ---------------------------------------------------------------------------
# Console reports
# ---------------------------------------------------------------------------


def report_reward_curve(rewards, window=10):
    print("\n=== Reward curve (rolling mean every window) ===")
    print(f"{'Episodes':>12}  {'Mean reward':>14}")
    n = len(rewards)
    for start in range(0, n, window):
        chunk = rewards[start : start + window]
        print(f"  {start:4d}–{start + len(chunk) - 1:<4d}    {np.mean(chunk):>12.2f}")
    print(f"  Overall improvement: {rewards[-1] - rewards[0]:+.2f}")


def report_policy_comparison(agent, bp):
    print("\n=== Learned vs analytical policy ===")
    test_states = get_test_states("D-2")
    c_analytical = get_analytical_policy("D-2")(test_states, {}, d2_calibration)["c"]

    R = d2_calibration["R"]
    y = d2_calibration["y"]
    m = test_states["a"].float() * R + y

    agent.actor.eval()
    with torch.no_grad():
        c_learned = agent.actor(m.unsqueeze(1).to(DEVICE)).flatten().cpu()
    agent.actor.train()

    mape = (torch.abs(c_learned - c_analytical) / c_analytical.abs()).mean().item()
    print(f"{'m':>8}  {'c_analytical':>14}  {'c_learned':>12}  {'abs err':>10}")
    for mi, ca, cl in zip(m, c_analytical, c_learned):
        print(f"  {mi:6.3f}    {ca:12.4f}    {cl:10.4f}    {abs(cl - ca):8.4f}")
    print(f"\n  MAPE: {mape:.3f}  ({mape * 100:.1f}%)")
    return m, c_analytical, c_learned, mape


def report_savings(agent, bp, n_steps=200):
    print("\n=== Savings behaviour (no noise) ===")
    env = Environment(
        bp, {"a": MeanOneLogNormal(sigma=1)}, rng=np.random.default_rng(SEED)
    )
    dr = agent.get_decision_rule(add_noise=False)

    a_nexts, cs, ms = [], [], []
    for _ in range(n_steps):
        _, action, _, next_state, _, obs = env.step(dr)
        a_next = next_state.get("a")
        a_nexts.append(
            a_next.item() if isinstance(a_next, torch.Tensor) else float(a_next)
        )
        c_val = next(iter(action.values()))
        cs.append(c_val.item() if isinstance(c_val, torch.Tensor) else float(c_val))
        m_val = next(iter(obs.values()))
        ms.append(m_val.item() if isinstance(m_val, torch.Tensor) else float(m_val))

    print(
        f"  a_next  — mean: {np.mean(a_nexts):8.4f}  min: {np.min(a_nexts):8.4f}  max: {np.max(a_nexts):8.4f}"
    )
    print(
        f"  c       — mean: {np.mean(cs):8.4f}  min: {np.min(cs):8.4f}  max: {np.max(cs):8.4f}"
    )
    print(
        f"  m (iset)— mean: {np.mean(ms):8.4f}  min: {np.min(ms):8.4f}  max: {np.max(ms):8.4f}"
    )
    print(
        f"  Fraction of steps with a_next > 0: {np.mean([a > 1e-4 for a in a_nexts]):.2%}"
    )
    return ms, cs


def report_critic_vs_analytical_q(agent):
    print("\n=== Critic Q(a,c) vs analytical Q*(a,c) ===")
    print(
        f"  {'a':>5}  {'c':>8}  {'Q_critic':>12}  {'Q*_analytical':>15}  {'error':>10}"
    )

    R = d2_calibration["R"]
    y = d2_calibration["y"]

    agent.critic.eval()
    with torch.no_grad():
        for a in [1.0, 2.0, 3.0]:
            m = a * R + y
            # evaluate at analytical optimal, at c=m/2, and at c=m
            for label, c in [("c_opt", None), ("m/2", m / 2), ("m (max)", m * 0.999)]:
                if c is None:
                    cal = d2_calibration
                    kappa = (
                        cal["R"] - (cal["DiscFac"] * cal["R"]) ** (1 / cal["CRRA"])
                    ) / cal["R"]
                    H = cal["y"] / (cal["R"] - 1)
                    c = kappa * (m + H)
                q_critic = agent.critic(
                    torch.FloatTensor([[a]]).to(DEVICE),
                    torch.FloatTensor([[c]]).to(DEVICE),
                ).item()
                q_star = analytical_q_d2(a, c)
                print(
                    f"  a={a:.1f}  {label:>6}  c={c:.4f}  "
                    f"Q_critic={q_critic:10.4f}  Q*={float(q_star):12.4f}  "
                    f"err={q_critic - float(q_star):+10.4f}"
                )
    agent.critic.train()


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def make_plots(
    rewards, m, c_analytical, c_learned, agent, output="ddpg_diagnostic_plots.png"
):
    R = d2_calibration["R"]
    y = d2_calibration["y"]

    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.32)

    # 1. Reward curve
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(rewards, alpha=0.35, color="steelblue", linewidth=0.8)
    w = max(1, len(rewards) // 10)
    smoothed = np.convolve(rewards, np.ones(w) / w, mode="valid")
    ax1.plot(
        range(w - 1, len(rewards)),
        smoothed,
        color="steelblue",
        linewidth=2,
        label=f"Rolling mean ({w} ep)",
    )
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Total reward")
    ax1.set_title("Reward curve")
    ax1.legend(fontsize=8)

    # 2. Policy comparison
    ax2 = fig.add_subplot(gs[0, 1])
    m_dense = torch.linspace(m.min().item(), m.max().item(), 100)
    c_an_dense = get_analytical_policy("D-2")(
        {"a": (m_dense - y) / R}, {}, d2_calibration
    )["c"]
    agent.actor.eval()
    with torch.no_grad():
        c_lrn_dense = agent.actor(m_dense.unsqueeze(1).to(DEVICE)).flatten().cpu()
    agent.actor.train()

    ax2.plot(m_dense, c_an_dense, "k-", linewidth=2, label="Analytical c*(m)")
    ax2.plot(m_dense, c_lrn_dense, "r--", linewidth=2, label="Learned c(m)")
    ax2.plot(
        m_dense,
        m_dense,
        color="gray",
        linewidth=1,
        linestyle=":",
        label="Upper bound c=m",
    )
    ax2.set_xlabel("m (cash-on-hand)")
    ax2.set_ylabel("c (consumption)")
    ax2.set_title("Policy: learned vs analytical")
    ax2.legend(fontsize=8)

    # 3. Q*(a,c) vs critic Q(a,c) — one panel per asset level
    ax3 = fig.add_subplot(gs[1, :])  # full-width bottom panel
    colors = ["steelblue", "darkorange", "green"]

    agent.critic.eval()
    for a_val, col in zip([1.0, 2.0, 3.0], colors):
        m_val = a_val * R + y
        c_grid = np.linspace(1e-3, m_val * 0.999, 80)

        # Analytical Q*
        q_star = analytical_q_d2(a_val, c_grid)

        # Critic Q
        with torch.no_grad():
            s_t = torch.FloatTensor([[a_val]] * len(c_grid)).to(DEVICE)
            a_t = torch.FloatTensor([[c] for c in c_grid]).to(DEVICE)
            q_critic = agent.critic(s_t, a_t).flatten().cpu().numpy()

        # Analytical optimal action
        kappa = (
            R - (d2_calibration["DiscFac"] * R) ** (1 / d2_calibration["CRRA"])
        ) / R
        H = y / (R - 1)
        c_opt = kappa * (m_val + H)

        ax3.plot(
            c_grid,
            q_star,
            color=col,
            linestyle="-",
            linewidth=2,
            label=f"Q* a={a_val:.1f}",
        )
        ax3.plot(
            c_grid,
            q_critic,
            color=col,
            linestyle="--",
            linewidth=2,
            label=f"Q_critic a={a_val:.1f}",
        )
        ax3.axvline(c_opt, color=col, linestyle=":", linewidth=1.2, alpha=0.7)
    agent.critic.train()

    ax3.set_xlabel("c (consumption)")
    ax3.set_ylabel("Q value")
    ax3.set_title(
        "Q*(a, c) [solid] vs critic Q(a, c) [dashed]  —  dotted verticals = c_analytical"
    )
    ax3.legend(fontsize=7, ncol=2)

    fig.suptitle(
        f"DDPG on D-2  |  {len(rewards)} episodes  |  device={DEVICE}  |  seed={SEED}",
        fontsize=11,
    )
    plt.savefig(output, dpi=120, bbox_inches="tight")
    print(f"\n  Plots saved to {output}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"Device: {DEVICE}")
    print("Training DDPG on D-2 (50 episodes × 100 steps, hidden_dim=64)...")
    agent, rewards, bp = run_training(num_episodes=50, max_steps=100, hidden_dim=64)

    report_reward_curve(rewards)
    m, c_analytical, c_learned, mape = report_policy_comparison(agent, bp)
    report_savings(agent, bp)
    report_critic_vs_analytical_q(agent)
    make_plots(rewards, m, c_analytical, c_learned, agent)
