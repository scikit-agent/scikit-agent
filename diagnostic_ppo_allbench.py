"""
Diagnostic: train :class:`PPOAgent` on every benchmark / conftest case for
which an analytical (or hand-coded "optimal") policy is available, evaluate
the learned policy on a small obs grid, and print a comparison table.

Goal: surface which models PPO nails out of the box and which it struggles
with, given identical training budgets and default hyperparameters.

Cases that can't be driven by ``PPOAgent`` (multi-control, empty iset,
finite-horizon w/ time-as-state, models without a closed form) are listed
in the table as ``SKIPPED`` with a reason.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from skagent.algos.sb3 import PPOAgent
from skagent.bellman import BellmanPeriod
from skagent.distributions import Uniform
from skagent.models.benchmarks import (
    d2_analytical_policy,
    d2_block,
    d2_calibration,
    d3_analytical_policy,
    d3_block,
    d3_calibration,
    u1_analytical_policy,
    u1_block,
    u1_calibration,
    u2_analytical_policy,
    u2_block,
    u2_calibration,
)

from tests import conftest  # case_0 ... case_11


logging.getLogger("skagent").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

# Keep per-model training cheap so the script finishes in ~2 min total.
TRAIN_STEPS = 10_000
SEED = 0
PPO_KWARGS = {"n_steps": 256, "batch_size": 64, "n_epochs": 4}
MAX_EPISODE_STEPS = 64


@dataclass
class Spec:
    name: str
    bp: BellmanPeriod
    initial: dict
    optimal_fn: Callable  # (obs_row: np.ndarray) -> float
    obs_grid: np.ndarray  # shape (N, |iset|)
    gym_kwargs: dict = field(default_factory=dict)
    notes: str = ""


@dataclass
class SkipSpec:
    name: str
    reason: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _scalar(x):
    if hasattr(x, "item"):
        try:
            return float(x.item())
        except Exception:
            pass
    return float(np.asarray(x).reshape(-1)[0])


def conftest_spec(
    case_name: str,
    initial: dict,
    obs_grid: np.ndarray,
    gym_kwargs: Optional[dict] = None,
    notes: str = "",
) -> Spec:
    """Build a Spec from a conftest case dict, using its ``optimal_dr``."""
    case = getattr(conftest, case_name)
    bp: BellmanPeriod = case["bp"]
    optimal_dr_c = case["optimal_dr"]["c"]

    def fn(obs_row: np.ndarray) -> float:
        # Conftest lambdas take iset values positionally (matching Control.iset).
        return _scalar(optimal_dr_c(*[float(v) for v in obs_row]))

    return Spec(
        name=case_name,
        bp=bp,
        initial=initial,
        optimal_fn=fn,
        obs_grid=obs_grid.astype(np.float32),
        gym_kwargs=gym_kwargs or {},
        notes=notes,
    )


def benchmark_spec(
    name: str,
    block,
    calibration: dict,
    initial: dict,
    optimal_fn: Callable,
    obs_grid: np.ndarray,
    discount_variable: str = "DiscFac",
    gym_kwargs: Optional[dict] = None,
    notes: str = "",
) -> Spec:
    bp = BellmanPeriod(block, discount_variable, calibration)
    return Spec(
        name=name,
        bp=bp,
        initial=initial,
        optimal_fn=optimal_fn,
        obs_grid=obs_grid.astype(np.float32),
        gym_kwargs=gym_kwargs or {},
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Benchmark analytical wrappers (iset-positional, returning scalar c).
# ---------------------------------------------------------------------------


def d2_opt(obs_row):  # iset=[m]
    m = float(obs_row[0])
    a = (m - d2_calibration["y"]) / d2_calibration["R"]
    return _scalar(d2_analytical_policy({"a": a}, {}, d2_calibration)["c"])


def d3_opt(obs_row):  # iset=[m]
    m = float(obs_row[0])
    a = (m - d3_calibration["y"]) / d3_calibration["R"]
    return _scalar(d3_analytical_policy({"a": a}, {}, d3_calibration)["c"])


def u1_opt(obs_row):  # iset=[m]
    m = float(obs_row[0])
    y = u1_calibration["y_mean"]
    A = (m - y) / u1_calibration["R"]
    return _scalar(u1_analytical_policy({"A": A, "y": y}, {}, u1_calibration)["c"])


def u2_opt(obs_row):  # iset=[m]
    import torch

    m = float(obs_row[0])
    a = (m - 1.0) / u2_calibration["R"]
    return _scalar(
        u2_analytical_policy({"a": torch.tensor([a])}, {}, u2_calibration)["c"]
    )


# ---------------------------------------------------------------------------
# Spec list
# ---------------------------------------------------------------------------

# Generous default bounds for cases whose Control has none (or only one side).
WIDE = {"default_lower": -5.0, "default_upper": 5.0}

# Obs grids (chosen to roughly bracket where each iset wanders in practice).
G_A = np.linspace(0.1, 1.0, 10).reshape(-1, 1)  # iset=[a]
G_A_THETA = np.array(  # iset=[a, theta]
    [[a, th] for a in np.linspace(0.0, 1.0, 4) for th in np.linspace(-1.0, 1.0, 4)]
)
G_M = np.linspace(0.1, 1.5, 10).reshape(-1, 1)  # iset=[m] (small)
G_M_D2 = np.linspace(0.5, 10.0, 20).reshape(-1, 1)  # iset=[m] (D2/D3/U1/U2)
G_G_M = np.array(  # iset=[g, m]
    [[g, m] for g in np.linspace(-2.0, 2.0, 4) for m in np.linspace(-2.0, 2.0, 4)]
)

INIT_A = {"a": Uniform(low=0.1, high=1.0)}
INIT_MG = {"m": Uniform(low=-1.0, high=1.0), "g": Uniform(low=-1.0, high=1.0)}  # case_4
INIT_AB = {
    "a": Uniform(low=-1.0, high=1.0),
    "b": Uniform(low=-1.0, high=1.0),
}  # case_11
INIT_AG_05 = {"a": Uniform(low=0.5, high=2.0)}  # D-2/U-2
INIT_D3 = {"a": Uniform(low=0.5, high=2.0), "liv": 1.0}  # D-3 needs alive flag
INIT_AY = {"A": Uniform(low=0.5, high=2.0), "y": Uniform(low=0.9, high=1.1)}  # U-1

SPECS: list[Spec | SkipSpec] = [
    # --- conftest cases ------------------------------------------------
    conftest_spec(
        "case_0",
        INIT_A,
        G_A,
        gym_kwargs={"default_lower": -1.0, "default_upper": 1.0},
        notes="c*=0",
    ),
    conftest_spec("case_1", INIT_A, G_A_THETA, gym_kwargs=WIDE, notes="c*=θ"),
    conftest_spec(
        "case_2",
        INIT_A,
        G_A,
        gym_kwargs={"default_lower": -1.0, "default_upper": 1.0},
        notes="c*=0 (θ hidden)",
    ),
    conftest_spec("case_3", INIT_A, G_M, gym_kwargs=WIDE, notes="c*=m"),
    conftest_spec("case_4", INIT_MG, G_G_M, gym_kwargs=WIDE, notes="c*=g-m"),
    conftest_spec("case_5", INIT_A, G_A, notes="c*=a (constrained)"),
    conftest_spec("case_6", INIT_A, G_A, notes="c*=a (lower-bound binds)"),
    conftest_spec(
        "case_7",
        INIT_A,
        G_A,
        gym_kwargs={"default_upper": 5.0},
        notes="c*=1 (lower-bounded only)",
    ),
    conftest_spec("case_8", INIT_A, G_A, notes="c*=a (upper-bounded only)"),
    SkipSpec("case_9", "empty iset -> SB3 obs space has shape (0,)"),
    SkipSpec("case_10", "multi-control -> PPOAgent supports single control"),
    conftest_spec(
        "case_11",
        INIT_AB,
        G_A_THETA,
        gym_kwargs=WIDE,
        notes="c*=a+θ (b is a lag arrival state)",
    ),
    # --- benchmark models ----------------------------------------------
    SkipSpec(
        "D-1",
        "finite-horizon (state `t` ticks toward T); not directly "
        "compatible with single-truncation PPO loop",
    ),
    benchmark_spec(
        "D-2",
        d2_block,
        d2_calibration,
        INIT_AG_05,
        d2_opt,
        G_M_D2,
        notes="c*=κ(m+H), bounded by m",
    ),
    benchmark_spec(
        "D-3",
        d3_block,
        d3_calibration,
        INIT_D3,
        d3_opt,
        G_M_D2,
        notes="κ_s mortality variant of D-2",
    ),
    benchmark_spec(
        "U-1",
        u1_block,
        u1_calibration,
        INIT_AY,
        u1_opt,
        G_M_D2,
        notes="Hall PIH, η~Normal income shock",
    ),
    benchmark_spec(
        "U-2",
        u2_block,
        u2_calibration,
        INIT_AG_05,
        u2_opt,
        G_M_D2,
        notes="normalised log-utility PIH",
    ),
    SkipSpec("U-3", "no closed-form analytical policy"),
]


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


@dataclass
class Result:
    name: str
    iset: list
    mae: float
    rmse: float
    max_err: float
    opt_min: float
    opt_max: float
    train_secs: float
    notes: str = ""
    error: str = ""


def evaluate(spec: Spec) -> Result:
    import time

    t0 = time.time()
    agent = PPOAgent(
        spec.bp,
        spec.initial,
        max_episode_steps=MAX_EPISODE_STEPS,
        seed=SEED,
        gym_kwargs=spec.gym_kwargs,
        ppo_kwargs=PPO_KWARGS,
    )
    agent.learn(total_timesteps=TRAIN_STEPS)
    train_secs = time.time() - t0

    obs = spec.obs_grid
    c_learned = agent.predict_unscaled(obs)

    # Optimal action per row, then clip to per-state bounds the env enforces —
    # this is the policy PPO can actually realise in this env.
    c_opt = np.empty(obs.shape[0], dtype=np.float32)
    for i in range(obs.shape[0]):
        lo, hi = agent.env._bounds_at_iset(obs[i])
        c_opt[i] = float(np.clip(spec.optimal_fn(obs[i]), lo, hi))

    err = c_learned - c_opt
    return Result(
        name=spec.name,
        iset=agent.env.iset,
        mae=float(np.mean(np.abs(err))),
        rmse=float(np.sqrt(np.mean(err**2))),
        max_err=float(np.max(np.abs(err))),
        opt_min=float(c_opt.min()),
        opt_max=float(c_opt.max()),
        train_secs=train_secs,
        notes=spec.notes,
    )


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


HEADER = (
    f"{'name':<8} {'iset':<14} {'MAE':>8} {'RMSE':>8} {'MaxErr':>8} "
    f"{'c*range':>16} {'train(s)':>8}  notes"
)


def fmt_row(r: Result) -> str:
    iset = ",".join(r.iset)[:13]
    rng = f"[{r.opt_min:+.2f},{r.opt_max:+.2f}]"[:16]
    return (
        f"{r.name:<8} {iset:<14} {r.mae:>8.3f} {r.rmse:>8.3f} {r.max_err:>8.3f} "
        f"{rng:>16} {r.train_secs:>8.1f}  {r.notes}"
    )


def fmt_skip(s: SkipSpec) -> str:
    return f"{s.name:<8} {'SKIPPED':<14} {'':>8} {'':>8} {'':>8} {'':>16} {'':>8}  {s.reason}"


def fmt_error(name: str, msg: str) -> str:
    return f"{name:<8} {'ERROR':<14} {'':>8} {'':>8} {'':>8} {'':>16} {'':>8}  {msg}"


def main() -> None:
    print(
        f"PPO settings: {PPO_KWARGS}, max_episode_steps={MAX_EPISODE_STEPS}, "
        f"train_steps={TRAIN_STEPS}, seed={SEED}\n"
    )
    print(HEADER)
    print("-" * len(HEADER))

    results: list[Result] = []
    for spec in SPECS:
        if isinstance(spec, SkipSpec):
            print(fmt_skip(spec))
            continue
        try:
            r = evaluate(spec)
            results.append(r)
            print(fmt_row(r))
        except Exception as e:
            print(fmt_error(spec.name, f"{type(e).__name__}: {e}"))

    if results:
        print("-" * len(HEADER))
        print("\nBest (lowest MAE):")
        for r in sorted(results, key=lambda r: r.mae)[:3]:
            print(f"  {r.name:<8} MAE={r.mae:.4f}  ({r.notes})")
        print("\nWorst (highest MAE):")
        for r in sorted(results, key=lambda r: -r.mae)[:3]:
            print(f"  {r.name:<8} MAE={r.mae:.4f}  ({r.notes})")


if __name__ == "__main__":
    main()
