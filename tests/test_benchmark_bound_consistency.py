"""Regression test: a closed-form benchmark's block bounds must be consistent
with its analytical policy, INCLUDING the borrowing region.

The unconstrained perfect-foresight closed forms (D-2, D-3) borrow against
human wealth, so at low wealth the optimal end-of-period assets are negative
(``a' < 0``). The blocks previously imposed ``c <= m`` (equivalently
``a' >= 0``), a no-borrowing constraint the closed forms violate there: block and
oracle silently described different models. Single-step and analytical-policy
tests missed it because their test states (``a >= 0.5``) never reach the
borrowing region. This test evaluates each block's OWN control bounds through the
framework and asserts the analytical policy is feasible under them, on states
that do reach ``a' < 0``.
"""

from inspect import signature

import torch

from skagent.bellman import BellmanPeriod
from skagent.models import benchmarks as bm

# Unconstrained closed-form models whose oracle borrows against human wealth.
# Grids start below the no-borrowing kink so the unconstrained optimum implies
# a' < 0 for some rows (U-2 keeps a loose neural-solver cap and is not covered).
BORROWING_MODELS = {
    "D-2": torch.linspace(-0.15, 4.0, 25),
    "D-3": torch.linspace(-0.15, 4.0, 25),
}


def _eval_bound(bound_fn, scope):
    """Evaluate a Control bound the way the framework does: resolve its lambda's
    named parameters from the model parameters merged with the pre-decision
    state (a bound may reference either, e.g. the natural limit m + H)."""
    if bound_fn is None:
        return None
    return bound_fn(*[scope[name] for name in signature(bound_fn).parameters])


def test_block_bounds_consistent_with_analytical_policy():
    for model_id, a in BORROWING_MODELS.items():
        block = bm.get_benchmark_model(model_id)
        cal = bm.get_benchmark_calibration(model_id)
        policy = bm.get_analytical_policy(model_id)

        states = {"a": a}
        shocks = {}

        bp = BellmanPeriod(block, "DiscFac", cal)
        pre = bp.compute_pre_state("c", states, shocks=shocks, parameters=cal)
        c = policy(states, shocks, cal)["c"]

        ctrl = block.dynamics["c"]
        scope = {**cal, **pre}
        lower = _eval_bound(ctrl.lower_bound, scope)
        upper = _eval_bound(ctrl.upper_bound, scope)

        if lower is not None:
            lower = torch.as_tensor(lower, dtype=c.dtype)
            assert torch.all(c >= lower - 1e-9), (
                f"{model_id}: analytical consumption falls below the block lower bound"
            )
        assert upper is not None, f"{model_id}: expected an upper bound on consumption"
        upper = torch.as_tensor(upper, dtype=c.dtype)
        assert torch.all(c <= upper + 1e-9), (
            f"{model_id}: analytical consumption exceeds the block upper bound "
            f"(borrowing-constraint mismatch: block says c <= {float(upper.min()):.3f}, "
            f"oracle wants up to c = {float(c.max()):.3f})"
        )

        # Guard the guard: the states must actually reach the borrowing region,
        # otherwise the old (buggy) c <= m bound would have passed this test too.
        a_next = pre["m"] - c
        assert torch.any(a_next < 0.0), (
            f"{model_id}: test grid never reaches a' < 0; would not catch the bug"
        )
