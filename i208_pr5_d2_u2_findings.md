# i208 Phase-2 PR5 — D-2 and U-2 closed-form discrepancies

**Status:** findings for specialist review. The `vbi.solve_bellman` machinery is
implemented and validated; the two closed-form benchmark checks below do **not**
pass, and the reasons are model-level (benchmark definitions vs. their stated
analytical policies), not solver bugs. The D-2/U-2 closed-form tests were
therefore deliberately **not** added to the suite pending this review.

Context: PR5 adds `vbi.solve_bellman` (value-function iteration driving
`bellman_step` to a fixed point) and `vbi.value_array_to_function` (rebuild a
continuation from a value grid). See `design.md` §3–§4, §9 row 5.

---

## TL;DR

- **`solve_bellman` is validated.** It matches the D-4 oracle
  (`d4_vfi_reference_policy`) to ~1%, and on D-2 it matches an _independent
  dense cash-on-hand VFI_ to <1e-3. Two independent exact solvers agree.
- **D-2:** both numerical solvers disagree with `d2_analytical_policy` by
  ~6–10%. Near-certain cause: **`d2_block` imposes a no-borrowing constraint
  (`c ≤ m` ⟺ next-period assets `a' ≥ 0`), but `d2_analytical_policy` is the
  _unconstrained_ perfect-foresight closed form** that borrows against human
  wealth. The constraint binds at low wealth and depresses consumption
  everywhere. D-2-as-coded is effectively a constrained problem (like D-4),
  which has no simple closed form.
- **U-2:** the iterated solve collapses onto the consumption **upper bound**
  (`0.1·m + 2`) at every grid point, even though that bound does not bind for
  the true policy. A single backup under a good continuation gives an interior
  answer, so the optimizer is fine — the value **iteration** is converging to
  the wrong fixed point. Less fully diagnosed than D-2; candidate causes below.

---

## What is validated (not in question)

| Check                                                                           | Result                                              |
| ------------------------------------------------------------------------------- | --------------------------------------------------- |
| D-4 `solve_bellman` vs `d4_vfi_reference_policy` oracle                         | max abs err ~1.0e-2 (grid interpolation), converges |
| D-2 `solve_bellman` (a-grid) vs independent dense m-grid VFI                    | agree to <1e-3                                      |
| Single `bellman_step` under the true analytic continuation (D-2, existing test) | matches closed form to 1e-3                         |

The optimizer, discount-factor resolution, reward summation, transition, and the
iteration wiring are all exercised and correct. `value_array_to_function` uses
`RegularGridInterpolator` with linear extrapolation (clamping to the boundary
was tried and collapses the policy — see note at the end).

---

## D-2

**Model (`d2_block`, `d2_calibration`):** deterministic CRRA, `DiscFac = 0.96`,
`CRRA (σ) = 2`, `R = 1.03`, `y = 1`.

```
m  = a·R + y
c  = Control(["m"], lower_bound = 0, upper_bound = m)     # c ≤ m  ⟺  a' ≥ 0
a' = m - c
u  = crra_utility(c, 2) = c^(1-2)/(1-2) = -1/c
```

**Stated closed form (`d2_analytical_policy`):** the unconstrained
perfect-foresight rule `c = κ·(m + H)`, `κ = (R − (βR)^{1/σ})/R ≈ 0.03458`,
`H = y/(R−1) = 33.33`.

### Numbers (consumption `c` at arrival assets `a`)

| a   | `solve_bellman` (a-grid) | independent dense m-grid VFI | `d2_analytical_policy` |
| --- | ------------------------ | ---------------------------- | ---------------------- |
| 1.0 | ≈1.180                   | 1.1186                       | 1.2228                 |
| 2.0 | 1.182                    | 1.1799                       | 1.2584                 |
| 3.0 | ≈1.232                   | 1.2320                       | 1.2940                 |

The two numerical solvers agree with each other; both fall ~6–10% below the
closed form.

### Ruled out (does not shrink the gap)

- **Grid resolution** — error is flat across 30 → 240 grid points (not O(h²), so
  not interpolation error).
- **Extrapolation policy** — linear extrapolation and boundary clamping give
  identical results; wider grids (`a_max` 8 → 40) do not help.
- **Optimizer method** — L-BFGS-B, Nelder-Mead, Powell, TNC, SLSQP all
  identical.
- **Missing consumption floor** — separately fixed (`d2_block` now has
  `lower_bound = 0`; previously scipy could chase `c → −∞`, where `−1/c → +∞`).
  This fixed the overflow/instability but not the ~8% gap.

### Root cause (near-certain)

`d2_block`'s `upper_bound = m` enforces `a' = m − c ≥ 0` — a **no-borrowing /
liquidity constraint** — but `d2_analytical_policy` is the _unconstrained_ rule,
which borrows against human wealth `H = 33.3`. At the unconstrained optimum
next-period assets are `a' = (1−κ)m − κH`, which is **negative for `m ≲ 1.19`**
(i.e. `a ≲ 0.19`): the frictionless agent wants to borrow at low wealth, and the
block forbids it.

A binding constraint at low wealth lowers the value function in those states,
which propagates through the Bellman recursion and **reduces consumption in all
states** (including where the constraint does not currently bind) — the standard
liquidity-constraint depression of consumption relative to the PIH/PF benchmark
(Deaton 1991; Carroll). Both numerical solvers impose `a' ≥ 0`; the closed form
does not. Hence they agree with each other and disagree with the formula.

### Decision needed

Pick one — this is a benchmark-definition choice, not a solver fix:

1. **Treat D-2 as a constrained model** (as coded). Then `d2_analytical_policy`
   is the wrong oracle; D-2 should validate against a numerical reference like
   D-4's `d4_vfi_reference_policy` (which `solve_bellman` already matches).
2. **Make D-2 match the unconstrained closed form.** Relax the borrowing
   constraint: raise the upper bound to allow borrowing against human wealth
   (e.g. `c ≤ m + H`, or the natural-borrowing-limit form), so the frictionless
   `c = κ(m + H)` is feasible.

**Confirmation test** (one line, not yet run per the stop-inquiry request):
re-run `solve_bellman` on a D-2 variant with the borrowing constraint relaxed
and verify it converges to `κ(m + H)`.

---

## U-2

**Model (`u2_block`, `u2_calibration`):** normalized log-utility PIH,
`DiscFac = 0.96`, `R = 1.03`, `sigma_psi = 0` (so `ψ ≡ 1`).

```
m  = R·a / clamp(ψ, 1e-8) + 1        # torch op in the dynamics
c  = Control(["m"], lower_bound = 0.01, upper_bound = 0.1·m + 2)   # "loose, anti-Ponzi"
a' = m - c
u  = crra_utility(c, 1) = log(c)
```

**Stated closed form (`u2_analytical_policy`):** `c = (1−β)(m + h)`,
`h = 1/(R−1) = 33.33`, MPC `= 1 − β = 0.04`. So `c ≈ 0.04·m + 1.37`.

### Observations

- **Iterated `solve_bellman` sits on the consumption cap `0.1·m + 2` at every
  grid point** (e.g. `a=0`: c=2.10=cap vs want 1.37; `a=1.82`: c=2.287=cap vs
  want 1.45; `a=10`: c=3.13=cap vs want 1.79). The cap does **not** bind for the
  true policy (`0.04m+1.37 ≪ 0.1m+2`), yet the solver rides it — i.e. the solver
  _over-consumes_ (opposite of D-2's under-consumption).
- **A single `bellman_step` under a hand-built good continuation gives an
  interior answer** (`a=1`: 1.49 vs 1.41; `a=2`: 1.54 vs 1.46), so the per-point
  optimizer is not the problem. The value **iteration** is converging to a wrong
  fixed point / flat value function whose implied policy is the cap.
- Independent of grid width and of warm-start (disabling warm-start does not
  change the outcome). Optimizer `ABNORMAL`-termination warnings appear at
  high-`a` grid points, whose corrupted values may feed back through iteration.

### Candidate causes (hypotheses — not confirmed)

1. **Value-iteration slope collapse.** With `βR = 0.9888` (near 1), huge human
   wealth (`h = 33`), and log utility, the iterated value function may be too
   flat in `a` on the tested grids, so `βR·V'(a') < u'(c)` even at the cap →
   consume to the cap. May need a much wider/finer grid or many more iterations
   than D-2/D-4.
2. **Torch in the dynamics.** `m = R·a/clamp(ψ)+1` and `u = log(c)` return torch
   tensors; the scipy objective coerces to float, but numerical differentiation
   over a piecewise-linear (grid-interpolated) continuation may be the source of
   the `ABNORMAL` optimizer exits at high `a`.
3. **The cap itself.** `0.1·m + 2` was chosen as an anti-Ponzi bound for the
   _neural_ solver (network transforms keep `c` interior); it may interact badly
   with an exact grid solver and the large human wealth.

### Suggested next steps for U-2

- Dump the converged value function and its slope over the grid; check whether
  `V'(a)` is plausibly `≈ u'(c_true)/βR`.
- Re-run with the `ABNORMAL` points logged (they are already warned) and see if
  a gradient-free method or a smoother (e.g. cubic) value interpolation removes
  the collapse.
- Build an independent dense-grid U-2 VFI (as done for D-2) as the arbiter: does
  the true fixed point give `≈ (1−β)(m+h)`, and does `solve_bellman` approach it
  with enough grid/iterations?

---

## Reproduction

Scratch experiments (session scratchpad, not committed):
`.../scratchpad/exp_*.py` — notably `exp_ref.py` (independent dense m-grid VFI
for D-2), `exp_clampvsextrap.py` (grid/fill independence), `exp_u2b.py` (U-2
single-backup vs iteration), `exp_matrix.py` (optimizer-method matrix).

Minimal D-2 comparison:

```python
import numpy as np, skagent.algos.vbi as vbi
from skagent.bellman import BellmanPeriod
import skagent.models.benchmarks as bm

cal = bm.d2_calibration
bp = BellmanPeriod(bm.d2_block, "DiscFac", cal)
dr, va, _ = vbi.solve_bellman(
    bp, {"a": np.linspace(0, 8, 60)}, scope=cal, tol=1e-7, max_iter=6000
)
for a in [1.0, 2.0, 3.0]:
    m = a * cal["R"] + cal["y"]
    print(a, dr["c"](m), bm.d2_analytical_policy({"a": a}, {}, cal)["c"])
```

---

## Note on the continuation interpolant

`value_array_to_function` uses linear **extrapolation** past the grid edges
(`RegularGridInterpolator(..., fill_value=None)`). Boundary **clamping** (flat
extrapolation, à la `numpy.interp`) was tried first and collapses D-2/U-2 onto a
bound, because a flat continuation zeroes the marginal value of saving. This
choice is orthogonal to the D-2/U-2 discrepancies above (both fills give the
same D-2 numbers) but is worth knowing when reading the code.
