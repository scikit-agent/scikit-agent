# i66 TODO — VBI tests & solver gaps

## Done

- Cleaned up `src/skagent/algos/vbi.py`:
  - Removed stale `# old! (should be negative)` comment on `negated_value`.
  - Fixed upper-bound default `1e-12` → `1e12` ("a very high number").
  - Removed leftover debug `print`s (`print(feq)`, `print(bounds)`,
    `print(policy_data)`).
  - Changed open lower-bound default `-1e-6` → `-1e12` (effectively −∞,
    symmetric with upper).
- Added comprehensive `test_vbi_conftest` class in `tests/test_vbi.py` (8 tests,
  all passing alongside the 3 legacy tests). Each conftest case is solved with
  VBI under a terminal (zero) continuation and checked against its analytic
  `optimal_dr` to within `1e-3` at interior grid points:

  | Case     | Exercises                                         | Optimum     |
  | -------- | ------------------------------------------------- | ----------- |
  | `case_0` | unconstrained interior optimum                    | `c = 0`     |
  | `case_1` | shock-dependent policy (`theta` in info set)      | `c = theta` |
  | `case_3` | consume cash-on-hand                              | `c = m`     |
  | `case_5` | double-bounded, upper binds                       | `c = a`     |
  | `case_6` | double-bounded, lower binds                       | `c = a`     |
  | `case_7` | lower-bound only → tests open upper-bound default | `c = 1`     |
  | `case_8` | upper-bound only → tests open lower-bound default | `c = a`     |
  | `case_9` | empty information set (constant rule)             | `c = 3`     |

  Cases 7 and 8 specifically validate the open-bound defaults (`±1e12`).

## TODO — benchmark models cannot be tested against VBI yet

The benchmark catalogue in `src/skagent/models/benchmarks.py` is all _discounted
dynamic programs_. The current `vbi.solve` cannot reproduce their analytic
policies. Structural gaps to close before benchmark policy-matching tests are
possible:

1. **No discount factor.** The Bellman backup is literally `r + cv`
   (`model.py:441`), with no β. Every benchmark's Euler equation depends on β.
2. **Only the first reward is used** — `calc_reward(...).values()[0]`
   (`model.py:435`), so e.g. D-1's two-period `u1 + u2` objective can't be
   expressed.
3. **Deterministic blocks crash.** The arrival value function calls
   `discretized_shock_dstn` on empty shocks, which throws `IndexError` (D-3/D-4
   have no shocks).
4. **VFI iteration breaks.** Feeding a fitted decision rule back as a
   continuation fails because `ar_from_data` uses a `**args` signature that
   `simulate_dynamics` can't introspect (`KeyError: 'args'`).
5. **Single control only** — `vbi.solve` raises for blocks with >1 control.

### Suggested next step

Extend VBI / `model.py` to close the gaps above:

- discount factor in the Bellman recursion,
- multi-reward summation,
- deterministic-block (empty-shock) arrival value,
- a re-feedable decision rule so value-function iteration works,

then add benchmark policy-matching tests (e.g. D-3 VFI should converge to
`c = κ·m` with `κ = (R − (βR)^{1/σ})/R`).
