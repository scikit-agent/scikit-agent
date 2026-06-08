# Benchmark Models

How do you know a numerical solver returned the right answer? One way is to run
it on a problem whose answer you already know. The
{py:mod}`skagent.models.benchmarks` registry collects such problems: benchmark
dynamic programming models, most of them with an optimal policy known in closed
form. Each entry pairs a working {py:class}`~skagent.block.DBlock` with that
policy. A few models, such as the buffer-stock problem, carry no closed form and
serve instead for numerical validation against known limiting behavior. The
registry has three uses for anyone building or testing solvers with
scikit-agent.

**Validation.** To check a solver, compare its output against the closed-form
policy on a set of test states.
{py:func}`~skagent.models.benchmarks.validate_analytical_solution` does exactly
this: it evaluates the analytical policy on a standard grid, checks feasibility
together with any model-specific identities, and returns a `"PASSED"` or
`"FAILED"` status with diagnostics on the consumption range.

**Regression testing.** The same comparison runs in the test suite, so a
regression that breaks an analytical identity surfaces early.

**Pedagogy.** The closed forms retrace the development of dynamic optimization
theory in roughly the order it was discovered. The runnable
{doc}`../auto_examples/models/plot_benchmark_models` walks through the economics
model by model, with equations and plots; this page stays focused on what the
registry contains and how to call it.

## Roster

The models below split into two groups. Six ship in the `BENCHMARK_MODELS`
registry, each reachable through
{py:func}`~skagent.models.benchmarks.list_benchmark_models` by the short
registry key shown in the table. Those keys, such as `D-1` or `U-2`, are
internal identifiers rather than names anyone uses at the whiteboard, so the
table leads with each model's descriptive name. The remaining four models ship
as standalone modules.

**Registry models** (fetch with
{py:func}`~skagent.models.benchmarks.get_benchmark_model` and the registry key):

| Model                                          | Key   | Utility   | Income                    | Closed form            |
| ---------------------------------------------- | ----- | --------- | ------------------------- | ---------------------- |
| Finite-horizon log utility                     | `D-1` | Log       | Wealth only               | Remaining-horizon MPC  |
| Infinite-horizon perfect foresight             | `D-2` | CRRA      | Constant $y$              | Linear in total wealth |
| Blanchard mortality                            | `D-3` | CRRA      | Constant $y$              | Same with $s\beta$     |
| Hall random walk                               | `U-1` | Quadratic | Stochastic, $\beta R = 1$ | PIH annuity rule       |
| Log utility with permanent shocks (normalized) | `U-2` | Log       | Geometric random walk     | $c = (1-\beta)(m + h)$ |
| Buffer stock                                   | `U-3` | CRRA      | Geometric random walk     | None (numerical only)  |

**Standalone modules (not in `BENCHMARK_MODELS`):**

| Module                                                 | Problem                        | Closed form                 |
| ------------------------------------------------------ | ------------------------------ | --------------------------- |
| {py:mod}`~skagent.models.fisher`                       | Fisher two-period              | CRRA Euler + budget         |
| {py:mod}`~skagent.models.perfect_foresight`            | PF with mortality and growth   | Linear in total wealth      |
| {py:mod}`~skagent.models.perfect_foresight_normalized` | Same, normalized variables     | Linear in normalized wealth |
| {py:mod}`~skagent.models.resource_extraction`          | Reed (1979) renewable resource | Constant escapement         |

## Using the Registry

Look up each entry by its registry key. The typical workflow is to fetch a
model, evaluate its analytical policy, and validate a candidate solver against
it:

```python
from skagent.models.benchmarks import (
    list_benchmark_models,
    get_benchmark_model,
    get_benchmark_calibration,
    get_analytical_policy,
    has_analytical_policy,
    validate_analytical_solution,
)

list_benchmark_models()
# {'D-1': '...', 'D-2': '...', 'D-3': '...',
#  'U-1': '...', 'U-2': '...', 'U-3': '...'}

block = get_benchmark_model("D-2")
calibration = get_benchmark_calibration("D-2")

# Closed-form entries expose an analytical policy; query before fetching.
if has_analytical_policy("D-2"):
    policy = get_analytical_policy("D-2")

result = validate_analytical_solution("D-2", tolerance=1e-8)
result["validation"]  # 'PASSED'
```

Models without a closed form report `False` from
{py:func}`~skagent.models.benchmarks.has_analytical_policy`, and requesting
their analytical policy raises {py:exc}`ValueError`. For those, solve
numerically with {py:class}`~skagent.loss.EulerEquationLoss` and
{py:func}`~skagent.algos.maliar.maliar_training_loop`; see {doc}`algorithms`.

## See Also

- {doc}`../auto_examples/models/plot_benchmark_models` is the runnable
  companion: a model-by-model tour that loads each closed-form policy, shows its
  equations, validates it on the standard grid, and plots the lesson each model
  teaches.
- {doc}`../api/models` for the full API reference, including every registry
  helper and the standalone model modules.
- {doc}`algorithms` for the numerical solvers used on models that have no closed
  form.
