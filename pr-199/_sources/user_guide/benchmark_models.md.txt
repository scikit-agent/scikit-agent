# Benchmark Models

How much of this year's income should a household spend, and how much should it
save for later? Consumption economics studies that one decision, made again and
again over a lifetime, when income is uncertain, life is finite, and spending
today comes at the expense of spending tomorrow. The benchmark models on this
page are the classic answers. Each starts from the simplest possible case and
adds a single realistic complication: income that grows, a chance of not
surviving into the next period, or a shock that moves income permanently. The
question stays the same throughout, namely how the optimal saving rule should
respond. Read in order, the models retrace how economists came to understand
saving over the past century.

The {py:mod}`skagent.models.benchmarks` registry collects these problems in a
form a solver can use. Most entries pair a working
{py:class}`~skagent.block.DBlock` with an analytical decision function whose
optimal policy is known in closed form. The buffer-stock model is the exception:
it has no closed-form solution and is included for numerical validation against
its limiting MPC properties. The registry serves three purposes.

**Validation.** A numerical solver can be checked against the closed-form policy
on a set of test states. The function
{py:func}`~skagent.models.benchmarks.validate_analytical_solution` evaluates the
analytical policy on a standard grid, checks feasibility together with any
model-specific identities, and returns a `"PASSED"` or `"FAILED"` status
alongside diagnostics on the consumption range.

**Regression testing.** The same comparison runs in the test suite, so any
regression that breaks an analytical identity is caught early.

**Pedagogy.** Working through the closed forms, from finite-horizon log utility
through Hall's random walk, retraces the development of dynamic consumption
theory in roughly the order it was discovered.

The closed forms are written in the language of discrete-time dynamic
programming, but the prose for each model is meant to stand on its own. Readers
who want the underlying machinery can start with Carroll (2024),
{py:mod}`~skagent.models.perfect_foresight_normalized` for the workhorse
formulation, and the {doc}`../auto_examples/index` for runnable code.

## Standard Timing Convention

Throughout this page and in the {py:mod}`~skagent.models.benchmarks` module,
periods are indexed by $t \in \{0, 1, 2, \ldots\}$ and the following symbols
carry fixed meaning.

| Symbol                                               | Meaning                                                      |
| ---------------------------------------------------- | ------------------------------------------------------------ |
| $A_{t-1}$                                            | Beginning-of-period assets (arrival state, before interest)  |
| $R$                                                  | Gross return on assets, $R = 1 + r > 1$                      |
| $y_t$                                                | Non-capital income realized in period $t$                    |
| $m_t = R\, A_{t-1} + y_t$                            | Cash-on-hand (market resources)                              |
| $c_t$                                                | Consumption (the control)                                    |
| $A_t = m_t - c_t$                                    | End-of-period assets                                         |
| $H_t = \mathbb{E}_t \sum_{s\geq 1} R^{-s}\, y_{t+s}$ | Human wealth                                                 |
| $W_t = m_t + H_t$                                    | Total wealth                                                 |
| $u(c)$                                               | Period utility                                               |
| $\beta$                                              | Discount factor                                              |
| TVC                                                  | $\lim_{T\to\infty} \mathbb{E}_0\, \beta^T u'(c_T)\, A_T = 0$ |

Normalized models additionally use lowercase $m, c, a$ for ratios to permanent
income $P_t$.

## Roster

The models below split into two groups. Six ship in the `BENCHMARK_MODELS`
registry, each reachable through
{py:func}`~skagent.models.benchmarks.list_benchmark_models` by the short
registry key shown in the table. Those keys, such as `D-1` or `U-2`, are
internal identifiers rather than names anyone uses at the whiteboard, so the
discussion below leads with each model's descriptive name. The remaining four
models ship as standalone modules.

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

## Deterministic Benchmarks

### Finite-Horizon Log Utility

The remaining-horizon MPC $(1 - \beta) / (1 - \beta^{T-t})$ is the cleanest
illustration of _time non-stationarity from a fixed terminal date_. The
infinite-horizon limit recovers the constant-MPC rule $c_t = (1 - \beta)\, W_t$,
which is the $\sigma = 1$ special case of the infinite-horizon perfect-foresight
model below.

```{eval-rst}
.. autofunction:: skagent.models.benchmarks.d1_analytical_policy
   :no-index:
```

### Infinite-Horizon CRRA Perfect Foresight

This is the workhorse model for almost everything that follows. The MPC
$\kappa = (R - (\beta R)^{1/\sigma}) / R$ collapses to $1 - \beta$ under log
utility ($\sigma = 1$), and to $r/R$ whenever $\beta R = 1$, regardless of
$\sigma$. The latter limit coincides with the permanent-income MPC out of total
wealth in Hall's random walk below: there the same value $r/R$ arises from a
quite different model (quadratic utility, stochastic income), and the two
derivations meet at the same number through the algebraic identity
$(\beta R)^{1/\sigma} = 1$. Human wealth $H = y / r$ converts the infinite
stream of future income into a single number that the linear rule can act on.

```{eval-rst}
.. autofunction:: skagent.models.benchmarks.d2_analytical_policy
   :no-index:
```

### Blanchard Discrete-Time Mortality

Adding i.i.d. survival risk does not break tractability: it scales the effective
discount factor from $\beta$ to $s\beta$. The resulting MPC $\kappa_s$ strictly
exceeds the no-mortality MPC $\kappa$, formalizing the intuition that a lower
survival probability erodes patience.

```{eval-rst}
.. autofunction:: skagent.models.benchmarks.d3_analytical_policy
   :no-index:
```

### Fisher Two-Period Model

The simplest dynamic-programming problem with a closed-form solution. Useful as
a numerical sanity check in two dimensions and as the example that connects the
intertemporal-choice diagram in introductory macroeconomics to the
infinite-horizon machinery used throughout the rest of this page.

```{eval-rst}
.. automodule:: skagent.models.fisher
   :no-members:
   :no-index:
```

The block is exposed as {py:data}`skagent.models.fisher.block` with calibration
{py:data}`skagent.models.fisher.calibration`.

### Perfect Foresight (Level Variables)

Adds permanent income growth $G$ and i.i.d. survival to the infinite-horizon
perfect-foresight model. The closed-form solution remains linear in total
wealth, but the human-wealth term now reflects growing income:
$H_t = G\, P_t / (R - G)$.

```{eval-rst}
.. automodule:: skagent.models.perfect_foresight
   :no-members:
   :no-index:
```

Two blocks are exposed: {py:data}`skagent.models.perfect_foresight.block` (with
the survival shock) and
{py:data}`skagent.models.perfect_foresight.block_no_shock` (deterministic).

### Perfect Foresight (Normalized)

The same problem expressed in variables divided by permanent income. The state
space collapses from $(M, P)$ to the single ratio $m$, which is what makes both
analytical and neural-network solutions practical for richer models.

```{eval-rst}
.. automodule:: skagent.models.perfect_foresight_normalized
   :no-members:
   :no-index:
```

## Stochastic Benchmarks with Closed Forms

### Hall's (1978) Random Walk

The historically pivotal observation: under quadratic utility and the neutral
SDF $\beta R = 1$, the Euler equation forces consumption to be a martingale,
regardless of how complicated the income process is. This result motivated the
empirical PIH literature, which tested whether $\Delta c_{t+1}$ is
unforecastable from period-$t$ information.

```{eval-rst}
.. autofunction:: skagent.models.benchmarks.u1_analytical_policy
   :no-index:
```

### Log Utility with Permanent Income Shocks

Geometric random-walk permanent income looks intractable in level variables. The
normalization $m = M/P$, $c = C/P$ collapses the state space, and the closed
form $c = (1 - \beta)(m + 1/r)$ falls out cleanly. The same normalization
applied to the borrowing-constrained CRRA case, the buffer-stock model below,
yields a problem with _no_ closed form: the constraint plus uncertainty break
the linearity.

```{eval-rst}
.. autofunction:: skagent.models.benchmarks.u2_analytical_policy
   :no-index:
```

### Resource Extraction (Reed 1979)

The constant-escapement policy $u_t^{\ast} = \max(0, x_t - S^{\ast})$ is the
canonical closed-form solution in renewable-resource management. With
multiplicative shocks and stock-dependent unit cost $c_0/x$, the optimal
escapement $S^{\ast}$ solves a single algebraic first-order condition rather
than a Bellman equation:

$$
S^{\ast} \;=\; \frac{c_0\, (1 - \delta)}{p\, (1 - \delta r)},
$$

where $\delta$ is the discount factor and $r$ is the _mean biological growth
rate_ of the resource stock, not the net interest rate $R - 1$ used elsewhere on
this page. The model requires the impatience condition $\delta r < 1$.

```{eval-rst}
.. automodule:: skagent.models.resource_extraction
   :no-members:
   :no-index:
```

The optimal decision rule is constructed by
{py:func}`skagent.models.resource_extraction.make_optimal_decision_rule`.

## Beyond Closed Forms: The Buffer-Stock Model

The buffer-stock model (CRRA utility, permanent and transitory income shocks,
binding borrowing constraint $c \leq m$) has no closed-form policy. Its limiting
properties are nonetheless known: the MPC stays in $(0, 1)$, decreases in
wealth, and converges to the perfect-foresight $\kappa$ as wealth grows large.
These limits are what the test harness in
{py:func}`~skagent.models.benchmarks.validate_analytical_solution` checks when a
numerical solver claims to have solved it.

## Validating Analytical Solutions

Each entry in the registry can be looked up by ID:

```python
from skagent.models.benchmarks import (
    list_benchmark_models,
    get_benchmark_model,
    get_benchmark_calibration,
    get_analytical_policy,
    validate_analytical_solution,
)

print(list_benchmark_models())
# {'D-1': '...', 'D-2': '...', 'D-3': '...',
#  'U-1': '...', 'U-2': '...', 'U-3': '...'}

block = get_benchmark_model("D-2")
calib = get_benchmark_calibration("D-2")
policy = get_analytical_policy("D-2")

result = validate_analytical_solution("D-2", tolerance=1e-8)
print(result["validation"])  # 'PASSED'
```

Requesting the analytical policy for the buffer-stock model raises
{py:exc}`ValueError`, since none exists; the error message points to
{py:class}`~skagent.loss.EulerEquationLoss` (combined with
{py:func}`~skagent.algos.maliar.maliar_training_loop`) as the recommended
numerical alternative.

The full registry helpers are documented below.

```{eval-rst}
.. autofunction:: skagent.models.benchmarks.list_benchmark_models
   :no-index:
.. autofunction:: skagent.models.benchmarks.get_benchmark_model
   :no-index:
.. autofunction:: skagent.models.benchmarks.get_benchmark_calibration
   :no-index:
.. autofunction:: skagent.models.benchmarks.get_analytical_policy
   :no-index:
.. autofunction:: skagent.models.benchmarks.get_test_states
   :no-index:
.. autofunction:: skagent.models.benchmarks.validate_analytical_solution
   :no-index:
.. autofunction:: skagent.models.benchmarks.euler_equation_test
   :no-index:
.. autofunction:: skagent.models.benchmarks.get_analytical_lifetime_reward
   :no-index:
```

## See Also

- {doc}`../auto_examples/models/plot_benchmark_models` is the runnable companion
  to this page: a six-step pedagogical tour that loads each closed-form policy,
  validates it on the standard test grid, and walks through the lessons each
  model contributes (finite-horizon limits, the mortality wedge, Hall's
  martingale, normalization, and why the buffer-stock model has no closed form).
- {doc}`../auto_examples/index` for the full gallery of runnable examples.
- {doc}`algorithms` for the numerical methods used to solve models that do not
  have closed forms.
- {doc}`../api/models` for the full API reference.
