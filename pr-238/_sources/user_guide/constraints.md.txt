# Constraining an Optimization Problem

Most dynamic programs constrain their decisions: a consumer cannot borrow more
than a credit limit, a portfolio share must stay between zero and one. This page
shows how to declare those bounds in scikit-agent and which solvers enforce
them.

You declare a bound once, on the {py:class}`~skagent.block.Control` object. From
that single declaration two mechanisms act, and they answer different questions:

- a **policy network** can build feasibility into its output layer, so every
  candidate policy it proposes is feasible;
- a **training loss** can encode the optimality condition that holds where a
  constraint binds.

The two are independent. You can use the network bounds alone, and for many
problems that is enough. The loss-side condition is currently available on one
loss, {py:class}`~skagent.loss.EulerEquationLoss`, and matters when you train on
an Euler objective and the constraint actually binds.

## Declaring Bounds

A bound on a {py:class}`~skagent.block.Control` is either a number or a callable
whose argument names are variables in the control's information set. A number is
a constant bound; a callable is a state-dependent one. Either side may be
omitted, leaving the control unbounded on that side.

```python
c = ska.Control(
    ["m"],
    lower_bound=1e-3,  # constant floor
    upper_bound=lambda m: m,  # state-dependent ceiling
)
```

This declares a decision $c$ that must stay between a small positive floor and
the pre-decision state $m$. In a consumption-saving model the upper bound
$c \le m$ is a no-borrowing constraint. Everything below reads this one
declaration, so a model never states its constraints twice.

## Feasibility by Construction: Bounded Policy Networks

{py:class}`~skagent.ann.BlockPolicyNet` and
{py:class}`~skagent.ann.BlockPolicyValueNet` transform the raw network output so
the policy respects the declared bounds at every point of training. Writing $z$
for the raw output of the last layer and $\underline{x}$, $\overline{x}$ for the
lower and upper bounds evaluated at the current state, the transform depends on
which bounds the control declares:

| Declared bounds | Transform                                                   | Output range                    |
| --------------- | ----------------------------------------------------------- | ------------------------------- |
| Both            | $\underline{x} + \sigma(z)\,(\overline{x} - \underline{x})$ | $(\underline{x}, \overline{x})$ |
| Lower only      | $\underline{x} + \operatorname{softplus}(z)$                | $(\underline{x}, \infty)$       |
| Upper only      | $\overline{x} - \operatorname{softplus}(z)$                 | $(-\infty, \overline{x})$       |
| None            | $z$                                                         | unbounded                       |

Here $\sigma$ is the logistic sigmoid and
$\operatorname{softplus}(z) = \log(1 + e^z)$. The bounds are _open_: the output
approaches but never equals either endpoint, so a reward like $\log(c)$ is never
evaluated at an infeasible point mid-training. What the transform does _not_ do
is signal whether a constraint should bind; a policy can hug a bound without the
loss being told a Lagrange multiplier belongs there. The transform is on by
default and can be disabled with `apply_open_bounds=False`.

## Optimality at the Bound: Fischer-Burmeister Complementarity

Where a constraint binds, the first-order condition changes. With an Euler
residual $f$ and an upper-bound slack $s = \overline{x} - x$, optimality is the
complementarity condition

$$
f \geq 0, \qquad s \geq 0, \qquad f \cdot s = 0,
$$

so the residual need not vanish at the bound; the slack must. The equivalent
equation $\min(f, s) = 0$ is not differentiable, which is the wrong shape for a
training loss. Following equation (25) of Maliar, Maliar, and Winant (2021),
{py:func}`~skagent.utils.fischer_burmeister` replaces it with

$$
\operatorname{FB}(f, s) = f + s - \sqrt{f^2 + s^2 + \varepsilon},
$$

which is zero (up to $\sqrt{\varepsilon}$) exactly where the complementarity
conditions hold. The small $\varepsilon$ under the root is a numerical
regularizer: it keeps the gradient finite at $f = s = 0$, so the residual is
differentiable everywhere, at the cost of
$\operatorname{FB}(0, 0) =
-\sqrt{\varepsilon}$ rather than exactly zero. With
the default $\varepsilon = 10^{-12}$ this offset is about $10^{-6}$, below
typical convergence tolerances but worth accounting for in a very tight test.

Set `constrained=True` on {py:class}`~skagent.loss.EulerEquationLoss` to build
this residual:

```python
loss = EulerEquationLoss(bellman_period, constrained=True)
```

The loss reads each control's declared bounds and forms the complementarity
residual that matches them:

- upper bound only: $\operatorname{FB}(f,\; \overline{x} - x)$;
- lower bound only: $\operatorname{FB}(-f,\; x - \underline{x})$, the residual
  sign flipping because a binding lower bound pushes the decision the other way;
- both bounds: a two-sided form,
  $\operatorname{FB}\!\big(\overline{x} - x,\; -\operatorname{FB}(x -
  \underline{x},\; -f)\big)$,
  which reduces to either one-sided residual when the opposite bound is slack;
- no bound: a one-sided penalty $\operatorname{relu}(-f)$ on violations of
  $f \geq 0$.

The squared expected residual is estimated in all-in-one fashion as the product
of the residuals at two independent next-period shock draws, which keeps the
estimate unbiased.

## Using Both Together

The mechanisms compose: the bounded network keeps every training iterate
feasible, while the Fischer-Burmeister loss supplies the optimality condition at
the bound. A minimal Euler-method setup on a constrained consumption-saving
block looks like this:

```python
import skagent as ska
from skagent.ann import BlockPolicyNet, train_block_nn
from skagent.loss import EulerEquationLoss

# c is bounded below by a floor and above by the no-borrowing ceiling c <= m.
bellman_period = ska.BellmanPeriod(block, "DiscFac", calibration)

policy_net = BlockPolicyNet(bellman_period)  # feasible by construction
euler_loss = EulerEquationLoss(
    bellman_period, constrained=True
)  # bound-aware optimality

policy_net, _, _ = train_block_nn(policy_net, train_grid, euler_loss, epochs=1)
```

The constrained perfect-foresight benchmark (`D-4` in the
{doc}`benchmark_models` registry) is the in-package demonstration: a policy
network trained this way matches a value-function-iteration oracle to well
within one percent on average, on a problem whose borrowing constraint binds at
low wealth and where an unconstrained Euler objective cannot identify the policy
level at all. For a runnable walkthrough, see
{doc}`../auto_examples/algorithms/plot_maliar_training_loop`.

A practical rule of thumb: if the constraint cannot bind in the region of the
state space you care about, the network bounds alone are enough. If it can bind
and you train on an Euler objective, add `constrained=True`.

## Where Each Mechanism Is Available

| Mechanism            | Where it lives                                  | Bounds handled  |
| -------------------- | ----------------------------------------------- | --------------- |
| Open-bounds network  | `BlockPolicyNet`, `BlockPolicyValueNet`         | lower and upper |
| Complementarity loss | `EulerEquationLoss(constrained=True)`           | lower and upper |
| Grid box constraints | `skagent.algos.vbi` (value backwards induction) | lower and upper |

The value backwards induction solver ({py:mod}`skagent.algos.vbi`) reads the
same `Control` bounds and passes them to `scipy.optimize.minimize` as box
constraints, with no smooth reformulation needed because no gradient flows
through the policy.
