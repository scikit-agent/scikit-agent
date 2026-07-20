# Constraining an Optimization Problem

Most dynamic programs worth solving constrain their decisions. An agent who can
choose any value of a control faces a different, and often degenerate, problem
from one whose choice must stay inside a feasible set; for example, a consumer
who cannot borrow solves a genuinely harder problem than one who can. For
gradient-based solvers the constraint is a complication in its own right:
training a neural network means differentiating through the policy, and a hard
constraint introduces exactly the kind of kink that gradients handle badly.

scikit-agent addresses this with one declaration and two enforcement mechanisms.
Bounds are declared once, on the {py:class}`~skagent.block.Control` object. A
policy network can then build feasibility into its output layer, and a training
loss can encode the optimality conditions that hold where the constraint binds.
The two mechanisms answer different questions. The first guarantees that every
candidate policy is feasible; the second teaches the solver where the constraint
should bind. They compose, and for problems with binding constraints they work
best together.

## Declaring Bounds

Each bound on a {py:class}`~skagent.block.Control` is a callable whose argument
names refer to variables in the control's information set. A constant bound is a
zero-argument callable.

```python
c = ska.Control(
    ["m"],
    lower_bound=lambda: 1e-3,
    upper_bound=lambda m: m,
)
```

For example, this declares a decision $c$ that must stay between a small
positive floor and the pre-decision state $m$; in a consumption-saving model the
upper bound is a borrowing constraint. Either bound may be omitted, in which
case the control is unbounded on that side. Passing a raw number such as
`upper_bound=1.0` raises a `TypeError` when a policy network is built from the
block; write `lambda: 1.0` instead. Everything below reads this single
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
approaches but never equals either endpoint, which is why the mechanism is
called open-bounds scaling. The payoff is robustness. Every policy the optimizer
visits is feasible, so a reward like $\log(c)$ is never evaluated at an
infeasible point mid-training. The limitation is that the transform carries no
information about _whether_ the constraint should bind; a policy can hug a bound
without the loss ever being told that a Lagrange multiplier belongs there. The
transform is on by default and can be disabled with `apply_open_bounds=False`.

## Optimality at the Bound: Fischer-Burmeister Complementarity

Where a constraint binds, the first-order condition changes. With an Euler
residual $f$ and constraint slack $s = \overline{x} - x$, optimality requires
the complementarity conditions

$$
f \geq 0, \qquad s \geq 0, \qquad f \cdot s = 0,
$$

so the residual need not vanish at the bound; the slack must. The equivalent
equation $\min(f, s) = 0$ is not differentiable, which is the wrong shape for a
training loss. Following equation (25) of Maliar, Maliar, and Winant (2021),
{py:func}`~skagent.utils.fischer_burmeister` replaces it with the smooth
Fischer-Burmeister function

$$
\operatorname{FB}(f, s) = f + s - \sqrt{f^2 + s^2} = 0,
$$

which is zero exactly where the complementarity conditions hold and
differentiable everywhere. Setting `constrained=True` on
{py:class}`~skagent.loss.EulerEquationLoss` builds this residual for every
control that declares an `upper_bound`:

```python
loss = EulerEquationLoss(bellman_period, constrained=True)
```

The squared expected residual is estimated in all-in-one fashion as the product
$\operatorname{FB}(f_a, s) \cdot \operatorname{FB}(f_b, s)$ at two independent
next-period shock draws, which keeps the estimate unbiased. A control without an
explicit `upper_bound` falls back to the one-sided penalty
$\operatorname{relu}(-f_a) \cdot \operatorname{relu}(-f_b)$, which punishes only
violations of $f \geq 0$.

One scope restriction is worth stating plainly: the constrained loss currently
models the upper-bound side of the complementarity condition only. A
`lower_bound` declared on a `Control` is respected by the network transform
above, but it does not yet enter the complementarity residual; bilateral support
requires flipping the residual sign for lower-binding cases
($\operatorname{FB}(-f, x - \underline{x})$) and is left as a follow-up.

## Using Both Together

The mechanisms are complementary rather than competing. The bounded network
keeps every training iterate feasible; the Fischer-Burmeister loss supplies the
optimality condition that tells the solver where the bound binds and by how
much. The constrained perfect-foresight benchmark (`D-4` in the
{doc}`benchmark_models` registry) is the in-package demonstration: a policy
network trained on `EulerEquationLoss(constrained=True)` matches a
value-function-iteration oracle to well within one percent on average, on a
problem whose borrowing constraint binds at low wealth and where an
unconstrained Euler objective cannot identify the policy level at all. For a
runnable walkthrough on the buffer-stock model, see
{doc}`../auto_examples/algorithms/plot_maliar_training_loop`.

A practical rule of thumb: if the constraint cannot bind in the region of the
state space you care about, the default network bounds are enough. If it can
bind and you train on an Euler or first-order-condition objective, add
`constrained=True` so the loss knows what optimality looks like at the boundary.

## Grid Solvers

Constraint handling is not specific to neural methods. The value backwards
induction solver ({py:mod}`skagent.algos.vfi`) reads the same `Control` bounds
and passes them to `scipy.optimize.minimize` as box constraints on each grid
point's optimization, with no smooth reformulation needed because no gradient
flows through the policy.
