# Solving Models with the Maliar Method

Many dynamic programs have no closed-form solution, and discretizing the state
space scales badly as states accumulate. The Maliar method (Maliar, Maliar, and
Winant 2021) sidesteps both problems: it represents the decision rule as a
neural network and turns the search for a policy into minimizing an
equation-residual loss over a sample of states. This page explains what the
method minimizes, the two residual losses scikit-agent provides, and how to run
the training loop.

## The All-in-One Expectation Operator

A dynamic optimality condition holds in expectation over next period's shocks.
Write $f$ for the residual of that condition, the quantity that is zero when the
policy is optimal. We want to drive the squared expected residual
$(\mathbb{E}[f])^2$ to zero across the sampled states.

The obvious estimator, squaring a single simulated residual, is biased. Because
$\mathbb{E}[f^2] = (\mathbb{E}[f])^2 + \mathrm{Var}(f)$, minimizing it rewards a
policy for suppressing the residual's sampling noise rather than for satisfying
the condition in expectation, and that extra variance term distorts the solution
of any stochastic model. The Maliar method instead draws two independent shock
realizations, evaluates the residual at each while holding the current control
fixed, and multiplies the two. Call these residuals $f_a$ and $f_b$; because the
draws are independent,

$$
\mathbb{E}[f_a\, f_b] = \mathbb{E}[f_a]\,\mathbb{E}[f_b] = (\mathbb{E}[f])^2,
$$

so the product is an unbiased estimator of the target. scikit-agent calls this
the "all-in-one" operator. The two draws enter the training grid under a
two-copy naming convention: a shock `psi` appears as `psi_0` and `psi_1`, and
`shock_copies=2` requests them.

## Two Residual Losses

scikit-agent provides two losses built on this operator. They differ in which
optimality condition the residual $f$ encodes.

### The Euler Loss

The Euler loss, {py:class}`~skagent.loss.EulerEquationLoss`, encodes the
first-order condition. For a consumption-saving model it equates the marginal
utility of consuming today with the discounted expected marginal utility of
consuming tomorrow. Let $u$ be the utility function, $c_t$ consumption in period
$t$, $\beta$ the discount factor (read from the model's discount variable), and
$R$ the gross return on savings. The residual is

$$
f = u'(c_t) - \beta\, \mathbb{E}_t\!\left[\,R\, u'(c_{t+1})\,\right],
$$

which is zero at the optimum. A first-order condition constrains the _slope_ of
the policy, how the control responds to the state, but leaves its overall
_level_ unpinned. Training on the Euler residual alone can therefore learn a
policy of the right shape sitting at the wrong height. The loss needs only a
decision rule, so it pairs with a policy-only network,
{py:class}`~skagent.ann.BlockPolicyNet`.

### The Bellman Loss

The Bellman loss, {py:class}`~skagent.loss.BellmanEquationLoss`, encodes the
Bellman equation itself. Let $V$ be the value function, $s$ the current state,
and $s'$ next period's state. The residual is

$$
f = V(s) - \left[\,u(s, c) + \beta\, V(s')\,\right],
$$

the gap between the value a state is assigned and the reward-plus-continuation
it actually delivers. Because it references $V$ directly, this loss anchors the
policy's level as well as its slope, curing the level ambiguity that the Euler
residual leaves behind. It requires an explicit value function, so it pairs with
{py:class}`~skagent.ann.BlockPolicyValueNet`, a shared-backbone network carrying
a policy head and a value head that one optimizer trains together.

### Choosing Between Them

| Property                  | Euler loss          | Bellman loss          |
| ------------------------- | ------------------- | --------------------- |
| Optimality condition      | first-order (Euler) | Bellman equation      |
| Requires a value function | no                  | yes                   |
| Pairs with                | `BlockPolicyNet`    | `BlockPolicyValueNet` |
| Identifies                | slope only          | slope and level       |

Reach for the Euler loss when it is lighter to set up and the policy's level is
pinned some other way; reach for the Bellman loss when the level itself must be
correct, as when validating a trained policy against a known closed-form
solution.

## Running the Training Loop

One call to {py:func}`~skagent.algos.maliar.maliar_training_loop` runs the full
outer loop. It builds a {py:class}`~skagent.ann.BlockPolicyNet`, alternates a
batch of stochastic-gradient updates with a forward-simulation step that redraws
the training states from the model's own ergodic set, and stops once the
parameters settle or a maximum iteration count is reached.

```python
import skagent.algos.maliar as maliar
import skagent.loss as loss

euler_loss = loss.EulerEquationLoss(bp, parameters=calibration, constrained=True)

policy, states = maliar.maliar_training_loop(
    bp,
    euler_loss,
    states_0,
    calibration,
    shock_copies=2,  # independent draws for the all-in-one operator
    max_iterations=40,
    tolerance=1e-6,
    network_width=32,
)
```

Resampling is what makes this more than gradient descent on a fixed grid:
trained only on the initial grid, the network would learn the policy where the
grid happens to sit, not where the model actually spends its time. The loop
returns the trained policy network and the final panel of training states.

Because {py:func}`~skagent.algos.maliar.maliar_training_loop` builds a
policy-only network internally, it pairs naturally with the Euler loss. To train
with a value head instead, build a {py:class}`~skagent.ann.BlockPolicyValueNet`
and drive it with {py:func}`~skagent.ann.train_block_nn` and
{py:class}`~skagent.loss.BellmanEquationLoss` directly, resampling the states
between batches yourself. The known-solution example below does exactly this.

## Constrained Controls

When a control is bounded and the constraint can bind, pass `constrained=True`
to {py:class}`~skagent.loss.EulerEquationLoss`, as above. The loss then encodes
the complementarity condition that holds at the bound through a smooth
Fischer-Burmeister residual. The {doc}`constraints` page covers the mechanics,
and how the loss-side condition composes with feasibility built into the
network's output layer.

## Worked Examples

Two runnable examples in the gallery carry these pieces end to end:

- {doc}`../auto_examples/algorithms/plot_maliar_training_loop` trains the Euler
  loop on U-3, Carroll's buffer-stock consumption-saving model, which has no
  closed form, and checks the learned consumption function against buffer-stock
  invariants.
- {doc}`../auto_examples/algorithms/plot_train_against_known_solution` trains
  the Bellman loss with a value head on U-2, a permanent-income model whose
  policy is known in closed form, and reports a mean relative error near one
  percent.

For the class and function reference, see {doc}`../api/algorithms` and
{doc}`../api/loss`.
