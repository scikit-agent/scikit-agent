# Solving Models with the Maliar Method

Grid-based methods such as {doc}`value backwards induction <algorithms>` solve a
model by covering its state space with points, and that approach collapses as
states accumulate. Ten points per state is 100 points in two states, and a
million in six. Refining the grid or adding a state multiplies the work rather
than adding to it, which is the curse of dimensionality.

The Maliar method (Maliar, Maliar, and Winant 2021) gets around it in two ways.
It represents the decision rule as a neural network, so the policy is stored in
a fixed set of weights instead of a structure that grows with the state space.
It then redraws its own training states by simulating the model forward, so the
network keeps being retrained on the region the model actually visits rather
than on whatever region a fixed grid happened to cover.

The method is also model-based, and that separates it from the
reinforcement-learning approaches described in {doc}`algorithms`. An RL
algorithm such as PPO treats the model as a black box, learning only from
rewards it observes by interacting with it. The Maliar method reads the model's
equations directly and differentiates them, so what it drives to zero is the
model's own optimality condition rather than a return estimated from sampled
experience.

This page explains what the method minimizes, the two residual losses
scikit-agent provides, and how to run the training loop.

## The All-in-One Expectation Operator

Training a network means minimizing a number, so the first question is what
number measures a policy's badness. This section builds that number. It takes
one step that is not the obvious one, and the reason is worth following, because
skipping it silently produces the wrong answer.

Start with what the model gives you. An optimality condition here is a formula
that takes a state, the choice made in it, and one random draw of next period's
shocks, and returns a number called the _residual_, written $f$. The condition
does not say that $f$ is zero for each individual draw. It says $f$ is zero _on
average_ across draws. So the quantity to drive to zero is $(\mathbb{E}[f])^2$,
the squared average residual, and the square is there so overshooting counts as
badly as undershooting.

Now the trap. The obvious way to estimate this is to simulate one shock draw and
square the residual you get. That does not estimate $(\mathbb{E}[f])^2$; it
estimates $\mathbb{E}[f^2]$, and the two differ by a term that is positive
whenever the residual genuinely varies across draws:

$$
\mathbb{E}[f^2] = (\mathbb{E}[f])^2 + \mathrm{Var}(f).
$$

An optimizer handed $\mathbb{E}[f^2]$ can lower it two ways: by satisfying the
condition on average, which is the intent, or by making the residual vary less
from draw to draw, which is not. Nothing stops it from buying the cheaper of the
two. The result is a policy tuned partly to suppress the model's own randomness,
and in any model with real uncertainty that policy is wrong.

The fix costs one extra shock draw. Draw next period's shocks twice,
independently, evaluate the residual under each while holding this period's
control fixed, and multiply the two results rather than squaring either one.
Call them $f_a$ and $f_b$. Because the draws are independent, the expectation of
their product splits into the product of their expectations:

$$
\mathbb{E}[f_a\, f_b] = \mathbb{E}[f_a]\,\mathbb{E}[f_b] = (\mathbb{E}[f])^2,
$$

which is the target on the nose, with no variance term riding along.
scikit-agent calls this the "all-in-one" operator. In a deterministic model the
two draws coincide, the product reduces to $f^2$, and nothing is lost, since
there is no sampling noise to be fooled by in the first place.

One piece of mechanics is easy to misread. The training grid does carry two
copies of every shock, so a shock `psi` appears as `psi_0` and `psi_1`, and
`shock_copies=2` is what requests them. Those two copies are the period $t$ and
period $t+1$ shocks, though, not the all-in-one pair: the residual spans two
periods and needs a draw for each. The second next-period draw, the one that
makes $f_b$ independent of $f_a$, is generated inside the loss itself, so you
never supply it.

## Two Residual Losses

scikit-agent provides two losses built on this operator. They differ in which
optimality condition the residual $f$ encodes.

### The Euler Loss

The Euler loss, {py:class}`~skagent.loss.EulerEquationLoss`, encodes the
first-order condition of the Bellman equation, which ties marginal rewards in
adjacent periods together. It applies to any block, not just consumption models,
so it is worth seeing in the general form the solver actually differentiates.
Let $u$ be the period reward, $x_t$ a control, $s_{t+1}$ the next-period arrival
states, $m'$ the pre-decision states, and $\beta$ the discount factor (read from
the model's discount variable). The residual is

$$
f = u'(x_t) + \beta\, u'(x_{t+1}) \sum_s
    \frac{\partial s_{t+1}}{\partial x_t}\,
    \frac{\partial m'}{\partial s_{t+1}},
$$

and the policy is optimal where $\mathbb{E}_t[f] = 0$. The sum runs over every
channel by which today's control moves tomorrow's states, and each derivative in
it is taken from the block's own dynamics, so nothing has to be derived by hand
for a new model.

Consumption-saving is the special case most readers will recognize. The control
is consumption $c_t$, the reward is utility, and the single channel to next
period runs through savings earning a gross return $R$. Spending one more unit
today leaves one less unit of assets, so the derivative product in the sum is
just $-R$, and the general residual collapses to the textbook Euler equation

$$
f = u'(c_t) - \beta\, R\, u'(c_{t+1}),
$$

marginal utility today set against discounted marginal utility tomorrow.

A first-order condition constrains the _slope_ of the policy, how the control
responds to the state, but leaves its overall _level_ unpinned. Training on the
Euler residual alone can therefore learn a policy of the right shape sitting at
the wrong height. The loss needs only a decision rule, so it pairs with a
policy-only network, {py:class}`~skagent.ann.BlockPolicyNet`.

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

The loop returns the trained policy network and the final panel of training
states, the latter being where resampling has carried the training distribution
by the time training stops.

### Setting the Arguments

One of these arguments is fixed by the method and the rest are genuinely tuned,
which is a useful distinction when a run misbehaves.

`shock_copies` is the fixed one. It sets how many independent realizations of
each shock the training grid carries, named `psi_0`, `psi_1`, and so on. Both
losses shipped with scikit-agent are two-period residuals, comparing period $t$
against period $t+1$, so both require exactly the two copies that
`shock_copies=2` supplies. In a model with shocks, passing fewer raises a
`KeyError` naming the shock keys it could not find. This is not a knob that
trades accuracy against cost. Raising it above 2 just draws shocks the loss
never reads, and the extra next-period draw behind the all-in-one product is
generated inside the loss rather than requested here. A custom loss spanning
more periods would state its own requirement.

The remaining arguments are ordinary training knobs. Start each one small,
confirm the model is wired up correctly, and scale up from there:

- `network_width` is the hidden-layer width, and it sets how much curvature the
  policy can represent. Widen it when the learned rule looks flatter than the
  true one.
- `max_iterations` and `tolerance` govern the outer loop, which stops when
  either the parameters or the loss move by less than `tolerance` between
  iterations, or when the iteration cap is reached. Which of the two ended a run
  matters: stopping at the cap means training had not converged, so raise the
  cap before trusting the result.
- `simulation_steps` sets how far the states are simulated forward between
  resampling rounds. Larger values reach more of the state space per round, at
  proportionate cost.
- `epochs_per_iteration` and `lr` are the inner gradient-descent budget and step
  size, and behave as they do in any other network training loop.

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
