# Solvers Guide

An algorithm solves _one_ decision: given everything else about a model, it
finds the action, or the rule, that maximizes some payoff. Most models ask for
more than that. They have several decisions, or several agents, or a population
of agents whose choices depend on each other — and then someone has to say which
decision is solved next, what the others are held at while it is, and when to
stop.

That is what this page is about. {doc}`algorithms` covers the numerical
technologies; this covers how a problem is posed to them.

## Two pieces: a method and a schedule

A **method** solves one decision against a supplied profile of everyone else's
rules. A **schedule** decides which decision that is, and repeats.

The two are separate objects, and they pair freely: any schedule accepts any
method. That is the point of the split — swapping a policy network for an exact
backup should change how accurately a decision is solved, not which decision
gets solved when.

A method carries whatever _its own algorithm_ needs, which is why the schedules
do not have to:

| Method                                                       | What it uses              | What it needs to be built with    |
| ------------------------------------------------------------ | ------------------------- | --------------------------------- |
| {py:class}`~skagent.solver.NeuralBestResponse`               | a policy network          | a training panel, epochs          |
| {py:class}`~skagent.solver.ExactBestResponse`                | an exact backup on a grid | a state grid, a continuation      |
| {py:class}`~skagent.algos.tabular.TabularBestResponseSolver` | a tabulated payoff table  | candidate actions, a sample count |

Every method offers the same three operations:
`best_response(decision, policies)`, `rule_distance(new, old, iset)`, and
`initial_policies()` — the profile to start from, before anything has been
solved.

## Choosing a schedule

### You know the order: `solve_in_order`

{py:func}`~skagent.solver.solve_in_order` solves the decisions you name, in the
order you name them. Repeat a symbol to refine it after its neighbours have
moved.

```python
import skagent.block as block
import skagent.grid as grid
from skagent.ground import GroundedBlock
from skagent.solver import NeuralBestResponse, solve_in_order

calibration = {"k": 3, "beta": 0.9}

b = block.DBlock(
    name="two controls",
    dynamics={
        "c": block.Control(["a"], agent="agent"),
        "d": block.Control([], agent="agent"),  # empty information set
        "u": lambda a, c, d, k: -((a - c) ** 2) - (k - d) ** 2,
    },
    reward={"u": "agent"},
)
states = grid.Grid.from_config({"a": {"min": -2, "max": 2, "count": 11}})

method = NeuralBestResponse(GroundedBlock(b, calibration), states, epochs=200)
decision_rules = solve_in_order(method, ["c", "d", "c"])
# optimal: c = a and d = 3, so the reward u is approximately 0
```

The return value maps each control symbol to its decision rule, ready for
`reward_function` or for simulation.

Two things to know. A decision you leave out of the order comes back at its
_starting_ rule and has not been solved — the returned profile does not
distinguish the two. And the sweep stops because your list ran out, not because
anything converged: a repeated symbol buys a fixed number of refinement passes,
not a fixed point.

### The model knows the order: `solve_in_relevance_order`

Which decision must be solved before which is usually a property of the model
rather than a choice. A decision that strategically relies on another cannot be
solved until that other one is. The block's relevance graph records this (see
{doc}`../api/analysis`), and {py:func}`~skagent.solver.solve_in_relevance_order`
reads the order off it:

```python
from skagent.algos.tabular import TabularBestResponseSolver
from skagent.solver import solve_in_relevance_order

method = TabularBestResponseSolver(GroundedBlock(b, calibration))
decision_rules = solve_in_relevance_order(method)
```

One pass per decision suffices here, because everything a decision relies on is
settled by the time its turn comes. If the graph has a cycle — decisions that
rely on each other — there is no such order, and this schedule refuses rather
than picking one. That refusal is the honest answer: a cycle is a
simultaneous-move equilibrium problem, and needs the next schedule.

### The decisions depend on each other: `solve_symmetric_equilibrium`

When a population of agents each responds to what the others do, no order
exists. What you want is a _fixed point_: a rule that is its own best response.
{py:func}`~skagent.solver.solve_symmetric_equilibrium` iterates toward one —
solve, substitute the answer back in as everyone else's rule, repeat — and stops
when the rule stops moving.

This needs a **projection** first, described below.

```python
import numpy as np
import skagent.models.cournot as cournot
from skagent.solver import ExactBestResponse, project, solve_symmetric_equilibrium

market = GroundedBlock(cournot.cournot_block, cournot.collusion_calibration(size=3))
projected = project(market)

method = ExactBestResponse(
    projected,
    {"c_actor": np.array([4.0])},
    scope={**projected.calibration, "c_other": 4.0},
)
rule, info = solve_symmetric_equilibrium(method, damping=0.5)

info["converged"]  # True; False would mean it did not settle, not a fallback answer
rule(np.array([4.0]))  # 4.5, the Cournot-Nash quantity for three firms
```

**Damping is not optional here, and it is not a speed knob.** Each round moves
only a fraction `L` of the way toward the best response:

```
next = (1 - L) * current + L * best_response(current)
```

Undamped iteration only converges when the best response is a contraction. In
Cournot competition it is not, once there are enough firms: with three the
iterates oscillate between two quantities forever, and with four they run to the
bounds you declared on the control and bounce between those. Neither raises. A
damped iteration reaches the equilibrium instead, and damping cannot move the
answer — a rule equal to its own damped update is a rule equal to its own best
response, so it changes only how you get there.

Because the schedule tests the _rule_ rather than counting iterations, a run
that fails to settle reports `converged: False` instead of handing back its last
iterate.

## Projecting a population

A model with an entity class describes many agents at once — several firms, many
households. A method solves one decision, so something has to turn "what should
all the firms do" into "what should _this_ firm do, given the others".

{py:func}`~skagent.solver.project` does that. It splits the entity class into
the instance being solved and the rest, and gives you back an ordinary block
with two controls: `<control>_actor` and `<control>_other`.

The one thing worth understanding about it: your aggregating equations are
**copied unchanged**. If your model says `Q = q.mean()`, the projection does not
rewrite that into a formula about one firm and `N - 1` others. It reassembles
the population and lets your own equation run on it. So a mean, a sum, a maximum
or any other reduction you wrote projects without the library needing to know
which one you used — and the solved firm's own share of the aggregate is there
by construction, rather than being something the projection computes.

Two current limits. The others are held at a single shared rule, so the
equilibrium found is a _symmetric_ one; a game whose only equilibria are
asymmetric cannot be expressed this way. And a masked reduction — a mean over
only the instances meeting some condition — does not currently work with the
neural method, though it is fine with the others.

---

_For runnable versions, see the
{doc}`Algorithms examples gallery </auto_examples/algorithms/index>`._
