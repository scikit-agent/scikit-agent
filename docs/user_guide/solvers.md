# Solvers Guide

An algorithm solves _one_ decision: given everything else about a model, it
finds the action, or the rule, that maximizes some payoff. Most models ask for
more than that. They have several decisions, or several agents, or a whole
population of agents whose choices depend on one another. Someone then has to
say which decision is solved next, which rules the other decisions are held at
while it is solved, and when the process should stop.

That is the subject of this page. The {doc}`algorithms` guide covers the
numerical methods themselves; this page covers how a problem with more than one
decision is posed to them.

## Two pieces: a method and a schedule

A **method** solves a single decision, given a supplied profile of rules for
everyone else. A **schedule** decides which decision is solved next, and it
repeats that step until the model is solved.

The two are separate objects, and they pair freely, because any schedule accepts
any method. That separation is the point of the design: swapping a policy
network for an exact backup should change how accurately a decision is solved,
and it should not change which decision is solved when.

Each method carries whatever _its own algorithm_ needs, so the schedules do not
have to carry it:

| Method                                                       | What it uses              | What it needs to be built with    |
| ------------------------------------------------------------ | ------------------------- | --------------------------------- |
| {py:class}`~skagent.solver.NeuralBestResponse`               | a policy network          | a training panel, epochs          |
| {py:class}`~skagent.solver.ExactBestResponse`                | an exact backup on a grid | a state grid, a continuation      |
| {py:class}`~skagent.algos.tabular.TabularBestResponseSolver` | a tabulated payoff table  | candidate actions, a sample count |

Every method offers the same three operations. The first,
`best_response(decision, policies)`, solves one decision against the rules
supplied for the others. The second, `rule_distance(new, old, iset)`, measures
how far two rules differ. The third, `initial_policies()`, returns the profile
to start from, before anything has been solved.

## Choosing a schedule

### You know the order: `solve_in_order`

{py:func}`~skagent.solver.solve_in_order` solves the decisions you name, in the
order you name them. Repeating a symbol schedules a further pass over that
decision, once its neighbours have moved.

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

The return value maps each control symbol to its decision rule, which is ready
to pass to `reward_function` or to a simulation.

Two points are worth knowing about this schedule. First, a decision you leave
out of the order is returned at its _starting_ rule, which means that it has not
been solved, and the returned profile does not distinguish a solved rule from an
unsolved one. Second, the sweep stops because your list of decisions ran out,
and not because anything converged, so a repeated symbol buys a fixed number of
refinement passes rather than a fixed point.

### The model knows the order: `solve_in_relevance_order`

Which decision must be solved before which is usually a property of the model
rather than a choice the user makes. A decision that strategically relies on
another cannot be solved until that other decision has been solved. The block's
relevance graph records these dependencies (see {doc}`../api/analysis`), and
{py:func}`~skagent.solver.solve_in_relevance_order` reads the order off that
graph:

```python
from skagent.algos.tabular import TabularBestResponseSolver
from skagent.solver import solve_in_relevance_order

method = TabularBestResponseSolver(GroundedBlock(b, calibration))
decision_rules = solve_in_relevance_order(method)
```

One pass per decision suffices here, because everything a decision relies on is
already settled by the time its turn comes. If the graph contains a cycle, which
means that two or more decisions rely on each other, then no such order exists,
and this schedule raises an error rather than choosing an order arbitrarily.
That refusal is the honest answer, because a cycle is a simultaneous-move
equilibrium problem, and it calls for the next schedule.

### The decisions depend on each other: `solve_symmetric_equilibrium`

When every agent in a population responds to what the other agents do, no such
order exists. What is wanted instead is a _fixed point_, which is a rule that is
its own best response. {py:func}`~skagent.solver.solve_symmetric_equilibrium`
iterates toward one: it solves the decision, substitutes the answer back in as
everyone else's rule, and repeats until the rule stops moving.

This schedule first needs a **projection**, which the next section describes.

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

**Damping is not optional here, and it is not a speed control.** Each round
moves the rule only a fraction `L` of the way toward the best response:

```
next = (1 - L) * current + L * best_response(current)
```

Undamped iteration converges only when the best response is a contraction. In
Cournot competition the best response is not a contraction once there are enough
firms. With three firms the iterates oscillate between two quantities forever,
and with four firms they run out to the bounds declared on the control and then
bounce between them. Neither failure raises an error. A damped iteration reaches
the equilibrium instead, and damping cannot change the answer, because a rule
that equals its own damped update is also a rule that equals its own best
response. Damping therefore changes only how the iteration gets there.

Because the schedule tests whether the _rule_ has stopped moving rather than
counting iterations, a run that fails to settle reports `converged: False`
instead of handing back its last iterate.

## Projecting a population

A model with an entity class describes many agents at once, such as several
firms or many households. A method solves one decision, so something has to turn
the question "what should all the firms do?" into the question "what should
_this_ firm do, given what the others do?"

{py:func}`~skagent.solver.project` performs that translation. It splits the
entity class into the instance being solved and the rest of the class, and it
returns an ordinary block with two controls, `<control>_actor` and
`<control>_other`.

The one thing worth understanding about the projection is that your aggregating
equations are **copied unchanged**. If your model says `Q = q.mean()`, the
projection does not rewrite that equation into a formula about one firm and
`N - 1` other firms. It reassembles the population and lets your own equation
run on it. A mean, a sum, a maximum, or any other reduction you wrote therefore
projects without the library needing to know which one you used, and the solved
firm's own share of the aggregate is present by construction rather than being
something the projection computes.

The projection has two current limits. The other instances are held at a single
shared rule, so the equilibrium it finds is a _symmetric_ one, and a game whose
only equilibria are asymmetric cannot be expressed in this shape. A masked
reduction, such as a mean taken over only the instances that meet some
condition, does not currently work with the neural method, although it does work
with the others.

---

_For runnable versions, see the
{doc}`Algorithms examples gallery </auto_examples/algorithms/index>`._
