"""Tests for skagent.solver (solve_multiple_controls)."""

import numpy as np
import pytest
import torch

import skagent.bellman as bellman
import skagent.block as block
import skagent.grid as grid
import skagent.ground as ground
import skagent.models.cournot as cournot
import skagent.models.macid as macid
from skagent.solver import (
    ExactBestResponse,
    NeuralBestResponse,
    project,
    solve_multiple_controls,
    solve_symmetric_equilibrium,
)

# Deterministic test seed - change this single value to modify all seeding
# Using same seed as test_maliar.py for consistency across test suite
TEST_SEED = 10077693

# Device selection (but no global state modification at import time)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CALIBRATION = {"k": 3, "beta": 0.9}


def two_control_period(calibration=None):
    """A static block whose reward is maximized at ``c = a`` and ``d = k``.

    Two controls, one of which conditions on nothing, so the sweep has to solve
    both and neither is solved by the other's pass alone.
    """
    b = block.DBlock(
        name="two controls",
        dynamics={
            "c": block.Control(["a"], agent="agent"),
            "d": block.Control([], agent="agent"),
            "u": lambda a, c, d, k: -((a - c) ** 2) - (k - d) ** 2,
        },
        reward={"u": "agent"},
    )
    return bellman.BellmanPeriod(
        b, "beta", dict(CALIBRATION) if calibration is None else calibration
    )


def states():
    return grid.Grid.from_config({"a": {"min": -2, "max": 2, "count": 11}})


class TestSolveMultipleControls:
    def test_every_control_reaches_its_optimum(self):
        torch.manual_seed(TEST_SEED)
        givens = states()

        rules = solve_multiple_controls(
            ["c", "d", "c"], two_control_period(), givens, epochs=100
        )

        a = givens["a"].flatten()
        c = rules["c"](a).detach().cpu().numpy().flatten()
        d = rules["d"]().detach().cpu().numpy().flatten()

        assert np.max(np.abs(c - a.cpu().numpy())) < 0.05
        assert d == pytest.approx(CALIBRATION["k"], abs=0.05)


class TestAgentAttribution:
    """Each network maximizes the payoff of its own control's agent."""

    def test_the_prisoners_dilemma_reaches_mutual_defection(self):
        """A two-agent game, solved by nets that each serve their own player."""
        torch.manual_seed(TEST_SEED)
        period = bellman.BellmanPeriod(macid.prisoners_dilemma_block, None, {})
        givens = grid.Grid.from_config({"z": {"min": 0.0, "max": 1.0, "count": 32}})

        rules = solve_multiple_controls(
            ["D1", "D2", "D1", "D2"], period, givens, epochs=150
        )

        actions = [
            float(rules[sym]().detach().cpu().numpy().mean()) for sym in ("D1", "D2")
        ]
        # Defection is dominant, so the equilibrium is the upper corner. Trained
        # against the summed reward instead, both players would cooperate; trained
        # against the first reward symbol, the second player would serve the
        # first and cooperate alone.
        assert actions == pytest.approx([1.0, 1.0], abs=0.02)

    def test_an_unattributed_control_among_several_owners_raises(self):
        """Better to refuse than to train a net on someone else's objective."""
        blk = block.DBlock(
            name="unattributed",
            dynamics={
                "a1": block.Control([]),
                "a2": block.Control([], agent="p2"),
                "u1": lambda a1: -a1,
                "u2": lambda a2: -a2,
            },
            reward={"u1": "p1", "u2": "p2"},
        )
        period = bellman.BellmanPeriod(blk, None, {})
        givens = grid.Grid.from_config({"z": {"min": 0.0, "max": 1.0, "count": 4}})

        with pytest.raises(ValueError, match="no agent attribution"):
            solve_multiple_controls(["a1"], period, givens, epochs=1)


class TestTheCalibrationArgument:
    """The period already carries a calibration; a second copy may disagree.

    Nothing downstream reconciles the two, so a caller who passes both can hand
    a period one calibration and evaluate its losses at another.
    """

    def test_a_disagreeing_calibration_raises(self):
        with pytest.warns(DeprecationWarning), pytest.raises(ValueError, match="'k'"):
            solve_multiple_controls(
                ["c"], two_control_period(), states(), {"k": 4, "beta": 0.9}, epochs=1
            )

    def test_passing_the_calibration_is_deprecated(self):
        torch.manual_seed(TEST_SEED)

        with pytest.warns(DeprecationWarning, match="bellman_period"):
            solve_multiple_controls(
                ["c"], two_control_period(), states(), dict(CALIBRATION), epochs=1
            )


# --- Cournot: projecting a population, and iterating to its equilibrium ----

COST = 4.0


def cournot_ground(size=3):
    return ground.GroundedBlock(
        cournot.cournot_block, cournot.collusion_calibration(size=size)
    )


def cournot_panel(count=128, low=COST, high=COST):
    """A panel carrying both sides' shocks, which the loss needs in full."""
    return grid.Grid.from_config(
        {
            "c_actor": {"min": low, "max": high, "count": count},
            "c_other": {"min": low, "max": high, "count": count},
        }
    )


class TestTheProjectionIsDerivedFromTheModel:
    """Nothing about Cournot is written into the transform."""

    def test_it_splits_the_class_and_joins_it_back(self):
        projected = project(cournot_ground()).block
        # Each per-instance equation copied per side, one synthesized join, and
        # the aggregating equations left as the author wrote them.
        assert list(projected.get_dynamics()) == [
            "q_actor",
            "q_other",
            "q",
            "Q",
            "P",
            "u_actor",
            "u_other",
        ]
        assert list(projected.get_shocks()) == ["c_actor", "c_other"]
        # The two sides own their payoffs separately, which is what lets a
        # solver maximize one instance's rather than the class's total.
        assert projected.reward == {"u_actor": "firm_actor", "u_other": "firm_other"}
        assert projected.deciding_agent("q_actor") == "firm_actor"

    @pytest.mark.parametrize(
        "own,rivals,payoff",
        [(4.5, 4.5, 6.75), (3.0, 3.0, 9.0), (6.0, 3.0, 12.0)],
        ids=["cournot-nash", "joint-monopoly", "one-defects"],
    )
    def test_the_projected_payoffs_are_the_published_ones(self, own, rivals, payoff):
        # cournot.PROFILES is hand-derived and supplied, so it is an oracle for
        # the projection rather than a restatement of it.
        projected = project(cournot_ground()).block
        values = projected.transition(
            {**cournot.collusion_calibration(), "c_actor": COST, "c_other": COST},
            {"q_actor": lambda c_actor: own, "q_other": lambda c_other: rivals},
        )
        assert float(
            np.atleast_1d(projected.calc_reward(values, agent="firm_actor")["u_actor"])[
                0
            ]
        ) == pytest.approx(payoff, abs=1e-6)


class TestTheAggregateIsPerSampleNotPerPanel:
    """An aggregating equation is written against the entity axis alone."""

    def test_a_batched_solve_does_not_reduce_the_panel_too(self):
        # The author wrote `Q = q.mean()` with no axis, because the simulator
        # guarantees the equation sees instances only. A batched method has a
        # sample axis as well, and reducing it too would return one aggregate
        # for the whole panel -- right whenever every sample happens to agree,
        # and wrong otherwise. So the panel here is deliberately not degenerate.
        projected = project(cournot_ground(size=3)).block
        quantities = torch.tensor([5.0, 1.0, 9.0])
        rivals = torch.tensor([3.0, 3.0, 3.0])
        values = projected.transition(
            {
                **cournot.collusion_calibration(),
                "c_actor": quantities,
                "c_other": rivals,
            },
            {"q_actor": lambda c_actor: quantities, "q_other": lambda c_other: rivals},
        )
        assert values["Q"].detach().cpu().numpy() == pytest.approx(
            [(5 + 3 + 3) / 3, (1 + 3 + 3) / 3, (9 + 3 + 3) / 3]
        )


class TestTheProjectionRefusesWhatItCannotSplit:
    def test_a_block_with_no_entity_raises(self):
        with pytest.raises(ValueError, match="exactly one entity class"):
            project(ground.GroundedBlock(macid.prisoners_dilemma_block, {}))

    def test_a_calibration_that_does_not_size_the_class_raises(self):
        with pytest.raises(ValueError, match="no size for entity class"):
            project(ground.GroundedBlock(cournot.cournot_block, {"A": 10.0, "b": 1.0}))

    def test_a_population_of_one_raises(self):
        # A monopolist has no others to be projected away from.
        with pytest.raises(ValueError, match="there are no others"):
            project(cournot_ground(size=1))


def neural_method(projected, epochs=300):
    torch.manual_seed(TEST_SEED)
    return NeuralBestResponse(projected, cournot_panel(), epochs=epochs)


def exact_method(projected):
    return ExactBestResponse(
        projected,
        {"c_actor": np.array([COST])},
        scope={**projected.calibration, "c_other": COST},
    )


def solved_quantity(rule):
    # The two methods' rules do not accept the same input type -- a policy net
    # wants a tensor, the backup's interpolant an array -- which is one more
    # place the method axis is not yet uniform.
    try:
        found = rule(np.array([COST]))
    except TypeError:
        found = rule(torch.full((8,), COST))
    if isinstance(found, torch.Tensor):
        return float(found.detach().cpu().numpy().mean())
    return float(np.atleast_1d(found).ravel()[0])


class TestEitherMethodReachesCournotNash:
    """The schedule takes the method's word for the solve and the distance, so
    swapping the method must not move the answer."""

    @pytest.mark.parametrize(
        "build", [neural_method, exact_method], ids=["neural", "exact"]
    )
    @pytest.mark.parametrize("size", [2, 3, 4])
    def test_it_converges_to_the_analytic_nash_quantity(self, build, size):
        projected = project(cournot_ground(size))
        rule, info = solve_symmetric_equilibrium(
            build(projected), damping=2.0 / (size + 1), max_iterations=12
        )
        assert info["converged"]
        assert solved_quantity(rule) == pytest.approx(
            cournot.nash_quantity(size=size), abs=0.02
        )

    @pytest.mark.parametrize(
        "build", [neural_method, exact_method], ids=["neural", "exact"]
    )
    def test_undamped_at_four_firms_reports_failure_rather_than_a_number(self, build):
        # The best-response slope is -(N-1)/2, so at four firms the undamped
        # iteration diverges. The residual is on the RULE, so this comes back as
        # a refusal to claim convergence rather than as whatever the last
        # iterate happened to be.
        projected = project(cournot_ground(4))
        _, info = solve_symmetric_equilibrium(
            build(projected), damping=1.0, max_iterations=8
        )
        assert not info["converged"]
        assert info["distances"] == sorted(info["distances"])


class TestTheScheduleRefusesAnUnprojectedProblem:
    def test_a_block_with_no_solved_instance_raises(self):
        period = ground.GroundedBlock(macid.prisoners_dilemma_block, {})
        method = ExactBestResponse(period, {})
        with pytest.raises(ValueError, match="one control named for the solved"):
            solve_symmetric_equilibrium(method)
