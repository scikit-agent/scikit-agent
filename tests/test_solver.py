"""Tests for skagent.solver: projections, schedules and methods."""

import numpy as np
import pytest
import torch

import skagent.bellman as bellman
import skagent.block as block
import skagent.algos.vfi as vfi
import skagent.grid as grid
import skagent.ground as ground
import skagent.models.aiyagari as aiyagari
import skagent.models.benchmarks as benchmarks
import skagent.models.cournot as cournot
import skagent.models.macid as macid
import skagent.models.privacy as privacy
import skagent.solver as skagent_solver
from skagent.solver import (
    _blocks_by_class,
    _joining_equation,
    ExactBestResponse,
    ExactStationaryBestResponse,
    NeuralBestResponse,
    StationaryBestResponse,
    project_mean_field,
    project_nash,
    solve_in_order,
    solve_in_relevance_order,
    solve_mean_field,
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


def two_control_ground(calibration=None):
    return ground.GroundedBlock(
        two_control_period().block,
        dict(CALIBRATION) if calibration is None else calibration,
    )


class TestSolvingInAGivenOrder:
    """The schedule takes its order from the caller; any method supplies the solves."""

    def test_a_policy_network_reaches_every_control_optimum(self):
        torch.manual_seed(TEST_SEED)
        givens = states()
        method = NeuralBestResponse(two_control_ground(), givens, epochs=200)

        rules = solve_in_order(method, ["c", "d", "c"])

        a = givens["a"].flatten()
        c = rules["c"](a).detach().cpu().numpy().flatten()
        assert np.max(np.abs(c - a.cpu().numpy())) < 0.05
        assert rules["d"]().detach().cpu().numpy().flatten() == pytest.approx(
            CALIBRATION["k"], abs=0.05
        )

    def test_an_exact_backup_reaches_the_same_optima(self):
        # Same schedule, same order, a different method. The backup needs bounds
        # to search between, which is its own configuration and not the model's.
        bounded = block.DBlock(
            name="two controls",
            dynamics={
                "c": block.Control(
                    ["a"], lower_bound=-2.0, upper_bound=2.0, agent="agent"
                ),
                "d": block.Control([], lower_bound=0.0, upper_bound=5.0, agent="agent"),
                "u": lambda a, c, d, k: -((a - c) ** 2) - (k - d) ** 2,
            },
            reward={"u": "agent"},
        )
        method = ExactBestResponse(
            ground.GroundedBlock(bounded, dict(CALIBRATION)),
            {"a": np.linspace(-2, 2, 9)},
            scope=dict(CALIBRATION),
        )

        rules = solve_in_order(method, ["c", "d", "c"])

        assert float(np.atleast_1d(rules["c"](np.array([1.0]))).ravel()[0]) == (
            pytest.approx(1.0, abs=1e-3)
        )
        assert float(np.atleast_1d(rules["d"]()).ravel()[0]) == pytest.approx(
            CALIBRATION["k"], abs=1e-3
        )

    def test_a_decision_left_out_of_the_order_is_not_solved(self):
        # It comes back at its STARTING rule, a constant and visibly
        # provisional. An untrained policy network would be callable, numeric,
        # and indistinguishable from a solved rule.
        method = ExactBestResponse(
            two_control_ground(), {"a": np.linspace(-2, 2, 9)}, scope=dict(CALIBRATION)
        )
        rules = solve_in_order(method, ["c"])
        assert float(np.atleast_1d(rules["d"]()).ravel()[0]) == 0.0


class TestAgentAttribution:
    """Each solve maximizes the payoff of its own control's agent."""

    def test_the_prisoners_dilemma_reaches_mutual_defection(self):
        """A two-agent game, solved by nets that each serve their own player."""
        torch.manual_seed(TEST_SEED)
        method = NeuralBestResponse(
            ground.GroundedBlock(macid.prisoners_dilemma_block, {}),
            grid.Grid.from_config({"z": {"min": 0.0, "max": 1.0, "count": 32}}),
            epochs=150,
        )

        rules = solve_in_order(method, ["D1", "D2", "D1", "D2"])

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
        method = NeuralBestResponse(
            ground.GroundedBlock(blk, {}),
            grid.Grid.from_config({"z": {"min": 0.0, "max": 1.0, "count": 4}}),
            epochs=1,
        )
        with pytest.raises(ValueError, match="no agent attribution"):
            solve_in_order(method, ["a1"])


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
        projected = project_nash(cournot_ground()).block
        # Each per-instance equation copied per side, one synthesized join, and
        # the aggregating equations left as the author wrote them. Both sides
        # run before what rejoins them, and the payoffs after it.
        assert list(projected.get_dynamics()) == [
            "q_other",
            "q_actor",
            "q",
            "Q",
            "P",
            "u_other",
            "u_actor",
        ]
        assert sorted(projected.get_shocks()) == ["c_actor", "c_other"]
        # The two sides own their payoffs separately, which is what lets a
        # solver maximize one instance's rather than the class's total.
        assert projected.reward == {"u_actor": "firm_actor", "u_other": "firm_other"}
        assert projected.deciding_agent("q_actor") == "firm_actor"

    def test_it_preserves_a_controls_action_space(self):
        population = block.DBlock(
            dynamics={
                "d": block.Control([], agent="player", action_space={0, 2}),
                "u": lambda d: d,
            },
            reward={"u": "player"},
            entity=block.Entity("player"),
        )
        projected = project_nash(ground.GroundedBlock(population, {"player": 2})).block

        assert projected.get_control("d_actor").action_space == frozenset({0.0, 2.0})
        assert projected.get_control("d_other").action_space == frozenset({0.0, 2.0})

    def test_the_others_stay_a_population_and_the_solved_instance_does_not(self):
        # The rest of the class is still several, so it keeps a class of its
        # own, sized one short of the original. The solved instance is ONE
        # instance rather than a population of one, so its symbols carry no
        # class, and that asymmetry is what the two sides are for. A rejoined
        # symbol is the whole class again, at the size the author gave it.
        projected = project_nash(cournot_ground(size=3))

        assert sorted(projected.block.entities()) == ["firm", "firm_other"]
        assert projected.calibration["firm_other"] == 2
        assert projected.calibration["firm"] == 3

        signatures = projected.block.signatures()
        assert signatures["q_other"] == frozenset({"firm_other"})
        assert signatures["c_other"] == frozenset({"firm_other"})
        assert signatures["q_actor"] == frozenset()
        assert signatures["q"] == frozenset({"firm"})

    def test_the_authors_own_aggregation_survives_the_split(self):
        # The population block reports one crossing: the market reading the
        # firms' quantities. The projection has to still report it, over the
        # same class, or the model it hands a solver is one with no aggregation
        # in it at all.
        population = cournot_ground(size=3)
        projected = project_nash(population)

        assert population.block.crossings()["Q"][0][:2] == ("q", frozenset({"firm"}))
        assert projected.block.crossings()["Q"][0][:2] == ("q", frozenset({"firm"}))
        # And the join is a crossing of its own: the rejoined symbol reads the
        # rivals' out of their class.
        assert projected.block.crossings()["q"][0][:2] == (
            "q_other",
            frozenset({"firm_other"}),
        )

    @pytest.mark.parametrize(
        "own,rivals,payoff",
        [(4.5, 4.5, 6.75), (3.0, 3.0, 9.0), (6.0, 3.0, 12.0)],
        ids=["cournot-nash", "joint-monopoly", "one-defects"],
    )
    def test_the_projected_payoffs_are_the_published_ones(self, own, rivals, payoff):
        # cournot.PROFILES is hand-derived and supplied, so it is an oracle for
        # the projection rather than a restatement of it.
        projected = project_nash(cournot_ground()).block
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
        projected = project_nash(cournot_ground(size=3)).block
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


class TestADecisionOverTheWholeClassSurvivesTheProjection:
    """A projection splits the class; a decision that reads all of it does not."""

    def test_it_is_copied_against_the_rejoined_symbols(self):
        projected = project_nash(
            ground.GroundedBlock(privacy.local_block, privacy.calibration(1.0, size=5))
        ).block
        controls = projected.get_controls()

        # The analyst's estimate reads the whole subject class, so it is one
        # decision on both sides of the split rather than two. The joins the
        # projection synthesizes are what its information set now names, so the
        # declaration is copied exactly as the author wrote it.
        assert controls["f"].iset == ["c", "d"]
        assert {"c", "d"} <= set(projected.get_dynamics())
        assert {"c_actor", "c_other"} == {
            sym for sym in controls if sym.startswith("c_")
        }


class TestTheProjectionRefusesWhatItCannotSplit:
    def test_a_block_with_no_entity_raises(self):
        with pytest.raises(ValueError, match="exactly one entity class"):
            project_nash(ground.GroundedBlock(macid.prisoners_dilemma_block, {}))

    def test_a_calibration_that_does_not_size_the_class_raises(self):
        with pytest.raises(ValueError, match="no size for entity class"):
            project_nash(
                ground.GroundedBlock(cournot.cournot_block, {"A": 10.0, "b": 1.0})
            )

    def test_a_population_of_one_raises(self):
        # A monopolist has no others to be projected away from.
        with pytest.raises(ValueError, match="there are no others"):
            project_nash(cournot_ground(size=1))


def test_the_old_name_warns_and_projects_as_the_new_one():
    with pytest.warns(DeprecationWarning, match="project_nash"):
        old = skagent_solver.project(cournot_ground())
    new = project_nash(cournot_ground())
    assert list(old.block.get_dynamics()) == list(new.block.get_dynamics())
    assert old.calibration == new.calibration


class TestTheMeanFieldProjection:
    """The price-taker's problem: the aggregate is read, not computed."""

    CAPITAL = 5.5

    def economy(self):
        return {**aiyagari.aiyagari_calibration(sigma=0.3), "DiscFac": 0.96}

    def test_it_is_the_household_problem_at_the_aggregates_prices(self):
        # Dropping the crossing and giving K a value is the hand construction:
        # the household block alone, with the prices K sets written into its
        # calibration. The two periods must compute the same things.
        alpha, delta = aiyagari.CAPITAL_SHARE, aiyagari.DEPRECIATION
        prices = {
            "R": alpha * self.CAPITAL ** (alpha - 1) - delta,
            "W": (1 - alpha) * self.CAPITAL**alpha,
        }
        projected = project_mean_field(
            ground.GroundedBlock(aiyagari.aiyagari_block, self.economy())
        )
        derived = bellman.BellmanPeriod(
            projected.block, "DiscFac", {**projected.calibration, "K": self.CAPITAL}
        )
        by_hand = bellman.BellmanPeriod(
            aiyagari.household_block, "DiscFac", {**self.economy(), **prices}
        )

        assert derived.arrival_states == by_hand.arrival_states == {"a"}
        for a, theta, c in [(0.0, 0.5, 0.3), (2.0, 1.0, 1.1), (9.0, 1.7, 2.4)]:
            got = derived.post_function({"a": a}, {"c": c}, shocks={"theta": theta})
            want = by_hand.post_function({"a": a}, {"c": c}, shocks={"theta": theta})
            for sym in ("z", "u", "a"):
                assert got[sym] == pytest.approx(want[sym])

    def test_a_block_with_no_entity_raises(self):
        with pytest.raises(ValueError, match="exactly one entity class"):
            project_mean_field(
                ground.GroundedBlock(aiyagari.household_block, self.economy())
            )

    def test_a_class_nothing_reads_out_of_raises(self):
        # The households alone, without the market: a population, but no
        # aggregate for an instance to take as given.
        households = block.RBlock(
            name="households",
            entity=block.Entity("household"),
            blocks=[aiyagari.household_block],
        )
        with pytest.raises(ValueError, match="reads nothing out of"):
            project_mean_field(ground.GroundedBlock(households, self.economy()))

    def test_a_reduction_that_is_a_decision_raises(self):
        # The privacy game's analyst decides over the whole class of reports,
        # which is a choice and not an aggregate a subject takes as given.
        with pytest.raises(ValueError, match="in a decision"):
            project_mean_field(
                ground.GroundedBlock(
                    privacy.local_block, privacy.calibration(1.0, size=10)
                )
            )


class SavingAtAFixedRate(StationaryBestResponse):
    """A stationary method that answers every problem with one savings rule.

    The rule ignores prices, so the economy settles at the capital the rule's
    own dynamics reach whatever the candidate, and the price loop's root is
    :func:`skagent.models.aiyagari.stationary_capital` with no solve to wait on.
    """

    def __init__(self, ground, rate):
        self.ground = ground
        self.rate = rate

    def best_response(self, decision, policies):
        return aiyagari.savings_rule(self.rate)

    def with_ground(self, ground):
        return SavingAtAFixedRate(ground, self.rate)


class TestThePriceLoop:
    """The schedule root-finds the aggregate against the full model's simulation."""

    RATE = aiyagari.savings_rate_for(0.04)

    def population(self, sigma=0.3):
        return ground.GroundedBlock(
            aiyagari.aiyagari_block,
            {**aiyagari.aiyagari_calibration(sigma=sigma), "DiscFac": 0.96},
        )

    def solve(self, population, method=None):
        target = aiyagari.stationary_capital(self.RATE)
        method = method or SavingAtAFixedRate(project_mean_field(population), self.RATE)
        return solve_mean_field(
            method, population, bracket=(0.5 * target, 2 * target), tol=1e-4
        )

    def test_it_finds_the_capital_the_rule_settles_at(self):
        # A finite class adds noise to the simulated aggregate, which is all
        # that separates the root from the closed form.
        _rule, info = self.solve(self.population())
        assert info["converged"]
        assert info["aggregate"] == pytest.approx(
            aiyagari.stationary_capital(self.RATE), rel=1e-2
        )

    def test_a_round_is_a_deterministic_function_of_the_candidate(self):
        # Every round simulates from the same seed, so two runs ask Brent's
        # method the same questions and get the same answers.
        _, first = self.solve(self.population())
        _, second = self.solve(self.population())
        assert first["rounds"] == second["rounds"]

    def test_a_one_period_method_is_refused(self):
        # A one-period best response consumes everything, and reporting the
        # economy it leads to as the stationary equilibrium would be wrong.
        population = self.population()
        method = ExactBestResponse(
            project_mean_field(population), {"a": np.linspace(0, 10, 5)}
        )
        with pytest.raises(TypeError, match="solves one period"):
            self.solve(population, method)

    def test_a_bracket_without_a_sign_change_is_refused(self):
        population = self.population()
        target = aiyagari.stationary_capital(self.RATE)
        with pytest.raises(ValueError, match="does not change sign"):
            solve_mean_field(
                SavingAtAFixedRate(project_mean_field(population), self.RATE),
                population,
                bracket=(1.5 * target, 2 * target),
            )


AIYAGARI_BETA = 0.96
AIYAGARI_K_RA = (
    aiyagari.CAPITAL_SHARE / (1 / AIYAGARI_BETA - 1 + aiyagari.DEPRECIATION)
) ** (1 / (1 - aiyagari.CAPITAL_SHARE))


def aiyagari_equilibrium(sigma, warm_start=True):
    """Aiyagari's price-taking equilibrium, solved by VFI on a coarse grid."""
    population = ground.GroundedBlock(
        aiyagari.aiyagari_block,
        {
            **aiyagari.aiyagari_calibration(size=1000, sigma=sigma),
            "DiscFac": AIYAGARI_BETA,
        },
    )
    method = ExactStationaryBestResponse(
        project_mean_field(population),
        {"a": 60 * np.linspace(0, 1, 30) ** 2},
        "DiscFac",
        disc_params={"theta": {"N": 5}},
        confine_to_grid=True,
    )
    return solve_mean_field(
        method,
        population,
        bracket=(AIYAGARI_K_RA, 1.1 * AIYAGARI_K_RA),
        tol=1e-3,
        warm_start=warm_start,
    )


@pytest.fixture(scope="module")
def aiyagari_with_risk():
    return aiyagari_equilibrium(0.3)


@pytest.mark.oracle
class TestAiyagarisEquilibrium:
    """The anchors of the representative agent's capital, ``K_RA``, at which
    the interest rate equals the discount rate."""

    def test_with_little_income_risk_it_is_the_representative_agents(self):
        _rule, info = aiyagari_equilibrium(0.1)
        assert info["converged"]
        assert info["aggregate"] == pytest.approx(AIYAGARI_K_RA, rel=2e-3)

    def test_income_risk_raises_it_above_the_representative_agents(
        self, aiyagari_with_risk
    ):
        # Precautionary saving: about 2% at sigma = 0.3, ten times the band of
        # the test above.
        _rule, info = aiyagari_with_risk
        assert info["converged"]
        assert info["aggregate"] > 1.01 * AIYAGARI_K_RA

    def test_a_warm_start_reaches_the_cold_starts_capital(self, aiyagari_with_risk):
        # Each round starting from the last round's value is an approximation
        # of where to start, so it may change the rounds but not the answer
        # beyond the loop's tolerance.
        _rule, warm = aiyagari_with_risk
        _rule, cold = aiyagari_equilibrium(0.3, warm_start=False)
        assert warm["aggregate"] == pytest.approx(cold["aggregate"], abs=1e-3)


class TestTheStationaryBestResponse:
    def test_under_solve_in_order_it_is_solve_bellman(self):
        # The method is solve_bellman behind the schedules' protocol, so a
        # schedule driving it on a recurring benchmark gets the same rule.
        cal = benchmarks.d4_calibration
        grid = {"a": np.linspace(0.0, 7.5, 9)}
        method = ExactStationaryBestResponse(
            ground.GroundedBlock(benchmarks.d4_block, cal), grid, "DiscFac", tol=1e-3
        )
        direct, _, _ = vfi.solve_bellman(
            bellman.BellmanPeriod(benchmarks.d4_block, "DiscFac", cal),
            grid,
            scope=cal,
            tol=1e-3,
            max_iter=500,
            search="bounded",
            policy_evaluations=20,
        )

        # D-4's bound reads the state, which a starting constant cannot, and
        # the one decision is the whole profile, so the profile starts empty.
        rules = solve_in_order(method, ["c"], policies={})

        cash = np.array([a * cal["R"] + cal["y"] for a in (0.25, 1.0, 3.0, 6.0)])
        assert rules["c"](cash) == pytest.approx(direct["c"](cash))
        assert method.value.attrs["converged"]

    def test_an_undiscounted_dynamic_problem_warns(self):
        # Without a discount factor the problem may have no fixed point for
        # the iteration to reach, so solving it silently would hide that.
        cal = benchmarks.d4_calibration
        method = ExactStationaryBestResponse(
            ground.GroundedBlock(benchmarks.d4_block, cal),
            {"a": np.linspace(0.0, 7.5, 3)},
            None,
            max_iter=2,
        )
        with pytest.warns(UserWarning, match="undiscounted"):
            with pytest.raises(RuntimeError):
                method.best_response("c", {})


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
        projected = project_nash(cournot_ground(size))
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
        projected = project_nash(cournot_ground(4))
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

    def test_a_population_asked_for_an_order_is_sent_to_the_fixed_point(self):
        # One decision that relies on itself is a component of one, so the
        # relevance schedule would otherwise solve it in a single pass against
        # a profile no instance is playing.
        panel = grid.Grid.from_config({"c": {"min": COST, "max": COST, "count": 8}})
        method = NeuralBestResponse(cournot_ground(), panel, epochs=1)
        with pytest.raises(NotImplementedError, match="entity class 'firm'"):
            solve_in_relevance_order(method)


# --- a rival whose cost is private, and what two rules are compared over ----


def bayesian_nash_rule(size, low, high, intercept=cournot.A, slope=cournot.B):
    """Cournot with private costs, where demand reads the MEAN quantity.

    Each firm knows its own cost and the cost distribution, not its rivals'
    draws, so it maximizes expected profit against the others' expected
    quantity. Averaging the first-order condition over the population gives
    ``E[q]``, and substituting it back gives one firm's rule.
    """
    expected = size * (intercept - (low + high) / 2) / (slope * (size + 1))

    def rule(cost):
        return (
            size
            * (intercept - cost - slope * (size - 1) * expected / size)
            / (2 * slope)
        )

    return rule


class TestARivalsPrivateDrawIsIntegratedRatherThanPinned:
    """A rival's cost has no single value, and the equilibrium needs none."""

    def test_the_solved_rule_is_the_bayesian_nash_one(self):
        # The rival's cost is left out of scope, so the backup integrates it
        # rather than conditioning on a realization the actor cannot see. It is
        # also in the partner rule's information set, so the schedule's residual
        # is measured over it -- which is what makes leaving it out possible.
        low, high, size = 2.0, 6.0, 3
        projected = project_nash(
            ground.GroundedBlock(
                cournot.cournot_block,
                cournot.heterogeneous_calibration(size=size, low=low, high=high),
            )
        )
        method = ExactBestResponse(
            projected,
            {"c_actor": np.linspace(low, high, 3)},
            disc_params={"c_other": {"N": 3}},
        )

        rule, info = solve_symmetric_equilibrium(
            method, damping=0.5, tolerance=1e-4, max_iterations=40
        )

        assert info["converged"]
        analytic = bayesian_nash_rule(size, low, high)
        for cost in (low, (low + high) / 2, high):
            found = float(np.atleast_1d(rule(cost)).ravel()[0])
            assert found == pytest.approx(analytic(cost), abs=1e-3)

    def test_a_symbol_that_is_nowhere_says_where_it_could_be(self):
        projected = project_nash(cournot_ground())
        method = ExactBestResponse(projected, {"c_actor": np.array([COST])})

        with pytest.raises(ValueError, match="Grid it, pin it, or declare it"):
            method.rule_distance(lambda x: x, lambda x: x, ["not_a_symbol"])


class TestBlocksAreLaidOutByEntityClass:
    """A block declares one class, so classified symbols imply a block tree."""

    def test_contiguous_runs_of_one_class_become_one_block(self):
        blocks = _blocks_by_class(
            "layout",
            [
                ("theta", None, "seller"),
                ("S", lambda theta: theta, "seller"),
                ("p", lambda S: S, None),
                ("u", lambda p: p, "seller"),
            ],
            shocks={"theta": "a declaration"},
            rewards={"u": "seller"},
        )

        assert [list(b.get_dynamics()) for b in blocks] == [["S"], ["p"], ["u"]]
        assert [b.entity.name if b.entity else None for b in blocks] == [
            "seller",
            None,
            "seller",
        ]
        # A shock is declared by the block that holds its symbol, and a reward
        # travels with the symbol that carries it.
        assert blocks[0].get_shocks() == {"theta": "a declaration"}
        assert blocks[2].reward == {"u": "seller"}

    def test_the_order_survives_the_split(self):
        # The point of keeping runs contiguous: an equation that has to run
        # before another still does, whichever class each belongs to.
        blocks = _blocks_by_class(
            "layout",
            [
                ("first", lambda: 1, None),
                ("second", lambda first: first, "plate"),
                ("third", lambda second: second, None),
            ],
            shocks={},
            rewards={},
        )

        assert [sym for b in blocks for sym in b.get_dynamics()] == [
            "first",
            "second",
            "third",
        ]


class TestTheJoinPutsTheEntityAxisLast:
    """The rejoined symbol is ``(samples..., instances)``, on either backend.

    The two arguments arrive as bare arrays, so the shapes are all the join has
    to tell a sample axis from an entity axis. Reading one as the other is not
    an error that surfaces: it returns a population of the wrong size and every
    number downstream stays plausible.
    """

    RIVALS = 2

    def join(self):
        return _joining_equation("q_actor", "q_other", self.RIVALS)

    @pytest.mark.parametrize("box", [np.asarray, torch.tensor], ids=["numpy", "torch"])
    def test_a_constant_rival_fills_the_class(self, box):
        joined = self.join()(box(2.0), box(3.0))

        assert tuple(joined.shape) == (1 + self.RIVALS,)
        assert np.asarray(joined) == pytest.approx([2.0, 3.0, 3.0])

    @pytest.mark.parametrize("box", [np.asarray, torch.tensor], ids=["numpy", "torch"])
    def test_rivals_that_differ_stay_different(self, box):
        # One market whose two rivals did different things -- not two markets.
        joined = self.join()(box(2.0), box([3.0, 4.0]))

        assert tuple(joined.shape) == (1 + self.RIVALS,)
        assert np.asarray(joined) == pytest.approx([2.0, 3.0, 4.0])

    @pytest.mark.parametrize("box", [np.asarray, torch.tensor], ids=["numpy", "torch"])
    def test_a_batch_is_one_market_per_sample(self, box):
        joined = self.join()(box([2.0, 5.0, 9.0]), box(3.0))

        assert tuple(joined.shape) == (3, 1 + self.RIVALS)
        assert np.asarray(joined) == pytest.approx(
            np.array([[2.0, 3.0, 3.0], [5.0, 3.0, 3.0], [9.0, 3.0, 3.0]])
        )

    @pytest.mark.parametrize("box", [np.asarray, torch.tensor], ids=["numpy", "torch"])
    def test_a_batch_of_markets_whose_rivals_differ(self, box):
        joined = self.join()(box([2.0, 5.0]), box([[3.0, 4.0], [6.0, 7.0]]))

        assert tuple(joined.shape) == (2, 1 + self.RIVALS)
        assert np.asarray(joined) == pytest.approx(
            np.array([[2.0, 3.0, 4.0], [5.0, 6.0, 7.0]])
        )

    @pytest.mark.parametrize("box", [np.asarray, torch.tensor], ids=["numpy", "torch"])
    def test_as_many_samples_as_rivals_is_refused(self, box):
        # The one shape that reads both ways. Answering it either way would be
        # a different model, and nothing in the array says which.
        with pytest.raises(ValueError, match="explicit entity axis"):
            self.join()(box([2.0, 5.0]), box([3.0, 4.0]))
