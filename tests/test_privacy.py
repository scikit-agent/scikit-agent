"""The differential-privacy games: the two models, their oracle, and the design."""

import numpy as np
import pytest

import skagent.models.privacy as privacy
from skagent.ground import GroundedBlock
from skagent.simulation.monte_carlo import Simulator
from skagent.solver import ExactBestResponse, project

SIZE = 200
"""Subjects per run. Small enough to simulate, large enough that the count's
distribution is what the closed forms say it is."""

SAMPLES = 400
"""Independent plays per run, which is what an expected error is averaged over."""

DELTA = 3.0
"""Individual variation, and the widest of the paper's three."""


def play(block, calibration, rules, samples=SAMPLES, seed=0):
    """One period of *block*, *samples* times over."""
    sim = Simulator(
        calibration, block, rules, {}, sample_count=samples, T_sim=1, seed=seed
    )
    sim.initialize_sim()
    return {sym: np.asarray(path) for sym, path in sim.simulate().items()}


def supplied(sigma, calibration):
    """Both closed-form rules: the subjects' threshold and the analyst's mean."""
    return {
        "c": privacy.sharing_rule(sigma),
        "f": privacy.analyst_rule(calibration["a_mean"]),
    }


def squared_error(history, estimate="f"):
    """The analyst's realized mean squared error over the plays simulated."""
    return float(np.mean((history[estimate].ravel() - history["a"].ravel()) ** 2))


class TestTheTwoGamesAreDeclaredAsThePaperWritesThem:
    """One model, two trust assumptions, and one equation's worth of difference."""

    def test_the_models_differ_only_in_where_the_noise_enters(self):
        local = privacy.local_block.signatures()
        central = privacy.central_block.signatures()

        # Locally every subject privatizes its own report, so the noise is an
        # attribute of the subject; centrally the analyst adds it once, to the
        # estimate, so it is axis-free and the published estimate is a symbol of
        # its own.
        assert local["gamma"] == frozenset({"subject"})
        assert central["gamma"] == frozenset()
        assert "f_dp" not in local and central["f_dp"] == frozenset()

        # And nothing else moved: the same data, decision, payoff and estimate.
        shared = {"a", "zeta", "p", "b", "c", "u", "d", "f", "g"}
        assert shared <= set(local) and shared <= set(central)

    def test_the_analyst_reads_the_class_and_a_subject_reads_only_itself(self):
        subject = privacy.local_block.get_control("c")
        analyst = privacy.local_block.get_control("f")
        signatures = privacy.local_block.signatures()

        # The subject decides on its own data and its own concern, both of which
        # it alone has. The analyst decides on the whole class, which is why its
        # rule is supplied rather than learned.
        assert subject.iset == ["b", "p"]
        assert all(signatures[sym] == frozenset({"subject"}) for sym in subject.iset)
        assert analyst.iset == ["c", "d"]
        assert signatures["f"] == frozenset()

    def test_the_game_is_a_dag_that_says_which_decision_comes_first(self):
        graph = privacy.local_block.relevance_graph(privacy.calibration(1.0))

        # No subject's payoff runs through another's decision, so there is no
        # fixed point here: the estimate accounts for the sharing decision, the
        # sharing decision accounts for nothing, and the order is the graph's.
        assert set(graph.nodes()) == {"c", "f"}
        assert graph.is_acyclic()
        assert [sorted(part) for part in graph.condensation()] == [["c"], ["f"]]


class TestTheSuppliedRulesAreThePapersClosedForms:
    def test_the_threshold_rule_shares_when_the_benefit_covers_the_concern(self):
        sigma = 1.0
        calibration = privacy.calibration(sigma, size=SIZE, delta=DELTA)
        history = play(privacy.local_block, calibration, supplied(sigma, calibration))

        # Sharing is worth q and costs p / sigma, so the fraction who share is
        # the concern distribution evaluated at the benefit.
        assert history["c"].mean() == pytest.approx(
            privacy.share_probability(sigma), abs=0.01
        )

        # A subject that did not share reports nothing and is paid nothing.
        silent = history["c"] == 0.0
        assert np.all(history["d"][silent] == 0.0)
        assert np.all(history["u"][silent] == 0.0)

    def test_the_estimate_averages_the_reports_it_received(self):
        sigma = 1.0
        calibration = privacy.calibration(sigma, size=SIZE, delta=DELTA)
        history = play(privacy.local_block, calibration, supplied(sigma, calibration))

        # Who shares turns on privacy concern, which is drawn independently of
        # the data, so the reports that arrive are a random sample of it and
        # their mean is an unbiased estimate of the population mean.
        bias = float(np.mean(history["f"].ravel() - history["a"].ravel()))
        assert bias == pytest.approx(0.0, abs=0.02)

    def test_an_empty_database_is_estimated_by_the_prior_mean(self):
        # The paper's branch on nobody sharing, and here it is not a branch: the
        # prior enters as one pseudo-observation, so it is the whole estimate
        # exactly when the weights sum to zero.
        nobody = np.zeros(5)
        assert privacy.estimate(nobody, nobody, 0.3) == pytest.approx(0.3)

        # And one report is enough to leave the prior behind.
        one = np.array([0.0, 1.0, 0.0, 0.0, 0.0])
        assert privacy.estimate(one, one * 2.5, 0.3) == pytest.approx(2.5)


class TestTheAnalystsErrorMatchesItsClosedForm:
    """The paper's equation (3) and its central analogue, against simulation."""

    @pytest.mark.parametrize("sigma", [0.5, 1.5])
    def test_the_local_error_is_what_equation_three_says(self, sigma):
        calibration = privacy.calibration(sigma, size=SIZE, delta=DELTA)
        history = play(privacy.local_block, calibration, supplied(sigma, calibration))

        # Locally the mechanism's noise is averaged down with everything else,
        # so it enters the error divided by the number of reports.
        assert squared_error(history) == pytest.approx(
            privacy.local_error(sigma, size=SIZE, delta=DELTA), rel=0.2
        )

    def test_the_central_error_carries_the_noise_undivided(self):
        sigma = 0.05
        calibration = privacy.calibration(sigma, size=SIZE, delta=DELTA)
        history = play(privacy.central_block, calibration, supplied(sigma, calibration))

        # Centrally the noise is added once, to the estimate, so no number of
        # subjects averages it away and it dominates the error the moment it is
        # larger than the variation being estimated.
        assert squared_error(history, "f_dp") == pytest.approx(
            privacy.central_error(sigma, size=SIZE, delta=DELTA), rel=0.2
        )
        assert privacy.central_error(sigma, size=SIZE, delta=DELTA) > sigma**2

    @pytest.mark.parametrize(
        "error, grid",
        [
            (privacy.local_error, np.linspace(1e-4, 2.0, 400)),
            (privacy.central_error, np.linspace(1e-5, 0.02, 400)),
        ],
        ids=["local", "central"],
    )
    def test_an_accuracy_maximizing_designer_still_asks_for_noise(self, error, grid):
        # The paper's Figures 3 and 4, as the claim they are drawn to make: a
        # designer who cares only about accuracy chooses NON-ZERO noise, because
        # noise is what brings in the subjects whose reports the estimate is
        # averaged over. Both trust models, at different scales.
        best = int(np.argmin([error(s, delta=DELTA) for s in grid]))
        assert 0 < best < len(grid) - 1

    def test_only_the_central_optimum_turns_on_how_many_subjects_there_are(self):
        def best_noise(error, grid, size):
            values = [error(s, size=size, delta=DELTA) for s in grid]
            return grid[int(np.argmin(values))]

        # Locally the mechanism's noise and the variation being estimated are
        # both divided by the number of reports, so the trade between them is
        # the same in a small population as in a large one.
        local = np.linspace(1e-4, 1.0, 500)
        assert best_noise(privacy.local_error, local, 200) == pytest.approx(
            best_noise(privacy.local_error, local, 1000), rel=0.05
        )

        # Centrally the noise is not divided, so more subjects shrink what is
        # gained by adding it and the designer asks for less.
        central = np.linspace(1e-5, 0.02, 500)
        assert best_noise(privacy.central_error, central, 200) > best_noise(
            privacy.central_error, central, 1000
        )


class TestTheDesignerFacesATradeAndSettlesIt:
    def test_more_noise_helps_the_subjects_and_hurts_the_analyst(self):
        grid = np.linspace(0.5, 8.0, 30)
        welfare = [privacy.bounded_subject_utility(s) for s in grid]
        error = [privacy.local_error(s, delta=DELTA) for s in grid]

        # Which is the whole reason the design is not a corner: the two agents
        # want the parameter moved in opposite directions.
        assert welfare == sorted(welfare)
        assert error == sorted(error)

    def test_the_objective_is_maximized_at_an_interior_noise_scale(self):
        grid = np.linspace(0.25, 20.0, 80)
        objective = [privacy.designer_objective(s, delta=DELTA) for s in grid]
        best = int(np.argmax(objective))

        # The paper's result: a designer weighing both parties chooses a noise
        # scale that is neither none nor unbounded.
        assert 0 < best < len(grid) - 1
        assert objective[best] > objective[0]
        assert objective[best] > objective[-1]

    def test_the_chosen_noise_scale_is_a_privacy_guarantee(self):
        # What the designer's number means to a subject: more noise is a smaller
        # epsilon, which is a stronger promise.
        assert privacy.privacy_epsilon(1.0) > privacy.privacy_epsilon(5.0)
        assert privacy.privacy_epsilon(1.0, delta=2.0) == pytest.approx(
            2 * privacy.privacy_epsilon(1.0, delta=1.0)
        )


class TestASolverFindsWhatThePaperDerives:
    """The subjects' half solved rather than supplied, against the closed form."""

    def method(self, sigma, size=20):
        projected = project(
            GroundedBlock(privacy.local_block, privacy.calibration(sigma, size=size))
        )
        grid = {
            "p_actor": np.linspace(-2.0, 2.0, 9),
            # Two points rather than one: a rule needs an axis to vary along,
            # and the solved rule turning out to be constant along this one is
            # the claim of the second test below.
            "b_actor": np.array([-1.0, 1.0]),
        }
        scope = {
            **projected.calibration,
            "a": 0.0,
            "zeta_actor": 0.0,
            "zeta_other": 0.0,
            "p_other": 0.0,
            "gamma_actor": 0.0,
            "gamma_other": 0.0,
        }
        return ExactBestResponse(projected, grid, scope=scope)

    def solved(self, sigma):
        method = self.method(sigma)
        return method.best_response(
            "c_actor",
            {"c_other": privacy.sharing_rule(sigma), "f": privacy.analyst_rule(0.0)},
        )

    @pytest.mark.parametrize("sigma", [0.5, 1.5])
    def test_the_exact_backup_returns_the_threshold_rule(self, sigma):
        rule = self.solved(sigma)
        closed = privacy.sharing_rule(sigma)
        concerns = np.linspace(-2.0, 2.0, 9)
        indifferent = np.isclose(concerns, privacy.Q * sigma)

        # The utility is linear in the decision, so the optimum of the relaxed
        # [0, 1] control is at a vertex and the solver returns the paper's rule
        # exactly rather than an approximation of it -- everywhere the subject
        # is not indifferent between sharing and not.
        solved = np.array([float(np.ravel(rule(0.0, p))[0]) for p in concerns])
        want = np.array([float(closed(0.0, p)) for p in concerns])
        assert np.array_equal(solved[~indifferent], want[~indifferent])

        # And where the benefit exactly covers the concern the bracket is zero,
        # so every action is optimal and there is nothing to prefer. The closed
        # form breaks that tie towards not sharing; the relaxation does not break
        # it at all, which is the one place the two answers differ.
        assert 0.0 < solved[indifferent][0] < 1.0

    def test_the_solved_rule_ignores_the_subjects_own_data(self):
        # A subject observes its own data and the solver finds that it should
        # not act on it, which is what makes the sharing decision reveal
        # nothing about what is being estimated.
        rule = self.solved(1.5)
        for p in (-1.0, 0.0, 1.0):
            assert float(np.ravel(rule(-1.0, p))[0]) == float(np.ravel(rule(1.0, p))[0])
