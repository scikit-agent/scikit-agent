"""
This file implements unit tests for the Monte Carlo simulation module
"""

import unittest

from skagent.distributions import (
    Bernoulli,
    DiscreteDistribution,
    IndexDistribution,
    MeanOneLogNormal,
    TimeVaryingDiscreteDistribution,
    set_rng,
)
from skagent.block import Aggregate, Control, DBlock, simulate_dynamics
from skagent.simulation.monte_carlo import (
    MonteCarloSimulator,
    draw_shocks,
)
import numpy as np

cons_shocks = {
    "agg_gro": Aggregate(MeanOneLogNormal(1)),
    "psi": IndexDistribution(MeanOneLogNormal, {"sigma": [1.0, 1.1]}),
    "theta": MeanOneLogNormal(1),
    "live": Bernoulli(p=0.98),
}

cons_pre = {
    "R": 1.05,
    "aNrm": 1,
    "gamma": 1.1,
    "psi": 1.1,  # TODO: draw this from a shock,
    "theta": 1.1,  # TODO: draw this from a shock
}

cons_dynamics = {
    "G": lambda gamma, psi: gamma * psi,
    "Rnrm": lambda R, G: R / G,
    "bNrm": lambda Rnrm, aNrm: Rnrm * aNrm,
    "mNrm": lambda bNrm, theta: bNrm + theta,
    "cNrm": Control(["mNrm"]),
    "aNrm": lambda mNrm, cNrm: mNrm - cNrm,
}

cons_dr = {"cNrm": lambda mNrm: mNrm / 2}


def _nested_shocks():
    """Shocks that draw through distributions they hold, rather than themselves.

    Both kinds keep one child per condition and draw through the child, so a
    generator set on the parent alone never reaches the values.
    """
    return {
        "psi": IndexDistribution(MeanOneLogNormal, {"sigma": [0.2, 0.4]}),
        "eta": TimeVaryingDiscreteDistribution(
            [
                DiscreteDistribution([0.0, 1.0], [0.5, 0.5]),
                DiscreteDistribution([0.0, 2.0], [0.5, 0.5]),
            ]
        ),
    }


class test_draw_shocks(unittest.TestCase):
    def test_draw_shocks(self):
        drawn = draw_shocks(cons_shocks, np.array([0, 1]))

        self.assertEqual(len(drawn["theta"]), 2)
        self.assertEqual(len(drawn["psi"]), 2)
        self.assertTrue(isinstance(drawn["agg_gro"], float))

    def test_the_seed_reaches_a_nested_distribution(self):
        """The same seed repeats a draw, and a different one changes it."""
        conditions = np.array([0, 1, 0, 1])

        first = draw_shocks(_nested_shocks(), conditions, rng=np.random.default_rng(11))
        again = draw_shocks(_nested_shocks(), conditions, rng=np.random.default_rng(11))
        other = draw_shocks(
            _nested_shocks(), conditions, rng=np.random.default_rng(999)
        )

        for sym in first:
            np.testing.assert_allclose(first[sym], again[sym])
            self.assertFalse(np.allclose(first[sym], other[sym]), sym)

    def test_seeding_the_shock_and_seeding_the_draw_agree(self):
        """The two ways to seed a draw are one operation, not two.

        One generator for the whole call, as ``draw_shocks`` passes it: the
        shocks share a stream and take from it in the order they are drawn.
        """
        conditions = np.array([0, 1, 0, 1])

        seeded_first = _nested_shocks()
        shared = np.random.default_rng(11)
        for shock in seeded_first.values():
            set_rng(shock, shared)
        before = draw_shocks(seeded_first, conditions)

        at_the_draw = draw_shocks(
            _nested_shocks(), conditions, rng=np.random.default_rng(11)
        )

        for sym in before:
            np.testing.assert_allclose(before[sym], at_the_draw[sym])


class test_simulate_dynamics(unittest.TestCase):
    def test_simulate_dynamics(self):
        post = simulate_dynamics(cons_dynamics, cons_pre, cons_dr)

        self.assertAlmostEqual(post["cNrm"], 0.98388429)


class test_MonteCarloSimulatorWithLiveShock(unittest.TestCase):
    def setUp(self):
        self.calibration = {
            "G": 1.05,
        }
        self.block = DBlock(
            **{
                "shocks": {
                    "theta": MeanOneLogNormal(1),
                    "agg_R": Aggregate(MeanOneLogNormal(1)),
                    "live": Bernoulli(p=0.98),
                },
                "dynamics": {
                    "b": lambda agg_R, G, a: agg_R * G * a,
                    "m": lambda b, theta: b + theta,
                    "c": Control(["m"]),
                    "a": lambda m, c: m - c,
                },
            }
        )

        self.initial = {"a": MeanOneLogNormal(1), "live": 1}

        self.dr = {"c": lambda m: m / 2}

    def test_simulate(self):
        self.simulator = MonteCarloSimulator(
            self.calibration,
            self.block,
            self.dr,
            self.initial,
            sample_count=3,
        )

        self.simulator.initialize_sim()
        history = self.simulator.simulate()

        a1 = history["a"][5]
        b1 = (
            history["a"][4] * history["agg_R"][5] * self.calibration["G"]
            + history["theta"][5]
            - history["c"][5]
        )

        # Use allclose for numerical tolerance instead of exact equality
        self.assertTrue(np.allclose(a1, b1, rtol=1e-12, atol=1e-12))


class test_MonteCarloSimulator(unittest.TestCase):
    def setUp(self):
        self.calibration = {
            "G": 1.05,
        }
        self.block = DBlock(
            **{
                "shocks": {
                    "theta": MeanOneLogNormal(1),
                    "agg_R": Aggregate(MeanOneLogNormal(1)),
                },
                "dynamics": {
                    "b": lambda agg_R, G, a: agg_R * G * a,
                    "m": lambda b, theta: b + theta,
                    "c": Control(["m"]),
                    "a": lambda m, c: m - c,
                },
            }
        )

        self.initial = {"a": MeanOneLogNormal(1)}

        self.dr = {"c": lambda m: m / 2}

    def test_simulate(self):
        self.simulator = MonteCarloSimulator(
            self.calibration,
            self.block,
            self.dr,
            self.initial,
            sample_count=3,
        )

        self.simulator.initialize_sim()
        history = self.simulator.simulate()

        a1 = history["a"][5]
        b1 = (
            history["a"][4] * history["agg_R"][5] * self.calibration["G"]
            + history["theta"][5]
            - history["c"][5]
        )

        # Use allclose for numerical tolerance instead of exact equality
        self.assertTrue(np.allclose(a1, b1, rtol=1e-12, atol=1e-12))


class test_MonteCarloSimulatorWithReward(unittest.TestCase):
    """Test MonteCarloSimulator with a block that has a reward dictionary."""

    def setUp(self):
        self.calibration = {
            "G": 1.05,
            "CRRA": 2.0,
        }
        self.block = DBlock(
            **{
                "shocks": {
                    "theta": MeanOneLogNormal(1),
                    "agg_R": Aggregate(MeanOneLogNormal(1)),
                },
                "dynamics": {
                    "b": lambda agg_R, G, a: agg_R * G * a,
                    "m": lambda b, theta: b + theta,
                    "c": Control(["m"], agent="consumer"),
                    "a": lambda m, c: m - c,
                    # Reward variable computed in dynamics
                    "u": lambda c, CRRA: c ** (1 - CRRA) / (1 - CRRA),
                },
                # Reward dictionary maps reward variable to agent role
                "reward": {"u": "consumer"},
            }
        )

        self.initial = {"a": MeanOneLogNormal(1)}

        self.dr = {"c": lambda m: m / 2}

    def test_simulate_with_reward(self):
        """Test that MonteCarloSimulator works with blocks that have reward dictionaries."""
        self.simulator = MonteCarloSimulator(
            self.calibration,
            self.block,
            self.dr,
            self.initial,
            sample_count=3,
        )

        self.simulator.initialize_sim()
        history = self.simulator.simulate()

        # Verify that reward variable 'u' is tracked
        self.assertIn("u", history)
        self.assertEqual(history["u"].shape, (10, 3))

        # Verify dynamics are computed correctly
        a1 = history["a"][5]
        b1 = (
            history["a"][4] * history["agg_R"][5] * self.calibration["G"]
            + history["theta"][5]
            - history["c"][5]
        )
        self.assertTrue(np.allclose(a1, b1, rtol=1e-12, atol=1e-12))

        # Verify reward is computed correctly
        u1 = history["u"][5]
        c1 = history["c"][5]
        u1_expected = c1 ** (1 - self.calibration["CRRA"]) / (
            1 - self.calibration["CRRA"]
        )
        self.assertTrue(np.allclose(u1, u1_expected, rtol=1e-12, atol=1e-12))


class test_MonteCarloSimulatorWithConsumerModel(unittest.TestCase):
    """Test MonteCarloSimulator with the actual consumer model from the issue."""

    def test_simulate_consumer_problem(self):
        """Test the exact scenario from the issue."""
        from skagent.distributions import MeanOneLogNormal
        import skagent.models.consumer as cons
        from skagent.simulation.monte_carlo import MonteCarloSimulator

        initial = {"a": MeanOneLogNormal(1)}
        dr = {"c": lambda m: m / 2}

        simulator = MonteCarloSimulator(
            cons.calibration,
            cons.cons_problem,
            dr,
            initial,
            sample_count=3,
        )

        simulator.initialize_sim()
        history = simulator.simulate()

        # Verify the simulation completed successfully
        self.assertIsNotNone(history)
        self.assertIn("c", history)
        self.assertIn("a", history)

        # Verify history structure
        self.assertEqual(history["c"].shape, (10, 3))
        self.assertEqual(history["a"].shape, (10, 3))
