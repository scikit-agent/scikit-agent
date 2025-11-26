"""
This file implements unit tests for the Monte Carlo simulation module
"""

import unittest

from skagent.distributions import Bernoulli, IndexDistribution, MeanOneLogNormal
from skagent.block import Aggregate, Control, DBlock, simulate_dynamics
from skagent.simulation.monte_carlo import (
    AgentTypeMonteCarloSimulator,
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


class test_draw_shocks(unittest.TestCase):
    def test_draw_shocks(self):
        drawn = draw_shocks(cons_shocks, np.array([0, 1]))

        self.assertEqual(len(drawn["theta"]), 2)
        self.assertEqual(len(drawn["psi"]), 2)
        self.assertTrue(isinstance(drawn["agg_gro"], float))


class test_simulate_dynamics(unittest.TestCase):
    def test_simulate_dynamics(self):
        post = simulate_dynamics(cons_dynamics, cons_pre, cons_dr)

        self.assertAlmostEqual(post["cNrm"], 0.98388429)


class test_AgentTypeMonteCarloSimulator(unittest.TestCase):
    def setUp(self):
        self.calibration = {  # TODO
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
        self.simulator = AgentTypeMonteCarloSimulator(
            self.calibration,
            self.block,
            self.dr,
            self.initial,
            agent_count=3,
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

    def test_make_shock_history(self):
        self.simulator = AgentTypeMonteCarloSimulator(
            self.calibration,
            self.block,
            self.dr,
            self.initial,
            agent_count=3,
        )

        self.simulator.make_shock_history()

        newborn_init_1 = self.simulator.newborn_init_history.copy()
        shocks_1 = self.simulator.shock_history.copy()

        self.simulator.initialize_sim()
        self.simulator.simulate()

        self.assertEqual(newborn_init_1, self.simulator.newborn_init_history)
        self.assertTrue(np.all(self.simulator.history["theta"] == shocks_1["theta"]))


class test_AgentTypeMonteCarloSimulatorAgeVariance(unittest.TestCase):
    def setUp(self):
        self.calibration = {  # TODO
            "G": 1.05,
        }
        self.block = DBlock(
            **{
                "shocks": {
                    "theta": MeanOneLogNormal(1),
                    "agg_R": Aggregate(MeanOneLogNormal(1)),
                    "live": Bernoulli(p=0.98),
                    "psi": IndexDistribution(MeanOneLogNormal, {"sigma": [1.0, 1.1]}),
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
        self.dr = {"c": [lambda m: m * 0.5, lambda m: m * 0.9]}

    def test_simulate(self):
        self.simulator = AgentTypeMonteCarloSimulator(
            self.calibration,
            self.block,
            self.dr,
            self.initial,
            agent_count=3,
        )

        self.simulator.initialize_sim()
        history = self.simulator.simulate(sim_periods=2)

        a1 = history["a"][1]
        b1 = history["m"][1] - self.dr["c"][1](history["m"][1])

        self.assertTrue((a1 == b1).all())


class test_MonteCarloSimulator(unittest.TestCase):
    def setUp(self):
        self.calibration = {  # TODO
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
            agent_count=3,
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
            agent_count=3,
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

        initial = {"a": MeanOneLogNormal(1), "live": 0}
        dr = {"c": lambda m: m / 2}

        simulator = MonteCarloSimulator(
            cons.calibration,
            cons.cons_problem,
            dr,
            initial,
            agent_count=3,
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
