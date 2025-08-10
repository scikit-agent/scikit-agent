"""
This file implements unit tests for the Monte Carlo simulation module
"""

import unittest

from skagent.distributions import Bernoulli, IndexDistribution, MeanOneLogNormal
from skagent.model import Aggregate, Control, DBlock, simulate_dynamics
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


class test_HeterogeneousCalibrationSimulation(unittest.TestCase):
    def test_distribution_calibration_used_in_dynamics(self):
        import numpy as np
        from skagent.distributions import Normal, MeanOneLogNormal
        from skagent.model import Control, DBlock
        from skagent.simulation.monte_carlo import MonteCarloSimulator

        block = DBlock(
            **{
                "shocks": {"eps": Normal(0.0, 1.0)},
                "dynamics": {
                    "x": lambda a, alpha, eps: alpha * a + eps,
                    "c": Control(["x"]),
                    "a": lambda x, c: x - c,
                },
            }
        )

        calib = {"alpha": Normal(1.0, 0.0)}  # degenerate at 1.0
        initial = {"a": MeanOneLogNormal(0.0)}
        sim = MonteCarloSimulator(
            calib,
            block,
            {"c": lambda x: 0.0},
            initial,
            agent_count=3,
            seed=123,
        )
        sim.initialize_sim()
        hist = sim.simulate(sim_periods=2)
        # At t=1: x - eps == alpha * a_prev. With degenerate alpha=1, equals a at t=0.
        np.testing.assert_allclose(
            hist["x"][1] - hist["eps"][1], hist["a"][0], rtol=1e-12, atol=1e-12
        )

    def test_age_varying_then_heterogeneous(self):
        import numpy as np
        from skagent.distributions import MeanOneLogNormal
        from skagent.model import Control, DBlock
        from skagent.simulation.monte_carlo import AgentTypeMonteCarloSimulator

        block = DBlock(
            **{
                "shocks": {"eps": MeanOneLogNormal(0.0)},
                "dynamics": {
                    "y": lambda a, beta, eps: beta * a + eps,
                    "c": Control(["y"]),
                    "a": lambda y, c: y - c,
                    "live": lambda: 1,  # keep alive
                },
            }
        )
        calib = {"beta": [0.5, 1.0, 1.5]}
        initial = {"a": MeanOneLogNormal(0.0), "live": 1}

        sim = AgentTypeMonteCarloSimulator(
            calib, block, {"c": lambda y: 0.0}, initial, agent_count=3, seed=0
        )
        sim.initialize_sim()
        hist = sim.simulate(sim_periods=2)
        # At t=1: y - eps == beta(age=1) * a_prev. beta(age=1) = 1.0
        np.testing.assert_allclose(
            hist["y"][1] - hist["eps"][1], 1.0 * hist["a"][0], rtol=1e-12, atol=1e-12
        )

    def test_per_agent_vector_calibration(self):
        import numpy as np
        from skagent.distributions import MeanOneLogNormal
        from skagent.model import Control, DBlock
        from skagent.simulation.monte_carlo import MonteCarloSimulator

        block = DBlock(
            **{
                "shocks": {"eps": MeanOneLogNormal(0.0)},
                "dynamics": {
                    "x": lambda a, gamma, eps: gamma * a + eps,
                    "c": Control(["x"]),
                    "a": lambda x, c: x - c,
                },
            }
        )
        calib = {"gamma": [0.9, 1.0, 1.1]}
        initial = {"a": MeanOneLogNormal(0.0)}
        sim = MonteCarloSimulator(
            calib, block, {"c": lambda x: 0.0}, initial, agent_count=3, seed=0
        )
        sim.initialize_sim()
        hist = sim.simulate(sim_periods=2)
        a_prev, x1, eps1 = hist["a"][0], hist["x"][1], hist["eps"][1]
        np.testing.assert_allclose(
            x1 - eps1, np.array([0.9, 1.0, 1.1]) * a_prev, rtol=1e-12, atol=1e-12
        )
