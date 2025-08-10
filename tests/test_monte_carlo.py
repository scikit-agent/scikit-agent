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
import pandas as pd
from skagent.models.benchmarks import (
    d3_block,
    d3_calibration,
    d4_block,
    d4_calibration,
)
from skagent.simulation.monte_carlo import sweep

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


class test_sweep_infinite_horizon(unittest.TestCase):
    def test_sweep_d3_vary_discfac_crra(self):
        block = d3_block
        base = d3_calibration.copy()
        # Simple linear policy; not necessarily optimal, but sufficient for simulation
        dr = {"c": lambda m: 0.3 * m}
        initial = {"a": 0.5}

        param_grid = {
            "DiscFac": [0.94, 0.96],
            "CRRA": [1.5, 2.0],
        }

        H = sweep(
            block=block,
            base_calibration=base,
            dr=dr,
            initial=initial,
            param_grid=param_grid,
            agent_count=50,
            T_sim=400,
            burn_in=0.5,
            variables=["a", "c", "m", "u"],
            seed=123,
        )

        self.assertEqual(H.shape[0], 4)
        for col in ["DiscFac", "CRRA", "a_mean", "c_mean", "m_mean", "u_mean"]:
            self.assertIn(col, H.columns)

        # Reproducibility with same seed
        H2 = sweep(
            block=block,
            base_calibration=base,
            dr=dr,
            initial=initial,
            param_grid=param_grid,
            agent_count=50,
            T_sim=400,
            burn_in=0.5,
            variables=["a", "c", "m", "u"],
            seed=123,
        )
        pd.testing.assert_frame_equal(
            H.reset_index(drop=True), H2.reset_index(drop=True)
        )

    def test_sweep_d4_vary_survivalprob(self):
        block = d4_block
        base = d4_calibration.copy()
        dr = {"c": lambda m: 0.35 * m}
        initial = {"a": 0.4}

        param_grid = {
            "SurvivalProb": [0.98, 0.99],
        }

        H = sweep(
            block=block,
            base_calibration=base,
            dr=dr,
            initial=initial,
            param_grid=param_grid,
            agent_count=40,
            T_sim=300,
            burn_in=0.4,
            variables=["a", "c", "m", "u"],
            seed=777,
        )

        self.assertEqual(H.shape[0], 2)
        for col in ["SurvivalProb", "a_mean", "c_mean", "m_mean", "u_mean"]:
            self.assertIn(col, H.columns)

        # Identity check in sample averages: a ≈ m - c
        diff = (H["m_mean"] - H["c_mean"] - H["a_mean"]).abs()
        self.assertTrue(np.all(diff < 1e-6))

    def test_burn_in_effect_changes_moments(self):
        block = d3_block
        base = d3_calibration.copy()
        dr = {"c": lambda m: 0.3 * m}
        initial = {"a": 0.5}
        param_grid = {"DiscFac": [0.96]}

        H0 = sweep(
            block=block,
            base_calibration=base,
            dr=dr,
            initial=initial,
            param_grid=param_grid,
            agent_count=30,
            T_sim=200,
            burn_in=0.0,
            variables=["a", "c", "m", "u"],
            seed=1,
        )
        H1 = sweep(
            block=block,
            base_calibration=base,
            dr=dr,
            initial=initial,
            param_grid=param_grid,
            agent_count=30,
            T_sim=200,
            burn_in=0.6,
            variables=["a", "c", "m", "u"],
            seed=1,
        )
        # At least one column of moments should differ when changing burn-in
        moment_cols = [c for c in H0.columns if c not in param_grid.keys()]
        self.assertTrue(
            any(not np.isclose(H0[c].values, H1[c].values) for c in moment_cols)
        )
