import numpy as np

from skagent.distributions import Lognormal
import skagent.models.consumer as cons
import skagent.models.perfect_foresight as pfm
import skagent.models.perfect_foresight_normalized as pfnm
from skagent.simulation.monte_carlo import Simulator


import unittest


class test_pfm(unittest.TestCase):
    def setUp(self):
        self.mcs = Simulator(
            pfm.calibration,
            pfm.block,
            {"c": lambda m: 0.5 * m},
            # danger: normalized decision rule for unnormalized problem
            {  # initial states
                "a": Lognormal(-6, 0),
                #'live' : 1,
                "p": 1.0,
            },
            agent_count=3,
            T_sim=120,
        )

    def test_simulate(self):
        ## smoke test
        self.mcs.initialize_sim()
        self.mcs.simulate()


class test_pfnm(unittest.TestCase):
    def setUp(self):
        self.mcs = Simulator(  ### Use fm, blockified
            pfnm.calibration,
            pfnm.block,
            {
                "c_nrm": lambda m_nrm: 0.5 * m_nrm
            },  # Simple consumption function for smoke test
            {  # initial states
                "a_nrm": Lognormal(-6, 0),
                #'live' : 1,
                "p": 1.0,
            },
            agent_count=3,
            T_sim=120,
        )

    def test_simulate(self):
        ## smoke test
        self.mcs.initialize_sim()
        self.mcs.simulate()


class test_consumer_models(unittest.TestCase):
    def setUp(self):
        self.cs = Simulator(
            cons.calibration,
            cons.cons_problem,
            {
                "c": lambda m: 0.5 * m,  # simple consumption function for smoke test
            },
            {  # initial states (normalized problem: p not needed)
                "k": Lognormal(-6, 0),
            },
            agent_count=2,
            T_sim=5,
        )

        self.pcs = Simulator(
            cons.calibration,
            cons.cons_portfolio_problem,
            {
                "c": lambda m: m / 2,
                "stigma": lambda a: a / (2 + a),  # dummy risky-share rule
            },
            {  # initial states (normalized problem: p not needed)
                "k": Lognormal(-6, 0),
                "R": 1.03,
            },
            agent_count=2,
            T_sim=5,
        )

        self.mcs = Simulator(
            cons.calibration,
            cons.mortal_cons_problem,
            {"c": lambda m: m / 3},
            {"k": Lognormal(-6, 0), "p": 1.0, "age": 0},
            agent_count=2,
            T_sim=5,
            seed=0,  # fixed: with these settings both agents die at t=3
        )

    def test_simulate(self):
        self.cs.initialize_sim()
        self.cs.simulate()

        # R is a fixed calibration parameter for the non-portfolio model, so it
        # is never written into the simulated history.
        self.assertEqual(self.cs.calibration["R"], 1.03)
        self.assertFalse("R" in self.cs.history)

        # For the portfolio model R is produced dynamically, so it varies.
        self.pcs.initialize_sim()
        self.pcs.simulate()
        self.assertTrue(self.pcs.history["R"][0][0] != 1.03)

        # the portfolio simulation must not produce NaN anywhere in m
        self.assertFalse(np.any(np.isnan(self.pcs.history["m"])))

    def test_mortality_dynamics(self):
        """mortality_block resets dead agents to newborns and ages survivors."""
        self.mcs.initialize_sim()
        self.mcs.simulate()

        hist = self.mcs.history

        # The seed must actually produce at least one death, otherwise the reset
        # branch checked below is never exercised.
        self.assertTrue(
            (hist["live"] == 0).any(),
            "expected at least one death; reset path is otherwise untested",
        )

        for t in range(1, self.mcs.T_sim):
            for i in range(self.mcs.agent_count):
                if hist["live"][t][i] == 0:
                    # Death: age resets to 0 and k is a freshly drawn newborn
                    # endowment (k_init), not the surviving end-of-period assets.
                    self.assertAlmostEqual(hist["age"][t][i], 0.0, places=10)
                    self.assertAlmostEqual(
                        hist["k"][t][i], hist["k_init"][t][i], places=10
                    )
                    self.assertNotAlmostEqual(
                        hist["k"][t][i], hist["a"][t][i], places=10
                    )
                else:
                    # Survival: age advances by exactly one and end-of-period
                    # assets become next period's capital (k = a within period).
                    self.assertAlmostEqual(
                        hist["age"][t][i], hist["age"][t - 1][i] + 1.0, places=10
                    )
                    self.assertAlmostEqual(hist["k"][t][i], hist["a"][t][i], places=10)

    def test_mortality_frequency(self):
        """Empirical survival frequency tracks LivPrb across many agents."""
        sim = Simulator(
            cons.calibration,
            cons.mortal_cons_problem,
            {"c": lambda m: m / 3},
            {"k": Lognormal(-6, 0), "p": 1.0, "age": 0},
            agent_count=500,
            T_sim=20,
            seed=1,
        )
        sim.initialize_sim()
        sim.simulate()

        survival_rate = sim.history["live"].mean()
        self.assertAlmostEqual(survival_rate, cons.calibration["LivPrb"], delta=0.01)
