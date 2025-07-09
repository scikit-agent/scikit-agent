from skagent.distributions import Lognormal
import skagent.models.consumer as cons
import skagent.models.perfect_foresight as pfm
import skagent.models.perfect_foresight_normalized as pfnm
from skagent.simulation.monte_carlo import AgentTypeMonteCarloSimulator


import unittest


class test_pfm(unittest.TestCase):
    def setUp(self):
        self.mcs = AgentTypeMonteCarloSimulator(
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
        self.mcs = AgentTypeMonteCarloSimulator(  ### Use fm, blockified
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
        self.cs = AgentTypeMonteCarloSimulator(  ### Use fm, blockified
            cons.calibration,
            cons.cons_problem,  ### multiple cons blocks!
            {
                "c": lambda m: 0.5 * m,  # Simple consumption function for smoke test
                # danger: normalized decision rule for unnormalized problem
            },
            {  # initial states
                "k": Lognormal(-6, 0),
                #'live' : 1,
                # "p": 1.0, - not needed, using normalized problem
            },
            agent_count=2,
            T_sim=5,
        )

        self.pcs = AgentTypeMonteCarloSimulator(  ### Use fm, blockified
            cons.calibration,
            cons.cons_portfolio_problem,  ### multiple cons blocks!
            {
                "c": lambda m: m / 2,
                # danger: normalized decision rule for unnormalized problem
                "stigma": lambda a: a / (2 + a),
                # just a dummy share func
            },
            {  # initial states
                "k": Lognormal(-6, 0),
                #'live' : 1,
                # "p": 1.0,- not needed, using normalized problem
                "R": 1.03,
            },
            agent_count=2,
            T_sim=5,
        )

    def test_simulate(self):
        self.cs.initialize_sim()
        self.cs.simulate()

        self.assertEqual(self.cs.calibration["R"], 1.03)
        self.assertFalse("R" in self.cs.history)

        self.pcs.initialize_sim()
        self.pcs.simulate()

        self.assertFalse("R" in self.cs.history)

        # test to see if the R value is
        # as calibrated for cons
        # and dynamic for the portfolio model

        self.assertTrue(self.pcs.history["R"][0][0] != 1.03)
