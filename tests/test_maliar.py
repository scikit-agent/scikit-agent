import numpy as np
import skagent.algos.maliar as solver
import skagent.model as model
import unittest

parameters = {"q": 1.1}

block_data = {
    "name": "test block - maliar",
    "dynamics": {
        "c": model.Control(["a"]),
        "a": lambda a, c, e, q: q * a - c + e,
        "e": lambda e: e,
        "u": lambda c: np.log(c),
    },
    "reward": {"u": "consumer"},
}

states_0 = {
    "a": 1,
    "e": 0.1,
}

# a dummy policy
decision_rules = {"c": lambda a: a / 2}
decisions = {"c": 0.5}


class TestSolverFunctions(unittest.TestCase):
    def setUp(self):
        self.block = model.DBlock(**block_data)

    def test_create_transition_function(self):
        transition_function = solver.create_transition_function(self.block, ["a", "e"])

        states_1 = transition_function(states_0, decisions, parameters=parameters)

        self.assertAlmostEqual(states_1["a"], 0.7)
        self.assertEqual(states_1["e"], 0.1)

    def test_create_decision_function(self):
        decision_function = solver.create_decision_function(self.block, decision_rules)

        decisions_0 = decision_function(states_0, parameters=parameters)

        self.assertEqual(decisions_0["c"], 0.5)

    def test_create_reward_function(self):
        reward_function = solver.create_reward_function(self.block)

        reward_0 = reward_function(states_0, decisions, parameters=parameters)

        self.assertAlmostEqual(reward_0["u"], -0.69314718)

    def test_estimate_discounted_lifetime_reward(self):
        dlr_0 = solver.estimate_discounted_lifetime_reward(
            self.block,
            0.9,
            decision_rules,
            states_0,
            0,
            parameters=parameters,
        )

        self.assertEqual(dlr_0, 0)

        dlr_1 = solver.estimate_discounted_lifetime_reward(
            self.block,
            0.9,
            decision_rules,
            states_0,
            1,
            parameters=parameters,
        )

        self.assertAlmostEqual(dlr_1, -0.69314718)

        dlr_2 = solver.estimate_discounted_lifetime_reward(
            self.block,
            0.9,
            decision_rules,
            states_0,
            2,
            parameters=parameters,
        )

        self.assertAlmostEqual(dlr_2, -1.63798709)
