from conftest import case_1
import conftest as bft
import numpy as np
import skagent.algos.maliar as solver
import skagent.model as model
import torch
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

        states_1 = transition_function(states_0, {}, decisions, parameters=parameters)

        self.assertAlmostEqual(states_1["a"], 0.7)
        self.assertEqual(states_1["e"], 0.1)

    def test_create_decision_function(self):
        decision_function = solver.create_decision_function(self.block, decision_rules)

        decisions_0 = decision_function(states_0, {}, parameters=parameters)

        self.assertEqual(decisions_0["c"], 0.5)

    def test_create_reward_function(self):
        reward_function = solver.create_reward_function(self.block)

        reward_0 = reward_function(states_0, {}, decisions, parameters=parameters)

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


class TestLifetimeReward(unittest.TestCase):
    """
    More tests of the lifetime reward function specifically.
    """

    def setUp(self):
        self.block_2 = model.DBlock(**bft.lr_test_block_data_2)

        self.states_0 = {"a": 0}

    def test_block_1(self):
        dlr_1 = solver.estimate_discounted_lifetime_reward(
            case_1["block"],
            0.9,
            case_1["optimal_dr"],
            self.states_0,
            1,
            shocks_by_t={"theta": torch.FloatTensor(np.array([[0]]))},
        )

        self.assertEqual(dlr_1, 0)

        # big_t is 2
        dlr_1_2 = solver.estimate_discounted_lifetime_reward(
            case_1["block"],
            0.9,
            case_1["optimal_dr"],
            self.states_0,
            2,
            shocks_by_t={"theta": torch.FloatTensor(np.array([[0], [0]]))},
        )

        self.assertEqual(dlr_1_2, 0)

    def test_block_2(self):
        dlr_2 = solver.estimate_discounted_lifetime_reward(
            self.block_2,
            0.9,
            bft.lr_test_block_data_2_optimal_dr,
            self.states_0,
            1,
            shocks_by_t={
                "theta": torch.FloatTensor(np.array([[0]])),
                "psi": torch.FloatTensor(np.array([[0]])),
            },
        )

        self.assertEqual(dlr_2, 0)
