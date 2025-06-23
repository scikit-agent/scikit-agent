import unittest

from HARK.distributions import Bernoulli, DiscreteDistribution
import skagent.model as model
from skagent.model import Control
import skagent.models.consumer as cons

# TODO: let the shock constructor reference this parameter.
LivPrb = 0.98

test_block_A_data = {
    "name": "test block A",
    "shocks": {
        "live": Bernoulli(p=LivPrb),
    },
    "dynamics": {
        "y": lambda p: p,
        "m": lambda Rfree, a, y: Rfree * a + y,
        "c": Control(["m"], agent="consumer"),
        "p": lambda PermGroFac, p: PermGroFac * p,
        "a": lambda m, c: m - c,
        "u": lambda c, CRRA: c ** (1 - CRRA) / (1 - CRRA),
    },
    "reward": {"u": "consumer"},
}

test_block_B_data = {
    "name": "test block B",
    "shocks": {"SB": Bernoulli(p=0.1)},
    "dynamics": {"pi": lambda a, Rfree: (Rfree - 1) * a},
    "reward": {"pi": "lender"},
}

test_block_C_data = {"name": "test block B", "shocks": {"SC": Bernoulli(p=0.2)}}

test_block_D_data = {
    "name": "test block D",
    "shocks": {"SD": Bernoulli(p=0.3)},
    "dynamics": {"z": Control(["y"], agent="foo-agent")},
}


class test_Control(unittest.TestCase):
    def setUp(self):
        self.test_control_A = model.Control(
            ["a"], upper_bound=lambda a: a, lower_bound=lambda a: 0, agent="myagent"
        )

    def test_attributes(self):
        self.assertEqual(self.test_control_A.agent, "myagent")


class test_DBlock(unittest.TestCase):
    def setUp(self):
        self.test_block_A = model.DBlock(**test_block_A_data)
        self.cblock = cons.consumption_block_normalized
        self.cblock.construct_shocks(cons.calibration)

        # prior states relative to the decision, so with realized shocks.
        self.dpre = {"k": 2, "R": 1.05, "PermGroFac": 1.1, "theta": 1, "CRRA": 2}

        # simple decision rule
        self.dr = {"c": lambda m: m}

    def test_init(self):
        self.assertEqual(self.test_block_A.name, "test block A")

    def test_discretize(self):
        dbl = self.cblock.discretize({"theta": {"N": 5}})

        self.assertEqual(len(dbl.shocks["theta"].pmv), 5)

    def test_transition(self):
        post = self.cblock.transition(self.dpre, self.dr)

        self.assertEqual(post["a"], 0)

    def test_calc_reward(self):
        self.assertEqual(self.cblock.calc_reward({"c": 1, "CRRA": 2})["u"], -1.0)

    def test_state_rule_value_function(self):
        savf = self.cblock.get_state_rule_value_function_from_continuation(lambda a: 0)

        dv0 = savf(self.dpre, self.dr)

        self.assertEqual(dv0, -0.34375)

        cv = 1
        # note change in continuation value here.
        dv1 = self.cblock.get_decision_value_function(self.dr, lambda a: cv)(self.dpre)

        self.assertEqual(dv1, dv0 + cv)

    def test_arrival_value_function(self):
        av = self.cblock.get_arrival_value_function(
            {"theta": {"N": 5}}, {"c": lambda m: m}, lambda a: 0
        )

        av({"k": 1, "R": 1.05, "PermGroFac": 1.1, "theta": 1, "CRRA": 2})

    def test_attributions(self):
        block_a_attributions = self.test_block_A.get_attributions()

        self.assertEqual(block_a_attributions["consumer"], ["c", "u"])

        cblock_attribtuions = self.cblock.get_attributions()

        self.assertEqual(cblock_attribtuions["consumer"], ["c", "u"])

    def test_iter_dblocks(self):
        """Test that DBlock.iter_dblocks() yields itself."""
        result = list(self.test_block_A.iter_dblocks())

        self.assertEqual(len(result), 1)
        self.assertIs(result[0], self.test_block_A)
        self.assertIsInstance(result[0], model.DBlock)


class test_RBlock(unittest.TestCase):
    def setUp(self):
        self.test_block_B = model.DBlock(**test_block_B_data)
        self.test_block_C = model.DBlock(**test_block_C_data)
        self.test_block_D = model.DBlock(**test_block_D_data)

        self.cpp = cons.cons_portfolio_problem

    def test_init(self):
        r_block_tree = model.RBlock(
            blocks=[
                self.test_block_B,
                model.RBlock(blocks=[self.test_block_C, self.test_block_D]),
            ]
        )

        r_block_tree.get_shocks()
        self.assertEqual(len(r_block_tree.get_shocks()), 3)

    def test_discretize(self):
        self.cpp.construct_shocks(cons.calibration)
        cppd = self.cpp.discretize({"theta": {"N": 5}, "risky_return": {"N": 6}})

        self.assertEqual(len(cppd.get_shocks()["theta"].pmv), 5)
        self.assertEqual(len(cppd.get_shocks()["risky_return"].pmv), 6)

        self.assertFalse(
            isinstance(self.cpp.get_shocks()["theta"], DiscreteDistribution)
        )

    def test_get_attributions(self):
        r_block_tree = model.RBlock(
            blocks=[self.test_block_B, self.test_block_C, self.test_block_D]
        )

        attrs = r_block_tree.get_attributions()

        self.assertEqual({"foo-agent": ["z"], "lender": ["pi"]}, attrs)

    def test_iter_dblocks_single_block(self):
        """Test RBlock.iter_dblocks() with a single DBlock."""
        rblock = model.RBlock(name="test_rblock_single", blocks=[self.test_block_B])

        result = list(rblock.iter_dblocks())

        self.assertEqual(len(result), 1)
        self.assertIs(result[0], self.test_block_B)
        self.assertIsInstance(result[0], model.DBlock)

    def test_iter_dblocks_complex_nested_structure(self):
        """Test RBlock.iter_dblocks() with a complex nested RBlock structure."""
        # Create the exact structure requested in the comment
        r_block_tree = model.RBlock(
            blocks=[
                self.test_block_B,
                model.RBlock(blocks=[self.test_block_C, self.test_block_D]),
            ]
        )

        result = list(r_block_tree.iter_dblocks())

        # Should get all 3 DBlocks from the nested structure
        self.assertEqual(len(result), 3)

        # Verify all expected blocks are present
        self.assertIn(self.test_block_B, result)
        self.assertIn(self.test_block_C, result)
        self.assertIn(self.test_block_D, result)

        # Check that all results are DBlock instances
        for block in result:
            self.assertIsInstance(block, model.DBlock)

        # Verify the order follows depth-first traversal
        # Expected order: test_block_B, then test_block_C, then test_block_D
        expected_blocks = [self.test_block_B, self.test_block_C, self.test_block_D]
        self.assertEqual(result, expected_blocks)
