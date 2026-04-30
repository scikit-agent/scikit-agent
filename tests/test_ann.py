from conftest import (
    case_0,
    case_1,
    case_2,
    case_3,
    case_5,
    case_6,
    case_7,
    case_8,
    case_9,
    case_10,
    case_11,
)
import numpy as np
import os
import skagent.ann as ann
import skagent.bellman as bellman
import skagent.grid as grid
import skagent.loss as loss
import skagent.models.perfect_foresight as pfm
import torch
import unittest

# Deterministic test seed - change this single value to modify all seeding
# Using same seed as test_maliar.py for consistency across test suite
TEST_SEED = 10077693

# Device selection (but no global state modification at import time)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class test_ann_lr(unittest.TestCase):
    def setUp(self):
        # Set deterministic state for each test (avoid global state interference in parallel runs)
        torch.manual_seed(TEST_SEED)
        np.random.seed(TEST_SEED)
        # Ensure PyTorch uses deterministic algorithms when possible
        torch.use_deterministic_algorithms(True, warn_only=True)
        # Set CUDA deterministic behavior for reproducible tests
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    def test_case_0(self):
        edlrl = loss.EstimatedDiscountedLifetimeRewardLoss(
            case_0["bp"],
            1,
            parameters=case_0["calibration"],
        )

        states_0_N = case_0["givens"]

        bpn = ann.BlockPolicyNet(case_0["bp"], width=16)
        ann.train_block_nn(bpn, states_0_N, edlrl, epochs=250)

        c_ann = bpn.decision_function(states_0_N.to_dict(), {}, {})["c"]

        # Is this result stochastic? How are the network weights being initialized?
        self.assertTrue(
            torch.allclose(c_ann, torch.zeros(c_ann.shape).to(device), atol=0.0015)
        )

    def test_case_1(self):
        edlrl = loss.EstimatedDiscountedLifetimeRewardLoss(
            case_1["bp"],
            1,
            parameters=case_1["calibration"],
        )

        given_0_N = case_1["givens"][1]

        bpn = ann.BlockPolicyNet(case_1["bp"], width=16)
        ann.train_block_nn(bpn, given_0_N, edlrl, epochs=500)

        c_ann = bpn.decision_function(
            # TODO -- make this from the Grid
            {"a": given_0_N["a"]},
            {"theta": given_0_N["theta_0"]},
            {},
        )["c"]

        errors = c_ann.flatten() - given_0_N.to_dict()["theta_0"]

        # Is this result stochastic? How are the network weights being initialized?
        self.assertTrue(
            torch.allclose(errors, torch.zeros(errors.shape).to(device), atol=0.03)
        )

    def test_case_1_2(self):
        """
        Running case 1 with big_t == 2
        """
        edlrl = loss.EstimatedDiscountedLifetimeRewardLoss(
            case_1["bp"],
            2,
            parameters=case_1["calibration"],
        )

        given_0_N = case_1["givens"][2]

        bpn = ann.BlockPolicyNet(case_1["bp"], width=16)
        ann.train_block_nn(bpn, given_0_N, edlrl, epochs=500)

        c_ann = bpn.decision_function(
            {"a": given_0_N["a"]},
            {"theta": given_0_N["theta_0"]},
            {},
        )["c"]

        errors = c_ann.flatten() - given_0_N["theta_0"]

        # Is this result stochastic? How are the network weights being initialized?
        self.assertTrue(
            torch.allclose(errors, torch.zeros(errors.shape).to(device), atol=0.03)
        )

    def test_case_2(self):
        edlrl = loss.EstimatedDiscountedLifetimeRewardLoss(
            case_2["bp"],
            1,
            parameters=case_2["calibration"],
        )

        given_0_N = case_2["givens"]

        bpn = ann.BlockPolicyNet(case_2["bp"], width=8)
        ann.train_block_nn(bpn, given_0_N, edlrl, epochs=100)

        # optimal DR is c = 0 = E[theta]

        # Just a smoke test. Since the information set to the control
        # actually gives no information, training isn't effective...

    def test_case_3(self):
        # Construct shocks with deterministic RNG for reproducible test
        case_3["block"].construct_shocks(
            case_3["calibration"], rng=np.random.default_rng(TEST_SEED)
        )  # this should mutate the block object referenced by the BP

        edlrl = loss.EstimatedDiscountedLifetimeRewardLoss(
            case_3["bp"],
            1,
            parameters=case_3["calibration"],
        )

        given_0_N = case_3["givens"][1]

        bpn = ann.BlockPolicyNet(case_3["bp"], width=8)
        ann.train_block_nn(bpn, given_0_N, edlrl, epochs=300)

        c_ann = bpn.decision_function(
            {"a": given_0_N["a"]},
            {
                "theta": given_0_N["theta_0"],
                "psi": given_0_N["psi_0"],
            },
            {},
        )["c"]
        given_m = given_0_N["a"] + given_0_N["theta_0"]

        self.assertTrue(torch.allclose(c_ann.flatten(), given_m.flatten(), atol=0.03))

    def test_case_3_2(self):
        edlrl = loss.EstimatedDiscountedLifetimeRewardLoss(
            case_3["bp"],
            2,
            parameters=case_3["calibration"],
        )

        given_0_N = case_3["givens"][2]

        bpn = ann.BlockPolicyNet(case_3["bp"], width=8)
        ann.train_block_nn(bpn, given_0_N, edlrl, epochs=300)

        c_ann = bpn.decision_function(
            {"a": given_0_N["a"]},
            {
                "theta": given_0_N["theta_0"],
                "psi": given_0_N["psi_0"],
            },
            {},
        )["c"]
        given_m = given_0_N["a"] + given_0_N["theta_0"]

        self.assertTrue(torch.allclose(c_ann.flatten(), given_m.flatten(), atol=0.04))

    def test_case_5_double_bounded_upper_binds(self):
        edlrl = loss.EstimatedDiscountedLifetimeRewardLoss(
            case_5["bp"],
            1,
            parameters=case_5["calibration"],
        )

        given_0_N = case_5["givens"]

        bpn = ann.BlockPolicyNet(case_5["bp"], width=8)
        ann.train_block_nn(bpn, given_0_N, edlrl, epochs=300)

        c_ann = bpn.decision_function(
            {"a": given_0_N["a"]},
            {"theta": given_0_N["theta_0"]},
            {},
        )["c"]

        self.assertTrue(
            torch.allclose(c_ann.flatten(), given_0_N["a"].flatten(), atol=0.03)
        )

    def test_case_6_double_bounded_lower_binds(self):
        edlrl = loss.EstimatedDiscountedLifetimeRewardLoss(
            case_6["bp"],
            1,
            parameters=case_6["calibration"],
        )

        given_0_N = case_6["givens"]

        bpn = ann.BlockPolicyNet(case_6["bp"], width=8)
        ann.train_block_nn(bpn, given_0_N, edlrl, epochs=300)

        c_ann = bpn.decision_function(
            {"a": given_0_N["a"]},
            {"theta": given_0_N["theta_0"]},
            {},
        )["c"]

        self.assertTrue(
            torch.allclose(c_ann.flatten(), given_0_N["a"].flatten(), atol=0.03)
        )

    def test_case_7_only_lower_bound(self):
        edlrl = loss.EstimatedDiscountedLifetimeRewardLoss(
            case_7["bp"],
            1,
            parameters=case_7["calibration"],
        )

        given_0_N = case_7["givens"]

        bpn = ann.BlockPolicyNet(case_7["bp"], width=8)
        ann.train_block_nn(bpn, given_0_N, edlrl, epochs=300)

        c_ann = bpn.decision_function(
            {"a": given_0_N["a"]},
            {"theta": given_0_N["theta_0"]},
            {},
        )["c"]

        self.assertTrue(
            torch.allclose(
                c_ann.flatten(), torch.zeros(c_ann.shape).to(device) + 1, atol=0.03
            )
        )

    def test_case_8_only_upper_bound(self):
        edlrl = loss.EstimatedDiscountedLifetimeRewardLoss(
            case_8["bp"],
            1,
            parameters=case_8["calibration"],
        )

        given_0_N = case_8["givens"]

        bpn = ann.BlockPolicyNet(case_8["bp"], width=8)
        ann.train_block_nn(bpn, given_0_N, edlrl, epochs=300)

        c_ann = bpn.decision_function(
            {"a": given_0_N["a"]},
            {"theta": given_0_N["theta_0"]},
            {},
        )["c"]

        self.assertTrue(
            torch.allclose(c_ann.flatten(), given_0_N["a"].flatten(), atol=0.03)
        )

    def test_case_9_empty_information_set(self):
        loss_fn = loss.EstimatedDiscountedLifetimeRewardLoss(
            case_9["bp"],
            2,
            parameters=case_9["calibration"],
        )

        given_0_N = case_9["givens"]

        bpn = ann.BlockPolicyNet(case_9["bp"], width=8)
        ann.train_block_nn(bpn, given_0_N, loss_fn, epochs=300)

        c_ann = bpn.decision_function(
            {"a": given_0_N["a"]},
            {},
            {},
        )["c"]

        self.assertTrue(
            torch.allclose(
                c_ann.flatten(), torch.full_like(c_ann.flatten(), 3.0), atol=0.04
            )
        )

    def test_lifetime_reward_perfect_foresight(self):
        ### Model data

        pfblock = pfm.block_no_shock
        pfbp = bellman.BellmanPeriod(pfblock, "DiscFac", pfm.calibration)

        ### Loss function
        edlrl = loss.EstimatedDiscountedLifetimeRewardLoss(
            pfbp, 1, parameters=pfm.calibration
        )

        ### Setting up the training

        states_0_N = grid.Grid.from_config(
            {
                "a": {"min": 0, "max": 3, "count": 5},
                "p": {"min": 0, "max": 1, "count": 4},
            }
        )

        bpn = ann.BlockPolicyNet(pfbp, width=8)
        ann.train_block_nn(bpn, states_0_N, edlrl, epochs=100)

        ## This is just a smoke test.


class test_ann_multiple_controls(unittest.TestCase):
    def setUp(self):
        pass

    def test_case_10_multiple_controls(self):
        # Control policy networks for each control in the block.
        cpns = {}

        # Invent Policy Neural Networks for each Control variable.
        for control_sym in case_10["bp"].get_controls():
            cpns[control_sym] = ann.BlockPolicyNet(
                case_10["bp"], control_sym=control_sym
            )

        dict_of_decision_rules = {
            k: v
            for d in [
                cpns[control_sym].get_decision_rule(length=case_10["givens"].n())
                for control_sym in cpns
            ]
            for k, v in d.items()
        }

        # train for control_sym1 with decision rule from the other net
        # for 'c'
        ann.train_block_nn(
            cpns["c"],
            case_10["givens"],
            loss.StaticRewardLoss(
                case_10["bp"],
                case_10["calibration"],
                dict_of_decision_rules,
            ),
            epochs=200,
        )

        # train for control_sym2 with decision rule from other net
        ann.train_block_nn(
            cpns["d"],
            case_10["givens"],
            loss.StaticRewardLoss(
                case_10["bp"],
                case_10["calibration"],
                dict_of_decision_rules,
            ),
            epochs=200,
        )

        # Train the policy neural network for 'c' again to refine its decision rule.
        # This step ensures that 'c' is optimized with the updated decision rules
        # from the other networks, improving the overall policy performance.

        ann.train_block_nn(
            cpns["c"],
            case_10["givens"],
            loss.StaticRewardLoss(
                case_10["bp"],
                case_10["calibration"],
                dict_of_decision_rules,
            ),
            epochs=100,
        )

        rewards = case_10["bp"].reward_function(
            {"a": case_10["givens"]["a"]},
            {},
            parameters=case_10["calibration"],
            decision_rules=dict_of_decision_rules,
        )

        self.assertTrue(
            torch.allclose(
                rewards["u"], torch.zeros(rewards["u"].shape).to(device), atol=0.03
            )
        )


class test_block_policy_value_net(unittest.TestCase):
    """Test the new value function capabilities in ann.py"""

    def setUp(self):
        """Set up a simple test block for value function testing."""
        # Set deterministic state for each test (avoid global state interference in parallel runs)
        torch.manual_seed(TEST_SEED)
        np.random.seed(TEST_SEED)
        # Ensure PyTorch uses deterministic algorithms when possible
        torch.use_deterministic_algorithms(True, warn_only=True)
        # Set CUDA deterministic behavior for reproducible tests
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    def test_block_policy_value_net__case_1(self):
        """Test comprehensive joint training that mirrors policy training patterns."""

        # Create networks
        policy_value_net = ann.BlockPolicyValueNet(case_1["bp"])

        drs, vf = policy_value_net.get_policy_and_value_functions(10)

        self.assertIn("c", drs)

        # Test value network predictions
        test_states = {"a": torch.tensor([2.0])}
        value_estimates = vf(test_states, {"theta": torch.tensor([1.0])}, {})
        self.assertIsInstance(value_estimates, torch.Tensor)
        self.assertEqual(value_estimates.shape, (1,))

    def test_block_policy_value_net__case_11(self):
        """BlockPolicyValueNet forward pass returns (policy, value) 2-tuple with correct shapes."""
        pvnet = ann.BlockPolicyValueNet(case_11["bp"])

        # Build an input tensor matching the iset size on the same device as the network
        n_samples = 5
        n_iset = len(pvnet.iset)
        x = torch.randn(n_samples, n_iset, device=device)

        policy, value = pvnet(x)

        self.assertEqual(policy.shape, (n_samples, 1))
        self.assertEqual(value.shape, (n_samples, 1))
        self.assertIsInstance(policy, torch.Tensor)
        self.assertIsInstance(value, torch.Tensor)


class TestTrainBlockNNValidation(unittest.TestCase):
    """Test input validation in train_block_nn."""

    def setUp(self):
        torch.manual_seed(TEST_SEED)
        self.bpn = ann.BlockPolicyNet(case_0["bp"], width=8)
        self.inputs = case_0["givens"]
        # Loss function that produces gradients: mean-square of the network output
        self.loss_fn = lambda df, g: df["c"](*[g[k].flatten() for k in ["a"]]) ** 2

    def test_zero_epochs_raises(self):
        with self.assertRaises(ValueError, msg="epochs must be a positive integer"):
            ann.train_block_nn(self.bpn, self.inputs, self.loss_fn, epochs=0)

    def test_negative_epochs_raises(self):
        with self.assertRaises(ValueError):
            ann.train_block_nn(self.bpn, self.inputs, self.loss_fn, epochs=-1)

    def test_zero_lr_raises(self):
        with self.assertRaises(ValueError, msg="lr must be > 0"):
            ann.train_block_nn(self.bpn, self.inputs, self.loss_fn, lr=0.0)

    def test_negative_lr_raises(self):
        with self.assertRaises(ValueError):
            ann.train_block_nn(self.bpn, self.inputs, self.loss_fn, lr=-0.001)

    def test_zero_grad_clip_raises(self):
        with self.assertRaises(ValueError, msg="grad_clip must be > 0 or None"):
            ann.train_block_nn(self.bpn, self.inputs, self.loss_fn, grad_clip=0.0)

    def test_none_grad_clip_allowed(self):
        _, loss = ann.train_block_nn(
            self.bpn, self.inputs, self.loss_fn, epochs=1, grad_clip=None
        )
        self.assertIsNotNone(loss)
