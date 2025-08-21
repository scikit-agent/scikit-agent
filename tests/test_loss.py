from conftest import case_0
import numpy as np
import os
import skagent.ann as ann
from skagent.loss import CustomLoss, static_reward
import torch
import unittest

# Deterministic test seed - change this single value to modify all seeding
# Using same seed as test_maliar.py for consistency across test suite
TEST_SEED = 10077693

# Device selection (but no global state modification at import time)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TestLossFunctions(unittest.TestCase):
    def setUp(self):
        # Set deterministic state for each test (avoid global state interference in parallel runs)
        torch.manual_seed(TEST_SEED)
        np.random.seed(TEST_SEED)
        # Ensure PyTorch uses deterministic algorithms when possible
        torch.use_deterministic_algorithms(True, warn_only=True)
        # Set CUDA deterministic behavior for reproducible tests
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    def test_case_0(self):
        cl = CustomLoss(
            static_reward,
            case_0["block"],
            parameters=case_0["calibration"],
        )

        states_0_N = case_0["givens"]

        bpn = ann.BlockPolicyNet(case_0["block"], width=16)
        ann.train_block_policy_nn(bpn, states_0_N, cl, epochs=250)

        c_ann = bpn.decision_function(states_0_N.to_dict(), {}, {})["c"]

        # Is this result stochastic? How are the network weights being initialized?
        self.assertTrue(
            torch.allclose(c_ann, torch.zeros(c_ann.shape).to(device), atol=0.0015)
        )
