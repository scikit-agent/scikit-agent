import numpy as np
import skagent.ann as ann
import skagent.models.perfect_foresight as pfm
import torch

import unittest

## CUDA handling

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(torch.cuda.is_available())


class test_ann(unittest.TestCase):
    def test_lifetime_reward_perfect_foresight(self):
        ### Model data

        pfblock = pfm.block_no_shock

        ### Other data

        states_0 = {
            "a": 1,
            "p": 0.1,
        }
        state_variables = list(states_0.keys())

        ### Loss function

        # TODO : have this take some number of random shocks
        edlrl = ann.get_estimated_discounted_lifetime_reward_loss(
            state_variables, pfblock, 0.9, 1, parameters=pfm.calibration
        )

        ### Setting up the training

        training_N = 10
        states_0_N = torch.FloatTensor(
            np.random.random((training_N, len(state_variables))) * np.array([3, 1])
        )
        states_0_N = states_0_N.to(device)
        ## TODO: make the training states reflect the right ranges on the state space.
        ## TODO: include big_t shocks in this -- for lifetime reward

        net = ann.PolicyNet(state_variables, pfblock.get_controls(), width=8)

        ### Trainiing
        ann.train_nn(net, states_0_N, edlrl, epochs=100)
