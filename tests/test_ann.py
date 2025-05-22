import skagent.ann as ann
import skagent.grid as grid
import skagent.models.perfect_foresight as pfm
from skagent.model import Control, DBlock
import torch

import unittest

## CUDA handling
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(torch.cuda.is_available())


test_calibration = {"theta": 1}

test_block = DBlock(
    **{
        "name": "test",
        "dynamics": {
            "b": lambda a, theta: a + theta,
            "c": Control(["b"], agent="consumer"),
            "u": lambda b, c: -((b - c) ** 2),
            "a": lambda b, theta: b + theta,
        },
        "reward": {"u": "consumer"},
    }
)


class test_ann(unittest.TestCase):
    def test_basic(self):
        states_0 = {
            "a": 0,
        }
        state_variables = list(states_0.keys())

        ### Loss function

        # TODO : have this take some number of random shocks
        edlrl = ann.get_estimated_discounted_lifetime_reward_loss(
            state_variables, test_block, 0.9, 1, parameters=test_calibration
        )

        ### Setting up the training
        states_0_N = grid.torched(
            grid.make_grid(
                {
                    "a": {"min": 0, "max": 4, "count": 21},
                }
            )
        )
        ## TODO: include big_t shocks in this -- for lifetime reward

        net = ann.PolicyNet(state_variables, test_block.get_controls(), width=16)

        ### Trainiing
        ann.train_nn(net, states_0_N, edlrl, epochs=200)

        self.assertTrue(torch.allclose(states_0_N + 1, net(states_0_N), atol=0.05))

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

        states_0_N = grid.torched(
            grid.make_grid(
                {
                    "a": {"min": 0, "max": 3, "count": 10},
                    "p": {"min": 0, "max": 1, "count": 4},
                }
            )
        )
        ## TODO: include big_t shocks in this -- for lifetime reward

        net = ann.PolicyNet(state_variables, pfblock.get_controls(), width=8)

        ### Trainiing
        ann.train_nn(net, states_0_N, edlrl, epochs=100)

        ## This is just a smoke test.
