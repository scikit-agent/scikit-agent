from HARK.distributions import Normal
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


test_block_with_shock = DBlock(
    **{
        "name": "test with shock",
        "shocks": {
            "theta": (Normal, {"mean": 0, "sigma": 1}),
        },
        "dynamics": {
            "b": lambda a, theta: a + theta,
            "c": Control(["b"], agent="consumer"),
            "u": lambda b, c: -((b - c) ** 2),
            "a": lambda b: b,
        },
        "reward": {"u": "consumer"},
    }
)


class test_ann(unittest.TestCase):
    def test_basic(self):
        state_variables = ["a"]

        ### Loss function
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

        net = ann.PolicyNet(state_variables, {}, test_block.get_controls(), width=16)

        ### Trainiing
        ann.train_nn(net, states_0_N, edlrl, epochs=200)

        self.assertTrue(torch.allclose(states_0_N + 1, net(states_0_N), atol=0.05))

    def test_basic_with_shock(self):
        states_0 = {
            "a": 0,
        }
        state_variables = list(states_0.keys())

        ### Loss function

        big_t = 1

        # TODO : have this take some number of random shocks
        edlrl = ann.get_estimated_discounted_lifetime_reward_loss(
            state_variables,
            test_block_with_shock,
            0.9,
            big_t,
            parameters=test_calibration,
        )

        ### Setting up the training
        states_0_N = grid.torched(
            grid.make_grid(
                {
                    "a": {"min": 0, "max": 4, "count": 21},
                    "theta": {
                        "min": -1,
                        "max": 1,
                        "count": 7,
                    },  # Ideally this is an equiprobable discretization
                    # there is only one of these now because big_t = 1.
                }
            )
        )

        net = ann.PolicyNet(
            state_variables,
            test_block_with_shock.get_shocks(),
            test_block_with_shock.get_controls(),
            width=16,
        )

        ### Trainiing
        ann.train_nn(net, states_0_N, edlrl, epochs=200)

        self.assertTrue(torch.allclose(states_0_N, net(states_0_N), atol=0.05))

    def test_lifetime_reward_perfect_foresight(self):
        ### Model data

        pfblock = pfm.block_no_shock
        state_variables = ["a", "p"]

        ### Loss function
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

        net = ann.PolicyNet(state_variables, {}, pfblock.get_controls(), width=8)

        ### Trainiing
        ann.train_nn(net, states_0_N, edlrl, epochs=100)

        ## This is just a smoke test.
