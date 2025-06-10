from conftest import case_0, case_1, case_2, case_3
import skagent.algos.maliar as maliar
import skagent.ann as ann
import skagent.grid as grid
import skagent.models.perfect_foresight as pfm
import torch
import unittest


torch.manual_seed(10077696)
# np.random.seed(seed_value)

## CUDA handling
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(torch.cuda.is_available())


class test_ann_lr(unittest.TestCase):
    def setUp(self):
        pass

    def test_case_0(self):
        edlrl = maliar.get_estimated_discounted_lifetime_reward_loss(
            ["a"],
            case_0["block"],
            0.9,
            1,
            parameters=case_0["calibration"],
        )

        states_0_N = case_0["givens"]

        bpn = ann.BlockPolicyNet(case_0["block"], width=16)
        ann.train_block_policy_nn(bpn, states_0_N, edlrl, epochs=250)

        c_ann = bpn.decision_function(states_0_N.to_dict(), {}, {})["c"]

        print(c_ann)

        # Is this result stochastic? How are the network weights being initialized?
        self.assertTrue(
            torch.allclose(c_ann, torch.zeros(c_ann.shape).to(device), atol=0.0015)
        )

    def test_case_1(self):
        edlrl = maliar.get_estimated_discounted_lifetime_reward_loss(
            ["a"],
            case_1["block"],
            0.9,
            1,
            parameters=case_1["calibration"],
        )

        given_0_N = case_1["givens"][1]

        bpn = ann.BlockPolicyNet(case_1["block"], width=16)
        ann.train_block_policy_nn(bpn, given_0_N, edlrl, epochs=350)

        c_ann = bpn.decision_function(
            # TODO -- make this from the Grid
            {"a": given_0_N["a"]},
            {"theta": given_0_N["theta_0"]},
            {},
        )["c"]

        errors = c_ann.flatten() - given_0_N.to_dict()["theta_0"]

        # Is this result stochastic? How are the network weights being initialized?
        self.assertTrue(
            torch.allclose(errors, torch.zeros(errors.shape).to(device), atol=0.015)
        )

    def test_case_1_2(self):
        """
        Running case 1 with big_t == 2
        """
        edlrl = maliar.get_estimated_discounted_lifetime_reward_loss(
            ["a"],
            case_1["block"],
            0.9,
            2,
            parameters=case_1["calibration"],
        )

        given_0_N = case_1["givens"][2]

        bpn = ann.BlockPolicyNet(case_1["block"], width=16)
        ann.train_block_policy_nn(bpn, given_0_N, edlrl, epochs=200)

        c_ann = bpn.decision_function(
            {"a": given_0_N["a"]},
            {"theta": given_0_N["theta_0"]},
            {},
        )["c"]

        errors = c_ann.flatten() - given_0_N["theta_0"]

        print(errors)
        # Is this result stochastic? How are the network weights being initialized?
        self.assertTrue(
            torch.allclose(errors, torch.zeros(errors.shape).to(device), atol=0.03)
        )

    def test_case_2(self):
        edlrl = maliar.get_estimated_discounted_lifetime_reward_loss(
            ["a"],
            case_2["block"],
            0.9,
            1,
            parameters=case_2["calibration"],
        )

        given_0_N = case_2["givens"]

        bpn = ann.BlockPolicyNet(case_2["block"], width=8)
        ann.train_block_policy_nn(bpn, given_0_N, edlrl, epochs=100)

        # optimal DR is c = 0 = E[theta]

        # Just a smoke test. Since the information set to the control
        # actually gives no information, training isn't effective...

    def test_case_3(self):
        edlrl = maliar.get_estimated_discounted_lifetime_reward_loss(
            ["a"],
            case_3["block"],
            0.9,
            1,
            parameters=case_3["calibration"],
        )

        given_0_N = case_3["givens"][1]

        bpn = ann.BlockPolicyNet(case_3["block"], width=8)
        ann.train_block_policy_nn(bpn, given_0_N, edlrl, epochs=300)

        c_ann = bpn.decision_function(
            {"a": given_0_N["a"]},
            {
                "theta": given_0_N["theta_0"],
                "psi": given_0_N["psi_0"],
            },
            {},
        )["c"]
        given_m = given_0_N["a"] + given_0_N["theta_0"]

        torch.allclose(c_ann.flatten(), given_m.flatten(), atol=0.03)

    def test_case_3_2(self):
        edlrl = maliar.get_estimated_discounted_lifetime_reward_loss(
            ["a"],
            case_3["block"],
            0.9,
            2,
            parameters=case_3["calibration"],
        )

        given_0_N = case_3["givens"][2]

        bpn = ann.BlockPolicyNet(case_3["block"], width=8)
        ann.train_block_policy_nn(bpn, given_0_N, edlrl, epochs=300)

        c_ann = bpn.decision_function(
            {"a": given_0_N["a"]},
            {
                "theta": given_0_N["theta_0"],
                "psi": given_0_N["psi_0"],
            },
            {},
        )["c"]
        given_m = given_0_N["a"] + given_0_N["theta_0"]

        torch.allclose(c_ann.flatten(), given_m.flatten(), atol=0.04)

    def test_lifetime_reward_perfect_foresight(self):
        ### Model data

        pfblock = pfm.block_no_shock
        state_variables = ["a", "p"]

        ### Loss function
        edlrl = maliar.get_estimated_discounted_lifetime_reward_loss(
            state_variables, pfblock, 0.9, 1, parameters=pfm.calibration
        )

        ### Setting up the training

        states_0_N = grid.Grid.from_config(
            {
                "a": {"min": 0, "max": 3, "count": 5},
                "p": {"min": 0, "max": 1, "count": 4},
            }
        )

        bpn = ann.BlockPolicyNet(pfblock, width=8)
        ann.train_block_policy_nn(bpn, states_0_N, edlrl, epochs=100)
        ## This is just a smoke test.
