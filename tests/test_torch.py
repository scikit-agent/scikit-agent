import unittest

import numpy as np
import skagent.model as model
from skagent.model import Control
import torch

torch.manual_seed(10077691)

sf = torch.nn.Sigmoid()

test_block_data = {
    "name": "test block - torch",
    # "shocks": {
    #    "live": Bernoulli(p=LivPrb),
    # },
    "dynamics": {
        "m": lambda a, q: a * q,
        "c": Control(["m"], upper_bound="a"),
        "a": lambda m, c, e: m * (1 - sf(c)) + e,
        #'e' : lambda e : e,
        "u": lambda c, CRRA: c ** (1 - CRRA) / (1 - CRRA),
    },
    # "reward": {"u" : "consumer"},
}

parameters = {"e": 1, "q": 1.1, "CRRA": 1}


class test_torch_equations(unittest.TestCase):
    def setUp(self):
        self.test_block = model.DBlock(**test_block_data)

    def test_stuff(self):
        pre = {"a": torch.FloatTensor(np.linspace(1, 21, 10))}
        pre.update(parameters)

        dr = {"c": lambda m: m - 10}
        self.assertEqual(self.test_block.name, "test block - torch")

        post = model.simulate_dynamics(self.test_block.dynamics, pre, dr)

        t1 = post["a"][0]
        t2 = torch.FloatTensor([2.0999])[0]
        print(
            f"t1: {t1}, dtype: {t1.dtype}, device: {t1.device}, requires_grad: {t1.requires_grad}"
        )
        print(
            f"t2: {t2}, dtype: {t2.dtype}, device: {t2.device}, requires_grad: {t2.requires_grad}"
        )

        self.assertTrue(torch.allclose(t1, t2, rtol=1e-4, atol=1e-8))
