import unittest

import numpy as np
import skagent.grid as grid
import torch


class test_make_grid(unittest.TestCase):
    def test_simple(self):
        simple = {"a": {"max": 1, "min": 0, "count": 10}}

        g = grid.make_grid(simple)

        self.assertEqual(g.size, 10)

    def test_double(self):
        double = {
            "a": {"max": 1, "min": 0, "count": 10},
            "b": {"max": 1, "min": 0, "count": 10},
        }

        g = grid.make_grid(double)

        self.assertEqual(g.shape, (100, 2))

    def test_triple(self):
        triple = {
            "a": {"max": 1, "min": 0, "count": 5},
            "b": {"max": 1, "min": 0, "count": 5},
            "c": {"max": 1, "min": 0, "count": 5},
        }

        g = grid.make_grid(triple)

        self.assertEqual(g.shape, (125, 3))


class test_grid_alt_constructors(unittest.TestCase):
    def setUp(self):
        self.config = {
            "a": {"max": 1, "min": 0, "count": 5},
            "b": {"max": 1, "min": 0, "count": 3},
        }

        self.dict_a_np = {"a": np.array([0.25, 0.5, 0.75])}

        self.dict_a_torch = {"a": torch.FloatTensor(np.array([0.25, 0.5, 0.75]))}

        self.dict_b_np = {"b": np.array([0, 0.5, 1])}

        self.dict_b_torch = {"b": torch.FloatTensor(np.array([0, 0.5, 1]))}

        self.dict_b_np_2d = {"b": np.array([[0, 0.5, 1]])}

    def test_from_config(self):
        g = grid.Grid.from_config(self.config)

        self.assertEqual(g.len(), 2)
        self.assertEqual(g.n(), 15)

    def test_from_dict_numpy(self):
        g = grid.Grid.from_dict(self.dict_b_np)

        self.assertEqual(g.len(), 1)
        self.assertEqual(g.n(), 3)

    def test_from_dict_torch(self):
        g = grid.Grid.from_dict(self.dict_b_torch)

        self.assertEqual(g.len(), 1)
        self.assertEqual(g.n(), 3)

    def test_from_dict_mixed_np_first(self):
        kv = {**self.dict_a_np, **self.dict_b_torch}

        g = grid.Grid.from_dict(kv)

        self.assertEqual(g.len(), 2)
        self.assertEqual(g.n(), 3)

    def test_from_dict_mixed_torch_first(self):
        kv = {**self.dict_a_torch, **self.dict_b_np}

        g = grid.Grid.from_dict(kv)

        self.assertEqual(g.len(), 2)
        self.assertEqual(g.n(), 3)

    def test_update_from_dict_np_first(self):
        g = grid.Grid.from_dict(self.dict_a_np)
        g2 = g.update_from_dict(self.dict_b_torch)

        self.assertEqual(g2.len(), 2)
        self.assertEqual(g2.n(), 3)

    def test_update_from_dict_torch_first(self):
        g = grid.Grid.from_dict(self.dict_a_torch)
        g2 = g.update_from_dict(self.dict_b_np)

        self.assertEqual(g2.len(), 2)
        self.assertEqual(g2.n(), 3)

    def test_update_from_dict_np_reshape(self):
        g = grid.Grid.from_dict(self.dict_a_torch)
        g2 = g.update_from_dict(self.dict_b_np_2d)

        self.assertEqual(g2.len(), 2)
        self.assertEqual(g2.n(), 3)
