import unittest

import skagent.grid as grid


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
