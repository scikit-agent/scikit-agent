import unittest
import pydot


class test_pydot(unittest.TestCase):
    def test_stuff(self):
        graph = pydot.Dot("my_graph", graph_type="graph", bgcolor="yellow")

        # Add nodes
        my_node = pydot.Node("a", label="Foo")
        graph.add_node(my_node)

        output_graphviz_svg = graph.create_svg()

        assert output_graphviz_svg is not None
