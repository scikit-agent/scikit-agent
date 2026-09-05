"""
Test cases for ModelVisualizer to ensure it correctly processes
ModelAnalyzer output and generates valid graph diagrams.
"""

import copy
import sys
import unittest

sys.path.append("../src")

from skagent.block import DBlock
from skagent.distributions import Bernoulli
from skagent.model_analyzer import ModelAnalyzer
from skagent.model_visualizer import ModelVisualizer


class TestModelVisualizerCore(unittest.TestCase):
    """Test core ModelVisualizer functionality."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a simple test block
        self.simple_block = DBlock(
            name="simple",
            shocks={"eps": (Bernoulli, {"p": 0.5})},
            dynamics={
                "x": lambda eps: eps,
                "y": lambda x, param: x + param,
            },
            reward={"y": "agent1"},
        )

        self.calibration = {"param": 1.0}

        # Generate analysis
        analyzer = ModelAnalyzer(self.simple_block, self.calibration)
        analyzer.analyze()
        self.analysis_dict = analyzer.to_dict()

    def test_initialization_and_colors(self):
        """
        Verify initialization: meta/edges/plates are directly mapped,
        and each agent generates a 7-character hex color.
        """
        viz = ModelVisualizer(self.analysis_dict)

        # Basic attribute mapping
        self.assertEqual(viz.meta, self.analysis_dict["node_meta"])
        self.assertEqual(viz.edges, self.analysis_dict["edges"])
        self.assertEqual(viz.plates, self.analysis_dict["plates"])

        # Color table: global/other → default_other_color, other agents → #xxxxxx
        for agent, color in viz.agent_colors.items():
            self.assertIsInstance(color, str)
            self.assertTrue(color.startswith("#"))
            self.assertEqual(len(color), 7)

    def test_make_node_styling(self):
        """
        _make_node applies _node_shape based on kind,
        and fills every kind that is drawn filled.
        """
        viz = ModelVisualizer(self.analysis_dict)

        for name, meta in self.analysis_dict["node_meta"].items():
            node = viz._make_node(name)

            # Node is cached
            self.assertIn(name, viz.nodes)

            attrs = node.get_attributes()

            # Shape matches kind
            self.assertEqual(attrs["shape"], viz._node_shape(meta["kind"]))

            if meta["kind"] == "param":
                # A parameter is unbordered and unfilled: it is the calibration
                # showing through rather than a variable of the model.
                self.assertNotIn("filled", attrs["style"])
                self.assertNotIn("fillcolor", attrs)
            else:
                self.assertEqual(attrs["fillcolor"], viz.agent_colors[meta["agent"]])
                self.assertIn("filled", attrs["style"])

    def test_a_shock_is_a_double_bordered_ellipse(self):
        """The shape vocabulary distinguishes a draw from a computed value."""
        viz = ModelVisualizer(self.analysis_dict)

        shock = viz._make_node("eps").get_attributes()
        state = viz._make_node("x").get_attributes()

        self.assertEqual(shock["shape"], "ellipse")
        self.assertEqual(shock["peripheries"], "2")
        self.assertEqual(state["shape"], "ellipse")
        self.assertNotIn("peripheries", state)

    def test_an_unread_parameter_is_not_drawn(self):
        """A parameter no equation reads is dropped; a read one is kept."""
        analysis = copy.deepcopy(self.analysis_dict)
        analysis["node_meta"]["unused"] = {
            "kind": "param",
            "agent": "global",
            "plate": None,
            "observed": False,
        }

        viz = ModelVisualizer(analysis)
        viz.create_graph()

        self.assertNotIn("unused", viz.nodes)
        self.assertIn("param", viz.nodes)  # read by y = x + param

    def test_a_discount_factor_is_drawn_even_when_unread(self):
        """No equation reads a discount factor, and it is still the problem's."""
        analysis = copy.deepcopy(self.analysis_dict)
        analysis["node_meta"]["beta"] = {
            "kind": "discount",
            "agent": "global",
            "plate": None,
            "observed": False,
        }

        viz = ModelVisualizer(analysis)
        viz.create_graph()

        self.assertIn("beta", viz.nodes)
        self.assertEqual(viz.nodes["beta"].get_attributes()["shape"], "hexagon")

    def test_an_empty_title_leaves_the_graph_unlabelled(self):
        """So that a caller drawing its own caption does not draw two."""
        self.assertIsNone(
            ModelVisualizer(self.analysis_dict, title="").create_graph().get_label()
        )
        self.assertEqual(
            ModelVisualizer(self.analysis_dict, title="Model")
            .create_graph()
            .get_label(),
            "Model",
        )

    def test_lag_variable_styling(self):
        """
        Lag variables (ending with *) should apply previous_period style.
        """
        # Create analysis with lag edge
        analysis = {
            "node_meta": {
                "x": {
                    "kind": "state",
                    "agent": "global",
                    "plate": None,
                    "observed": False,
                },
                "x*": {
                    "kind": "state",
                    "agent": "global",
                    "plate": None,
                    "observed": False,
                },
            },
            "edges": {"instant": [], "param": [], "shock": [], "lag": [("x", "x")]},
            "plates": {},
            "formulas": {},
        }

        viz = ModelVisualizer(analysis)

        # Current period node
        node_current = viz._make_node("x")
        attrs_current = node_current.get_attributes()

        # Lag period node (with *)
        node_lag = viz._make_node("x*")
        attrs_lag = node_lag.get_attributes()

        # Lag node should have different styling (previous_period applied)
        # This assumes previous_period style exists in config
        self.assertNotEqual(attrs_current, attrs_lag)

    def test_create_graph_nodes(self):
        """
        create_graph creates nodes for each original variable,
        and generates prev-period nodes (name*) for lag edges.
        """
        viz = ModelVisualizer(self.analysis_dict)
        viz.create_graph()  # Don't save graph - only interested in side effects

        # Original nodes
        expected = set(self.analysis_dict["node_meta"].keys())

        # Lag edges produce prev-period nodes
        for src, tgt in self.analysis_dict["edges"]["lag"]:
            expected.add(f"{src}*")

        # viz.nodes keys should match
        self.assertEqual(set(viz.nodes.keys()), expected)

    def test_edge_counts_and_styles(self):
        """
        create_graph treats instant/param/shock as current_period,
        lag as previous_period, and applies styles from config.
        """
        viz = ModelVisualizer(self.analysis_dict)
        graph = viz.create_graph()
        edges = graph.get_edges()

        total_expected = sum(
            len(self.analysis_dict["edges"][k])
            for k in ["instant", "param", "shock", "lag"]
        )
        self.assertEqual(len(edges), total_expected)

        # Each edge should have at least one styling attribute
        for e in edges:
            attrs = e.get_attributes()
            self.assertTrue(any(key in attrs for key in ("color", "style", "penwidth")))

    def test_color_determinism(self):
        """
        Multiple instantiations of ModelVisualizer with the same analysis
        produce consistent agent_colors (due to fixed seed).
        """
        viz1 = ModelVisualizer(self.analysis_dict)
        viz2 = ModelVisualizer(dict(self.analysis_dict))

        self.assertEqual(viz1.agent_colors, viz2.agent_colors)


class TestDiscountFactor(unittest.TestCase):
    """A block does not know which parameter is its discount factor."""

    def test_visualize_takes_the_discount_symbol(self):
        block = DBlock(
            name="discounted",
            dynamics={"x": lambda param: param},
            reward={"x": "agent1"},
        )
        calibration = {"param": 1.0, "beta": 0.96}

        plain = block.visualize(calibration).create_graph()
        named = block.visualize(calibration, discount="beta").create_graph()

        # Unnamed, beta is an unread parameter and is not drawn at all. Named,
        # it is drawn, and as the one shape reserved for it.
        self.assertNotIn("beta", [n.get_name() for n in plain.get_nodes()])
        (beta,) = [n for n in named.get_nodes() if n.get_name() == "beta"]
        self.assertEqual(beta.get_attributes()["shape"], "hexagon")


class TestModelVisualizerPlates(unittest.TestCase):
    """Test plate/subgraph functionality."""

    def test_plate_subgraphs_and_assignment(self):
        """
        Manually construct multi-plate analysis,
        ensure create_graph generates cluster_{agent} subgraphs,
        and places nodes in correct plates.
        """
        analysis = {
            "node_meta": {
                "x": {"kind": "state", "agent": "A", "plate": "A", "observed": False},
                "y": {"kind": "state", "agent": "B", "plate": "B", "observed": False},
                "z": {
                    "kind": "state",
                    "agent": "global",
                    "plate": None,
                    "observed": False,
                },
            },
            "edges": {"instant": [], "param": [], "shock": [], "lag": []},
            "plates": {
                "A": {"label": "AgentA", "size": "N_A"},
                "B": {"label": "AgentB", "size": "N_B"},
            },
            "formulas": {},
        }

        viz = ModelVisualizer(analysis)
        graph = viz.create_graph()

        subgraphs = {sg.get_name(): sg for sg in graph.get_subgraphs()}

        # Should contain cluster_A and cluster_B
        self.assertIn("cluster_A", subgraphs)
        self.assertIn("cluster_B", subgraphs)

        # They should contain corresponding nodes x and y
        names_A = {n.get_name() for n in subgraphs["cluster_A"].get_nodes()}
        names_B = {n.get_name() for n in subgraphs["cluster_B"].get_nodes()}

        self.assertEqual(names_A, {"x"})
        self.assertEqual(names_B, {"y"})


class TestModelVisualizerComplex(unittest.TestCase):
    """Test complex scenarios with multiple node and edge types."""

    def test_complex_manual_graph(self):
        """
        Comprehensive test: shock/state/control/reward/param +
        instant/param/shock/lag edges + plates.
        Verify it exports valid dot code.
        """
        analysis = {
            "node_meta": {
                "e": {"kind": "shock", "agent": "H", "plate": "H", "observed": False},
                "k": {"kind": "state", "agent": "F", "plate": "F", "observed": False},
                "c": {"kind": "control", "agent": "H", "plate": "H", "observed": True},
                "u": {"kind": "reward", "agent": "H", "plate": "H", "observed": True},
                "a": {
                    "kind": "param",
                    "agent": "global",
                    "plate": None,
                    "observed": False,
                },
            },
            "edges": {
                "instant": [("e", "c"), ("k", "u")],
                "param": [("a", "k")],
                "shock": [("e", "k")],
                "lag": [("k", "k")],
            },
            "plates": {
                "H": {"label": "House", "size": "N_H"},
                "F": {"label": "Firm", "size": "N_F"},
            },
            "formulas": {},
        }

        viz = ModelVisualizer(analysis)
        graph = viz.create_graph()

        # Node count: 5 original + 1 self-lag prev-period
        self.assertEqual(len(viz.nodes), 6)

        # Subgraph count: 2
        self.assertEqual(len(graph.get_subgraphs()), 2)

        # Edge count: instant(2) + param(1) + shock(1) + lag(1) = 5
        self.assertEqual(len(graph.get_edges()), 5)

        # Verify dot output
        dot_string = graph.to_string()
        self.assertTrue(dot_string.startswith("digraph"))
        self.assertIn("cluster_H", dot_string)
        self.assertIn("cluster_F", dot_string)
