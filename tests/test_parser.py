import os
import unittest

import skagent.block as model
import skagent.parser as parser
import yaml

from skagent.distributions import Bernoulli, MeanOneLogNormal
from skagent.rule import extract_dependencies


CONSUMER_YAML_PATH = os.path.join(
    os.path.dirname(__file__), "../src/skagent/models/consumer.yaml"
)

MALFORMED_BLOCK_PATH = os.path.join(
    os.path.dirname(__file__), "data/malformed_portfolio_block.yaml"
)


def load_consumer_config():
    with open(CONSUMER_YAML_PATH, "r") as f:
        return yaml.load(f, Loader=parser.skagent_loader())


class test_consumption_parsing(unittest.TestCase):
    def setUp(self):
        self.config = load_consumer_config()

    def test_parse(self):
        config = self.config

        self.assertEqual(config["calibration"]["DiscFac"], 0.96)
        self.assertEqual(config["blocks"][0]["name"], "consumption normalized")

        ## construct and test the consumption block
        cons_norm_block = model.DBlock(**config["blocks"][0])
        cons_norm_block.discretize(
            {"theta": {"N": 5}}, calibration=config["calibration"]
        )
        self.assertEqual(cons_norm_block.calc_reward({"c": 1, "CRRA": 2})["u"], -1.0)

        ## construct and test the portfolio block
        portfolio_block = model.DBlock(**config["blocks"][1])
        # ``construct_shocks`` returns the resolved shocks rather than storing
        # them, so the calibration goes to ``discretize`` directly.
        portfolio_block.discretize(
            {"risky_return": {"N": 5}}, calibration=config["calibration"]
        )

    def test_the_portfolio_block_has_its_dynamics(self):
        """The second block declares a share and a return, not shocks alone."""
        portfolio_block = model.DBlock(**self.config["blocks"][1])

        self.assertEqual(list(portfolio_block.get_shocks()), ["risky_return"])
        self.assertEqual(list(portfolio_block.get_dynamics()), ["stigma", "R"])

        stigma = portfolio_block.get_controls()["stigma"]
        self.assertIsInstance(stigma, model.Control)
        self.assertEqual(stigma.iset, ["a"])

        self.assertEqual(
            sorted(extract_dependencies(portfolio_block.get_dynamics()["R"])),
            ["Rfree", "risky_return", "stigma"],
        )

    def test_control_tag(self):
        """`!Control` produces a Control, not a token."""
        block = model.DBlock(**self.config["blocks"][0])

        c = block.get_controls()["c"]
        self.assertIsInstance(c, model.Control)
        self.assertEqual(c.iset, ["m"])
        self.assertEqual(c.agent, "consumer")

        # A bound declared as an expression reads its arguments from the iset.
        self.assertEqual(c.upper_bound(m=3.0), 3.0)
        self.assertIsNone(c.lower_bound)

    def test_control_tag_rejects_bad_declarations(self):
        """A malformed control is refused at parse time rather than dropped."""
        with self.assertRaises(ValueError):
            yaml.load("c: !Control {agent: consumer}", Loader=parser.skagent_loader())

        with self.assertRaises(ValueError):
            yaml.load(
                "c: !Control {iset: m, infoset: m}", Loader=parser.skagent_loader()
            )


class test_malformed_document(unittest.TestCase):
    """What a block whose keys are not a block's keys parses into today.

    The fixture is the ``portfolio choice`` block as it was written before its
    repair: ``dynamics`` one level too deep, and so a shock. Nothing refuses
    it, which is what this pins -- a document that loses its own dynamics is
    the case a validator is for, and the test says what it will have to catch.
    """

    def setUp(self):
        with open(MALFORMED_BLOCK_PATH, "r") as f:
            self.document = yaml.load(f, Loader=parser.skagent_loader())

    def test_the_dynamics_arrive_as_a_shock(self):
        self.assertIn("dynamics", self.document["shocks"])
        self.assertNotIn("dynamics", self.document)

    def test_the_block_it_builds_has_no_dynamics(self):
        block = model.DBlock(**self.document)

        self.assertEqual(block.get_dynamics(), {})
        self.assertEqual(block.get_controls(), {})


class test_authoring_equivalence(unittest.TestCase):
    """A document and a Python block describing the same model agree."""

    def setUp(self):
        config = load_consumer_config()
        self.calibration = config["calibration"]
        self.documented = model.DBlock(**config["blocks"][0])
        self.written = model.DBlock(
            name="consumption normalized",
            shocks={
                "live": (Bernoulli, {"p": "LivPrb"}),
                "theta": (MeanOneLogNormal, {"sigma": "TranShkStd"}),
            },
            dynamics={
                "b": lambda k, R, PermGroFac: k * R / PermGroFac,
                "m": lambda b, theta: b + theta,
                "c": model.Control(["m"], upper_bound=lambda m: m, agent="consumer"),
                "a": lambda m, c: m - c,
                "u": lambda c, CRRA: c ** (1 - CRRA) / (1 - CRRA),
            },
            reward={"u": "consumer"},
        )

    def test_controls_agree(self):
        documented = self.documented.get_controls()
        written = self.written.get_controls()

        self.assertEqual(list(documented), list(written))
        for sym, control in written.items():
            self.assertEqual(documented[sym].iset, control.iset)
            self.assertEqual(documented[sym].agent, control.agent)

    def test_shocks_agree(self):
        self.assertEqual(
            list(self.documented.get_shocks()), list(self.written.get_shocks())
        )

    def test_reward_attribution_agrees(self):
        self.assertEqual(self.documented.reward, self.written.reward)

    def test_graph_edges_agree(self):
        for sym, rule in self.written.get_dynamics().items():
            self.assertEqual(
                sorted(extract_dependencies(self.documented.get_dynamics()[sym])),
                sorted(extract_dependencies(rule)),
                f"'{sym}' has different dependencies in the document than in Python",
            )

    def test_arrival_states_agree(self):
        self.assertEqual(
            self.documented.get_arrival_states(calibration=self.calibration),
            self.written.get_arrival_states(calibration=self.calibration),
        )
