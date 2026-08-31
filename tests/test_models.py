import numpy as np

from skagent.distributions import Lognormal
import skagent.models.consumer as cons
import skagent.models.macid as macid
import skagent.models.perfect_foresight as pfm
import skagent.models.perfect_foresight_normalized as pfnm
from skagent.simulation.monte_carlo import Simulator


import unittest


class test_pfm(unittest.TestCase):
    def setUp(self):
        self.mcs = Simulator(
            pfm.calibration,
            pfm.block,
            {"c": lambda m: 0.5 * m},
            # danger: normalized decision rule for unnormalized problem
            {  # initial states
                "a": Lognormal(-6, 0),
                #'live' : 1,
                "p": 1.0,
            },
            sample_count=3,
            T_sim=120,
        )

    def test_simulate(self):
        ## smoke test
        self.mcs.initialize_sim()
        self.mcs.simulate()


class test_pfnm(unittest.TestCase):
    def setUp(self):
        self.mcs = Simulator(  ### Use fm, blockified
            pfnm.calibration,
            pfnm.block,
            {
                "c_nrm": lambda m_nrm: 0.5 * m_nrm
            },  # Simple consumption function for smoke test
            {  # initial states
                "a_nrm": Lognormal(-6, 0),
                #'live' : 1,
                "p": 1.0,
            },
            sample_count=3,
            T_sim=120,
        )

    def test_simulate(self):
        ## smoke test
        self.mcs.initialize_sim()
        self.mcs.simulate()


class test_consumer_models(unittest.TestCase):
    def setUp(self):
        self.cs = Simulator(
            cons.calibration,
            cons.cons_problem,
            {
                "c": lambda m: 0.5 * m,  # simple consumption function for smoke test
            },
            {  # initial states (normalized problem: p not needed)
                "k": Lognormal(-6, 0),
            },
            sample_count=2,
            T_sim=5,
        )

        self.pcs = Simulator(
            cons.calibration,
            cons.cons_portfolio_problem,
            {
                "c": lambda m: m / 2,
                "stigma": lambda a: a / (2 + a),  # dummy risky-share rule
            },
            {  # initial states (normalized problem: p not needed)
                "k": Lognormal(-6, 0),
                "R": 1.03,
            },
            sample_count=2,
            T_sim=5,
        )

        self.mcs = Simulator(
            cons.calibration,
            cons.mortal_cons_problem,
            {"c": lambda m: m / 3},
            {"k": Lognormal(-6, 0), "p": 1.0, "age": 0},
            sample_count=2,
            T_sim=5,
            seed=3,  # fixed: with these settings one agent dies at t=2
        )

    def test_simulate(self):
        self.cs.initialize_sim()
        self.cs.simulate()

        # R is a fixed calibration parameter for the non-portfolio model, so it
        # is never written into the simulated history.
        self.assertEqual(self.cs.calibration["R"], 1.03)
        self.assertFalse("R" in self.cs.history)

        # For the portfolio model R is produced dynamically, so it varies.
        self.pcs.initialize_sim()
        self.pcs.simulate()
        self.assertTrue(self.pcs.history["R"][0][0] != 1.03)

        # the portfolio simulation must not produce NaN anywhere in m
        self.assertFalse(np.any(np.isnan(self.pcs.history["m"])))

    def test_mortality_dynamics(self):
        """mortality_block resets dead agents to newborns and ages survivors."""
        self.mcs.initialize_sim()
        self.mcs.simulate()

        hist = self.mcs.history

        # The seed must actually produce at least one death, otherwise the reset
        # branch checked below is never exercised.
        self.assertTrue(
            (hist["live"] == 0).any(),
            "expected at least one death; reset path is otherwise untested",
        )

        for t in range(1, self.mcs.T_sim):
            for i in range(self.mcs.sample_count):
                if hist["live"][t][i] == 0:
                    # Death: age resets to 0 and k is a freshly drawn newborn
                    # endowment (k_init), not the surviving end-of-period assets.
                    self.assertAlmostEqual(hist["age"][t][i], 0.0, places=10)
                    self.assertAlmostEqual(
                        hist["k"][t][i], hist["k_init"][t][i], places=10
                    )
                    self.assertNotAlmostEqual(
                        hist["k"][t][i], hist["a"][t][i], places=10
                    )
                else:
                    # Survival: age advances by exactly one and end-of-period
                    # assets become next period's capital (k = a within period).
                    self.assertAlmostEqual(
                        hist["age"][t][i], hist["age"][t - 1][i] + 1.0, places=10
                    )
                    self.assertAlmostEqual(hist["k"][t][i], hist["a"][t][i], places=10)

    def test_mortality_frequency(self):
        """Empirical survival frequency tracks LivPrb across many agents."""
        sim = Simulator(
            cons.calibration,
            cons.mortal_cons_problem,
            {"c": lambda m: m / 3},
            {"k": Lognormal(-6, 0), "p": 1.0, "age": 0},
            sample_count=500,
            T_sim=20,
            seed=1,
        )
        sim.initialize_sim()
        sim.simulate()

        survival_rate = sim.history["live"].mean()
        self.assertAlmostEqual(survival_rate, cons.calibration["LivPrb"], delta=0.01)


class test_tree_killer(unittest.TestCase):
    def setUp(self):
        self.block = macid.tree_killer_block
        self.shocks = {
            sym: dist.draw(20_000)
            for sym, dist in self.block.construct_shocks(
                {}, rng=np.random.default_rng(1)
            ).items()
        }

    def alice_payoff(self, poison, doctor):
        """Alice's mean payoff when she builds the patio, given a poison
        intensity and Bob's (constant) tree-doctor intensity."""
        vals = self.block.transition(
            self.shocks,
            {
                "PT": lambda: poison,
                "TDoc": lambda TS: doctor,
                "BP": lambda PT, TDoc: 1.0,
            },
        )
        rewards = self.block.calc_reward(vals)
        owned = [v for sym, v in rewards.items() if self.block.reward[sym] == "alice"]
        return float(np.mean(sum(owned)))

    def test_simulate(self):
        """The mechanisms evaluate on the numpy path, and the chance nodes are
        binary as the source game requires."""
        vals = self.block.transition(
            self.shocks,
            {
                "PT": lambda: 0.5,
                "TDoc": lambda TS: 0.5,
                "BP": lambda PT, TDoc: 0.5,
            },
        )
        for sym in ["TS", "TDead"]:
            self.assertLessEqual(set(np.unique(vals[sym])), {0.0, 1.0})
        self.assertEqual(set(self.block.calc_reward(vals)), {"E", "V", "Tree", "Cost"})

    def test_poisoning_payoff_depends_on_the_doctor(self):
        """The payoffs are strategically non-degenerate: whether poisoning is
        worthwhile to Alice depends on how Bob plays."""
        self.assertGreater(self.alice_payoff(1.0, 0.0), self.alice_payoff(0.0, 0.0))
        self.assertLess(self.alice_payoff(1.0, 1.0), self.alice_payoff(0.0, 1.0))


class test_prisoners_dilemma(unittest.TestCase):
    def payoff(self, d1, d2):
        vals = macid.prisoners_dilemma_block.transition(
            {},
            {"D1": lambda: d1, "D2": lambda: d2},
        )
        return vals["U1"], vals["U2"]

    def test_agent_attributions(self):
        """Each control and utility belongs to the corresponding player."""
        self.assertEqual(
            macid.prisoners_dilemma_block.get_attributions(),
            {
                "player_1": ["D1", "U1"],
                "player_2": ["D2", "U2"],
            },
        )

    def test_standard_payoff_matrix(self):
        """The continuous formulation agrees with PD at its four corners."""
        expected = {
            (0.0, 0.0): (3.0, 3.0),
            (0.0, 1.0): (0.0, 5.0),
            (1.0, 0.0): (5.0, 0.0),
            (1.0, 1.0): (1.0, 1.0),
        }
        for actions, payoffs in expected.items():
            with self.subTest(actions=actions):
                self.assertEqual(self.payoff(*actions), payoffs)

    def test_player_swap_parity(self):
        """Swapping the players swaps their utilities, including in the interior."""
        u1, u2 = self.payoff(0.25, 0.75)
        swapped_u1, swapped_u2 = self.payoff(0.75, 0.25)
        self.assertAlmostEqual(u1, swapped_u2)
        self.assertAlmostEqual(u2, swapped_u1)


class test_iterated_prisoners_dilemma(unittest.TestCase):
    def test_multi_agent_structure(self):
        """Both players observe the last round and own one control and utility."""
        block = macid.iterated_prisoners_dilemma_block
        self.assertEqual(
            block.get_attributions(),
            {
                "player_1": ["D1", "U1"],
                "player_2": ["D2", "U2"],
            },
        )
        self.assertEqual(
            block.get_dynamics()["D1"].iset,
            ["previous_D1", "previous_D2"],
        )
        self.assertEqual(
            block.get_dynamics()["D2"].iset,
            ["previous_D1", "previous_D2"],
        )

    def test_previous_actions_are_arrival_state(self):
        """A new round arrives with both actions from the preceding round."""
        self.assertEqual(
            macid.iterated_prisoners_dilemma_block.get_arrival_states(),
            {"previous_D1", "previous_D2"},
        )

    def test_tit_for_tat_carries_actions_between_rounds(self):
        """Each player can copy the other player's action from the last round."""
        tit_for_tat = {
            "D1": lambda previous_D1, previous_D2: previous_D2,
            "D2": lambda previous_D1, previous_D2: previous_D1,
        }
        first = macid.iterated_prisoners_dilemma_block.transition(
            {"previous_D1": 0.0, "previous_D2": 1.0},
            tit_for_tat,
        )
        self.assertEqual(
            (first["D1"], first["D2"], first["U1"], first["U2"]),
            (1.0, 0.0, 5.0, 0.0),
        )
        self.assertEqual(
            (first["previous_D1"], first["previous_D2"]),
            (1.0, 0.0),
        )

        second = macid.iterated_prisoners_dilemma_block.transition(first, tit_for_tat)
        self.assertEqual(
            (second["D1"], second["D2"], second["U1"], second["U2"]),
            (0.0, 1.0, 0.0, 5.0),
        )
        self.assertEqual(
            (second["previous_D1"], second["previous_D2"]),
            (0.0, 1.0),
        )

    def test_payoffs_match_one_shot_model(self):
        """Repeating the game changes its state, not its per-round payoffs."""
        for d1, d2 in [(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)]:
            one_shot = macid.prisoners_dilemma_block.transition(
                {},
                {"D1": lambda d1=d1: d1, "D2": lambda d2=d2: d2},
            )
            repeated = macid.iterated_prisoners_dilemma_block.transition(
                {"previous_D1": 0.0, "previous_D2": 0.0},
                {
                    "D1": lambda previous_D1, previous_D2, d1=d1: d1,
                    "D2": lambda previous_D1, previous_D2, d2=d2: d2,
                },
            )
            with self.subTest(actions=(d1, d2)):
                self.assertEqual(
                    (repeated["U1"], repeated["U2"]),
                    (one_shot["U1"], one_shot["U2"]),
                )

    def test_tit_for_tat_runs_in_simulator(self):
        """The simulator carries the updated actions through multiple rounds."""
        sim = Simulator(
            {},
            macid.iterated_prisoners_dilemma_block,
            {
                "D1": lambda previous_D1, previous_D2: previous_D2,
                "D2": lambda previous_D1, previous_D2: previous_D1,
            },
            {"previous_D1": 0.0, "previous_D2": 1.0},
            agent_count=2,
            T_sim=4,
        )
        sim.initialize_sim()
        history = sim.simulate()

        np.testing.assert_array_equal(
            history["D1"], [[1.0, 1.0], [0.0, 0.0], [1.0, 1.0], [0.0, 0.0]]
        )
        np.testing.assert_array_equal(
            history["D2"], [[0.0, 0.0], [1.0, 1.0], [0.0, 0.0], [1.0, 1.0]]
        )
        np.testing.assert_array_equal(
            history["U1"], [[5.0, 5.0], [0.0, 0.0], [5.0, 5.0], [0.0, 0.0]]
        )
        np.testing.assert_array_equal(
            history["U2"], [[0.0, 0.0], [5.0, 5.0], [0.0, 0.0], [5.0, 5.0]]
        )

    def test_tit_for_tat_sustains_mutual_cooperation(self):
        """Two cooperative Tit-for-Tat players keep cooperating each round."""
        sim = Simulator(
            {},
            macid.iterated_prisoners_dilemma_block,
            {
                "D1": lambda previous_D1, previous_D2: previous_D2,
                "D2": lambda previous_D1, previous_D2: previous_D1,
            },
            {"previous_D1": 0.0, "previous_D2": 0.0},
            agent_count=2,
            T_sim=4,
        )
        sim.initialize_sim()
        history = sim.simulate()

        np.testing.assert_array_equal(history["D1"], np.zeros((4, 2)))
        np.testing.assert_array_equal(history["D2"], np.zeros((4, 2)))
        np.testing.assert_array_equal(history["U1"], np.full((4, 2), 3.0))
        np.testing.assert_array_equal(history["U2"], np.full((4, 2), 3.0))
