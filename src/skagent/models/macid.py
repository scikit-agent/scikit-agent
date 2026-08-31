"""
Multi-agent influence diagram (MAID) illustration models.

These are game-theoretic influence diagrams from the literature, encoded as
scikit-agent blocks to illustrate and exercise strategic-relevance analysis
(``Block.relevance_graph`` / ``Block.relies_on``). Unlike the consumption-saving
models in ``benchmarks.py``, these are games rather than dynamic programs: what
they pin down is graphical structure (information sets, agent ownership,
dependencies), and their functional forms and payoff magnitudes are illustrative.
The magnitudes are nonetheless chosen so that each game is strategically
non-degenerate -- no decision has an optimum that is independent of the others --
so that a solver exercises the structure rather than sidestepping it. Games whose
relevance graph is acyclic can be solved by
``skagent.algos.best_response.TabularBestResponseSolver``; a cyclic one needs a joint
equilibrium solution, which the library does not offer.

Encoding conventions (a deliberate departure from the source presentations):

- Chance nodes are given in structural-causal form rather than as conditional
  probability distributions P(node | parents): each chance node is a
  deterministic mechanism of its endogenous parents plus an explicit exogenous
  noise variable (a shock). This is equivalent in distribution (any CPD can be
  written as a function of its parents plus independent noise) but makes the
  noise a first-class graph node, matching scikit-agent's shock/dynamics
  vocabulary. Because the noise nodes are single-child exogenous roots, they
  cannot lie on any d-connecting path and so do not change the relevance graph.
- Binary decisions are relaxed to continuous ``[0, 1]`` controls, pending
  discrete-action support. For Prisoner's Dilemma, ``0`` means cooperate and
  ``1`` means defect; intermediate values use the multilinear extension of the
  standard payoff matrix and can be interpreted as defection probabilities.
  The iterated model's utilities are per-round payoffs; accumulation or
  discounting across rounds belongs to the simulator or solver using the block.
"""

from skagent.block import Control, DBlock
from skagent.distributions import Uniform

# Tree Killer (Koller & Milch 2001, Fig. 1)
# -----------------------------------------
# Alice considers poisoning her neighbour Bob's tree (PT) to improve the view
# from a patio she is deciding whether to build (BP); Bob observes whether the
# tree is sick (TS) and decides whether to call a tree doctor (TDoc). The
# relevance graph (KM Fig. 4a) is PT -> BP, PT -> TDoc, BP -> TDoc: Alice's
# poison decision relies on both other decisions, her patio decision relies on
# Bob's, and Bob's tree-doctor decision relies on nothing.
#
# Chance nodes TS and TDead use the structural-causal form (deterministic
# mechanism + exogenous shock) described in the module docstring; the tree
# becomes sick as a parent-conditioned Bernoulli via the inverse-CDF trick on a
# Uniform noise u_TS, and likewise TDead. Utility nodes are deterministic
# functions of their parents (as influence diagrams require).
#
# Dynamics are listed in topological order so no within-period dependency is
# mistaken for an arrival-state (lag) edge. Structure matches PyCID's
# story_macids.tree_doctor, the cross-check oracle.
tree_killer_block = DBlock(
    **{
        "name": "tree_killer",
        "shocks": {
            # noise driving the tree-sick and tree-death CPDs
            "u_TS": (Uniform, {"low": 0.0, "high": 1.0}),
            "u_TDead": (Uniform, {"low": 0.0, "high": 1.0}),
        },
        # Decisions are binary in the original game (poison or not, call the
        # doctor or not, build or not). scikit-agent has no discrete-action
        # support yet, so each is modelled as a continuous [0, 1] relaxation
        # (read as an intensity / probability of the action). The bound
        # functions take the control's information set as positional arguments,
        # per the Control convention, though the bounds here are constant.
        # Bounds do not affect relevance analysis.
        "dynamics": {
            "PT": Control(
                [],
                lower_bound=lambda: 0.0,
                upper_bound=lambda: 1.0,
                agent="alice",
            ),  # poison tree
            # P(sick) rises with poisoning: Bernoulli via inverse-CDF on u_TS.
            "TS": lambda PT, u_TS: (u_TS < 0.1 + 0.7 * PT) * 1.0,
            "TDoc": Control(
                ["TS"],
                lower_bound=lambda TS: 0.0,
                upper_bound=lambda TS: 1.0,
                agent="bob",
            ),  # call tree doctor
            # P(death) rises with sickness, falls if the doctor is called.
            "TDead": lambda TS, TDoc, u_TDead: (u_TDead < 0.1 + 0.7 * TS - 0.5 * TDoc)
            * 1.0,
            "BP": Control(
                ["PT", "TDoc"],
                lower_bound=lambda PT, TDoc: 0.0,
                upper_bound=lambda PT, TDoc: 1.0,
                agent="alice",
            ),  # build patio
            # Payoffs. Alice pays for the poison and values a patio with an
            # unobstructed view, so the dead tree is what makes building worth
            # the construction cost; Bob values the tree and pays the doctor.
            # The magnitudes are chosen so that no decision is optimal
            # independently of the others: poisoning pays only if the tree
            # actually dies, building pays only if the view is clear, and the
            # doctor is worth calling only for a sick tree.
            # Alice's poison expense
            "E": lambda PT: -PT,
            # Alice's patio: the view is worth 3.0 with the tree dead, and
            # building costs 0.5.
            "V": lambda TDead, BP: BP * (3.0 * TDead - 0.5),
            # Bob's tree-health utility
            "Tree": lambda TDead: -TDead,
            # Bob's doctor fee
            "Cost": lambda TDoc: -0.2 * TDoc,
        },
        # TODO(roadmap: multi-reward): each agent has an additively decomposed
        # utility (Alice: E + V; Bob: Tree + Cost), which is the intended syntax
        # for multiple reward variables per agent. The relevance machinery
        # aggregates these correctly, but the single-agent solver path currently
        # assumes one reward variable per block (see Block
        # get_state_rule_value_function_from_continuation). Handling additive
        # multi-utility in the solver is future roadmap work.
        "reward": {
            "E": "alice",
            "V": "alice",
            "Tree": "bob",
            "Cost": "bob",
        },
    }
)


# Prisoner's Dilemma
# ------------------
def _player_1_utility(D1, D2):
    return 3.0 + 2.0 * D1 - 3.0 * D2 - D1 * D2


def _player_2_utility(D1, D2):
    return 3.0 - 3.0 * D1 + 2.0 * D2 - D1 * D2


prisoners_dilemma_block = DBlock(
    **{
        "name": "prisoners_dilemma",
        "dynamics": {
            "D1": Control(
                [],
                lower_bound=lambda: 0.0,
                upper_bound=lambda: 1.0,
                agent="player_1",
            ),
            "D2": Control(
                [],
                lower_bound=lambda: 0.0,
                upper_bound=lambda: 1.0,
                agent="player_2",
            ),
            "U1": _player_1_utility,
            "U2": _player_2_utility,
        },
        "reward": {
            "U1": "player_1",
            "U2": "player_2",
        },
    }
)


iterated_prisoners_dilemma_block = DBlock(
    **{
        "name": "iterated_prisoners_dilemma",
        "dynamics": {
            # The players see the preceding round, but neither sees the other
            # player's current action before choosing their own.
            "D1": Control(
                ["previous_D1", "previous_D2"],
                lower_bound=lambda previous_D1, previous_D2: 0.0,
                upper_bound=lambda previous_D1, previous_D2: 1.0,
                agent="player_1",
            ),
            "D2": Control(
                ["previous_D1", "previous_D2"],
                lower_bound=lambda previous_D1, previous_D2: 0.0,
                upper_bound=lambda previous_D1, previous_D2: 1.0,
                agent="player_2",
            ),
            "U1": _player_1_utility,
            "U2": _player_2_utility,
            # Today's actions become the state observed in the next round.
            "previous_D1": lambda D1: D1,
            "previous_D2": lambda D2: D2,
        },
        "reward": {
            "U1": "player_1",
            "U2": "player_2",
        },
    }
)
