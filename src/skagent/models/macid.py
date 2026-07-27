"""
Multi-agent influence diagram (MAID) illustration models.

These are game-theoretic influence diagrams from the literature, encoded as
scikit-agent blocks to illustrate and exercise strategic-relevance analysis
(``Block.relevance_graph`` / ``Block.relies_on``). Unlike the consumption-saving
models in ``benchmarks.py``, they are not solved for a policy -- scikit-agent
has no equilibrium solver yet -- so only their graphical structure (information
sets, agent ownership, dependencies) is meaningful; the functional forms and
probabilities/payoffs are illustrative.

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
  discrete-action support.
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
            "u_TS": Uniform(0.0, 1.0),  # noise driving the tree-sick CPD
            "u_TDead": Uniform(0.0, 1.0),  # noise driving the tree-death CPD
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
            "TS": lambda PT, u_TS: (u_TS < 0.1 + 0.7 * PT).float(),
            "TDoc": Control(
                ["TS"],
                lower_bound=lambda TS: 0.0,
                upper_bound=lambda TS: 1.0,
                agent="bob",
            ),  # call tree doctor
            # P(death) rises with sickness, falls if the doctor is called.
            "TDead": lambda TS, TDoc, u_TDead: (
                u_TDead < 0.1 + 0.7 * TS - 0.5 * TDoc
            ).float(),
            "BP": Control(
                ["PT", "TDoc"],
                lower_bound=lambda PT, TDoc: 0.0,
                upper_bound=lambda PT, TDoc: 1.0,
                agent="alice",
            ),  # build patio
            "E": lambda PT: -PT,  # Alice's poisoning-effort/expense utility
            "V": lambda TDead, BP: BP * (1.0 - TDead),  # Alice's view utility
            "Tree": lambda TDead: -TDead,  # Bob's tree-health utility
            "Cost": lambda TDoc: -TDoc,  # Bob's doctor-cost utility
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
