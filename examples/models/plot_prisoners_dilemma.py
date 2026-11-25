r"""
##################################
The One-Shot Prisoner's Dilemma
##################################

Two prisoners choose simultaneously whether to cooperate with each other or
defect. Neither sees the other's choice before acting. Cooperation produces a
good joint outcome, but each player can improve their own payoff by defecting,
whatever the other player does.

We encode cooperate as action ``0`` and defect as action ``1``. The payoff
matrix, with player 1's payoff first, is

.. list-table::
   :header-rows: 1

   * -
     - Player 2 cooperates
     - Player 2 defects
   * - Player 1 cooperates
     - :math:`(3, 3)`
     - :math:`(0, 5)`
   * - Player 1 defects
     - :math:`(5, 0)`
     - :math:`(1, 1)`

The first part of this example solves the game. The second part executes two
chosen randomized policies, illustrating simulation rather than equilibrium
computation.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

from skagent.algos.tabular import TabularBestResponseSolver
from skagent.block import Control, DBlock
from skagent.ground import GroundedBlock
from skagent.simulation.monte_carlo import Simulator
from skagent.solver import solve_in_relevance_order


def prisoners_dilemma(*, randomize=False):
    """Build the one-shot game, optionally with randomized controls."""
    return DBlock(
        name="randomized_prisoners_dilemma" if randomize else "prisoners_dilemma",
        dynamics={
            "D1": Control(
                [], agent="player_1", action_space={0, 1}, randomize=randomize
            ),
            "D2": Control(
                [], agent="player_2", action_space={0, 1}, randomize=randomize
            ),
            "U1": lambda D1, D2: 3.0 + 2.0 * D1 - 3.0 * D2 - D1 * D2,
            "U2": lambda D1, D2: 3.0 - 3.0 * D1 + 2.0 * D2 - D1 * D2,
        },
        reward={"U1": "player_1", "U2": "player_2"},
    )


ACTIONS = {0: "cooperate", 1: "defect"}

# %%
# Solving the game
# =================
#
# A decision is **dominant** when it is best regardless of the other player's
# action. If player 2 cooperates, player 1 receives 5 by defecting instead of 3
# by cooperating. If player 2 defects, player 1 receives 1 by defecting instead
# of 0 by cooperating. Player 2 faces the same comparison, so defection is
# dominant for both players.
#
# ``TabularBestResponseSolver`` evaluates the two actions declared by each
# control. The game's structural relevance graph places the decisions in the
# same cyclic component, so ``solve_in_relevance_order`` iterates their best
# responses until the policies stop changing. The game has no shocks; the small
# shock sample only supplies the solver's common sample axis.

game = prisoners_dilemma()
solver = TabularBestResponseSolver(
    GroundedBlock(game, {}),
    shock_samples=16,
    rng=np.random.default_rng(0),
)
policies = solve_in_relevance_order(solver)

solved_actions = {symbol: int(rule()) for symbol, rule in policies.items()}
for symbol in ("D1", "D2"):
    print(f"{symbol}: {ACTIONS[solved_actions[symbol]]}")

outcome = game.transition({}, policies)
payoffs = (int(outcome["U1"]), int(outcome["U2"]))
print(f"payoffs: {payoffs}")

# %%
# The solver returns ``defect`` for both players and the payoff pair ``(1, 1)``.
# This is the one-shot game's pure-strategy Nash equilibrium: neither player can
# gain by changing their action alone. Both would receive 3 from mutual
# cooperation, but that outcome is not stable because either one could defect
# and receive 5.

# %%
# Executing randomized policies
# =============================
#
# A randomized control gives its policy a private uniform draw between zero and
# one. We now build a separate copy of the game and supply policies that turn
# those draws into legal actions. Player 1 defects with probability 0.35 and
# player 2 with probability 0.65.

randomized_game = prisoners_dilemma(randomize=True)
defection_probabilities = {"D1": 0.35, "D2": 0.65}
randomized_policies = {
    symbol: (lambda noise, probability=probability: (noise < probability).astype(int))
    for symbol, probability in defection_probabilities.items()
}

simulation = Simulator(
    {},
    randomized_game,
    randomized_policies,
    {},
    seed=7,
    sample_count=1_000,
    T_sim=1,
)
simulation.initialize_sim()
history = simulation.simulate()

counts = {}
for symbol in ("D1", "D2"):
    actions = np.asarray(history[symbol]).ravel()
    counts[symbol] = np.bincount(actions.astype(int), minlength=2)
    print(f"{symbol}: cooperate={counts[symbol][0]}, defect={counts[symbol][1]}")

# %%
# Both action categories are shown for each player. Their heights are empirical
# frequencies, so they lie near the probabilities supplied above.

labels = ["Cooperate", "Defect"]
x = np.arange(len(labels))
width = 0.34

fig, ax = plt.subplots(figsize=(6, 3.5))
for offset, (symbol, color) in zip(
    (-width / 2, width / 2),
    (("D1", "#0072B2"), ("D2", "#D55E00")),
):
    ax.bar(
        x + offset,
        counts[symbol] / 1_000,
        width,
        label=f"Player {symbol[-1]}",
        color=color,
    )

ax.set_xticks(x, labels)
ax.set_ylim(0, 1)
ax.set_ylabel("Empirical frequency")
ax.set_title("One round under supplied randomized policies")
ax.legend()
fig.tight_layout()

# %%
# These frequencies are not a mixed-strategy equilibrium computed by the
# solver. The first section solves the one-shot game and finds its
# pure-strategy equilibrium. This section instead executes policies whose
# defection probabilities we chose in advance.
