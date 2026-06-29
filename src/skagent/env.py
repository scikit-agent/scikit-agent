from skagent.bellman import BellmanPeriod
from skagent.simulation.monte_carlo import draw_shocks
import numpy as np
import torch


class Environment:
    def __init__(
        self, bp: BellmanPeriod, initial, rng: np.random.Generator | None = None
    ):
        self.bp = bp
        self.initial = initial
        self.rng = rng

        # this is the saved state of the environment
        # it corresponds to arrival states of the BP
        self.state = None
        self.reset()

    def reset(self):
        # TODO fix
        initial_vals = draw_shocks(
            self.initial,
            [0],  # only one agent
            rng=self.rng,
        )

        # ok this is pretty annoying
        initial_vals = {sym: torch.Tensor(initial_vals[sym]) for sym in initial_vals}

        self.state = initial_vals

        return initial_vals

    def step(self, decision_rule):
        # todo: better handling/internalizing this type shift
        shocks = {k: torch.from_numpy(v) for k, v in self.bp.draw_shocks(n=1).items()}

        post = self.bp.forward_function(
            self.state, shocks, {}, decision_rules=decision_rule
        )

        state_t = self.state
        action = {csym: post[csym] for csym in decision_rule}
        reward = {
            rsym: post[rsym]
            for rsym in self.bp.block.reward
            # if agent is None or self.block.reward[rsym] == agent # TODO deal with multiple agents
        }
        discount = self.bp.resolve_discount_factor(post)
        state_t_plus = {
            a_sym: post[a_sym].detach()
            if isinstance(post[a_sym], torch.Tensor)
            else post[a_sym]
            for a_sym in self.bp.get_arrival_states()
        }

        # info-set for each control (what the actor sees)
        control_sym = next(iter(decision_rule))
        iset = self.bp.block.dynamics[control_sym].iset
        obs = {
            sym: post[sym].detach()
            if isinstance(post[sym], torch.Tensor)
            else post[sym]
            for sym in iset
        }

        self.state = state_t_plus

        return state_t, action, reward, state_t_plus, discount, obs
