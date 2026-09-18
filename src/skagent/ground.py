"""
A block paired with the calibration it is read against.

A block declares dynamics, shocks and rewards without committing to values for
the symbols they refer to, which is what lets one block stand for the same model
at many calibrations. The consequence is that most questions about a model are
questions about a block *and* a calibration, and the two travel together through
every solver, simulator and environment in the library.
:class:`GroundedBlock` is the pair.
"""

from __future__ import annotations

import copy
import numbers
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from skagent.distributions import set_rng

if TYPE_CHECKING:
    from skagent.block import Block


class GroundedBlock:
    """A block together with the calibration and generator it is read against.

    Owns the resolution of the block's shock declarations into distributions:
    a shock declared as a ``(class, arguments)`` pair needs a calibration to
    resolve, so the resolved distributions belong to this pair rather than to
    the block.

    A calibration is fixed before the model is solved or simulated, so the pair
    resolves every shock whose arguments refer to calibrated symbols and no
    others. A shock argument referring to a value known only during a solve or
    a run is outside what this pair holds and raises; such a shock is resolved
    by the caller that has the value, against a scope overlaying it on the
    calibration.

    Parameters
    ----------
    block : Block
        The model's dynamics, shocks and rewards.
    calibration : dict[str, Any]
        Values for the symbols the block's declarations and dynamics refer to.
    rng : numpy.random.Generator, optional
        Generator this instance's shocks are drawn from, however they were
        declared. Two instances over one block hold separate distributions, so
        each draws its own path.

    Attributes
    ----------
    block : Block
        The underlying block model.
    calibration : dict[str, Any]
        The calibration parameters.
    rng : numpy.random.Generator | None
        The generator this instance's shock draws come from.
    """

    def __init__(
        self,
        block: Block,
        calibration: dict[str, Any],
        rng: np.random.Generator | None = None,
    ) -> None:
        self.block = block
        self.calibration = calibration
        self.rng = rng
        self._shocks: dict[str, Any] | None = None

    def shock_distributions(self) -> dict[str, Any]:
        """This instance's shocks, resolved against its calibration.

        The block declares shocks; resolving a declaration needs a calibration,
        which is what this class supplies. Resolved once and held, so that the
        generator advances across draws instead of restarting, and so that the
        block itself is left as its author wrote it. Every resolved shock draws
        from ``rng``, whether it was declared as a ``(class, arguments)`` pair
        or as a distribution instance.

        Returns
        -------
        dict[str, Distribution]

        Raises
        ------
        KeyError
            If a shock's arguments refer to a symbol the calibration does not
            assign.
        """
        if self._shocks is None:
            self._shocks = self.block.construct_shocks(self.calibration, rng=self.rng)
            if self.rng is not None:
                # ``construct_shocks`` injects the generator into the
                # constructor, which reaches a shock declared as a
                # ``(class, arguments)`` pair and not one declared as a
                # distribution INSTANCE. It deep-copies either way, so these
                # are this instance's own distributions and seeding them here
                # leaves the block's alone.
                for distribution in self._shocks.values():
                    set_rng(distribution, self.rng)
        return self._shocks

    def with_rng(self, rng: np.random.Generator | None) -> GroundedBlock:
        """A copy of this pair drawing from *rng* instead.

        A new instance rather than a repointed one, so that a holder currently
        drawing from this pair keeps its own path: the copy resolves its shocks
        afresh on first access and therefore shares no distribution with the
        original. The block, the calibration and anything a subclass adds are
        carried over unchanged -- a generator is a different sample of one
        model, not a different model.

        Parameters
        ----------
        rng : numpy.random.Generator or None
            Generator the copy's shocks draw from.

        Returns
        -------
        GroundedBlock
            Of the same type as *self*.
        """
        other = copy.copy(self)
        other.rng = rng
        other._shocks = None
        return other

    def expected_payoff(self, policies, measure, *, states=None, agent=None):
        """What a policy profile is worth, under *measure*.

        Runs the block under *policies* and takes the expectation of the
        rewards *agent* owns. The block's period is taken as it stands: this is
        the payoff of one pass through it, with no continuation.

        Parameters
        ----------
        policies : Mapping[str, Callable]
            A decision rule for every control of the block. This is the profile
            the expectation is taken under, so a rule left out is a symbol the
            block cannot compute.
        measure : Measure
            The reduction that turns the block's shocks into one number. There
            is no default: a sampled estimate and a discretized one carry
            different error, so a caller stating a tolerance has to have chosen
            which it is stating it about.
        states : Mapping[str, Any], optional
            Values for the symbols the block reads on arrival. A block with
            arrival states cannot be run without them.
        agent : str, optional
            Whose payoff to take. Omitted, every reward symbol in the block is
            summed, which is one agent's payoff only where the block has one
            agent.

        Returns
        -------
        ExpectedPayoff
            The expectation, beside the axis that was reduced to reach it.
        """

        owners = set(self.block.reward.values())
        if agent is not None and agent not in owners:
            raise ValueError(
                f"no reward in this block is attributed to agent {agent!r}, so "
                f"its payoff is an empty sum rather than zero; the agents paid "
                f"here are {sorted(owners)}"
            )

        def integrand(shock_values):
            pre = {**self.calibration, **(states or {}), **shock_values}
            vals = self.block.transition(pre, policies)
            return sum(self.block.calc_reward(vals, agent=agent).values())

        return measure.reduce(self, integrand)

    def draw_shocks(self, n: int) -> dict[str, Any]:
        """Draw *n* realizations of each of this instance's shocks.

        Parameters
        ----------
        n : int
            Number of realizations per shock.

        Returns
        -------
        dict[str, Any]
            A mapping from shock symbol to its draws.
        """
        from skagent.simulation.monte_carlo import draw_shocks

        return draw_shocks(self.shock_distributions(), n=n)


@dataclass(frozen=True)
class ExpectedPayoff:
    """An expected payoff, and the axis the expectation was taken over.

    The two reductions do not carry the same kind of error -- one falls with
    the number of draws and the other is a property of the rule the nodes came
    from -- so a number that has lost which axis produced it cannot be held to
    a tolerance. ``value`` is the expectation, ``axis`` is what was reduced to
    reach it (``"samples"`` for draws of the block's shocks, ``"nodes"`` for a
    discretization of them), and ``size`` is how many points of that axis it
    rests on. Reading it as a plain number is ``float(result)``.
    """

    value: Any
    axis: str
    size: int

    def __float__(self) -> float:
        return float(self.value)


class Measure(ABC):
    """How an expectation over a block's shocks is taken.

    A measure reduces one axis and says which: a subclass draws the shocks and
    reduces the draws, or discretizes them and reduces the nodes. What it
    reduces over is its own configuration -- a sample count, a table of
    discretization arguments -- so a caller passes a measure rather than the
    union of every measure's arguments, and a third reduction arrives as a
    class rather than as another branch.

    The two do not carry the same error, which is why there is a type here
    rather than a flag: sampling error falls with the number of draws, and a
    discretization's error is a property of the rule its nodes came from.
    """

    @abstractmethod
    def reduce(self, ground: GroundedBlock, integrand) -> ExpectedPayoff:
        """Reduce *integrand* over this measure's axis.

        Parameters
        ----------
        ground : GroundedBlock
            The pair whose shocks are being integrated. A measure reads its
            resolved distributions and, where it draws, its generator.
        integrand : Callable
            Takes a mapping from shock symbol to value -- one point of the
            axis, or the whole axis at once where the measure is vectorized --
            and returns what is being averaged.

        Returns
        -------
        ExpectedPayoff
        """


class Sampled(Measure):
    """Reduce the sample axis: the mean over *n* draws of the block's shocks.

    Parameters
    ----------
    n : int
        Realizations to draw. The estimate carries sampling error, which falls
        as *n* rises.
    rng : numpy.random.Generator, optional
        Generator to draw from, in place of the one the pair carries. Two
        reductions run against generators in the same state see the same
        draws, which is what makes two profiles comparable rather than
        separated by the noise of their own draws.
    """

    def __init__(self, n: int, *, rng: np.random.Generator | None = None) -> None:
        if not isinstance(n, numbers.Integral) or isinstance(n, bool) or n < 1:
            raise ValueError(f"n must be a positive integer, got {n!r}")
        self.n = int(n)
        self.rng = rng

    def reduce(self, ground: GroundedBlock, integrand) -> ExpectedPayoff:
        """The mean of *integrand* over this measure's draws of *ground*."""
        pair = ground if self.rng is None else ground.with_rng(self.rng)
        payoffs = integrand(pair.draw_shocks(self.n))
        mean = payoffs.mean() if hasattr(payoffs, "mean") else np.mean(payoffs)
        return ExpectedPayoff(mean, "samples", self.n)


class Discretized(Measure):
    """Reduce the nodes of a discretization, against their own weights.

    Parameters
    ----------
    disc_params : Mapping[str, dict], optional
        Arguments to ``Distribution.discretize``, per shock. A shock named here
        is discretized with those arguments; a continuous shock left out is
        discretized with its own defaults; a shock already discrete is taken as
        it stands.
    """

    def __init__(self, disc_params: dict[str, dict] | None = None) -> None:
        self.disc_params = dict(disc_params or {})

    def reduce(self, ground: GroundedBlock, integrand) -> ExpectedPayoff:
        """The weighted sum of *integrand* over the discretization's nodes."""
        from skagent.block import discretized_shock_dstn
        from skagent.distributions import DiscreteDistribution, expected

        shocks = ground.shock_distributions()
        if not shocks:
            # Nothing to integrate: the payoff of a block with no shocks is
            # already its own expectation, and there is no node to name.
            return ExpectedPayoff(integrand({}), "nodes", 1)

        # Every continuous shock is discretized, whether or not the caller
        # named it: the helper's default for an unnamed shock is to take it as
        # already discrete, which for a continuous one integrates against
        # whatever its raw form happens to expose.
        params = {
            sym: self.disc_params.get(sym, {})
            for sym, distribution in shocks.items()
            if sym in self.disc_params
            or not isinstance(distribution, DiscreteDistribution)
        }
        nodes = discretized_shock_dstn(shocks, params)
        names = list(nodes.var_names)

        def at_node(point):
            # One shock reaches the integrand as a bare value and several as a
            # mapping, so the node is named here rather than in the integrand.
            values = (
                {names[0]: point}
                if len(names) == 1 and not isinstance(point, dict)
                else {name: point[name] for name in names}
            )
            return integrand(values)

        return ExpectedPayoff(
            expected(func=at_node, dist=nodes), "nodes", len(nodes.weights)
        )
