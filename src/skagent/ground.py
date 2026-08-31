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
from typing import TYPE_CHECKING, Any

import numpy as np

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

    block: Block
    calibration: dict[str, Any]
    rng: np.random.Generator | None

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
            from skagent.simulation.monte_carlo import _set_rng_recursive

            self._shocks = self.block.construct_shocks(self.calibration, rng=self.rng)
            if self.rng is not None:
                # ``construct_shocks`` injects the generator into the
                # constructor, which reaches a shock declared as a
                # ``(class, arguments)`` pair and not one declared as a
                # distribution INSTANCE. It deep-copies either way, so these
                # are this instance's own distributions and seeding them here
                # leaves the block's alone.
                for distribution in self._shocks.values():
                    _set_rng_recursive(distribution, self.rng)
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
