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
