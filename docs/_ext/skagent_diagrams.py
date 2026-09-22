"""The user guide's model diagrams, and the staleness they would otherwise have.

Two hazards with one cause: a change to how a diagram is drawn is invisible to
anything holding a drawing already made. sphinx-gallery re-executes an example
when its SOURCE changes, so a library-only change leaves every cached figure
alone -- the build stays clean and the pictures go old -- and a figure written
into the source tree is not redrawn at all.

So the guide's figures are drawn on every build from the shipped models, and the
gallery's cache is dropped when the code that draws a diagram has changed. To
add a figure to the guide, add a row to ``DIAGRAMS`` and reference the filename
from the page; nothing else here needs touching.
"""

from __future__ import annotations

import hashlib
import importlib
import shutil
from dataclasses import dataclass
from pathlib import Path

DOCS = Path(__file__).resolve().parent.parent
SKAGENT = DOCS.parent / "src" / "skagent"

IMAGES = DOCS / "user_guide" / "images"
GALLERY = DOCS / "auto_examples"
FINGERPRINT = GALLERY / ".drawing-fingerprint"

#: The files that decide how a diagram is drawn. Fingerprinting these rather
#: than the whole library is deliberate: a whole-library key would re-run the
#: gallery on any change at all, minutes per build, for a hazard that only the
#: drawing path creates.
DRAWING_SOURCES = (
    SKAGENT / "model_visualizer.py",
    SKAGENT / "model_analyzer.py",
    SKAGENT / "model_visualization_config.yaml",
)


@dataclass(frozen=True)
class Diagram:
    """One figure in the guide: which shipped block to draw, and what it shows.

    Attributes
    ----------
    filename
        Written to ``docs/user_guide/images/``, and what the page references.
    module, block
        Import path of the module, and the block's name within it.
    shows
        Why this model and not another. Not rendered; it is the reason the row
        exists, kept beside the row so a later editor can weigh a replacement.
    calibration
        Name of a module attribute holding the calibration. A callable is
        called with no arguments, which is how a model that builds its
        calibration is reached without putting code in this table. ``None``
        passes an empty calibration.
    discount
        The calibration symbol serving as the discount factor, drawn as a
        hexagon. A block does not know which of its parameters this is.
    """

    filename: str
    module: str
    block: str
    shows: str
    calibration: str | None = None
    discount: str | None = None

    def draw(self) -> str:
        """Render this diagram to SVG."""
        module = importlib.import_module(self.module)
        calibration = {}
        if self.calibration is not None:
            calibration = getattr(module, self.calibration)
            if callable(calibration):
                calibration = calibration()

        svg = (
            getattr(module, self.block)
            .visualize(calibration, title="", discount=self.discount)
            .create_graph()
            .create_svg()
        )
        return svg.decode("utf-8") if isinstance(svg, bytes) else svg


DIAGRAMS = (
    Diagram(
        filename="diagram-shapes.svg",
        module="skagent.models.consumer",
        block="consumption_block",
        calibration="calibration",
        discount="DiscFac",
        shows="the only shipped block that puts every shape and both arrival "
        "values in one picture",
    ),
    Diagram(
        filename="diagram-plate.svg",
        module="skagent.models.cournot",
        block="cournot_block",
        calibration="collusion_calibration",
        shows="the smallest entity class with an aggregation leaving it",
    ),
    Diagram(
        filename="diagram-maid.svg",
        module="skagent.models.macid",
        block="tree_killer_block",
        shows="the smallest two-agent game, so the per-agent coloring is legible",
    ),
)


def draw_diagrams(app) -> None:
    """Redraw the guide's figures, so they cannot disagree with the library."""
    IMAGES.mkdir(parents=True, exist_ok=True)
    for diagram in DIAGRAMS:
        svg = diagram.draw()
        target = IMAGES / diagram.filename
        # Only write on a change, so an unchanged figure does not re-trigger
        # everything downstream that watches its timestamp.
        if not target.is_file() or target.read_text() != svg:
            target.write_text(svg)


def _drawing_fingerprint() -> str:
    digest = hashlib.sha256()
    for path in DRAWING_SOURCES:
        digest.update(path.read_bytes())
    return digest.hexdigest()


def clear_gallery_if_drawing_changed(app) -> None:
    """Drop the gallery's cache when the drawing code has changed under it."""
    current = _drawing_fingerprint()

    if GALLERY.is_dir():
        previous = FINGERPRINT.read_text().strip() if FINGERPRINT.is_file() else None
        if previous == current:
            return
        shutil.rmtree(GALLERY)

    GALLERY.mkdir(parents=True, exist_ok=True)
    FINGERPRINT.write_text(current)


def setup(app):
    app.connect("builder-inited", clear_gallery_if_drawing_changed)
    app.connect("builder-inited", draw_diagrams)
    return {"parallel_read_safe": True, "parallel_write_safe": True}
