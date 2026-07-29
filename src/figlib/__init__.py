"""figlib: a figure compiler for mathematical exposition.

Figures are programs (``figures/*.py``) with the contract in `program`;
``figcheck`` (see `cli`) runs one through render + gates. Read
``docs/skill.md`` before writing one.

There is no package façade: import from the module that owns the concern,
``from figlib.scene import Curve``. The map:

WRITING A FIGURE
    scene       typed primitives in math coords (Curve, MathLabel, Point,
                Vector, Brace, Callout, RasterField, ...) — the content model
    style       Role, Ink, WEIGHT_* stroke levels — semantics, not appearance
    theme       CLEAN | RISO; ink(Role), ramp, categorical, surface_shade
    format      MARGIN | COLUMN | WIDE page slots; canvas units are CSS px
    figure      Figure/Panel/Connector — multi-panel page grammar
    correspond  Correspondence — what a composite holds fixed; the residual
    geometry    compute-layer numerics that generate geometry
    builders    compute-side emitters (grids, streamlines, level ladders)
    plots       data plots as scene items: Axes, scales, series, bands
    schematic   Class B: typed Node/Port/Edge + a ranked layout pass
    matrix      matrices as Blocks drawn at their own aspect ratio;
                structure/value encoders, the expression row, shape gates
    surface3d   orthographic 3D -> depth-tagged 2D items; compose(), label3()
    sphere3d    sphere-specific producers for surface3d.compose()
    shading     chromatic light->shadow ramps in OKLCh

PIPELINE (rarely imported by a figure)
    program     the contract and run(); load_program, Report
    autoplace   deterministic free-nudge label solver, before the gates
    layout      math -> canvas transform (equal aspect, y-flip, margins)
    typeset     LaTeX -> SVG via ziamath, with exact metrics
    render      SVG emission + PNG rasterization; owns every bbox
    cli         figcheck

CHECKING
    gates       numerical / mechanical / color gates; Checks for multi-assert
    color       colorimetry that makes the color channel gateable
    expressivity  advisory under-expression signals (never fails a gate)
    regress     golden-figure sweep over the corpus
    readback    the cold-reader protocol: prompt_for(), record()
    report      textual scene inventory — layout debugging without pixels
"""

__version__ = "0.1.0"
