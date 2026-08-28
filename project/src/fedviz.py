"""fedviz — every figure, table and check for project 7.

Scenario-neutral. Call `use(scenario)` once and every figure takes that scenario's
words and colours; nothing here knows what the participants are.

House figure style (2026 redesign): white ground, one sans face, a big
plain-language title with a short roman brief under it, nothing else above the plot.
No rules, no boxes around axes, no colorbars: cells and bars are direct-labelled,
identity comes from one fixed colour per participant, and every conclusion sits in a
rounded takeaway band with a violet accent bar. Numbers are large because Colab
downscales rendered figures.

Colour system (validated with the palette checker, light surface):
  four hues   one per participant, supplied by the scenario, never reassigned
  violet      the federation itself: the shared model, takeaways
  charcoal    the pooled reference (training that is not allowed to exist)
  red/green   reserved for bad/good, never used as a series colour

Everything that is not a matplotlib figure is rendered as HTML, so figures, tables and
the cap explorer in 6.1 all survive a saved notebook with no ipywidgets dependency. The
explorer pre-renders one image per cap setting and switches between them with radio
inputs and CSS, because ipywidgets buttons stop working the moment the kernel they were
made by is gone, and a student reopening the notebook should not lose them.
"""

from __future__ import annotations

import os
import textwrap
import urllib.request
from html import escape as _escape

import numpy as np
import matplotlib as mpl
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, to_rgb
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.transforms import blended_transform_factory

import fedcore as fc
from fedcore import Vocab

try:
    from IPython.display import HTML, display
except ImportError:                                    # running outside a notebook
    HTML = None
    def display(x):                                    # noqa: D103
        print(x)

# ----------------------------------------------------------------- design tokens

INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"          # hairline gridlines
BASELINE = "#c3c2b7"
PANEL = "#f6f5f2"         # card / callout ground
TRACK = "#efeeea"         # the unfilled part of a share bar
NEUTRAL = "#d9d7d0"       # de-emphasis fill
CHARCOAL = "#3a3935"      # the pooled reference

ACCENT = "#4a3aa7"        # the federation: shared model, takeaways
GOOD = "#0ca30c"
GOOD_TEXT = "#006300"
BAD = "#d03b3b"
FLAG = "#9a6b00"          # worth-a-look amber, dark enough to read on white

#: What y == 1 is drawn in, everywhere. Red is reserved for it and nothing else.
ILL = "#d03b3b"

#: One colour per participant, set by `use()`. Never reassigned mid-notebook: identity
#: has to survive a reader flicking between two figures.
SITE_COLOUR = {}
SHORT_NAME = {}
FEATURES = []
READABLE = {}
POOLED_COLOUR = CHARCOAL

#: The scenario's words. Replaced by `use()`; this default only keeps imports working.
V = Vocab(site="site", sites="sites", member="record", members="records",
          positive="positive", positive_rate="positive rate", setting="consortium",
          outcome_question="?", negative="the rest",
          record="record", records="records")


def use(scenario):
    """Point every figure at one scenario: its words, its colours, its short names.

        import scenario, fedviz as fv
        fv.use(scenario)

    Everything downstream reads `fv.V` and `fv.SITE_COLOUR`, so a notebook switches
    datasets by changing this one line.
    """
    global V, SITE_COLOUR, SHORT_NAME, QUIZZES, FEATURES, READABLE, SPLITS
    V = scenario.VOCAB
    SITE_COLOUR = dict(scenario.SITE_COLOUR)
    SHORT_NAME = dict(getattr(scenario, "SHORT_NAME", {}))
    FEATURES = list(getattr(scenario, "FEATURES", []))
    READABLE = dict(getattr(scenario, "READABLE", {}))
    SPLITS = list(getattr(scenario, "SPLITS", DEFAULT_SPLITS))
    try:
        import importlib
        QUIZZES = dict(importlib.import_module("quizzes").QUIZZES)
    except (ImportError, AttributeError):
        QUIZZES = dict(getattr(scenario, "QUIZZES", {}))
    return V

#: Ordinal blues for "more of the same knob" series (local steps, noise levels).
BLUE_STEPS = ["#86b6ef", "#2a78d6", "#104281"]

#: Sequential ramp for magnitude (the accuracy grid).
SEQ = LinearSegmentedColormap.from_list(
    "fv_seq", ["#e4effc", "#9ec5f4", "#5598e7", "#256abf", "#0d366b"])

#: Diverging ramp for better/worse than a reference (the drift grid).
DIV = LinearSegmentedColormap.from_list(
    "fv_div", [(0.0, "#1c5cab"), (0.36, "#8fbcf0"), (0.5, "#f0efec"),
               (0.64, "#f2a48f"), (1.0, "#c22f2f")])

#: Only fonts bundled with matplotlib, so a Colab render matches a local one exactly.
SANS = ["DejaVu Sans"]
MONO = ["DejaVu Sans Mono"]

_STYLED = False


def _style():
    global _STYLED
    if _STYLED:
        return
    mpl.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "font.family": "sans-serif", "font.sans-serif": SANS,
        "mathtext.fontset": "dejavusans",
        "text.color": INK, "axes.labelcolor": INK_2, "axes.edgecolor": BASELINE,
        "xtick.color": INK_2, "ytick.color": INK_2,
        "axes.titlesize": 12.5, "axes.labelsize": 11,
        "xtick.labelsize": 10.5, "ytick.labelsize": 11,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.spines.left": False,
        "xtick.major.size": 0, "ytick.major.size": 0,
        "xtick.minor.size": 0, "ytick.minor.size": 0,
        "grid.color": GRID, "grid.linestyle": "-", "grid.linewidth": 0.9,
        "legend.frameon": False,
        "figure.dpi": 96, "savefig.bbox": "tight",
    })
    _STYLED = True


def _mix(colour, white):
    """`colour` blended toward white by fraction `white` in [0, 1]."""
    r, g, b = to_rgb(colour)
    return (r + (1 - r) * white, g + (1 - g) * white, b + (1 - b) * white)


#: Inches reserved above the plot for title and brief, and below it.
HEAD_IN = 1.10
TAKE_IN = 1.34
FOOT_IN = 0.72


def _margins(fig, takeaway=False, **kw):
    """Reserve fixed vertical space for the header and takeaway, whatever the height."""
    h = fig.get_figheight()
    fig.subplots_adjust(top=1 - HEAD_IN / h,
                        bottom=(TAKE_IN if takeaway else FOOT_IN) / h, **kw)


def _headline(fig, title, sub=None, x=0.045):
    """A big title and a short roman brief.

    The brief is dark, not an italic grey whisper: it is the one place a reader is
    told what they are looking at, so it has to survive Colab's downscaling.
    """
    h, w = fig.get_figheight(), fig.get_figwidth()
    fig.text(x, 1 - 0.20 / h, title, ha="left", va="top",
             fontsize=17, fontweight="bold", color=INK)
    if sub:
        # Wrap to the canvas. One scenario's nouns are longer than another's, and a
        # brief that runs off the right edge is the one thing a reader always notices.
        fig.text(x, 1 - 0.57 / h, "\n".join(textwrap.wrap(sub, int(w * 10.2))),
                 ha="left", va="top", fontsize=11.5, color=INK_2, linespacing=1.5)


def _takeaway(fig, text, per_line=94):
    """The conclusion: a rounded band with a violet accent bar, bold, left-aligned."""
    lines = textwrap.wrap(text, per_line)
    h = fig.get_figheight()
    y, height = 0.10 / h, (0.32 + 0.24 * len(lines)) / h
    fig.patches.append(FancyBboxPatch(
        (0.035, y), 0.93, height, transform=fig.transFigure,
        boxstyle="round,pad=0.004,rounding_size=0.010",
        facecolor=PANEL, edgecolor="none", zorder=0))
    fig.patches.append(FancyBboxPatch(
        (0.035, y), 0.0045, height, transform=fig.transFigure,
        boxstyle="round,pad=0.0,rounding_size=0.002",
        facecolor=ACCENT, edgecolor="none", zorder=1))
    fig.text(0.058, y + height / 2, "\n".join(lines), ha="left", va="center",
             fontsize=12, fontweight="bold", color=INK, linespacing=1.5)


# --------------------------------------------------------- geometry helpers

def _px_per_unit(ax):
    """Pixels per data unit in x and y. Call after limits and margins are final."""
    fig = ax.figure
    bb = ax.get_position()
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    return (bb.width * fig.get_figwidth() * fig.dpi / abs(x1 - x0),
            bb.height * fig.get_figheight() * fig.dpi / abs(y1 - y0))


def _pill(ax, x0, x1, y, h, colour, r_px=7.0, zorder=3):
    """A horizontal rounded bar from x0 to x1, visually equal corner radii."""
    if x1 - x0 <= 0:
        return
    xpp, ypp = _px_per_unit(ax)
    r = max(min(r_px, h * ypp / 2, (x1 - x0) * xpp / 2), 0.01) / xpp
    ax.add_patch(FancyBboxPatch(
        (x0, y - h / 2), x1 - x0, h,
        boxstyle=f"round,pad=0,rounding_size={r}", mutation_aspect=xpp / ypp,
        facecolor=colour, edgecolor="none", zorder=zorder))


def _vpill(ax, x, y0, y1, w, colour, r_px=7.0, zorder=3):
    """A vertical rounded bar from y0 up to y1."""
    if y1 - y0 <= 0:
        return
    xpp, ypp = _px_per_unit(ax)
    r = max(min(r_px, w * xpp / 2, (y1 - y0) * ypp / 2), 0.01) / xpp
    ax.add_patch(FancyBboxPatch(
        (x - w / 2, y0), w, y1 - y0,
        boxstyle=f"round,pad=0,rounding_size={r}", mutation_aspect=xpp / ypp,
        facecolor=colour, edgecolor="none", zorder=zorder))


def _cell(ax, x, y, w, h, colour, gap_px=3.0, r_px=6.0, edge=None, lw=0.0, z=2):
    """One rounded heat-grid cell with a white gap around it."""
    xpp, ypp = _px_per_unit(ax)
    gx, gy = gap_px / xpp, gap_px / ypp
    r = max(min(r_px, (h - 2 * gy) * ypp / 2, (w - 2 * gx) * xpp / 2), 0.01) / xpp
    ax.add_patch(FancyBboxPatch(
        (x + gx, y + gy), w - 2 * gx, h - 2 * gy,
        boxstyle=f"round,pad=0,rounding_size={r}", mutation_aspect=xpp / ypp,
        facecolor=colour, edgecolor=edge or "none", linewidth=lw, zorder=z))


def _card(ax, x, y, w, h, fill, edge="none", lw=1.2, r_px=10.0, z=2, shadow=False):
    """A rounded card in data coordinates, with an optional soft drop shadow."""
    xpp, ypp = _px_per_unit(ax)
    r = max(min(r_px, h * ypp / 2, w * xpp / 2), 0.01) / xpp
    patch = FancyBboxPatch(
        (x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
        mutation_aspect=xpp / ypp, facecolor=fill, edgecolor=edge,
        linewidth=lw, zorder=z)
    if shadow:
        patch.set_path_effects([pe.withSimplePatchShadow(
            offset=(0, -2.2), shadow_rgbFace="#3a3935", alpha=0.10, rho=1)])
    ax.add_patch(patch)


def _model_strip(ax, cx, y, colour, n=None, cell=0.16, gap=0.045, z=5, shades=None):
    """The model, drawn honestly: one square per parameter. Shade varies per weight.

    `n` defaults to the scenario's parameter count, so the strip is never a lie about
    how big the model is. `shades`: optional per-cell white-mix fractions (0 = full
    colour); the string "white" leaves a cell as an empty slot, which is how the
    update strip shows that most numbers did not move.
    """
    n = fc.N_PARAMS if n is None else n
    if shades is None:
        shades = [0.05, 0.38, 0.22, 0.52, 0.12, 0.45, 0.30, 0.58, 0.18, 0.42, 0.08]
    total = n * cell + (n - 1) * gap
    x = cx - total / 2
    for k in range(n):
        s = shades[k % len(shades)]
        fill = "white" if s == "white" else _mix(colour, s)
        _card(ax, x + k * (cell + gap), y, cell, cell, fill, r_px=2.5, z=z)


def _padlock(ax, x, y, s=0.14, colour=MUTED, z=6):
    """A tiny padlock: a rounded body plus a shackle arc."""
    xpp, ypp = _px_per_unit(ax)
    _card(ax, x - s / 2, y - s * 0.55, s, s * 0.78, colour, r_px=2, z=z)
    ax.add_patch(mpl.patches.Arc((x, y + s * 0.26), s * 0.62,
                                 s * 0.62 * (xpp / ypp), theta1=0, theta2=180,
                                 lw=1.4, color=colour, zorder=z))


def _named_rows(ax, ys, names, colours, size=11.5, dot_dx=-0.022, text_dx=-0.040):
    """Row labels left of the axis: a colour dot for identity, ink for the name."""
    tr = blended_transform_factory(ax.transAxes, ax.transData)
    for y, name, c in zip(ys, names, colours):
        ax.scatter([dot_dx], [y], transform=tr, s=52, color=c,
                   clip_on=False, zorder=6, linewidth=0)
        ax.text(text_dx, y, name, transform=tr, ha="right", va="center",
                fontsize=size, color=INK)


def _done(fig):
    """Hand back a finished figure.

    Closing it first takes it out of pyplot's registry. A notebook then shows it once,
    from the cell's return value, instead of twice: once when the inline backend flushes
    the registry at the end of the cell, and again from the value's own repr. Saving and
    rendering both still work on a closed figure.
    """
    plt.close(fig)
    return fig


def _nice_ticks(hi, target=5):
    """Round tick positions from 0 to `hi`, aiming for about `target` intervals.

    The two scenarios differ by a factor of three in size, so any fixed tick list is
    wrong for one of them.
    """
    raw = max(hi, 1) / target
    mag = 10 ** np.floor(np.log10(raw))
    step = mag * min((1, 2, 2.5, 5, 10), key=lambda m: abs(raw - mag * m))
    return np.arange(0, hi + step * 0.5, step)


def _bare(ax):
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def _short(name):
    return SHORT_NAME.get(name, name)


def _luminance(colour):
    r, g, b = to_rgb(colour)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def _chip(ax, x, y, text, colour=INK_2, size=9.5, fill="white"):
    ax.text(x, y, text, ha="center", va="center", fontsize=size,
            color=colour, zorder=8,
            bbox=dict(boxstyle="round,pad=0.42", facecolor=fill,
                      edgecolor=GRID, lw=0.9))


# ------------------------------------------------------------------------ HTML

_CSS = """
<style>
.fv { font-family: system-ui, -apple-system, 'Segoe UI', sans-serif;
      color:#0b0b0b; background:#ffffff;
      max-width: 54rem; font-size:.95rem; line-height:1.55;
      padding:1rem 1.15rem; border:1px solid #e1e0d9; border-radius:.75rem; }
.fv, .fv * { color:#0b0b0b; }
.fv h4 { margin:0 0 .6rem; font-size:1.02rem; font-weight:700; }
.fv h4:before { content:''; display:inline-block; width:.55em; height:.55em;
                background:#4a3aa7; border-radius:30%; margin-right:.5em; }
.fv table { border-collapse: collapse; width:100%; font-size:.92rem; }
.fv th, .fv td { text-align:left; padding:.45rem .65rem; }
.fv thead th { font-size:.72rem; letter-spacing:.08em; text-transform:uppercase;
               color:#898781; font-weight:600; border-bottom:1px solid #c3c2b7; }
.fv tbody td { border-bottom:1px solid #eceae4; }
.fv tbody tr:hover td { background:#f6f5f2; }
.fv td.n { font-family: ui-monospace, Menlo, monospace;
           font-variant-numeric: tabular-nums; text-align:right; }
.fv .quiz input.qr { position:absolute; opacity:0; width:0; height:0; }
.fv .opt { display:block; padding:.42rem .7rem .42rem 2rem; position:relative;
           border-radius:.5rem; cursor:pointer; border:1.5px solid transparent;
           margin:.18rem 0; transition:background .12s; }
.fv .opt:hover { background:#f2f1ec; }
.fv .opt:before { content:''; position:absolute; left:.62rem; top:.72rem; width:.72rem;
                  height:.72rem; border:2px solid #b6b5ae; border-radius:50%;
                  box-sizing:border-box; }
.fv .quiz input.qr:checked + label.right { background:#dff5e6; border-color:#0ca30c; }
.fv .quiz input.qr:checked + label.right:before { border-color:#0ca30c;
                  background:#0ca30c; }
.fv .quiz input.qr:checked + label.right:after { content:'  ✓ correct';
                  color:#006300; font-weight:700; font-size:.86rem; }
.fv .quiz input.qr:checked + label.wrong { background:#fbe6e6; border-color:#d03b3b; }
.fv .quiz input.qr:checked + label.wrong:before { border-color:#d03b3b;
                  background:#d03b3b; }
.fv .quiz input.qr:checked + label.wrong:after { content:'  ✗ not this one';
                  color:#c22f2f; font-weight:700; font-size:.86rem; }
.fv details { margin-top:.8rem; background:#f4f2fb; padding:.7rem 1rem;
              border-radius:.6rem; }
.fv summary { cursor:pointer; font-weight:700; color:#4a3aa7; }
.fv .note { color:#898781; font-size:.87rem; margin-top:.5rem; }
.fv .good { color:#006300 !important; font-weight:700; }
.fv .bad { color:#c22f2f !important; font-weight:700; }
.fv .flag { color:#9a6b00 !important; font-weight:700; }
.fv thead th { color:#6b6a65 !important; }
.fv .note { color:#6b6a65 !important; }
.fv summary { color:#4a3aa7 !important; }
.fv details { color:#0b0b0b; }
</style>
"""


def _html(body):
    out = _CSS + f"<div class='fv'>{body}</div>"
    if HTML is None:
        print(body)
        return None
    return HTML(out)


def _table(headers, rows, aligns=None):
    aligns = aligns or [""] * len(headers)
    head = "".join(f"<th>{h}</th>" for h in headers)
    body = ""
    for r in rows:
        body += "<tr>" + "".join(
            f"<td class='{a}'>{c}</td>" for c, a in zip(r, aligns)) + "</tr>"
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


# ================================================================ PART 0

def scale_walls(federation=None):
    """Part 0's opening figure: why this is a scaling problem, and which kind.

    Three walls, left to right, the first two muted and the third lit. The first two are
    answered by buying hardware and are drawn abstractly, because a reader does not need
    to know what sharding is to see that money solves them. The third is drawn in this
    scenario's own colours, because it is the one the notebook is about.

    Deliberately shows no exchange, no coordinator and no updates. That is part 3's
    figure, and putting the mechanism here would answer "how" before a reader has asked
    "why".
    """
    _style()
    fig, ax = plt.subplots(figsize=(12.2, 5.5))
    _margins(fig, takeaway=True, left=0.02, right=0.98)
    _bare(ax)
    ax.set_xlim(0, 12.2); ax.set_ylim(0, 4.05)

    W, H, Y0 = 3.62, 3.30, 0.55
    xs = [0.30, 4.29, 8.28]
    names = list(SITE_COLOUR) or ["A", "B", "C", "D"]
    cols = list(SITE_COLOUR.values()) or [ACCENT] * 4

    def panel(x, n, title, footer, lit):
        _card(ax, x, Y0, W, H, _mix(ACCENT, 0.955) if lit else "white",
              edge=ACCENT if lit else GRID, lw=2.2 if lit else 1.2, r_px=12)
        ax.text(x + 0.30, Y0 + H - 0.38, str(n), ha="center", va="center",
                fontsize=10, fontweight="bold", color="white",
                bbox=dict(boxstyle="circle,pad=0.32",
                          facecolor=ACCENT if lit else NEUTRAL, edgecolor="none"),
                zorder=6)
        ax.text(x + 0.60, Y0 + H - 0.38, title, ha="left", va="center",
                fontsize=12 if lit else 11.5, fontweight="bold",
                color=INK if lit else INK_2)
        ax.text(x + W / 2, Y0 + 0.28, footer, ha="center", va="center",
                fontsize=10.5 if lit else 10, fontweight="bold" if lit else "normal",
                color=INK if lit else MUTED)
        ax.text(x + W / 2, Y0 - 0.28, "not made of hardware" if lit
                else "made of hardware", ha="center", va="center", fontsize=9.5,
                fontweight="bold" if lit else "normal", color=INK if lit else MUTED)

    # ---------------------------------------------- 1 · the model is too big
    x = xs[0]
    panel(x, 1, "the model is too big", "answer: split it across machines", False)
    bx, by, bw, bh = x + 0.72, Y0 + 1.06, 1.10, 1.40
    _card(ax, bx, by, bw, bh, "none", edge=BASELINE, lw=1.4, r_px=6)
    ax.text(bx + bw / 2, by - 0.22, "one machine", ha="center", fontsize=8.6,
            color=MUTED)
    for k in range(3):
        _card(ax, bx + 0.22, by + 0.26 + k * 0.42, 1.62 + k * 0.16, 0.30,
              _mix(NEUTRAL, 0.05), r_px=4, z=4)

    # ---------------------------------------------- 2 · one machine is too slow
    x = xs[1]
    panel(x, 2, "one machine is too slow", "answer: add machines", False)
    for k in range(4):
        cx = x + 0.86 + k * 0.64
        _card(ax, cx - 0.22, Y0 + 1.36, 0.44, 0.86, _mix(NEUTRAL, 0.05), r_px=5)
    ax.text(x + W / 2 + 0.10, Y0 + 1.06, "more machines, the same data",
            ha="center", fontsize=8.6, color=MUTED)

    # ---------------------------------------------- 3 · the data cannot be gathered
    x = xs[2]
    panel(x, 3, "the data cannot be gathered", "no answer you can buy", True)
    top = Y0 + 2.16
    for k in range(4):
        cx = x + 0.60 + k * 0.80
        _card(ax, cx - 0.28, top - 0.30, 0.56, 0.64, _mix(cols[k % 4], 0.28), r_px=5)
        _padlock(ax, cx, top + 0.02, s=0.24, colour="white")
    pool_y = Y0 + 1.00
    _card(ax, x + 1.10, pool_y - 0.26, 1.60, 0.52, "white", edge=BASELINE,
          lw=1.3, r_px=6)
    ax.text(x + 1.90, pool_y, "one place", ha="center", va="center", fontsize=9,
            color=MUTED)
    for k in range(4):
        cx = x + 0.60 + k * 0.80
        ax.add_patch(FancyArrowPatch((cx, top - 0.36), (x + 1.90, pool_y + 0.30),
                                     arrowstyle="-", lw=1.2,
                                     color=_mix(BAD, 0.62), zorder=3))
    bx, by, r = x + 1.90, (top - 0.36 + pool_y + 0.30) / 2, 0.20
    _card(ax, bx - r * 1.5, by - r * 1.5, r * 3, r * 3, "white", edge="none", z=6)
    for dx in (-1, 1):
        ax.plot([bx - dx * r, bx + dx * r], [by - r, by + r], color=BAD, lw=3.0,
                solid_capstyle="round", zorder=7)

    _headline(fig, "Three walls of scale. This notebook is about the third.",
              "The first two are answered by buying more of something. The third "
              "is not.")
    _takeaway(fig, f"No amount of hardware puts these four books in one place. The "
                   f"{V.records} stay where they are, so the training has to go to "
                   f"them instead.")
    return _done(fig)


def market_map(federation, x_feature, y_feature, feature_names, readable=None,
               x_label=None, y_label=None, x_clip=None, y_clip=None):
    """Figure 1. Who each participant lends to, and what happened to them.

    One panel per participant, all on the same axes. Every dot is one customer: where it
    sits is who they are, and red means the outcome we are trying to predict. The whole
    market sits behind each panel in grey so a reader can see which slice of it this
    participant holds.

    This is the first figure a student sees, so it has to carry the problem rather than
    the solution. It shows four different customer populations, one shared question, and
    the fact that the answer depends on where you are on the map.
    """
    _style()
    names = list(feature_names)
    xi, yi = names.index(x_feature), names.index(y_feature)
    sites = list(federation)
    n = len(sites)
    allX = np.vstack([s.X for s in sites])
    xs_all, ys_all = allX[:, xi], allX[:, yi]
    xlo, xhi = (x_clip or (np.percentile(xs_all, 1), np.percentile(xs_all, 99)))
    ylo, yhi = (y_clip or (np.percentile(ys_all, 1), np.percentile(ys_all, 99)))

    h = 6.5
    fig, axes = plt.subplots(1, n, figsize=(2.55 * n + 1.0, h), sharex=True, sharey=True)
    # Hand-set rather than via _margins: this figure needs room above for a two line panel
    # heading and room below for an axis label, a legend and the takeaway band.
    fig.subplots_adjust(top=1 - 1.55 / h, bottom=2.40 / h, left=0.085, right=0.985,
                        wspace=0.14)
    for ax, site in zip(np.atleast_1d(axes), sites):
        c = SITE_COLOUR.get(site.name, ACCENT)
        ax.scatter(xs_all, ys_all, s=5, color="#e8e7e2", linewidth=0, zorder=1)
        pay = site.y == 0
        ax.scatter(site.X[pay, xi], site.X[pay, yi], s=8, color=_mix(c, 0.35),
                   linewidth=0, zorder=2)
        ax.scatter(site.X[~pay, xi], site.X[~pay, yi], s=13, color=ILL,
                   linewidth=0, zorder=3)
        ax.set_xlim(xlo, xhi); ax.set_ylim(ylo, yhi)
        ax.set_title(_short(site.name), fontsize=12, fontweight="bold", color=INK, pad=22)
        ax.text(0.5, 1.015, f"{site.n:,} {V.members}   ·   {site.share_positive:.0%} "
                f"{V.positive}", transform=ax.transAxes, ha="center", va="bottom",
                fontsize=9.5, color=INK_2)
        ax.grid(True, color=GRID, linewidth=0.8); ax.set_axisbelow(True)
        for side in ("top", "right", "left", "bottom"):
            ax.spines[side].set_visible(False)
    axes_list = list(np.atleast_1d(axes))
    axes_list[0].set_ylabel(y_label or (readable or {}).get(y_feature, y_feature),
                            fontsize=11)
    fig.text(0.535, (2.40 - 0.55) / h, x_label or (readable or {}).get(x_feature, x_feature),
             ha="center", va="center", fontsize=11, color=INK_2)

    handles = [plt.Line2D([], [], marker="o", ls="", color="#cfcec8", markersize=7,
                          label=f"every {V.member} in the {V.setting}"),
               plt.Line2D([], [], marker="o", ls="", color=MUTED, markersize=7,
                          label=f"this {V.site}'s {V.members}"),
               plt.Line2D([], [], marker="o", ls="", color=ILL, markersize=7,
                          label=f"{V.positive}")]
    fig.legend(handles=handles, fontsize=10, ncols=3, loc="center",
               bbox_to_anchor=(0.535, (2.40 - 0.98) / h), handletextpad=0.35,
               columnspacing=2.0, frameon=False)

    _headline(fig, f"Four {V.sites}, four kinds of {V.member}, one question.",
              f"Every dot is one {V.member}. Red means they ended up {V.positive}.")
    _takeaway(fig, f"The four {V.sites} lend to different people, so each one only ever sees "
                   "part of the picture. Red sits in the same corner of all four panels, "
                   "which is the first hint that they are all learning the same rule.")
    return _done(fig)


def model_card(values, weights, feature_names, readable=None, bias=0.0, outcome=None,
               filed=None, show=3):
    """Figure 2. What the model is, and how it scores one customer.

    Deliberately small. The whole model is fourteen numbers, so the figure says that
    first and shows them as fourteen marks; then it scores one customer with three of
    them and sums the rest into a line; then it draws the curve. A reader who takes one
    thing away should take away the fourteen.

    `values` are the model's standardised inputs, `filed` the same customer as a credit
    file writes them. Only the `show` largest contributions get a row.
    """
    _style()
    names = list(feature_names)
    w = np.asarray(weights, float)[:len(names)]
    v = np.asarray(values, float)[:len(names)]
    contrib = v * w
    score = float(contrib.sum() + bias)
    prob = 1.0 / (1.0 + np.exp(-score))

    order = np.argsort(-np.abs(contrib))
    top = sorted(order[:show].tolist())
    rest = [i for i in range(len(names)) if i not in top]
    rest_sum = float(contrib[rest].sum()) if rest else 0.0

    fig, ax = plt.subplots(figsize=(11.6, 5.3))
    _margins(fig, takeaway=True, left=0.030, right=0.980)
    _bare(ax); ax.set_xlim(0, 12.4); ax.set_ylim(6.9, -2.5)

    NAME_X, FILED_X, MUL_X, W_X, EQ_X, C_X = 0.30, 3.15, 3.42, 4.42, 4.68, 5.85

    # ---- the model is fourteen numbers, shown as fourteen marks
    cw, gap, y0 = 0.20, 0.085, -2.15
    for i in range(len(names)):
        _cell(ax, NAME_X + i * (cw + gap), y0, cw, 0.34, _mix(ACCENT, 0.42),
              gap_px=0, r_px=3, z=2)
    xa = NAME_X + len(names) * (cw + gap)
    ax.text(xa + 0.16, y0 + 0.17, "+", fontsize=11, va="center", ha="center", color=MUTED)
    _cell(ax, xa + 0.32, y0, cw, 0.34, _mix(MUTED, 0.55), gap_px=0, r_px=3, z=2)
    ax.text(xa + 0.72, y0 + 0.17, f"= {len(names) + 1} numbers, and that is the whole model",
            fontsize=10.5, va="center", color=INK, fontweight="bold")
    ax.text(NAME_X, y0 + 0.78, f"one for each of the {len(names)} things a bank has on "
            "file, plus a starting number", fontsize=9.4, color=MUTED, va="center")

    # ---- one customer, scored
    ax.text(NAME_X, 0.35, "SCORING ONE CUSTOMER", fontsize=8.6, fontweight="bold",
            color=INK_2)
    ax.text(FILED_X, 0.35, "ON FILE", fontsize=8.6, fontweight="bold", color=INK_2,
            ha="right")
    ax.text(W_X, 0.35, "ITS NUMBER", fontsize=8.6, fontweight="bold", color=INK_2,
            ha="right")

    def row(y, name, filed_txt, num, out, colour=INK, dim=False):
        ax.text(NAME_X, y, name, fontsize=10.2, va="center",
                color=MUTED if dim else INK, style="italic" if dim else "normal")
        if filed_txt is not None:
            ax.text(FILED_X, y, filed_txt, fontsize=9.8, va="center", ha="right",
                    color=INK_2, family="DejaVu Sans Mono")
            ax.text(MUL_X, y, "×", fontsize=9.5, va="center", ha="center", color=MUTED)
            ax.text(W_X, y, num, fontsize=9.8, va="center", ha="right", color=ACCENT,
                    family="DejaVu Sans Mono")
            ax.text(EQ_X, y, "=", fontsize=9.5, va="center", ha="center", color=MUTED)
        ax.text(C_X, y, out, fontsize=10.2, va="center", ha="right", color=colour,
                family="DejaVu Sans Mono")

    y = 1.25
    for i in top:
        row(y, (readable or {}).get(names[i], names[i]), filed[i] if filed else None,
            f"{w[i]:+.2f}", f"{contrib[i]:+.2f}",
            BAD if contrib[i] > 0 else GOOD_TEXT)
        y += 0.82
    row(y, f"the other {len(rest)} measurements", None, None, f"{rest_sum:+.2f}",
        MUTED, dim=True)
    y += 0.82
    row(y, "the starting number", None, None, f"{bias:+.2f}", MUTED, dim=True)
    y += 0.62
    ax.plot([NAME_X, C_X], [y, y], color=BASELINE, lw=1.1)
    y += 0.62
    ax.text(NAME_X, y, "this customer's score", fontsize=10.5, va="center", color=INK,
            fontweight="bold")
    ax.text(C_X, y, f"{score:+.2f}", fontsize=15, va="center", ha="right", color=INK,
            fontweight="bold", family="DejaVu Sans Mono")

    # ---- the curve, small and to the side
    yspan = 6.9 + 2.5
    xf = lambda x: x / 12.4
    yf = lambda t: (6.9 - t) / yspan
    cax = ax.inset_axes([xf(7.35), yf(4.75), xf(11.95) - xf(7.35), yf(0.95) - yf(4.75)])
    lim = max(4.2, abs(score) + 1.8)
    xs = np.linspace(-lim, lim, 400)
    cax.plot(xs, 1.0 / (1.0 + np.exp(-xs)), color=ACCENT, lw=2.4, zorder=4,
             solid_capstyle="round")
    cax.axhline(0.5, color=GRID, lw=1.0, zorder=1)
    cax.axvline(0, color=GRID, lw=1.0, zorder=1)
    cax.plot([-lim, score], [prob, prob], color=MUTED, lw=1.1, ls=":", zorder=3)
    cax.plot([score, score], [0, prob], color=MUTED, lw=1.1, ls=":", zorder=3)
    cax.plot([score], [prob], "o", ms=9, mfc=ACCENT, mec="white", mew=2.0, zorder=6)
    cax.set_xlim(-lim, lim); cax.set_ylim(-0.05, 1.12)
    cax.set_yticks([0, 0.5, 1.0], ["0%", "50%", "100%"], fontsize=8.6)
    cax.set_xticks([])
    cax.tick_params(length=0, colors=MUTED)
    for side in ("top", "right", "left", "bottom"):
        cax.spines[side].set_visible(False)
    lx, ha = (score - 0.28, "right") if score < 0 else (score + 0.28, "left")
    cax.text(lx, prob + 0.05, f"{prob:.0%}", fontsize=22, color=ACCENT,
             fontweight="bold", va="bottom", ha=ha, zorder=6)
    cax.text(score, 0.03, f"{score:+.2f} ", fontsize=9.2, color=INK, va="bottom",
             ha="right", family="DejaVu Sans Mono", zorder=6)

    ax.text(7.35, 0.35, "TURNING THAT SCORE INTO A CHANCE", fontsize=8.6,
            fontweight="bold", color=INK_2)
    ax.text(7.35, 5.55, "The curve keeps every answer between 0% and 100%.\n"
            "A score of 0 would be exactly 50%.",
            fontsize=9.6, color=INK_2, va="center", linespacing=1.7)
    if outcome is not None:
        ax.text(7.35, 6.55, f"What actually happened: this {V.member} was "
                f"{V.positive if outcome else V.negative}.",
                fontsize=9.6, color=INK, fontweight="bold", va="center")

    _headline(fig, f"The model is {len(names) + 1} numbers.",
              "A logistic regression. Multiply each thing on file by its number, add them "
              "up, and a curve turns the total into a chance.")
    _takeaway(fig, f"Training means choosing those {len(names) + 1} numbers so the "
                   "predictions match what actually happened. Part 1 is where you do it.")
    return _done(fig)


def rule_recovery(fits, truth, feature_names, readable=None, sites=None):
    """Part 2. Four participants, four estimates of the same rule.

    One panel per participant. The black tick is the number the data was really
    generated with. The coloured bar is what that participant worked out on its own.
    Bars reaching their ticks means everyone is solving the same problem, which is the
    single most important fact in the notebook and is easy to miss in prose.

    `fits` is {name: coefficient vector}, `truth` the vector it should recover.
    """
    _style()
    names = list(fits)
    truth = np.asarray(truth, float)
    k = len(feature_names)
    n = len(names)
    lo = min(float(np.min(np.asarray(v)[:k])) for v in fits.values())
    hi = max(float(np.max(np.asarray(v)[:k])) for v in fits.values())
    lo, hi = min(lo, truth.min()) - 0.18, max(hi, truth.max()) + 0.18

    fig, axes = plt.subplots(1, n, figsize=(2.5 * n + 2.0, 0.30 * k + 3.1),
                             sharey=True, sharex=True)
    h = fig.get_figheight()
    fig.subplots_adjust(top=1 - 1.45 / h, bottom=2.15 / h, left=0.215, right=0.985,
                        wspace=0.13)
    for ax, name in zip(np.atleast_1d(axes), names):
        c = SITE_COLOUR.get(name, ACCENT)
        v = np.asarray(fits[name], float)[:k]
        ax.axvline(0, color=BASELINE, lw=1.0, zorder=1)
        for i in range(k):
            _pill(ax, min(0, v[i]), max(0, v[i]), i, 0.52, _mix(c, 0.30), r_px=4)
            ax.plot([truth[i], truth[i]], [i - 0.36, i + 0.36], color=INK, lw=2.0,
                    zorder=6, solid_capstyle="butt")
        ax.set_xlim(lo, hi); ax.set_ylim(k - 0.4, -1.0)
        ax.set_title(_short(name), fontsize=11.5, fontweight="bold", color=INK, pad=16)
        if sites:
            ax.text(0.5, 1.01, f"{sites[name]:,} {V.members}", transform=ax.transAxes,
                    ha="center", va="bottom", fontsize=9, color=MUTED)
        ax.grid(axis="x", color=GRID, lw=0.8); ax.set_axisbelow(True)
        for side in ("top", "right", "left", "bottom"):
            ax.spines[side].set_visible(False)
        ax.tick_params(labelsize=9)
    first = list(np.atleast_1d(axes))[0]
    first.set_yticks(range(k), [(readable or {}).get(f, f) for f in feature_names],
                     fontsize=9.5)

    handles = [plt.Line2D([], [], color=INK, lw=2.4, label="the rule the data really uses"),
               plt.Line2D([], [], color=MUTED, lw=8,
                          label=f"what this {V.site} worked out on its own")]
    fig.legend(handles=handles, fontsize=10, ncols=2, loc="center",
               bbox_to_anchor=(0.6, (2.15 - 0.62) / h), frameon=False, columnspacing=2.2)
    _headline(fig, f"All four are solving the same problem.",
              f"Each bar is one measurement's number. The black tick is the value the data "
              f"was really built with.")
    _takeaway(fig, f"Every {V.site} is reaching for the same rule and getting a rough version "
                   f"of it. They differ in how close they get, not in what they are aiming "
                   f"at. That is what makes one shared model the right answer.")
    return _done(fig)


#: The divisions part 2 offers. A scenario may replace this through `use()`.
DEFAULT_SPLITS = [("even split", "iid"),
                  ("the real participants", "natural"),
                  ("one group gets 60% of positives", "concentrate:0.60"),
                  ("one group gets 80% of positives", "concentrate:0.80"),
                  ("one group gets 100% of positives", "concentrate:1.00")]
SPLITS = list(DEFAULT_SPLITS)

_SPLIT_CACHE = {}


def _split_groups(federation, regime, seed=0, min_size=0):
    """Index arrays for one division, with the real participants as a special case."""
    X, y = federation.pooled()
    if regime == "natural":
        out, at = [], 0
        for site in federation:
            out.append(np.arange(at, at + site.n)); at += site.n
        return out
    return fc.partition_indices(y, regime, seed=seed, min_size=min_size, X=X)


def split_report(federation, cohort, regime, local_steps=20, rounds=40, lr=0.5,
                 min_size=0):
    """Everything part 2 needs to know about one division of the same records.

    Trains six models: each group on its own, all four together by federation, and all
    four pooled. Then scores them on a fresh cohort divided by the same rule, so no model
    is ever scored on a record it trained on and every test set is thousands of records
    rather than a couple of hundred. Cached, because the explorer re-renders whenever a
    reader clicks a different division.

    Dividing the cohort the same way is what makes the columns mean something. A group is
    a rule for sorting records, not a fixed list of them, so the rule applies just as well
    to next year's applicants.
    """
    key = (id(federation), id(cohort), regime, local_steps, rounds, min_size)
    if key in _SPLIT_CACHE:
        return _SPLIT_CACHE[key]

    X, y = federation.pooled()
    Xc, yc = cohort.pooled()
    groups = _split_groups(federation, regime, min_size=min_size)
    tests = _split_groups(cohort, regime, min_size=min_size)

    train = [fc.Site(f"Group {k + 1}", X[g], y[g]) for k, g in enumerate(groups)]
    locals_ = [fc.train(fc.init(), s.X, s.y, 6000, lr=lr) for s in train]
    fed = fc.run(train, E=local_steps, rounds=rounds, lr=lr)["w"]
    pooled = fc.train(fc.init(), X, y, 6000, lr=lr)

    def score(w, Xs, ys):
        if len(np.unique(ys)) < 2:
            return float("nan")
        return float(fc.auc(w, Xs, ys))

    cols = [(Xc[t], yc[t]) for t in tests] + [(Xc, yc)]
    M = np.array([[score(w, Xs, ys) for Xs, ys in cols] for w in locals_ + [fed, pooled]])

    lp = fc.loss(pooled, Xc, yc)
    rec = dict(
        regime=regime, groups=groups, tests=tests,
        rates=[float(y[g].mean()) for g in groups],
        sizes=[len(g) for g in groups],
        matrix=M,
        row_labels=[f"Group {k + 1} on its own" for k in range(len(groups))]
                   + ["all four, federated", "all four pooled"],
        col_labels=[f"Group {k + 1}" for k in range(len(groups))] + ["everyone"],
        test_sizes=[len(t) for t in tests] + [len(yc)],
        test_positives=[int(yc[t].sum()) for t in tests] + [int(yc.sum())],
        cost=100 * (fc.loss(fed, Xc, yc) - lp) / lp,
        separability=fc.separability(X, groups)[0],
        chance=1.0 / len(groups),
    )
    rates = rec["rates"]
    rec["spread"] = max(rates) / min(rates) if min(rates) > 0 else float("inf")
    _SPLIT_CACHE[key] = rec
    return rec


def split_costs(federation, cohort, splits=None, **kw):
    """`split_report` for every division on the list."""
    return [(label, split_report(federation, cohort, regime, **kw))
            for label, regime in (splits or SPLITS)]


def _split_panels(fig, rect, rec, rawX, xi, yi, x_label, y_label, cols):
    """The top band: who each group lends to, four panels on one pair of axes.

    Shared limits, so the panels are directly comparable. Grey is everybody, colour is
    this group, red is a positive. Four panels that look alike but carry very different
    rates is the whole tell, and it needs no caption.
    """
    x0, y0, w, h = rect
    xs, ys = rawX[:, xi], rawX[:, yi]
    xlo, xhi = np.percentile(xs, 1), np.percentile(xs, 99)
    ylo, yhi = np.percentile(ys, 1), np.percentile(ys, 99)
    n = len(rec["groups"])
    gap = 0.016
    pw = (w - gap * (n - 1)) / n

    for k, g in enumerate(rec["groups"]):
        ax = fig.add_axes([x0 + k * (pw + gap), y0, pw, h])
        _bare(ax)
        ax.set_xlim(xlo, xhi); ax.set_ylim(ylo, yhi)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(True); sp.set_color(GRID); sp.set_linewidth(1.0)
        ax.scatter(xs, ys, s=2.6, color="#f1f0ec", linewidth=0, zorder=1)
        col = cols[k % len(cols)]
        pos = g[rec["_pos_mask"][k]]
        neg = g[~rec["_pos_mask"][k]]
        ax.scatter(xs[neg], ys[neg], s=5.5, color=_mix(col, 0.34), linewidth=0, zorder=2)
        ax.scatter(xs[pos], ys[pos], s=10, color=ILL, linewidth=0, zorder=3)

        rate = rec["rates"][k]
        ax.text(0.0, 1.185, f"Group {k + 1}", transform=ax.transAxes, ha="left",
                va="baseline", fontsize=12, fontweight="bold", color=col)
        ax.text(0.0, 1.045, f"{rec['sizes'][k]:,} {V.members}", transform=ax.transAxes,
                ha="left", va="baseline", fontsize=9.2, color=MUTED)
        ax.text(1.0, 1.045, f"{rate:.0%} {V.positive}", transform=ax.transAxes,
                ha="right", va="baseline", fontsize=11.5, fontweight="bold",
                color=ILL if rate > 0.20 else INK_2)
        if k == 0:
            ax.set_ylabel(y_label, fontsize=9.5, labelpad=5)
    fig.text(x0 + w / 2, y0 - 0.021, x_label, ha="center", va="top", fontsize=9.5,
             color=INK_2)


def _split_matrix(fig, rect, rec, cols, metric="AUC"):
    """The middle band: part 1's grid again, on whichever division is showing."""
    M = rec["matrix"]
    n_rows, n_cols = M.shape
    fed_i, pool_i = n_rows - 2, n_rows - 1
    ax = fig.add_axes(rect)
    _bare(ax)
    gap = 0.36
    ax.set_xlim(-0.02, n_cols + 0.02)
    ax.set_ylim(n_rows + 2 * gap + 0.12, -1.62)

    def row_y(i):
        return i + (gap if i >= fed_i else 0) + (gap if i >= pool_i else 0)

    ax.text(n_cols / 2, -1.36, "SCORED ON " + V.members.upper()
            + " THE MODEL NEVER TRAINED ON", ha="center", fontsize=9,
            fontweight="bold", color=INK_2)
    for j, name in enumerate(rec["col_labels"]):
        last = j == n_cols - 1
        ax.text(j + 0.5, -0.72, name, ha="center", fontsize=10.5, fontweight="bold",
                color=INK if last else cols[j % len(cols)])
        ax.text(j + 0.5, -0.52, f"{rec['test_sizes'][j]:,} {V.members}\n"
                f"{rec['test_positives'][j]:,} {V.positive}", ha="center", va="top",
                fontsize=8.4, color=MUTED, linespacing=1.45)
    tr = blended_transform_factory(ax.transAxes, ax.transData)
    ax.text(-0.04, -1.36, "MODEL TRAINED AT", transform=tr, ha="right",
            fontsize=9, fontweight="bold", color=INK_2)

    ys, names, rcols = [], [], []
    for i in range(n_rows):
        yv = row_y(i)
        ys.append(yv + 0.5)
        names.append(rec["row_labels"][i])
        rcols.append(ACCENT if i == fed_i else
                     POOLED_COLOUR if i == pool_i else cols[i % len(cols)])
        for j in range(n_cols):
            v = M[i, j]
            if v != v:                                    # nobody positive to score on
                _cell(ax, j, yv, 1, 1, "#f4f3f0", gap_px=3.5, r_px=7)
                ax.text(j + 0.5, yv + 0.5, f"nobody\n{V.positive}", ha="center",
                        va="center", fontsize=8.4, color=MUTED, linespacing=1.4)
                continue
            colv = M[:, j][~np.isnan(M[:, j])]
            lo, hi = colv.min(), colv.max()
            t = 0.5 if hi - lo < 1e-9 else (v - lo) / (hi - lo)
            fill = SEQ(0.10 + 0.80 * t)
            _cell(ax, j, yv, 1, 1, fill, gap_px=3.5, r_px=7)
            ax.text(j + 0.5, yv + 0.5, f"{v:.3f}", ha="center", va="center",
                    fontsize=12, color="white" if _luminance(fill) < 0.52 else INK,
                    fontweight="bold")

    # the federated row is the one a reader is here to look at
    _cell(ax, -0.02, row_y(fed_i) - 0.11, n_cols + 0.04, 1.22, "none",
          edge=ACCENT, lw=2.2, gap_px=0.0, r_px=9, z=6)
    for j in range(n_cols):
        colv = M[:, j]
        if np.isnan(colv).all():
            continue
        i = int(np.nanargmax(colv))
        _cell(ax, j, row_y(i), 1, 1, "none", edge=INK, lw=2.4, gap_px=1.0, r_px=8, z=7)

    _named_rows(ax, ys, names, rcols, size=10.5, dot_dx=-0.020, text_dx=-0.036)
    ax.text(-0.036, row_y(pool_i) + 1.10, "not allowed here. the target to beat.",
            transform=tr, ha="right", va="top", fontsize=8.4, color=MUTED)
    return ax


def split_view(federation, raw, cohort, regime="natural", x_feature=None, y_feature=None,
               feature_names=None, readable=None, x_label=None, y_label=None,
               local_steps=20, min_size=0, metric="AUC"):
    """Part 2's explorer, as one picture, for whichever division is selected.

    Three bands, top to bottom. Who each group lends to, as four panels on shared axes
    with the positives drawn in red. What every model scores, in part 1's language. Then
    a strip of the three numbers that describe this division.

    Deliberately carries no conclusion. A reader is meant to click through the divisions
    and work it out, and a figure that announces the answer takes that away.
    """
    _style()
    names = list(feature_names)
    xi, yi = names.index(x_feature), names.index(y_feature)
    rawX, _ = raw.pooled()
    _, y = federation.pooled()
    rec = split_report(federation, cohort, regime, local_steps=local_steps,
                       min_size=min_size)
    rec["_pos_mask"] = [y[g] == 1 for g in rec["groups"]]
    label = next((l for l, r in SPLITS if r == regime), regime)
    cols = list(SITE_COLOUR.values()) or [ACCENT] * len(rec["groups"])

    h = 11.6
    fig = plt.figure(figsize=(12.4, h))
    fig.text(0.055, 1 - 1.15 / h, "WHO EACH GROUP LENDS TO", fontsize=9,
             fontweight="bold", color=INK_2)
    fig.text(0.055, 1 - 1.37 / h,
             f"Grey is every {V.member} in the {V.setting}. Colour is this group's. "
             f"Red is a {V.member} {V.positive}.", fontsize=9.5, color=MUTED)
    _split_panels(fig, [0.055, 1 - 4.90 / h, 0.915, 2.55 / h], rec, rawX, xi, yi,
                  x_label or (readable or {}).get(x_feature, x_feature),
                  y_label or (readable or {}).get(y_feature, y_feature), cols)

    fig.text(0.055, 1 - 5.62 / h, "WHAT EACH MODEL SCORES", fontsize=9,
             fontweight="bold", color=INK_2)
    fig.text(0.055, 1 - 5.84 / h,
             f"{metric} on held-out {V.members}. Higher is better and 0.5 would be "
             f"guessing.", fontsize=9.5, color=MUTED)
    fig.text(0.055, 1 - 6.06 / h, "Black ring wins its column. Shading compares each "
             "cell against the others in its own column.", fontsize=9.5, color=MUTED)
    _split_matrix(fig, [0.300, 1 - 10.50 / h, 0.670, 4.10 / h], rec, cols, metric=metric)

    # ---- the strip: three numbers, no verdict
    sp = rec["spread"]
    gap_auc = rec["matrix"][-2, -1] - rec["matrix"][-1, -1]
    chips = [(V.positive_rate + "s",
              f"nobody {V.positive} in three groups" if sp == float("inf")
              else "much the same everywhere" if sp < 1.15
              else f"{sp:.0f} times apart"),
             ("telling the groups apart",
              f"{rec['separability']:.0%}, where chance is {rec['chance']:.0%}"),
             ("federated against pooled",
              f"{'0.000' if abs(gap_auc) < 5e-4 else format(gap_auc, '+.3f')} {metric}, "
              f"{rec['cost']:+.1f}% on the loss")]
    for i, (k, v) in enumerate(chips):
        x = 0.055 + i * 0.313
        fig.patches.append(FancyBboxPatch(
            (x, 0.30 / h), 0.285, 0.62 / h, transform=fig.transFigure,
            boxstyle="round,pad=0.004,rounding_size=0.010",
            facecolor=PANEL, edgecolor="none", zorder=0))
        fig.text(x + 0.014, 0.74 / h, k.upper(), fontsize=8.4, fontweight="bold",
                 color=MUTED, va="center")
        fig.text(x + 0.014, 0.48 / h, v, fontsize=11.5, fontweight="bold", color=INK,
                 va="center")

    _headline(fig, f"{label}.",
              f"The same {len(y):,} {V.members}, divided a different way. Nothing about "
              f"the {V.members} themselves has changed.")
    return _done(fig)


#: One radio-group name per explorer on the page, so two renders never share state.
_EXPLORER_N = [0]


def split_explorer(federation, raw, cohort, **kw):
    """Figure A: `split_view` for all six divisions, behind a row of clickable tabs.

    Pure HTML and CSS, the same bargain the quizzes make. Every division is rendered
    once and embedded as an image; radio inputs decide which one is visible, so a click
    is instant, needs no kernel, and still works in a saved or reopened notebook.
    ipywidgets buttons die the moment the kernel does, which is why they are not used.
    """
    import io
    import base64

    start = kw.pop("regime", SPLITS[0][1])
    panes = []
    for label, reg in SPLITS:
        fig = split_view(federation, raw, cohort, regime=reg, **kw)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=96, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        panes.append((label, reg, base64.b64encode(buf.getvalue()).decode("ascii")))

    _EXPLORER_N[0] += 1
    gid = f"fvx{_EXPLORER_N[0]}"
    inputs, tabs, imgs, css = "", "", "", ""
    for i, (label, reg, png) in enumerate(panes):
        oid = f"{gid}_{i}"
        checked = " checked" if reg == start else ""
        inputs += f"<input class='xr' type='radio' name='{gid}' id='{oid}'{checked}>"
        tabs += f"<label class='tab' for='{oid}'>{label}</label>"
        imgs += (f"<div class='pane'><img alt='{label}' "
                 f"src='data:image/png;base64,{png}'></div>")
        css += (f"#{oid}:checked ~ .tabs label[for='{oid}'] "
                "{ background:#4a3aa7; border-color:#4a3aa7; color:#ffffff !important; }"
                f"#{oid}:checked ~ .panes .pane:nth-of-type({i + 1}) "
                "{ display:block; }")

    style = ("<style>"
             ".fvx input.xr { position:absolute; opacity:0; width:0; height:0; }"
             ".fvx .tabs { display:flex; flex-wrap:wrap; gap:.4rem; margin:.5rem 0 .9rem; }"
             ".fvx .tab { padding:.38rem .8rem; border:1.5px solid #c3c2b7;"
             " border-radius:.55rem; cursor:pointer; font-weight:600; font-size:.9rem;"
             " transition:background .12s; }"
             ".fvx .tab:hover { background:#f2f1ec; }"
             ".fvx .pane { display:none; }"
             ".fvx .pane img { max-width:100%; height:auto; }"
             + css + "</style>")
    # the standard card clamps at 54rem, which would shrink the figure. Widen it.
    style = style.replace("</style>",
                          ".fv.wide { max-width: 1220px; }</style>")
    out = (_CSS + style + "<div class='fv wide'><div class='fvx'>"
           + inputs
           + "<h4>Click a division of the same customers</h4>"
           + f"<div class='tabs'>{tabs}</div>"
           + f"<div class='panes'>{imgs}</div>"
           + "</div></div>")
    if HTML is None:
        print(f"[split_explorer: {len(panes)} divisions rendered]")
        return None
    return HTML(out)


def split_summary(federation, cohort, splits=None, local_steps=20, min_size=0,
                  metric="AUC", pair=None):
    """Figure B. The two divisions that a spread number cannot tell apart.

    The explorer above already walks a reader through all six divisions. This figure
    does the opposite: it throws five of them away and sets the two that matter side by
    side, on the three numbers that decide the argument. Both spread the positive rate
    the same distance. One is explained by the records and costs nothing; the other was
    dealt out and costs real accuracy.

    `pair` names the two divisions. Without it, the two whose spreads are closest are
    chosen, which is the same comparison stated a different way.
    """
    _style()
    rows = split_costs(federation, cohort, splits=splits, local_steps=local_steps,
                       min_size=min_size)
    by_label = dict(rows)
    if pair is None:
        finite = [(l, r) for l, r in rows if r["spread"] != float("inf")]
        pair = min(((a[0], b[0]) for i, a in enumerate(finite) for b in finite[i + 1:]),
                   key=lambda p: (max(by_label[p[0]]["spread"], by_label[p[1]]["spread"])
                                  / min(by_label[p[0]]["spread"], by_label[p[1]]["spread"]),
                                  -abs(by_label[p[0]]["cost"] - by_label[p[1]]["cost"])))
    picked = [(label, by_label[label]) for label in pair]

    fig, ax = plt.subplots(figsize=(10.6, 4.55))
    _margins(fig, takeaway=True, left=0.04, right=0.96)
    _bare(ax); ax.set_xlim(0, 10); ax.set_ylim(4.72, -0.25)

    for k, (label, rec) in enumerate(picked):
        x0 = 0.1 + k * 5.05
        # Two independent questions, so two independent tests. A division is explained
        # when a probe beats chance by a wide margin; it is expensive when the loss moves.
        explained = rec["separability"] > 2 * rec["chance"]
        costly = rec["cost"] > 0.5
        _card(ax, x0, -0.1, 4.75, 4.2, PANEL, r_px=12, z=1)
        ax.text(x0 + 0.38, 0.50, label, fontsize=13.5, fontweight="bold", color=INK,
                zorder=3)

        ax.text(x0 + 0.38, 1.12, f"how uneven the {V.positive_rate} is", fontsize=9.6,
                color=MUTED, zorder=3)
        ax.text(x0 + 0.38, 1.78, f"{rec['spread']:.0f}×", fontsize=31, fontweight="bold",
                color=INK, va="center", zorder=3)
        ax.text(x0 + 1.62, 1.80, "the same, either way", fontsize=10, color=MUTED,
                va="center", zorder=3)

        ax.plot([x0 + 0.38, x0 + 4.42], [2.38, 2.38], color="#e2e0d8", lw=1.1, zorder=2)

        ax.text(x0 + 0.38, 2.86, f"do the {V.records} explain it?", fontsize=9.6,
                color=MUTED, zorder=3)
        ax.text(x0 + 0.38, 3.42, f"{rec['separability']:.0%}", fontsize=21,
                fontweight="bold", color=ACCENT, va="center", zorder=3)
        ax.text(x0 + 1.28, 3.44,
                "yes, almost always" if explained else "no, it was dealt out",
                fontsize=10, color=INK_2, va="center", zorder=3)

        ax.text(x0 + 4.42, 2.86, "the federation pays", fontsize=9.6, color=MUTED,
                ha="right", zorder=3)
        ax.text(x0 + 4.42, 3.42, f"{rec['cost']:+.1f}%", fontsize=21, fontweight="bold",
                color=BAD if costly else GOOD_TEXT, ha="right", va="center", zorder=3)

    # The one number that needs its units spelled out, kept out of the conclusion so the
    # conclusion stays one sentence long.
    ax.text(5.0, 4.60, "\u201cthe federation pays\u201d is how much worse the federated "
            "model\u2019s predictions are than pooling\u2019s, "
            f"at {local_steps} local steps per exchange",
            ha="center", va="bottom", fontsize=9.2, color=MUTED, zorder=3)

    mult = min(r["spread"] for _, r in picked)
    _headline(fig, "Uneven is not one thing.",
              f"Two ways of dividing the same {V.members}. Both stretch the "
              f"{V.positive_rate} {mult:.0f} times over, so by that measure they are "
              "equally uneven.")
    _takeaway(fig, "Same spread, opposite bills. What decides the cost is not how uneven "
                   f"a division is, but whether the {V.records} explain it.")
    return _done(fig)



def federation_diagram(federation=None):
    """Figure 1. Updates travel. Records do not.

    The model is drawn honestly everywhere it lives: a strip of 11 numbers, one
    copy at the coordinator and one copy inside every participant. What crosses the
    network is exactly that strip; the records sit under a padlock and never do.
    """
    _style()
    fig, ax = plt.subplots(figsize=(10.2, 7.0))
    _margins(fig, takeaway=True)
    _bare(ax); ax.set_xlim(0, 10); ax.set_ylim(0.42, 7.02)

    names = list(SITE_COLOUR)
    counts = {h.name: h.n for h in federation} if federation is not None else {}
    P = fc.N_PARAMS

    # ---- the coordinator: one shared model, in the federation's violet
    _card(ax, 3.25, 5.15, 3.5, 1.62, _mix(ACCENT, 0.93), r_px=13, shadow=True)
    ax.text(5.0, 6.45, "coordinator", ha="center", fontsize=13,
            fontweight="bold", color=INK)
    _model_strip(ax, 5.0, 5.92, ACCENT, cell=0.17, gap=0.05, n=P)
    ax.text(5.0, 5.68, f"the shared model — {P} numbers", ha="center",
            fontsize=8.5, color=INK_2)
    ax.text(5.0, 5.38, f"averages the {V.sites}' updates, weighted by size",
            ha="center", fontsize=9, color=INK_2)

    # ---- four participants, each with its copy of the model and its locked records
    xs = [1.35, 3.78, 6.22, 8.65]
    for x, name in zip(xs, names):
        c = SITE_COLOUR[name]
        _card(ax, x - 1.10, 0.55, 2.20, 3.05, _mix(c, 0.91), r_px=13, shadow=True)
        ax.text(x, 3.26, _short(name), ha="center", fontsize=11.5,
                fontweight="bold", color=INK)
        n = counts.get(name)
        ax.text(x, 2.94, f"{n} {V.members}" if n else V.records,
                ha="center", fontsize=9, color=INK_2)

        _model_strip(ax, x, 2.48, c, cell=0.135, gap=0.032, n=P)
        ax.text(x, 2.22, "its copy of the model", ha="center", fontsize=7.8,
                color=MUTED)

        for k in range(3):                      # the records, staying put
            _cell(ax, x - 0.72, 1.26 + 0.225 * k, 1.44, 0.20,
                  "white", gap_px=1.2, r_px=3, z=3)
        _padlock(ax, x - 0.70, 0.97, colour=INK_2)
        ax.text(x - 0.50, 0.95, "records never leave", ha="left", va="center",
                fontsize=8, color=INK_2)

        # the model goes out (violet, thin) …
        ax.add_patch(FancyArrowPatch(
            (5.0 + (x - 5.0) * 0.28 - 0.16, 5.10), (x - 0.26, 3.70),
            arrowstyle="-|>", mutation_scale=11, linewidth=1.5, color=ACCENT,
            alpha=0.55, capstyle="round", connectionstyle="arc3,rad=-0.08",
            zorder=4))
        # … each participant's update comes back (its own colour, emphatic)
        ax.add_patch(FancyArrowPatch(
            (x + 0.26, 3.70), (5.0 + (x - 5.0) * 0.28 + 0.16, 5.10),
            arrowstyle="-|>", mutation_scale=14, linewidth=2.3, color=c,
            alpha=0.95, capstyle="round", connectionstyle="arc3,rad=-0.08",
            zorder=4))

    _chip(ax, 1.45, 4.55, f"the model goes out\n— the same {P} numbers —", size=9)
    _chip(ax, 8.55, 4.55, f"updates come back\n— {P} numbers each —", size=9)

    _headline(fig, f"The {V.records} never move. Only the updates do.",
              f"One small model, {P} numbers. Every {V.site} holds a copy, trains "
              f"it on its own {V.members}, and sends back only how the numbers "
              f"should change.")
    _takeaway(fig, "Everything in this project is a consequence of that one constraint.")
    return _done(fig)


def site_bars(federation, per_square=None, per_row=25, rows_cap=6):
    """Figure 2. The participants are not alike, before any model exists.

    An isotype block per participant: one square is `per_square` records, red
    squares are the positives. Size and mix land in the same glance — the biggest
    participant is a wall, the smallest a sliver.
    """
    _style()
    sites = list(federation)
    if per_square is None:
        # Keep the biggest participant to `rows_cap` rows, so one large book does not
        # turn the figure into a wall of squares that nobody counts.
        big = max(h.n for h in sites)
        per_square = max(5, int(np.ceil(big / (per_row * rows_cap))))
    blocks = []                       # (site, negative squares, positive squares)
    for h in sites:
        total = round(h.n / per_square)
        ill = round(h.n * h.share_positive / per_square)
        blocks.append((h, total - ill, ill))
    rows_of = [max(1, -(-(a + b) // per_row)) for _, a, b in blocks]

    gap_rows = 1.25
    total_rows = sum(rows_of) + gap_rows * (len(blocks) - 1)
    ax_w_in = 10.2 * (0.97 - 0.185)
    ax_h_in = total_rows * (ax_w_in / (per_row + 0.5))
    fig_h = ax_h_in + HEAD_IN + FOOT_IN
    fig, ax = plt.subplots(figsize=(10.2, fig_h))
    _margins(fig, left=0.185, right=0.97)
    _bare(ax)
    ax.set_xlim(-0.25, per_row + 0.25)
    ax.set_ylim(total_rows, -0.2)

    y = 0.0
    for (h, healthy, ill), n_rows in zip(blocks, rows_of):
        cells = [NEUTRAL] * healthy + [ILL] * ill
        for k, colour in enumerate(cells):
            _cell(ax, k % per_row, y + k // per_row, 1, 1, colour,
                  gap_px=2.6, r_px=3.5)
        tr = blended_transform_factory(ax.transAxes, ax.transData)
        ax.scatter([-0.022], [y + 0.5], transform=tr, s=52,
                   color=SITE_COLOUR[h.name], clip_on=False, zorder=6, linewidth=0)
        ax.text(-0.040, y + 0.5, _short(h.name), transform=tr, ha="right",
                va="center", fontsize=11.5, fontweight="bold", color=INK)
        ax.text(-0.040, y + 1.22, f"{h.n} {V.members}", transform=tr, ha="right",
                va="center", fontsize=9.5, color=INK_2)
        ax.text(-0.040, y + 1.86, f"{h.share_positive:.0%} {V.positive}", transform=tr,
                ha="right", va="center", fontsize=9.5, fontweight="bold",
                color=ILL)
        y += n_rows + gap_rows

    handles = [plt.Line2D([], [], marker="s", ls="", color=NEUTRAL,
                          markersize=10, label=V.negative),
               plt.Line2D([], [], marker="s", ls="", color=ILL,
                          markersize=10, label=f"{V.positive} — one square is "
                                               f"{per_square} {V.members}")]
    fig.legend(handles=handles, fontsize=10, ncols=2, loc="lower right",
               bbox_to_anchor=(0.965, 0.02), handletextpad=0.3,
               columnspacing=1.2, borderaxespad=0)

    big = max(sites, key=lambda h: h.n)
    small = min(sites, key=lambda h: h.n)
    worst = max(sites, key=lambda h: h.share_positive)
    _headline(fig, f"The {len(sites)} {V.sites} are not alike.",
              f"{_short(big.name)} holds {big.n/small.n:.0f} times "
              f"{_short(small.name)}'s {V.members}, and {worst.share_positive:.0%} of "
              f"{_short(worst.name)}'s are {V.positive}.")
    return _done(fig)




def example_record(federation, index=0, site=None, feature_names=None,
                   readable=None):
    """One record, as a table a manager could read."""
    h = federation[site] if site is not None else federation[0]
    names = feature_names or [f"feature {i}" for i in range(h.X.shape[1])]
    labels = [(readable or {}).get(n, n) for n in names]
    def _num(v):
        v = float(v)
        if v in (0.0, 1.0):
            return str(int(v))                      # a yes or no field
        return f"{v:,.1f}" if abs(v) >= 10 else f"{v:.2f}"
    rows = [(lab, _num(v)) for lab, v in zip(labels, h.X[index])]
    verdict = (f"<span class='bad'>{V.positive}</span>" if h.y[index]
               else f"<span class='good'>not {V.positive}</span>")
    body = (f"<h4>One {V.member} at {h.name}</h4>"
            + _table(["measurement", "value"], rows, ["", "n"])
            + f"<p class='note'>What actually happened: {verdict}. "
              f"{len(rows)} numbers in, one answer out. "
              "This is the whole prediction task.</p>")
    return _html(body)


# ================================================================ QUIZZES

#: Quiz text names real participants and real measured numbers, so it belongs to a
#: scenario. Populated by `use()` from `scenario.QUIZZES`.
QUIZZES = {}


def quiz(key):
    """A multiple choice check the student can actually click.

    Radio inputs with CSS-only feedback, so it works in Colab and in a saved notebook
    without any javascript. Clicking an option says right or wrong straight away; the
    reveal underneath explains why.
    """
    if key not in QUIZZES:
        raise KeyError(f"{key!r} is not in this scenario's QUIZZES. "
                       f"Did you call fedviz.use(scenario)? "
                       f"Available: {sorted(QUIZZES)}")
    q, options, answer, why = QUIZZES[key]
    gid = "fvq_" + "".join(ch for ch in key if ch.isalnum())
    opts = ""
    for i, o in enumerate(options):
        oid = f"{gid}_{i}"
        cls = "right" if i == answer else "wrong"
        opts += (f"<input class='qr' type='radio' name='{gid}' id='{oid}'>"
                 f"<label class='opt {cls}' for='{oid}'>{o}</label>")
    body = (f"<h4>{q}</h4><div class='quiz'>{opts}</div>"
            f"<p class='note'>Pick one. It will tell you straight away.</p>"
            f"<details><summary>Why</summary><p>{why}</p></details>")
    return _html(body)


# ================================================================ PART 1

def cross_grid(models, test_sites, accuracy_fn, metric="AUC", title=None,
               pooled_note=None):
    """Figure 3. Every model scored on every participant's held-out records.

    Read it down a column: those are the models competing for one participant's
    customers. Shading is normalised inside each column for exactly that reason. On a
    shared colour scale every cell here would be the same blue, because the spread
    between the best and worst model for any one participant is a few hundredths, and
    the reader would learn nothing.

    The best model in each column is ringed, which is where the story lives: it is
    usually not the participant's own.
    """
    _style()
    names = [h.name for h in test_sites]
    labels = list(models)
    M = np.array([[accuracy_fn(models[m], h.X, h.y) for h in test_sites]
                  for m in labels])
    n_rows, n_cols = M.shape
    pooled_i = next((i for i, l in enumerate(labels) if "pool" in l.lower()), None)
    gap = 0.42 if pooled_i is not None else 0.0
    pct = metric.lower().startswith(("accur", "share"))
    fmt = (lambda v: f"{v:.0%}") if pct else (lambda v: f"{v:.3f}")

    fig, ax = plt.subplots(figsize=(10.6, 7.6))
    _margins(fig, takeaway=True, left=0.255, right=0.965)
    _bare(ax)
    ax.set_xlim(-0.02, n_cols + 0.02)
    ax.set_ylim(n_rows + gap + 2.05, -1.75)

    def row_y(i):
        return i + (gap if pooled_i is not None and i > pooled_i - 1 and i == pooled_i else 0)

    # ---- column headings, no ellipses
    ax.text(n_cols / 2, -1.52, "SCORED ON THESE CUSTOMERS", ha="center",
            fontsize=9.5, fontweight="bold", color=INK_2)
    for j, name in enumerate(names):
        ax.text(j + 0.5, -0.95, _short(name), ha="center", fontsize=11,
                fontweight="bold", color=SITE_COLOUR.get(name, INK))
        ax.text(j + 0.5, -0.52, f"{test_sites[j].n:,} held out", ha="center",
                fontsize=8.8, color=MUTED)
    tr = blended_transform_factory(ax.transAxes, ax.transData)
    ax.text(-0.05, -1.52, "MODEL TRAINED AT", transform=tr, ha="right",
            fontsize=9.5, fontweight="bold", color=INK_2)

    # ---- cells, shaded within each column so the comparison that matters is visible
    ys, rnames, rcols = [], [], []
    for i, label in enumerate(labels):
        y = row_y(i)
        ys.append(y + 0.5)
        is_pooled = i == pooled_i
        rnames.append("all four pooled" if is_pooled else _short(label))
        rcols.append(POOLED_COLOUR if is_pooled else SITE_COLOUR.get(label, INK))
        for j in range(n_cols):
            col = M[:, j]
            lo, hi = col.min(), col.max()
            t = 0.5 if hi - lo < 1e-9 else (M[i, j] - lo) / (hi - lo)
            fill = SEQ(0.10 + 0.80 * t)
            _cell(ax, j, y, 1, 1, fill, gap_px=3.5, r_px=7)
            ax.text(j + 0.5, y + 0.5, fmt(M[i, j]), ha="center", va="center",
                    fontsize=13, color="white" if _luminance(fill) < 0.52 else INK,
                    fontweight="bold")

    # ---- mark a participant's own model, then ring the winner of each column
    for j in range(min(n_rows, n_cols)):
        if names[j] == labels[j]:
            _cell(ax, j, row_y(j), 1, 1, "none", edge=SITE_COLOUR.get(names[j], INK),
                  lw=2.6, gap_px=5.0, r_px=5, z=5)
    for j in range(n_cols):
        i = int(np.argmax(M[:, j]))
        _cell(ax, j, row_y(i), 1, 1, "none", edge=INK, lw=2.4, gap_px=1.0, r_px=8, z=6)

    _named_rows(ax, ys, rnames, rcols, dot_dx=-0.026, text_dx=-0.046)
    if pooled_i is not None:
        ax.text(-0.046, row_y(pooled_i) + 0.72, pooled_note or
                "one model on every record\nat once. Not allowed here.\n"
                "It is the target to beat.",
                transform=tr, ha="right", va="top", fontsize=8.6, color=MUTED,
                linespacing=1.5)

    # ---- legend, well clear of the last row
    yl = n_rows + gap + 0.85
    _cell(ax, 0.02, yl - 0.16, 0.26, 0.32, "white", edge=INK, lw=2.2,
          gap_px=0.5, r_px=5, z=6)
    ax.text(0.40, yl, "wins its column", fontsize=9.5, color=INK, va="center")
    _cell(ax, 1.72, yl - 0.16, 0.26, 0.32, "none", edge=MUTED, lw=2.2,
          gap_px=0.5, r_px=4, z=6)
    ax.text(2.10, yl, f"the {V.site}'s own model", fontsize=9.5, color=INK, va="center")
    ax.text(0.02, yl + 0.62, "Shading compares each cell against the others in its own "
            "column, because the spread within a column is what matters here.",
            fontsize=9, color=MUTED, va="center")

    _headline(fig, title or "A model can be good at home and poor next door.",
              f"Every number is {metric} on {V.members} the model never trained on. "
              f"Higher is better, and 0.5 would be guessing.")
    _takeaway(fig, "Read down a column. Those five models are all competing for the same "
                   f"{V.members}, and the ring shows which one wins. The bottom row wins "
                   "every column, and it is the one nobody is allowed to build.")
    return _done(fig)


def global_vs_worst(entries, wilson_fn=None, metric="accuracy", ax=None):
    """Figure 4. The average hides the smallest participant.

    entries: (label, global_score, worst_score, worst_n) or, when the interval cannot
             be derived from the score, (label, global, worst, worst_n, (lo, hi)).

    A Wilson interval is only valid for a proportion. If the metric is AUC the caller
    has to supply the interval — pass `wilson_fn=None` and a fifth element — because
    treating a rank statistic as a share of successes would be a made-up number.
    """
    _style()
    alone = ax is None
    if alone:
        fig, ax = plt.subplots(figsize=(10.2, 7.0))
        _margins(fig, takeaway=True, left=0.10, right=0.79)
    x = np.arange(len(entries), dtype=float)
    ax.set_xlim(-0.5 if alone else -0.72, len(entries) - (0.25 if alone else 0.72))
    # the scale follows the numbers, so an embedded panel is not mostly empty space
    tops = [e[1] for e in entries]
    bots = [min(e[2], (e[4] or (e[2], e[2]))[0]) for e in entries]
    top, bot = max(tops) + 0.028, min(bots) - (0.075 if alone else 0.072)
    ax.set_ylim(bot, top)
    pct = metric.lower().startswith(("accur", "share"))
    fmt = (lambda v: f"{v:.0%}") if pct else (lambda v: f"{v:.3f}")
    ticks = [t for t in np.arange(0.4, 1.001, 0.05) if bot + 0.05 < t < top]
    ax.set_yticks(ticks, [fmt(v) for v in ticks])
    ax.grid(axis="y"); ax.set_axisbelow(True)
    ax.spines["bottom"].set_visible(False)

    for xi, entry in zip(x, entries):
        label, g, worst, n = entry[:4]
        if len(entry) > 4 and entry[4] is not None:
            lo, hi = entry[4]
        elif wilson_fn is not None:
            lo, hi = wilson_fn(round(worst * n), n)
        else:
            lo = hi = worst
        ax.plot([xi, xi], [worst, g], color=BASELINE, lw=2.2,
                solid_capstyle="round", zorder=2)
        ax.plot([xi, xi], [lo, hi], color=FLAG, lw=1.6, zorder=3)
        for v in (lo, hi):
            ax.plot([xi - 0.035, xi + 0.035], [v, v], color=FLAG, lw=1.6, zorder=3)
        ax.scatter([xi], [g], s=150, color=CHARCOAL, zorder=5,
                   edgecolor="white", linewidth=1.6)
        ax.scatter([xi], [worst], s=150, color=FLAG, zorder=5,
                   edgecolor="white", linewidth=1.6)
        ax.text(xi + 0.09, g, fmt(g), va="center", fontsize=12.5 if alone else 11,
                fontweight="bold", color=INK)
        ax.text(xi + 0.09, worst, fmt(worst), va="center", fontsize=12.5 if alone else 11,
                fontweight="bold", color=INK)
        if not alone and xi == x[0]:
            ax.text(xi - 0.14, g, f"everyone", ha="right", va="center", fontsize=8.8,
                    color=CHARCOAL, fontweight="bold")
            ax.text(xi - 0.14, worst, f"weakest {V.site}", ha="right", va="center",
                    fontsize=8.8, color=FLAG, fontweight="bold")
        gap = (f"{round((g - worst) * 100)} pts" if pct else f"{g - worst:.3f}")
        ax.text(xi - 0.09, (g + worst) / 2, f"{gap}\nhidden", ha="right",
                va="center", fontsize=9.5, color=FLAG, linespacing=1.3)
        main, _, detail = label.partition("\n")
        ax.text(xi, bot + (0.040 if alone else 0.038), main, ha="center",
                fontsize=11.5 if alone else 9.6, fontweight="bold", color=INK)
        if detail:
            ax.text(xi, bot + (0.020 if alone else 0.017), detail, ha="center",
                    fontsize=9.5 if alone else 8.4, color=INK_2)
    ax.set_xticks([])

    handles = [
        plt.Line2D([], [], marker="o", ls="", color=CHARCOAL, markersize=10,
                   label=f"everyone's {V.members} together"),
        plt.Line2D([], [], marker="o", ls="", color=FLAG, markersize=10,
                   label=f"the worst single {V.site}"),
    ]
    if not alone:
        return None
    ax.legend(handles=handles, fontsize=10.5, loc="center left",
              bbox_to_anchor=(1.01, 0.55), borderaxespad=0, handletextpad=0.4)
    _headline(fig, "The average hides the smallest participant.",
              f"Dark dot: scored across every held-out {V.member} at once. Amber dot: "
              f"scored at each {V.site} separately, keeping the lowest of the four.")
    _takeaway(fig, f"Both columns describe the same model. Which number you quote decides "
                   f"whether the weakest {V.site} is part of the conversation.")
    return _done(fig)


def part1_close(per_site_gains, metric="AUC"):
    """Figure 4. What every participant gains by pooling, on every draw of the data.

    One row per participant, one faint dot per rebuild, the solid dot the average, and a
    real axis under all of it so a reader can see how big these gains are rather than
    take the printed number on trust. Four participants and ten draws is a small
    experiment; the spread is shown because the figure should look like one.

    Whether everyone gains is a fact about one dataset, so the headline and the takeaway
    are computed from what is actually there.
    """
    _style()
    names = list(per_site_gains)
    n = len(names)
    vals = {k: np.asarray(v, float) for k, v in per_site_gains.items()}
    draws = max(len(v) for v in vals.values())
    best = max(vals, key=lambda k: vals[k].mean())

    fig, ax = plt.subplots(figsize=(11.4, 0.94 * n + 4.3))
    _margins(fig, takeaway=True, left=0.175, right=0.965)
    _bare(ax)
    allv = np.concatenate(list(vals.values()))
    lo, hi = min(allv.min(), 0.0), allv.max()
    pad = (hi - lo) * 0.09 or 0.005
    ax.set_xlim(lo - pad, hi + pad * 7.4)
    ax.set_ylim(n - 0.35, -1.05)

    # a real axis: the gains are hundredths of AUC and that is worth being able to see
    span = hi - lo
    step = next(t for t in (0.002, 0.005, 0.01, 0.02, 0.05, 0.1) if span / t <= 8)
    ticks = np.arange(0, hi + step * 0.5, step)
    if lo < -step * 0.5:
        ticks = np.concatenate([np.arange(-step, lo - step * 0.5, -step)[::-1], ticks])
    for t in ticks:
        ax.plot([t, t], [-0.55, n - 0.45], color=GRID, lw=1.0, zorder=1)
        ax.text(t, n - 0.30, "no gain" if abs(t) < 1e-12 else f"{t:+.2f}",
                ha="center", va="top", fontsize=10.4,
                color=INK_2 if abs(t) < 1e-12 else MUTED,
                fontweight="bold" if abs(t) < 1e-12 else "normal")
    ax.plot([0, 0], [-0.55, n - 0.45], color=BASELINE, lw=1.6, zorder=2)

    for i, name in enumerate(names):
        v = vals[name]
        c = SITE_COLOUR.get(name, ACCENT)
        if name == best:                       # the row the takeaway rests on
            _cell(ax, lo - pad, i - 0.40, (hi + pad * 7.4) - (lo - pad), 0.80,
                  _mix(ACCENT, 0.94), gap_px=0, r_px=6, z=0)
        ax.plot([v.min(), v.max()], [i, i], color=_mix(c, 0.74), lw=3.0,
                solid_capstyle="round", zorder=3)
        ax.scatter(v, [i] * len(v), s=62, color=_mix(c, 0.46), zorder=4,
                   edgecolor="white", linewidth=1.1)
        ax.scatter([v.mean()], [i], s=205, color=c, zorder=6, edgecolor="white",
                   linewidth=2.0)
        ax.text(hi + pad * 1.00, i, f"{v.mean():+.3f}", va="center", fontsize=13.4,
                fontweight="bold", color=INK)
        won = int((v > 0).sum())
        ax.text(hi + pad * 3.60, i, f"ahead on {won} of {draws}", va="center",
                fontsize=10.8, color=GOOD_TEXT if won == draws else INK_2)

    _named_rows(ax, np.arange(n), [_short(x) for x in names],
                [SITE_COLOUR.get(x, ACCENT) for x in names])
    ax.text(hi + pad * 1.00, -0.75, "average gain", fontsize=9.8, color=MUTED,
            va="center")
    ax.text(hi + pad * 3.60, -0.75, "how reliably", fontsize=9.8, color=MUTED, va="center")
    ax.text(lo - pad, -0.75, f"{metric} the pooled model adds", fontsize=9.8,
            color=MUTED, va="center")

    every = all(v.min() > 0 for v in vals.values())
    _headline(fig, f"Every {V.site} does better on the pooled model.",
              f"One dot per rebuild of the data, {draws} of them, and the solid dot is the "
              f"average of those {draws}.")
    _takeaway(fig, (f"{_short(best)} gains most and " +
                    (f"gains on all {draws} draws" if every else "gains on average") +
                    f", so the smallest {V.site} has the strongest case for joining. "
                    "Pooling is also the one thing the consortium is not allowed to do."))
    return _done(fig)


# ================================================================ PART 2

def partition_bars(partitions):
    """Figure 5. What "different data distributions" actually looks like.

    partitions: dict {regime label: Federation}. A label containing "real" is
    tagged as the division that actually exists.
    """
    _style()
    n = len(partitions)
    fig, axes = plt.subplots(n, 1, figsize=(9.8, 1.52 * n + 2.4), sharex=True)
    _margins(fig, left=0.185, right=0.97, hspace=0.75)
    axes = np.atleast_1d(axes)
    biggest = max(h.n for f in partitions.values() for h in f)

    for p, (ax, (label, fed)) in enumerate(zip(axes, partitions.items())):
        real = "real" in label.lower()
        sites = list(fed)
        ys = np.arange(len(sites))[::-1]
        ax.set_xlim(0, biggest * 1.16)
        ax.set_ylim(-0.6, len(sites) - 0.4)
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        for y, h in zip(ys, sites):
            healthy = h.n * (1 - h.share_positive)
            _pill(ax, 0, healthy, y, 0.58, NEUTRAL, r_px=5)
            _pill(ax, healthy + biggest * 0.006, h.n, y, 0.58, ILL, r_px=5)
            ax.text(h.n + biggest * 0.022, y, f"{h.n}", va="center",
                    fontsize=10, color=MUTED)
        names = [_short(h.name) for h in sites]
        cols = ([SITE_COLOUR.get(h.name, MUTED) for h in sites] if real
                else [BASELINE] * len(sites))
        _named_rows(ax, ys, names, cols, size=10.5, dot_dx=-0.020, text_dx=-0.036)
        ax.text(0, 1.24, label, transform=ax.transAxes, fontsize=11.5,
                fontweight="bold", color=INK, va="top")
        tag, fill, tc = (("THE REAL SPLIT", INK, "white") if real
                         else ("SYNTHETIC", PANEL, MUTED))
        ax.text(1.0, 1.24, tag, transform=ax.transAxes, fontsize=8,
                fontweight="bold", color=tc, ha="right", va="top",
                bbox=dict(boxstyle="round,pad=0.42", facecolor=fill,
                          edgecolor="none"))
    axes[-1].set_xticks(_nice_ticks(biggest))
    axes[-1].set_xlabel(V.members)

    handles = [plt.Line2D([], [], marker="s", ls="", color=NEUTRAL,
                          markersize=11, label=V.negative),
               plt.Line2D([], [], marker="s", ls="", color=ILL,
                          markersize=11, label=V.positive)]
    fig.legend(handles=handles, fontsize=10.5, ncols=2, loc="lower right",
               bbox_to_anchor=(0.965, 0.012), handletextpad=0.3,
               columnspacing=1.3, borderaxespad=0)

    _total = sum(h.n for h in list(partitions.values())[0])
    _headline(fig, f"The same {_total:,} {V.members}, divided {len(partitions)} ways.",
              "Only one of these divisions happened. The other three show what "
              "‘different data’ can mean.")
    return _done(fig)


def gain_stability(per_site_gains, metric="AUC", title=None, takeaway=None, ax=None):
    """Part 2. What each participant gains by federating, on every draw of the data.

    One row per participant. Every draw is a light dot, the mean is the solid one, so a
    reader sees the spread rather than being asked to trust an average. Four banks and
    ten draws is a small experiment and the figure should look like one.

    per_site_gains: {name: [gain on draw 1, gain on draw 2, ...]}

    Whether everyone gains is a fact about one dataset, so the title and the takeaway
    are computed from what is actually there, and a scenario can override both.
    """
    _style()
    names = list(per_site_gains)
    n = len(names)
    alone = ax is None
    if alone:
        fig, ax = plt.subplots(figsize=(10.6, 1.05 * n + 4.4))
        _margins(fig, takeaway=True, left=0.235, right=0.955)
    _bare(ax)
    allv = [v for vs in per_site_gains.values() for v in vs]
    lo, hi = min(allv + [0.0]), max(allv)
    pad = (hi - lo) * 0.16 or 0.01
    ax.set_xlim(lo - pad, hi + pad * (2.4 if ax is None else 1.15))
    ax.set_ylim(n + (1.30 if ax is None else -0.42), -0.92)
    ax.plot([0, 0], [-0.55, n - 0.45], color=BASELINE, lw=1.4, zorder=1)
    ax.text(0, -0.70, "no gain", ha="center", va="bottom", fontsize=9.5, color=MUTED)

    ys = np.arange(n)
    for i, name in enumerate(names):
        vs = np.asarray(per_site_gains[name], float)
        c = SITE_COLOUR.get(name, ACCENT)
        ax.plot([vs.min(), vs.max()], [i, i], color=_mix(c, 0.72), lw=2.4,
                solid_capstyle="round", zorder=2)
        ax.scatter(vs, [i] * len(vs), s=52, color=_mix(c, 0.5), zorder=3,
                   edgecolor="white", linewidth=1.0)
        ax.scatter([vs.mean()], [i], s=170, color=c, zorder=5, edgecolor="white",
                   linewidth=1.8)
        ax.text(hi + pad * 0.55, i, f"{vs.mean():+.3f}", va="center",
                fontsize=12 if alone else 11, fontweight="bold", color=INK)
        won = int((vs > 0).sum())
        if alone:
            ax.text(hi + pad * 1.55, i, f"ahead in {won} of {len(vs)}", va="center",
                    fontsize=9.5, color=GOOD_TEXT if won == len(vs) else INK_2)
    _named_rows(ax, ys, [_short(x) for x in names],
                [SITE_COLOUR.get(x, ACCENT) for x in names])

    best = max(names, key=lambda k: np.mean(per_site_gains[k]))
    bvals = np.asarray(per_site_gains[best], float)
    rest = [np.mean(per_site_gains[k]) for k in names if k != best] or [0.0]
    draws = max(len(v) for v in per_site_gains.values())
    every = all(np.asarray(v).min() > 0 for v in per_site_gains.values())
    all_mean_positive = all(np.mean(v) > 0 for v in per_site_gains.values())

    if title is None:
        if every:
            title = f"Every {V.site} gains, on every draw."
        elif all_mean_positive:
            title = f"Every {V.site} gains on average, but not on every draw."
        else:
            title = f"Federating helps some {V.sites} and not others."
    if alone:
        ax.text(lo - pad, n - 0.05, "WHAT ONE DOT IS", fontsize=9.5, fontweight="bold",
                color=INK_2, va="top")
        ax.text(lo - pad, n + 0.32,
                f"We rebuilt the {V.sites}' books {draws} times, so each rebuild gives them a "
                f"different set of {V.members}. For each rebuild we\n"
                f"trained one {V.site} on its own book, trained a second model on all four books "
                f"together, and scored both on the same\n"
                f"held-out {V.members}. One dot is the difference between those two "
                f"{metric} scores. Right of the line means the shared model won.",
                fontsize=9, color=INK_2, va="top", linespacing=1.7)
        _headline(fig, title, f"Each row is one {V.site}. Each dot is one rebuild of the "
                  f"data, and the solid dot is the average of the {draws}.")

    if takeaway is None:
        won = int((bvals > 0).sum())
        ratio = float(np.mean(bvals) / max(np.mean(rest), 1e-9))
        lead = (f"{_short(best)} gains {ratio:.1f} times what the others average"
                if np.mean(rest) > 0 else f"{_short(best)} is the only clear gainer")
        takeaway = (f"{lead}, and gains on {won} of {len(bvals)} draws. "
                    + ("Nobody is worse off on any draw, so the case for joining does not "
                       "depend on which year you look at."
                       if every else
                       "The spread matters as much as the average here: a single draw would "
                       "have told a different story for at least one participant."))
    if not alone:
        return None
    _takeaway(fig, takeaway)
    return _done(fig)


def fedsgd_round(federation=None):
    """Part 3's opening figure: one complete FedSGD round, six numbered moves.

    Replaces the old pair (federation_diagram + round_diagram). Those two split one
    story across two canvases, drew five local steps where part 3 teaches one, and the
    round diagram ended on the update being applied, which answers the question part 3
    is about to pose. This one ends on the question: four directions have arrived, and
    how they become one is deliberately not shown.

    Reading order is a clockwise circuit. Initialise at the coordinator's left, the
    model travels down, one local step happens inside each participant, directions
    travel up, combine poses its question at the coordinator's right, and a dashed arc
    carries "repeat" back to the start.
    """
    _style()
    fig, ax = plt.subplots(figsize=(12.4, 8.9))
    _margins(fig, takeaway=True, left=0.02, right=0.98)
    _bare(ax)
    ax.set_xlim(0, 12.4)
    ax.set_ylim(0, 9.0)

    names = list(SITE_COLOUR)
    sizes = {h.name: h.n for h in federation} if federation is not None else {}

    def badge(x, y, n):
        ax.scatter([x], [y], s=300, color=ACCENT, zorder=6)
        ax.text(x, y - 0.008, str(n), ha="center", va="center", fontsize=11,
                fontweight="bold", color="white", zorder=7)

    def step(x, y, n, name, line, ha="left", width=None):
        badge(x, y, n)
        tx = x + 0.32 if ha == "left" else x - 0.32
        ax.text(tx, y + 0.02, name, ha=ha, va="center", fontsize=10.5,
                fontweight="bold", color=INK)
        ax.text(tx, y - 0.34, line, ha=ha, va="top", fontsize=9,
                color=INK_2, linespacing=1.45, wrap=False)

    # ------------------------------------------------------------ coordinator
    C0, C1, CB, CT = 0.55, 11.85, 6.30, 7.85
    _card(ax, C0, CB, C1 - C0, CT - CB, PANEL, r_px=12)
    ax.text(C0 + 0.28, CT - 0.24, "THE COORDINATOR", fontsize=9,
            fontweight="bold", color=MUTED, va="center")

    # 1 · initialise, and the model itself
    step(1.05, CT - 0.78, 1, "INITIALISE", "fourteen numbers,\nall zeros to start")
    mx0, mx1, my = 3.10, 4.95, CT - 0.98
    _card(ax, mx0, my, mx1 - mx0, 0.62, ACCENT, r_px=9, z=4)
    ax.text((mx0 + mx1) / 2, my + 0.31, "the model · 14 numbers", ha="center",
            va="center", fontsize=9.5, fontweight="bold", color="white", zorder=5)

    # 5 · combine, ending on the question
    step(6.55, CT - 0.55, 5, "COMBINE", "")
    chip_y = CT - 1.12
    for k, name in enumerate(names):
        _card(ax, 6.45 + k * 0.52, chip_y, 0.40, 0.40, SITE_COLOUR[name], r_px=5, z=4)
    ax.text(8.72, chip_y + 0.20, "?", ha="center", va="center", fontsize=30,
            fontweight="bold", color=INK)
    ax.text(9.20, chip_y + 0.20, "four directions, one model.\nhow should they be "
            "combined?", ha="left", va="center", fontsize=10, fontweight="bold",
            color=INK, linespacing=1.5)

    # 6 · the repeat arc, dashed, back over the top to the model
    ax.add_patch(FancyArrowPatch((9.35, CT + 0.10), (4.05, CT + 0.10),
                                 connectionstyle="arc3,rad=0.18",
                                 arrowstyle="-|>", mutation_scale=13, lw=1.6,
                                 linestyle=(0, (4, 3)), color=MUTED, zorder=3))
    badge(4.45, CT + 0.86, 6)
    ax.text(4.77, CT + 0.88, "UPDATE AND REPEAT", fontsize=10.5, fontweight="bold",
            color=INK, va="center")
    ax.text(7.15, CT + 0.88, "once the rule is settled: apply it, send the model "
            "out again", fontsize=9, color=INK_2, va="center")

    # ------------------------------------------------------------ the banks
    n_b = len(names)
    bw, gap = 2.50, 0.30
    x0 = (12.4 - (n_b * bw + (n_b - 1) * gap)) / 2
    BB, BT = 1.15, 4.15
    for k, name in enumerate(names):
        bx = x0 + k * (bw + gap)
        col = SITE_COLOUR[name]
        _card(ax, bx, BB, bw, BT - BB, "white", edge=BASELINE, lw=1.2, r_px=10)
        ax.text(bx + 0.22, BT - 0.34, _short(name), fontsize=11.5,
                fontweight="bold", color=col, va="center")
        if sizes:
            ax.text(bx + 0.22, BT - 0.68, f"{sizes[name]:,} {V.members}",
                    fontsize=8.8, color=MUTED, va="center")

        # records: three grey bars and a padlock. They are drawn once and never move.
        ry = BB + 1.42
        for i in range(3):
            _card(ax, bx + 0.24, ry + i * 0.21, 1.30, 0.13, NEUTRAL, r_px=3, z=3)
        _padlock(ax, bx + 1.80, ry + 0.30, s=0.15, colour=INK_2)
        ax.text(bx + 0.24, ry - 0.26, f"{V.records} stay here", fontsize=8.8,
                fontweight="bold", color=INK_2, va="center")

        # what one local step makes: this participant's direction
        _card(ax, bx + 0.24, BB + 0.28, 2.02, 0.48, _mix(col, 0.12), r_px=7, z=3)
        ax.text(bx + 0.24 + 1.01, BB + 0.52, "its direction · 14 numbers",
                ha="center", va="center", fontsize=8.6, fontweight="bold",
                color="white" if _luminance(_mix(col, 0.12)) < 0.52 else INK,
                zorder=4)

    step(x0 + 0.10, 0.62, 3, "COMPUTE LOCALLY",
         "one training step on its own records makes one direction.   "
         "one step, not several: that is what makes this FedSGD.")

    # ------------------------------------------------- what travels, both ways
    for k, name in enumerate(names):
        bx = x0 + k * (bw + gap) + bw / 2
        ax.add_patch(FancyArrowPatch((bx - 0.38, CB - 0.04), (bx - 0.38, BT + 0.06),
                                     arrowstyle="-|>", mutation_scale=15, lw=3.0,
                                     color=ACCENT, zorder=3))
        ax.add_patch(FancyArrowPatch((bx + 0.38, BT + 0.02), (bx + 0.38, CB - 0.10),
                                     arrowstyle="-|>", mutation_scale=13, lw=2.1,
                                     color=SITE_COLOUR[name], zorder=3))

    # 2 and 4 sit beside the arrows, stacked and centred so nothing touches them
    mid = (CB + BT) / 2
    badge(0.90, mid + 0.62, 2)
    ax.text(0.90, mid + 0.22, "DISTRIBUTE", ha="center", va="center",
            fontsize=10.5, fontweight="bold", color=INK)
    ax.text(0.90, mid - 0.12, "the same model,\nto every " + V.site, ha="center",
            va="top", fontsize=9, color=INK_2, linespacing=1.45)
    badge(11.50, mid + 0.62, 4)
    ax.text(11.50, mid + 0.22, "RETURN", ha="center", va="center",
            fontsize=10.5, fontweight="bold", color=INK)
    ax.text(11.50, mid - 0.12, "14 numbers each.\n" + V.records + "\nnever travel",
            ha="center", va="top", fontsize=9, color=INK_2, linespacing=1.45)

    _headline(fig, "One federated round, step by step.",
              "Six moves. The method is FedSGD, one pass through the loop is an "
              "exchange, and the loop ends on a question.")
    _takeaway(fig, "One shared model travels between the coordinator and the "
                   f"{V.sites}. The {V.members}' records never do.")
    return _done(fig)


def round_diagram(federation=None, hero=None):
    """Figure 6. One exchange, drawn in full for one participant.

    The hero participant's round, explicitly: the shared model arrives, training steps
    on the local records visibly change some of its numbers, and only the
    difference — a strip that is mostly empty — travels back. The other three
    participants run the same round in miniature; every arrow is a straight line.
    """
    _style()
    fig, ax = plt.subplots(figsize=(11.0, 7.7))
    _margins(fig)
    _bare(ax); ax.set_xlim(0, 10.7); ax.set_ylim(0.25, 7.55)

    names = list(SITE_COLOUR)
    counts = {h.name: h.n for h in federation} if federation is not None else {}
    hero = hero or names[0]
    hero_c = SITE_COLOUR[hero]
    names = [hero] + [n for n in names if n != hero]

    #: the model before training, how training moves it, and after. Hand-tuned so that
    #: some numbers visibly move and most do not, then cycled to the model's real width.
    _S = [0.30, 0.38, 0.22, 0.52, 0.45, 0.28, 0.30, 0.58, 0.40, 0.42, 0.35]
    _D = [0.26, 0, -0.30, 0, 0.38, 0, 0, -0.30, 0.34, 0, 0.30]
    S = [_S[i % len(_S)] for i in range(fc.N_PARAMS)]
    D = [_D[i % len(_D)] for i in range(fc.N_PARAMS)]
    T = [min(max(s - d, 0.04), 0.88) for s, d in zip(S, D)]
    DELTA = ["white" if d == 0 else max(0.85 - abs(d) * 1.6, 0.20) for d in D]

    def badge(x, y, k, colour=CHARCOAL):
        ax.text(x, y, str(k), ha="center", va="center", fontsize=9.5,
                fontweight="bold", color="white", zorder=9,
                bbox=dict(boxstyle="circle,pad=0.28", facecolor=colour,
                          edgecolor="none"))

    def delta_chip(x, y, colour, s=0.27):
        _card(ax, x - s / 2, y - s / 2, s, s, colour, r_px=3.5, z=7)
        ax.text(x, y, "Δ", ha="center", va="center", fontsize=8.5,
                fontweight="bold", color="white", zorder=8)

    # =========================== the coordinator, averaging four Δ terms
    _card(ax, 2.9, 5.55, 4.6, 1.85, _mix(ACCENT, 0.93), r_px=13, shadow=True)
    badge(3.32, 7.10, 4, ACCENT)
    ax.text(3.56, 7.10, "weighted average", fontsize=11, fontweight="bold",
            color=INK, va="center")
    ax.text(7.28, 7.10, "coordinator", fontsize=8.5, color=MUTED,
            va="center", ha="right")
    for k, name in enumerate(names):
        x = 3.42 + k * 1.02
        delta_chip(x, 6.52, SITE_COLOUR[name])
        w_lab = f"×{counts[name]}" if counts else "× size"
        ax.text(x + 0.18, 6.52, w_lab, fontsize=8, color=INK_2, va="center")
        if k < len(names) - 1:
            ax.text(x + 0.80, 6.52, "+", fontsize=9, color=MUTED,
                    va="center", ha="center")
    ax.add_patch(FancyArrowPatch((5.2, 6.30), (5.2, 6.06), arrowstyle="-|>",
                                 mutation_scale=10, lw=1.4, color=INK_2, zorder=6))
    _model_strip(ax, 5.2, 5.88, ACCENT, cell=0.12, gap=0.033)
    ax.text(5.2, 5.68, "the new shared model — sent out again",
            fontsize=8, color=INK_2, ha="center")

    # =========================== the hero participant, one round in detail
    _card(ax, 0.25, 1.02, 4.0, 3.83, _mix(hero_c, 0.91), r_px=13, shadow=True)
    n_hero = counts.get(hero)
    ax.text(2.25, 4.58, _short(hero) + (f" — {n_hero} {V.members}" if n_hero else ""),
            ha="center", fontsize=11.5, fontweight="bold", color=INK)

    cx = 2.75
    _model_strip(ax, cx, 4.10, ACCENT, cell=0.125, gap=0.03, shades=S)
    ax.text(cx, 3.90, "the shared model arrives", fontsize=8, color=INK_2,
            ha="center")
    ax.add_patch(FancyArrowPatch((cx, 3.82), (cx, 3.64), arrowstyle="-|>",
                                 mutation_scale=10, lw=1.6, color=CHARCOAL,
                                 zorder=6))

    # the training stage: five gradient steps, fed by the padlocked records
    _card(ax, 1.90, 2.98, 1.70, 0.62, "white", r_px=8, z=3)
    badge(1.90, 3.60, 2)
    step_x = np.linspace(2.15, 3.35, 5)
    ax.plot(step_x, [3.38] * 5, color=_mix(ACCENT, 0.55), lw=1.2, zorder=4)
    ax.scatter(step_x, [3.38] * 5, s=34, color=ACCENT, zorder=5,
               edgecolor="white", linewidth=0.9)
    ax.text(2.75, 3.12, "5 training steps", fontsize=7.8, color=INK_2,
            ha="center")
    for k in range(3):
        _cell(ax, 0.45, 2.98 + 0.22 * k, 1.10, 0.19, "white",
              gap_px=1.2, r_px=3, z=3)
    _padlock(ax, 0.60, 2.72, colour=INK_2)
    ax.text(0.78, 2.70, "records stay", ha="left", va="center",
            fontsize=7.5, color=INK_2)
    ax.add_patch(FancyArrowPatch((1.60, 3.28), (1.86, 3.28), arrowstyle="-|>",
                                 mutation_scale=9, lw=1.6, color=hero_c,
                                 zorder=6))

    ax.add_patch(FancyArrowPatch((cx, 2.94), (cx, 2.76), arrowstyle="-|>",
                                 mutation_scale=10, lw=1.6, color=CHARCOAL,
                                 zorder=6))
    _model_strip(ax, cx, 2.58, hero_c, cell=0.125, gap=0.03, shades=T)
    ax.text(2.65, 2.38, "after training: some numbers moved", fontsize=8,
            color=INK_2, ha="center")

    # the difference — mostly empty slots, and that is the point
    badge(0.90, 2.06, 3)
    ax.text(1.12, 2.06, "what goes back: the difference", fontsize=8.5,
            fontweight="bold", color=INK, ha="left", va="center")
    _model_strip(ax, cx, 1.56, hero_c, cell=0.125, gap=0.03, shades=DELTA)
    total = 11 * 0.125 + 10 * 0.03
    for k, d in enumerate(D):
        if d:
            cxk = cx - total / 2 + k * 0.155 + 0.0625
            ax.text(cxk, 1.76, "▲" if d > 0 else "▼", fontsize=5.5,
                    color=hero_c, ha="center", va="center", zorder=7)
    ax.text(cx, 1.32, "Δ = after − before · mostly small numbers, zero records",
            fontsize=7.6, color=INK_2, ha="center", va="center")

    # ① out and ③ back — straight lines only
    ax.add_patch(FancyArrowPatch((3.30, 5.52), (2.60, 4.90), arrowstyle="-|>",
                                 mutation_scale=13, lw=2.0, color=ACCENT,
                                 alpha=0.85, zorder=6))
    badge(1.30, 5.35, 1, ACCENT)
    ax.text(1.50, 5.35, "the model goes out", fontsize=8.5, color=INK_2,
            va="center")
    ax.add_patch(FancyArrowPatch((3.62, 1.64), (4.35, 5.52), arrowstyle="-|>",
                                 mutation_scale=14, lw=2.2, color=hero_c,
                                 zorder=6))
    delta_chip(4.13, 4.35, hero_c)

    # =========================== the other three, same round in miniature
    mini_x = [5.60, 7.30, 9.00]
    landings = [5.70, 6.50, 7.30]
    for name, x0, land in zip(names[1:], mini_x, landings):
        c = SITE_COLOUR[name]
        mcx = x0 + 0.75
        _card(ax, x0, 0.75, 1.50, 1.95, _mix(c, 0.91), r_px=10, shadow=True)
        n = counts.get(name)
        ax.text(mcx, 2.44, _short(name) + (f" · {n}" if n else ""),
                ha="center", fontsize=9, fontweight="bold", color=INK)
        _model_strip(ax, mcx, 2.02, c, cell=0.07, gap=0.02)
        for k in range(2):
            _cell(ax, mcx - 0.45, 1.42 + 0.20 * k, 0.90, 0.16, "white",
                  gap_px=1.0, r_px=2.5, z=3)
        _padlock(ax, mcx - 0.38, 1.12, s=0.11, colour=INK_2)
        ax.text(mcx - 0.24, 1.10, "records stay", ha="left", va="center",
                fontsize=6.8, color=INK_2)
        ax.add_patch(FancyArrowPatch((mcx, 2.74), (land, 5.52),
                                     arrowstyle="-|>", mutation_scale=11,
                                     lw=1.8, color=c, alpha=0.9, zorder=5))
        t = 0.45
        delta_chip(mcx + t * (land - mcx), 2.74 + t * (5.52 - 2.74), c, s=0.22)
    ax.text(8.05, 0.42, "… and the other three run the same round, "
            "at the same time", fontsize=8.5, color=INK_2, ha="center",
            va="center")

    _headline(fig, "One exchange, drawn in full.",
              f"{_short(hero)}'s round, explicitly: the model arrives, training "
              "steps on its records move some of its numbers, and only that "
              "difference travels back. The coordinator averages.")
    return _done(fig)


def aggregation_table(rows):
    """Figure 7. Weighted reproduces pooled training. A plain average does not."""
    body = ("<h4>One federated step against the pooled step</h4>"
            + _table(["how the four directions were combined",
                      "difference from the pooled direction"], rows, ["", "n"])
            + "<p class='note'>The first row is the whole point of the derivation: it is not "
              "close, it is the same calculation.</p>")
    return _html(body)


def convergence_overlay(federated_curve, pooled_curve, converged=None,
                        label="federated"):
    """Two curves lying exactly on top of each other, which is the whole point.

    `pooled_curve` must be pooled training measured at THE SAME number of steps the
    federation has had exchanges. An earlier version compared the federated curve against
    pooled training run to convergence and drew the shortfall as a closing gap. That gap
    was twenty steps against twenty thousand, not federation against pooling, and it
    invited exactly the reading the next cell has to argue against: that federating gets
    you close. Matched step for step the two are identical, so the honest picture is one
    curve hidden underneath another.

    `converged` is optional and drawn as a faint rule: where pooled training ends up if
    you let it run, which is what "how soon can we stop" later measures against.
    """
    _style()
    fed = np.asarray(federated_curve, float)
    pool = np.asarray(pooled_curve, float)
    fig, ax = plt.subplots(figsize=(9.6, 5.6))
    _margins(fig, left=0.09, right=0.965)
    x = np.arange(1, len(fed) + 1)
    ax.grid(axis="y"); ax.set_axisbelow(True)

    lo = min(fed.min(), pool.min(), converged if converged is not None else fed.min())
    top = max(fed.max(), pool.max())
    if converged is not None:
        ax.axhline(converged, color=BASELINE, lw=1.5, ls=(0, (5, 4)), zorder=1)
        ax.text(x[-1], converged - (top - lo) * 0.028,
                "where pooled training ends up if you let it run", ha="right", va="top",
                fontsize=9.5, color=INK_2)

    # pooled underneath, thick; federated dashed on top, so both are visible at once
    ax.plot(x, pool, color=POOLED_COLOUR, lw=7.5, solid_capstyle="round", alpha=0.42,
            zorder=2)
    ax.plot(x, fed, color=ACCENT, lw=2.4, ls=(0, (5, 3.2)), zorder=3)
    ax.scatter(x[-1], fed[-1], s=64, color=ACCENT, zorder=5, edgecolor="white",
               linewidth=1.5)

    ax.set_ylim(lo - (top - lo) * 0.16, top + (top - lo) * 0.10)
    mid = max(1, len(x) // 3)
    ax.text(x[mid] + 0.6, fed[mid] + (top - lo) * 0.115,
            f"{label}, four {V.sites} exchanging directions", fontsize=10.5,
            color=ACCENT, fontweight="bold")
    ax.text(x[mid] + 0.6, fed[mid] + (top - lo) * 0.048,
            "pooled training, the same steps taken in one place",
            fontsize=10.5, fontweight="bold", color=INK_2)

    ax.set_xticks([t for t in (1, 5, 10, 15, 20, 25, 30) if t <= len(x)])
    ax.set_xlabel("network exchanges, and steps of pooled training")
    ax.set_ylabel("test loss  (lower is better)")

    gap = float(np.abs(fed - pool).max())
    _headline(fig, "Sending directions reproduces pooled training.",
              f"There are two curves here. The largest difference between them over "
              f"{len(fed)} exchanges is {gap:.0e}, so the dashed one covers the other "
              f"exactly.")
    return _done(fig)


def methods_overview(rows, takeaway=None):
    """The four methods, as four cards. A summary, not a diagram.

    rows: (part, name, what it does in plain words, what you measured it to do).

    Rebuilt from scratch. The version before this drew local steps as dot columns and
    network traffic as a second column, which was a lot of apparatus for a figure whose
    only job at the end of part 7 is to remind a reader of four things they have already
    done. Four cards, three short lines each, nothing to decode.
    """
    _style()
    n = len(rows)
    hh = 1.30 * n + 2.35
    fig, ax = plt.subplots(figsize=(10.6, hh))
    _margins(fig, takeaway=True, left=0.03, right=0.985)
    _bare(ax)
    ax.set_xlim(0, 10.6)
    ax.set_ylim(0, 1.30 * n)

    for i, (part, name, does, got) in enumerate(rows):
        y = 1.30 * (n - i - 1)
        _card(ax, 0.12, y + 0.13, 10.36, 1.04, PANEL, r_px=10)
        _pill(ax, 0.42, 1.62, y + 0.86, 0.40, _mix(ACCENT, 0.78), r_px=6)
        ax.text(1.02, y + 0.86, part, ha="center", va="center", fontsize=9,
                fontweight="bold", color=ACCENT, zorder=5)
        ax.text(1.92, y + 0.86, name, va="center", fontsize=13, fontweight="bold",
                color=INK)
        ax.text(1.92, y + 0.47, does, va="center", fontsize=10, color=INK_2)
        ax.text(10.20, y + 0.66, got, ha="right", va="center", fontsize=10.5,
                fontweight="bold", color=ACCENT, linespacing=1.5)

    _headline(fig, "Four methods, and what each one turned out to be for.",
              "Everything the notebook set out to teach, with the number you measured "
              "rather than the claim you were given.")
    _takeaway(fig, takeaway or
              "Not one of these is a privacy mechanism on its own. Three of them "
              "decide how the training is arranged, and only the last one changes "
              "what an update gives away.", per_line=88)
    return _done(fig)

def exchange_table(rows, headers):
    """The per-split exchange counts. Never averaged: the outcome is bimodal."""
    def fmt(v):
        if v is None:
            return "<span class='bad'>never</span>"
        return f"<span class='good'>{v}</span>" if v <= 4 else str(v)
    out = [[r[0]] + [fmt(v) for v in r[1:]] for r in rows]
    body = ("<h4>Exchanges needed to match pooled training</h4>"
            + _table(headers, out, [""] + ["n"] * (len(headers) - 1))
            + "<p class='note'>Reported per split rather than averaged. Above five local "
              "steps the outcome is not a slower decline, it is a coin flip.</p>")
    return _html(body)


#: Okabe–Ito hues, colour-blind safe, one per local-step setting.
OKABE = ["#0072B2", "#E69F00", "#009E73", "#CC79A7"]


def two_panel(curves, pooled_value, target=None):
    """Part 4's central figure: the same runs, counted on two meters.

    curves: dict {label: (exchanges array, steps-per-bank array, loss array)}, one entry
    per local-step setting. `target` is the practical stopping line, defaulting to one
    per cent above `pooled_value`. A ring marks where each run first crosses it, with
    the count printed beside the ring, so the figure carries the measured comparison
    rather than gesturing at it.

    Every setting keeps one colour across both panels: plain solid lines, named in a
    legend on the left panel and labelled directly on the right one, where the four
    starting points sit far apart.
    """
    _style()
    target = target if target is not None else pooled_value * 1.01
    h = 6.9
    fig, axes = plt.subplots(1, 2, figsize=(11.6, h), sharey=True)
    _margins(fig, takeaway=True, left=0.075, right=0.975, wspace=0.14)
    fig.subplots_adjust(bottom=1.92 / h)          # the takeaway band needs 1.4in;
                                                  # the axis titles need the rest

    hi = max(max(c[2]) for c in curves.values())
    lo = min(min(min(c[2]) for c in curves.values()), pooled_value)
    span = hi - lo
    rings = {0: [], 1: []}
    for k, (label, (ex, steps, ls)) in enumerate(curves.items()):
        col = OKABE[k % 4]
        cross = next((i for i, v in enumerate(ls) if v <= target), None)
        for pi, (ax, xs) in enumerate(((axes[0], ex), (axes[1], steps))):
            ax.plot(xs, ls, lw=2.4, color=col, solid_capstyle="round",
                    label=label if pi == 0 else None)
            if cross is not None:
                ax.scatter([xs[cross]], [ls[cross]], s=150, facecolor="white",
                           edgecolor=col, linewidth=2.2, zorder=6)
                rings[pi].append((xs[cross], ls[cross], col))
        # on the log panel the four starts sit far apart, so the name goes there
        axes[1].annotate(label, (steps[0], ls[0]), xytext=(8, 5),
                         textcoords="offset points", va="center", fontsize=9.5,
                         fontweight="bold", color=col)

    # ring counts, stacked upward wherever two rings sit close together
    for pi, ax in enumerate(axes):
        gap = (lambda a, b: np.log10(b / a)) if pi == 1 else (lambda a, b: b - a)
        thresh = 0.07 if pi == 1 else 0.075 * max(r[0] for r in rings[0])
        prev_x = None
        for x, y, col in sorted(rings[pi]):
            # a ring crowded by its left neighbour drops its count below the line
            dy = -27 if prev_x is not None and gap(prev_x, x) < thresh else 11
            ax.annotate(f"{x:,.0f}", (x, y), xytext=(0, dy),
                        textcoords="offset points", ha="center", fontsize=10.5,
                        fontweight="bold", color=col, zorder=7)
            prev_x = x

    for ax in axes:
        ax.axhline(target, color=INK_2, lw=1.3, ls=(0, (2, 2.4)), zorder=1)
        ax.grid(axis="y"); ax.set_axisbelow(True)
        ax.set_ylim(lo - span * 0.11, hi + span * 0.06)
    axes[0].legend(fontsize=9.5, loc="upper right", frameon=False,
                   handlelength=2.6, borderaxespad=0.2)
    axes[1].text(0.02, target - span * 0.030,
                 "the stopping target: one per cent above\nwhere pooled training "
                 "finishes", transform=blended_transform_factory(
                     axes[1].transAxes, axes[1].transData),
                 fontsize=9, color=INK_2, va="top", linespacing=1.4)
    axes[1].set_xscale("log")
    axes[1].set_xticks([1, 10, 100], ["1", "10", "100"])
    axes[1].minorticks_off()

    axes[0].set_xlabel("network exchanges")
    axes[1].set_xlabel("local training steps per bank  (log scale)")
    axes[0].set_ylabel("test loss  (lower is better)")
    axes[0].set_title("counted in exchanges", fontsize=11, loc="left",
                      color=INK_2, pad=10)
    axes[1].set_title(f"counted in local steps per {V.site}", fontsize=11,
                      loc="left", color=INK_2, pad=10)

    _headline(fig, "More local work reduces exchanges, but increases computation.",
              "The same four runs on two meters. A ring marks where each first "
              "passes the stopping target, with the count beside it.")
    _takeaway(fig, "Read the rings. Counted in exchanges, fifty local steps looks "
                   "nearly twenty times cheaper than one. Counted in steps per "
                   f"{V.site}, it costs almost three times as much. The work has not "
                   "shrunk, it has moved from one meter to the other.")
    return _done(fig)


def training_rhythm(sgd_exchanges, avg_exchanges, E=5, target_label="the same target"):
    """Part 4: FedSGD and FedAvg as two timelines, same goal, different rhythm.

    One lane per method. Down arrows are the shared model going out, up arrows are the
    update coming back, dots are local training steps. FedSGD alternates one step with
    one exchange; FedAvg takes E steps between two exchanges. The measured bill sits at
    the right of each lane. Colours match the two settings' curves in `two_panel`.
    """
    _style()
    fig, ax = plt.subplots(figsize=(12.4, 6.6))
    _margins(fig, takeaway=True, left=0.02, right=0.98)
    _bare(ax)
    ax.set_xlim(0, 12.4)
    ax.set_ylim(0, 6.4)

    def lane(y_coord, y_bank, colour, name, desc):
        ax.plot([0.55, 8.55], [y_coord, y_coord], color=BASELINE, lw=1.1, zorder=1)
        ax.plot([0.55, 8.55], [y_bank, y_bank], color=BASELINE, lw=1.1, zorder=1)
        ax.text(0.55, y_coord + 0.16, "coordinator", fontsize=8.2, color=MUTED)
        ax.text(0.55, y_bank - 0.30, f"one {V.site}", fontsize=8.2, color=MUTED)
        ax.text(0.55, y_coord + 0.78, name, fontsize=13, fontweight="bold",
                color=colour)
        ax.text(2.35, y_coord + 0.80, desc, fontsize=9.5, color=INK_2, va="center")

    def swap_down(x, y_coord, y_bank):
        ax.add_patch(FancyArrowPatch((x, y_coord), (x, y_bank + 0.05),
                                     arrowstyle="-|>", mutation_scale=11, lw=2.2,
                                     color=ACCENT, zorder=3))

    def swap_up(x, y_coord, y_bank, colour):
        ax.add_patch(FancyArrowPatch((x, y_bank), (x, y_coord - 0.05),
                                     arrowstyle="-|>", mutation_scale=11, lw=2.0,
                                     color=colour, zorder=3))

    # ---- FedSGD: step, exchange, step, exchange
    cs, yc, yb = OKABE[0], 5.05, 4.15
    lane(yc, yb, cs, "FedSGD", "one local step, then an exchange. every time.")
    x = 1.30
    for cycle in range(3):
        swap_down(x, yc, yb)
        ax.scatter([x + 0.42], [yb], s=70, color=cs, zorder=4)
        swap_up(x + 0.84, yc, yb, cs)
        x += 1.55
    ax.text(x + 0.12, (yc + yb) / 2, "· · ·", fontsize=15, color=MUTED, va="center")
    ax.text(1.72, yb - 0.34, "1 step", fontsize=8.4, color=cs, ha="center",
            fontweight="bold")

    # ---- FedAvg: one exchange, E steps, one exchange
    ca, yc2, yb2 = OKABE[1], 2.55, 1.65
    lane(yc2, yb2, ca, "FedAvg", f"{E} local steps between exchanges. "
         "nothing is sent while they run.")
    swap_down(1.30, yc2, yb2)
    _card(ax, 1.72, yb2 - 0.26, 3.30, 0.52, _mix(ca, 0.82), r_px=7, z=2)
    for i in range(E):
        ax.scatter([2.05 + i * 0.66], [yb2], s=70, color=ca, zorder=4)
    ax.text(1.72 + 1.65, yb2 - 0.44, f"{E} steps, no exchange", fontsize=8.4,
            color=ca, ha="center", fontweight="bold")
    swap_up(5.44, yc2, yb2, ca)
    swap_down(6.10, yc2, yb2)
    ax.text(6.55, (yc2 + yb2) / 2, "· · ·", fontsize=15, color=MUTED, va="center")

    # ---- what each rhythm was measured to cost, at the right
    for y_mid, colour, ex, st in ((4.60, cs, sgd_exchanges, sgd_exchanges),
                                  (2.10, ca, avg_exchanges, avg_exchanges * E)):
        _card(ax, 8.95, y_mid - 0.78, 3.25, 1.56, PANEL, r_px=10)
        ax.text(9.20, y_mid + 0.42, f"reaches {target_label} in", fontsize=9,
                color=INK_2, va="center")
        ax.text(9.20, y_mid + 0.02, f"{ex} exchanges", fontsize=13.5,
                fontweight="bold", color=colour, va="center")
        ax.text(9.20, y_mid - 0.42, f"{st} local steps per {V.site}", fontsize=10.5,
                fontweight="bold", color=INK, va="center")

    ax.text(0.55, 0.52, f"↓  the shared model goes out          "
            f"↑  the {V.site}'s update comes back          ●  one local "
            "training step", fontsize=9.5, color=INK_2)

    _headline(fig, "Same goal, different training rhythm.",
              "Down arrows are the model going out, up arrows are the update coming "
              "back, dots are local training steps.")
    _takeaway(fig, "Both send the same fourteen numbers and stop at the same target. "
                   "FedAvg holds a quarter as many conversations by working "
                   f"{E} times longer between them.")
    return _done(fig)


# ================================================================ PART 5

def drift_grid(matrix, regimes, local_steps, pooled_value, highlight_row=None,
               title=None, takeaway=None, rounds=None):
    """Figure 9. Where more local work stops being safe.

    Cells show test loss after many exchanges, as % distance from pooled training.
    `highlight_row` rings the division that actually exists.
    """
    _style()
    M = np.array(matrix)
    delta = (M / pooled_value - 1) * 100
    n_rows, n_cols = M.shape

    fig, ax = plt.subplots(figsize=(9.6, 6.4))
    _margins(fig, takeaway=True, left=0.24, right=0.955)
    _bare(ax)
    ax.set_xlim(-0.02, n_cols + 0.02)
    ax.set_ylim(n_rows + 1.75, -0.10)

    norm = mpl.colors.Normalize(vmin=-2.0, vmax=2.0)
    for i in range(n_rows):
        for j in range(n_cols):
            d = delta[i, j]
            fill = DIV(norm(d))
            text = "0.0%" if abs(d) < 0.05 else f"{d:+.1f}%"
            ax.text(j + 0.5, i + 0.5, text, ha="center",
                    va="center", fontsize=12,
                    color="white" if _luminance(fill) < 0.52 else INK,
                    fontweight="bold" if d > 0.5 else "normal", zorder=4)
            _cell(ax, j, i, 1, 1, fill, gap_px=3, r_px=7)
    if highlight_row is not None:
        _cell(ax, 0, highlight_row, n_cols, 1, "none", edge=INK, lw=2.0,
              gap_px=1.0, r_px=9, z=5)

    tr = blended_transform_factory(ax.transAxes, ax.transData)
    for i, name in enumerate(regimes):
        real = i == highlight_row
        ax.text(-0.025, i + 0.5, name, transform=tr, ha="right", va="center",
                fontsize=10.5, color=INK if real else INK_2,
                fontweight="bold" if real else "normal")
    for j, e in enumerate(local_steps):
        ax.text(j + 0.5, n_rows + 0.30, str(e), ha="center", fontsize=11,
                color=INK_2)
    ax.text(n_cols / 2, n_rows + 0.76, "local steps per exchange",
            ha="center", fontsize=11, color=INK_2)

    y_leg = n_rows + 1.42
    _cell(ax, 0.02, y_leg - 0.13, 0.26, 0.26, DIV(norm(-1.2)), gap_px=0.5,
          r_px=4, z=5)
    ax.text(0.38, y_leg, "better than pooled", fontsize=9.5, color=INK_2,
            va="center")
    _cell(ax, 1.80, y_leg - 0.13, 0.26, 0.26, DIV(norm(1.4)), gap_px=0.5,
          r_px=4, z=5)
    ax.text(2.16, y_leg, "worse than pooled", fontsize=9.5, color=INK_2,
            va="center")

    _headline(fig, title or f"Only the real {V.sites} punish more local work.",
              f"Test loss after {rounds or 60} exchanges, as a distance from "
              f"where pooled "
              f"training finishes ({pooled_value:.4f}).")
    _takeaway(fig, takeaway or f"Skew is harmless until each {V.site} does a lot of "
                   "local work between exchanges. Then it compounds.")
    return _done(fig)


def rescue_table(rows, headers):
    """Figure 10. The leash makes an aggressive setting survivable."""
    def fmt(v):
        return "<span class='bad'>never</span>" if v is None else f"<span class='good'>{v}</span>"
    out = [[r[0]] + [fmt(v) for v in r[1:]] for r in rows]
    body = ("<h4>Does the leash rescue the settings that fail?</h4>"
            + _table(headers, out, [""] + ["n"] * (len(headers) - 1))
            + "<p class='note'>It does not make the model better. It makes the run land at "
              "all, and it still needs more exchanges than simply using five local steps.</p>")
    return _html(body)


# ================================================================ PART 6

def reconstruction_table(original, recovered, feature_names, readable=None,
                         update=None, title=None):
    """Figure 11. One record coming back out of an update, with the arithmetic shown.

    When `update` is supplied the table carries the division itself: each feature's Δw,
    the ratio Δw ÷ Δb, and the record it should equal. Δb is a single number shared by
    every row, so it is stated once above the table rather than repeated thirteen times.
    A reader can check any row with a calculator, which is the point of the figure.

    Called without `update` it is the plain two-column comparison. The two forms stay in
    their own unit systems on purpose: a few features are ratios in the model and amounts
    on the file, so mixing them into one row would put a ratio's name beside an amount's
    value.

    Unused by the banks notebook since part 6 was shortened: the three-measurement
    walkthrough carries the same point in less space and shows the arithmetic. Kept for
    anyone who wants the full thirteen-row version.
    """
    def num(v, dp=4):
        v = float(v)
        if v in (0.0, 1.0):
            return str(int(v))
        return f"{v:,.2f}" if abs(v) >= 10 else f"{v:.{dp}f}"

    original = np.asarray(original, float)
    recovered = np.asarray(recovered, float)
    n = len(feature_names)
    err = float(np.max(np.abs(original - recovered)))

    if update is None:
        rows = [[(readable or {}).get(f, f), num(o), num(r)]
                for f, o, r in zip(feature_names, original, recovered)]
        head = ["measurement", f"the real {V.member}", "recovered from the update"]
        aligns = ["", "n", "n"]
        intro = ""
    else:
        update = np.asarray(update, float)
        dw, db = update[:n], update[n]
        head = ["measurement", "Δw  (sent)", "Δw ÷ Δb", f"the real {V.member}", ""]
        aligns = ["", "n", "n", "n", ""]
        rows = []
        for i, f in enumerate(feature_names):
            same = abs(dw[i] / db - original[i]) < 1e-9
            rows.append([(readable or {}).get(f, f), num(dw[i]), num(dw[i] / db),
                         num(original[i]),
                         "<span class='good'>match</span>" if same else "differs"])
        intro = (f"<p class='note'>Every row divides by the same number, the bias part "
                 f"of the update: <b>Δb = {db:+.4f}</b>. Check a row with a calculator: "
                 f"the third column is the second divided by that, and it equals the "
                 f"fourth.</p>")

    body = (f"<h4>{title or 'What one update gave back'}</h4>" + intro
            + _table(head, rows, aligns)
            + f"<p class='note'>Largest disagreement across all {n} measurements: "
              f"<b>{err:.1e}</b>. That is floating-point noise. "
              f"The update was the {V.member}.</p>")
    return _html(body)


def leakage_panel(update, recovered, contributors, n_customers, n_steps, lr,
                  is_default=False, feature_names=None):
    """One configuration of part 6's leakage explorer.

    `contributors` is the block of records that went into the update, `recovered` is
    what dividing the two parts of the update returns. Reports distances and one of the
    permitted status labels. Never says "safe", "private" or "anonymous": this figure
    measures one specific attack and says only whether that attack landed.
    """
    _style()
    fig, ax = plt.subplots(figsize=(11.0, 5.9))
    _margins(fig, takeaway=True, left=0.02, right=0.98)
    _bare(ax); ax.set_xlim(0, 11.0); ax.set_ylim(0, 5.0)

    d_near = float(min(np.abs(recovered - c).mean() for c in contributors))
    d_first = float(np.abs(recovered - contributors[0]).mean())
    d_group = float(np.abs(recovered - contributors.mean(0)).mean())
    exact = d_near < 1e-9

    # ---------------------------------------------------- 1 · the configuration
    _card(ax, 0.16, 3.86, 10.68, 0.94, _mix(BAD, 0.90) if exact else PANEL, r_px=10)
    ax.text(0.42, 4.50, f"{n_customers} {V.member if n_customers == 1 else V.members}"
            f"   ×   {n_steps} local step{'' if n_steps == 1 else 's'}",
            fontsize=15, fontweight="bold", color=INK, va="center")
    ax.text(0.42, 4.13, f"learning rate {lr}, no clipping, no noise", fontsize=9.5,
            color=INK_2, va="center")
    if is_default:
        _pill(ax, 6.30, 10.66, 4.50, 0.42, _mix(ACCENT, 0.80), r_px=6)
        ax.text(8.48, 4.50, "the consortium's actual configuration", ha="center",
                va="center", fontsize=10, fontweight="bold", color=ACCENT, zorder=5)
    else:
        ax.text(10.66, 4.50, "a configuration the consortium does not use",
                ha="right", va="center", fontsize=9.5, color=MUTED)

    # ---------------------------------------------------- 2 · what the bank does
    ax.text(0.42, 3.44, "WHAT THE BANK DOES", fontsize=8.6, fontweight="bold",
            color=INK_2)
    boxes = [(0.42, "the shared\nmodel arrives", NEUTRAL),
             (3.02, f"trains on {n_customers} of\nits {V.members}" if n_customers > 1
              else f"trains on 1\n{V.member} only", _mix(ACCENT, 0.70)),
             (5.62, "sends the model\nchange only", _mix(ACCENT, 0.35))]
    for x, label, col in boxes:
        _card(ax, x, 2.44, 2.10, 0.78, col, r_px=8)
        ax.text(x + 1.05, 2.83, label, ha="center", va="center", fontsize=9,
                fontweight="bold", color=INK, linespacing=1.4)
    for x in (2.62, 5.22):
        ax.add_patch(FancyArrowPatch((x, 2.83), (x + 0.36, 2.83), arrowstyle="-|>",
                                     mutation_scale=11, lw=1.8, color=BASELINE))
    _padlock(ax, 8.20, 2.83, s=0.20, colour=INK_2)
    ax.text(8.46, 2.83, f"the {V.records} never leave", fontsize=9.5, color=INK_2,
            va="center")

    # ------------------------------------- 3 · what the coordinator receives
    ax.text(0.42, 2.06, "WHAT THE COORDINATOR RECEIVES", fontsize=8.6,
            fontweight="bold", color=INK_2)
    dw, db = update[:len(update) - 1], update[-1]
    scale = max(np.abs(dw).max(), abs(db), 1e-12)
    cw = 0.30
    for i, val in enumerate(dw):
        t = abs(val) / scale
        _cell(ax, 0.42 + i * (cw + 0.045), 1.30, cw, 0.44,
              _mix(ACCENT if val >= 0 else BAD, 1 - 0.75 * t),
              gap_px=0.0, r_px=3, z=3, edge=BASELINE, lw=0.5)
    ax.text(0.42, 1.12, f"13 weight changes  ·  Δw", fontsize=8.6, color=MUTED)
    bx = 0.42 + 13 * (cw + 0.045) + 0.22
    _cell(ax, bx, 1.30, cw, 0.44, _mix(FLAG, 0.35), gap_px=0.0, r_px=3, z=3)
    ax.text(bx + cw / 2, 1.12, "Δb", ha="center", fontsize=8.6, color=MUTED)
    ax.text(bx + cw / 2, 1.86, f"{db:+.4f}", ha="center", fontsize=9,
            color=INK_2, va="center", family=MONO)

    # ---------------------------------------------------- 4 · the attack result
    ax.plot([6.42, 6.42], [0.98, 2.14], color=GRID, lw=1.0, zorder=1)
    ax.text(6.60, 2.06, "THE ATTACK:  Δw ÷ Δb", fontsize=8.6, fontweight="bold",
            color=INK_2)
    rows = [(f"to the closest contributing {V.member}", d_near, True),
            (f"to the {V.member} we started from", d_first, False),
            ("to the contributing group's average", d_group, False)]
    for j, (label, val, lead) in enumerate(rows):
        y = 1.62 - j * 0.34
        ax.text(6.60, y, label, fontsize=9.2,
                fontweight="bold" if lead else "normal", color=INK if lead else INK_2,
                va="center")
        ax.text(10.66, y, "0.000" if val < 1e-9 else f"{val:.3f}", ha="right",
                va="center", fontsize=10.5, fontweight="bold",
                color=BAD if (lead and val < 1e-9) else INK, family=MONO)

    if exact:
        status, colour = "Exact individual reconstruction", BAD
    elif d_group < d_near:
        status, colour = "Result is closer to the contributing group", INK
    else:
        status, colour = "No exact individual reconstruction", INK
    _card(ax, 0.16, 0.16, 10.68, 0.64, _mix(BAD, 0.88) if exact else PANEL, r_px=9)
    ax.text(0.42, 0.48, status, fontsize=12, fontweight="bold", color=colour,
            va="center")
    ax.text(10.66, 0.48, "0 is a match; 1.0 is about one standard deviation apart",
            ha="right", va="center", fontsize=9, color=INK_2)

    _headline(fig, "What does this update give away?",
              f"The bank builds an update from {n_customers} "
              f"{V.member if n_customers == 1 else V.members} over {n_steps} local "
              f"step{'' if n_steps == 1 else 's'}, and the coordinator divides its two "
              "parts.")
    _takeaway(fig, "A distance is evidence about one specific attack, not a privacy "
                   "guarantee. It says whether this division recovered somebody. It "
                   "does not say the update carries nothing.")
    return _done(fig)


def leakage_explorer(site, customers=(1, 2, 8, 32, 64, 150), steps=(1, 2, 5, 10),
                     lr=0.5, default=None, gradient=None, init=None):
    """Part 6's playground: build one bank's update, then try to invert it.

    Two independent rows of choices rather than a grid of cells. The grid asked for a
    pair to be picked before either axis meant anything, and it hid the finding: these
    two axes are not equal partners. One of them decides the outcome and the other does
    nothing at all, which only shows if you can hold one still and move the other.

    Pure HTML and CSS, the same bargain part 2's explorer makes. Both rows are radio
    groups, and because every input is a sibling of every other,
    `#customer:checked ~ #step:checked ~ .panes` selects the single pane where both
    choices hold. Every configuration is rendered once and embedded, so a click costs
    nothing and a saved notebook keeps working.

    Learning rate is deliberately not a control. It cancels in the ratio exactly, so a
    third row would triple the figures to teach one sentence, which the prose says
    instead. Clipping and noise are not controls either: part 7 measures them.
    """
    import io
    import base64

    default = default or (customers[-1], steps[2] if len(steps) > 2 else steps[-1])
    panes = {}
    for n in customers:
        for st in steps:
            w = init()
            for _ in range(st):
                w = w - lr * gradient(w, site.X[:n], site.y[:n])
            update = w - init()
            recovered = update[:len(update) - 1] / update[len(update) - 1]
            fig = leakage_panel(update, recovered, site.X[:n], n, st, lr,
                                is_default=(n, st) == default)
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=84, bbox_inches="tight",
                        facecolor="white")
            plt.close(fig)
            panes[(n, st)] = base64.b64encode(buf.getvalue()).decode("ascii")

    _EXPLORER_N[0] += 1
    gid = f"fvl{_EXPLORER_N[0]}"
    cid = [f"{gid}c{i}" for i in range(len(customers))]
    sid = [f"{gid}s{j}" for j in range(len(steps))]
    lit = "{background:#4a3aa7;border-color:#4a3aa7;color:#ffffff !important;}"

    # every customer input first, then every step input, so the second is always a
    # later sibling of the first and `~` can require both at once
    inputs = "".join(
        f"<input class='lr' type='radio' name='{gid}c' id='{cid[i]}'"
        f"{' checked' if n == default[0] else ''}>" for i, n in enumerate(customers))
    inputs += "".join(
        f"<input class='lr' type='radio' name='{gid}s' id='{sid[j]}'"
        f"{' checked' if st == default[1] else ''}>" for j, st in enumerate(steps))

    ctabs = "".join(
        f"<label class='tab' for='{cid[i]}'>{n:,}"
        + ("<span>the whole book</span>" if n >= site.n else "")
        + "</label>" for i, n in enumerate(customers))
    stabs = "".join(f"<label class='tab' for='{sid[j]}'>{st}</label>"
                    for j, st in enumerate(steps))

    css = "".join(f"#{o}:checked ~ .tabs label[for='{o}']{lit}" for o in cid + sid)
    imgs = ""
    for i, n in enumerate(customers):
        for j, st in enumerate(steps):
            k = i * len(steps) + j
            imgs += (f"<div class='pane'><img alt='{n} customers, {st} steps' "
                     f"src='data:image/png;base64,{panes[(n, st)]}'></div>")
            css += (f"#{cid[i]}:checked ~ #{sid[j]}:checked ~ .panes "
                    f".pane:nth-of-type({k + 1}){{display:block;}}")

    style = ("<style>"
             ".fvl input.lr{position:absolute;opacity:0;width:0;height:0;}"
             ".fvl h4{margin:.15rem 0 .4rem;}"
             ".fvl .lx-intro{font-size:.92rem;color:#3d3c39;line-height:1.55;"
             "margin:0 0 1rem;max-width:54rem;}"
             ".fvl .lx-lab{font-size:.86rem;font-weight:700;color:#52514e;"
             "margin:.1rem 0 .35rem;}"
             ".fvl .lx-lab em{font-style:normal;font-weight:400;color:#7a7871;"
             "margin-left:.4rem;}"
             ".fvl .tabs{display:flex;flex-wrap:wrap;gap:.4rem;margin:0 0 .85rem;}"
             ".fvl .tab{padding:.34rem .8rem;border:1.5px solid #c3c2b7;"
             "border-radius:.55rem;cursor:pointer;font-weight:600;font-size:.9rem;"
             "transition:background .12s;}"
             ".fvl .tab:hover{background:#f2f1ec;}"
             ".fvl .tab span{font-weight:400;font-size:.78rem;opacity:.72;"
             "margin-left:.35rem;}"
             ".fvl .lx-note{font-size:.85rem;color:#7a7871;margin:0 0 1rem;}"
             ".fvl .pane{display:none;}.fvl .pane img{max-width:100%;height:auto;}"
             + css + ".fv.wide{max-width:1180px;}</style>")

    body = (_CSS + style + "<div class='fv wide'><div class='fvl'>" + inputs
            + "<h4>Build one bank's update, then try to invert it</h4>"
            + "<p class='lx-intro'>The bank trains on some of its own "
            + f"{V.members}, takes a few steps, and sends the model change. The "
            + f"{V.records} themselves never leave. Set what went into that update "
            + "below, and the panel shows what the coordinator received and how close "
            + "&Delta;w &divide; &Delta;b lands to the "
            + f"{V.members} behind it.</p>"
            + "<div class='lx-lab'>1 &middot; Customers in the update"
            + f"<em>how many of {site.name}'s {V.records} the local model was trained "
            + "on</em></div>"
            + f"<div class='tabs'>{ctabs}</div>"
            + "<div class='lx-lab'>2 &middot; Local steps before sending"
            + f"<em>how much training happened on those {V.records}</em></div>"
            + f"<div class='tabs'>{stabs}</div>"
            + f"<p class='lx-note'>The consortium runs <b>{default[0]:,} "
            + f"{V.members} &times; {default[1]} local steps</b>. Move one row at a "
            + "time and watch which one changes the answer.</p>"
            + f"<div class='panes'>{imgs}</div></div></div>")
    if HTML is None:
        print(f"[leakage_explorer: {len(panes)} configurations]")
        return None
    return HTML(body)


def _cap_panel(labels, sizes, cap, hi):
    """One setting of part 7's cap explorer: a dozen named customers, one ceiling."""
    _style()
    n = len(sizes)
    fig, ax = plt.subplots(figsize=(9.2, 0.40 * n + 1.75))
    _margins(fig, left=0.115, right=0.985)
    fig.subplots_adjust(bottom=0.5 / (0.40 * n + 1.75))
    y = np.arange(n)[::-1]                        # biggest at the top
    kept = np.minimum(sizes, cap) if cap else np.asarray(sizes, float)
    ax.set_xlim(0, hi)
    ax.set_ylim(-0.7, n + 0.3)          # headroom for the cap's own label
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(BASELINE)

    for yi, s, k in zip(y, sizes, kept):
        ax.barh(yi, k, height=0.6, color=_mix(ACCENT, 0.42), zorder=3)
        if s > k + 1e-12:
            ax.barh(yi, s - k, left=k, height=0.6, color=_mix(BAD, 0.52), zorder=3)
            ax.text(s + hi * 0.012, yi, f"{s:.2f} → {k:.2f}", va="center",
                    fontsize=9.5, color=BAD, fontweight="bold")
        else:
            # inside the bar, because an untouched bar can end within a hair of the
            # cap line and a label sitting on that line is unreadable
            ax.text(s - hi * 0.012, yi, f"{s:.2f}", va="center", ha="right",
                    fontsize=9.5, color="white", fontweight="bold", zorder=6)
    if cap:
        ax.axvline(cap, color=ACCENT, lw=1.8, ls=(0, (5, 3)), zorder=5)
        ax.text(cap, n - 0.42, f"the cap, {cap:.2f}", ha="center", va="bottom",
                fontsize=9.5, fontweight="bold", color=ACCENT, zorder=6)

    ax.set_yticks(y, labels, fontsize=9.5)
    ax.set_xlabel(f"how much this {V.member} contributes to the update")
    shortened = int((np.asarray(sizes) > cap).sum()) if cap else 0
    _headline(fig, "The cap only ever takes from the top."
              if cap else "Nothing bounds the biggest contributor.",
              f"Twelve of D Community's 150 {V.members}, spread evenly across the range, "
              f"biggest first. " + (f"Violet still reaches the update, red is what the "
                                    f"cap removed, and {shortened} of the twelve are "
                                    f"shortened." if cap else
                                    "The largest contributes 2.7 times the smallest."))
    return _done(fig)


def cap_explorer(labels, sizes, caps=None, population=None, default="no cap"):
    """Part 6's mechanism 2: set the cap and watch what it does to each contribution.

    One row of choices, the same CSS-and-radio mechanism the quizzes use, so a click
    costs nothing and a saved notebook keeps working. The settings are quantiles of
    `population`, the whole consortium's contributions, rather than round numbers,
    because that is how a cap is supposed to be chosen.

    Drawn as a dozen labelled bars rather than the sorted curve over all 2,450 that
    came first. That curve was honest but useless for the one thing it most needed to
    show: at 2,450 points the biggest contributor occupied about three pixels, so the
    number the figure quoted could not be found on it.
    """
    import io
    import base64

    sizes = np.asarray(sizes, float)
    pop = np.asarray(population if population is not None else sizes, float)
    if caps is None:
        caps = [("no cap", None)]
        for q, name in ((25, "25th"), (50, "median"), (75, "75th"), (90, "90th")):
            caps.append((f"{np.percentile(pop, q):.2f}   the {name}",
                         float(np.percentile(pop, q))))
    hi = float(sizes.max()) * 1.30                # room for the value labels

    panes = []
    for label, cap in caps:
        fig = _cap_panel(labels, sizes, cap, hi)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=92, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        panes.append((label, base64.b64encode(buf.getvalue()).decode("ascii")))

    _EXPLORER_N[0] += 1
    gid = f"fvc{_EXPLORER_N[0]}"
    inputs = tabs = imgs = css = ""
    for i, (label, png) in enumerate(panes):
        oid = f"{gid}_{i}"
        checked = " checked" if label.startswith(default) else ""
        inputs += f"<input class='cr' type='radio' name='{gid}' id='{oid}'{checked}>"
        tabs += f"<label class='tab' for='{oid}'>{label}</label>"
        imgs += (f"<div class='pane'><img alt='{label}' "
                 f"src='data:image/png;base64,{png}'></div>")
        css += (f"#{oid}:checked ~ .tabs label[for='{oid}']"
                "{background:#4a3aa7;border-color:#4a3aa7;color:#ffffff !important;}"
                f"#{oid}:checked ~ .panes .pane:nth-of-type({i + 1})"
                "{display:block;}")

    style = ("<style>"
             ".fvc input.cr{position:absolute;opacity:0;width:0;height:0;}"
             ".fvc h4{margin:.15rem 0 .5rem;}"
             ".fvc .tabs{display:flex;flex-wrap:wrap;gap:.4rem;margin:0 0 .9rem;}"
             ".fvc .tab{padding:.34rem .8rem;border:1.5px solid #c3c2b7;"
             "border-radius:.55rem;cursor:pointer;font-weight:600;font-size:.9rem;"
             "transition:background .12s;white-space:pre;}"
             ".fvc .tab:hover{background:#f2f1ec;}"
             ".fvc .pane{display:none;}.fvc .pane img{max-width:100%;height:auto;}"
             + css + ".fv.wide{max-width:1000px;}</style>")
    body = (_CSS + style + "<div class='fv wide'><div class='fvc'>" + inputs
            + "<h4>Set the cap, and watch what it does to each of them</h4>"
            + f"<div class='tabs'>{tabs}</div>"
            + f"<div class='panes'>{imgs}</div></div></div>")
    if HTML is None:
        print(f"[cap_explorer: {len(panes)} settings]")
        return None
    return HTML(body)


def influence_share(sizes, median, low, high, takeaway=None):
    """Part 5. How much of a blended update one customer can still account for.

    Over a batch the division returns sum_i c_i x_i, with c_i the prediction error on
    customer i over the sum of all of them. Those weights add to one but they are not
    probabilities: they take either sign, and their denominator can approach zero, so no
    single weight is bounded by anything. This plots the largest of them.

    Three things carry the argument. The dashed reference is 1/n, what an honest
    equal-weight average would give every customer. The solid line is what actually
    happens, several times higher at every batch size. The band is the range across
    model positions, and where it crosses 100% the denominator has nearly vanished and
    the division has returned something no customer would recognise.

    This replaced a figure that plotted the distance from the original customer. That
    distance stops meaning anything past one customer, which its own headline admitted,
    and it left part 6's clipping to be motivated by assertion. A share is a bound you
    can put a number on, which is exactly what clipping does.
    """
    _style()
    hh = 6.2
    fig, ax = plt.subplots(figsize=(9.8, hh))
    _margins(fig, takeaway=True, left=0.105, right=0.965)
    # a three-line takeaway sits higher than the default band, so lift the axes clear
    # of it or it paints over the x tick labels
    fig.subplots_adjust(bottom=(TAKE_IN + 0.58) / hh)
    x = np.arange(len(sizes), dtype=float)
    equal = [1.0 / n for n in sizes]
    ax.set_xlim(-0.45, len(sizes) - 0.55)
    ax.set_yscale("log")
    ax.set_ylim(min(min(low), min(equal)) * 0.55, max(high) * 2.1)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.spines["bottom"].set_color(BASELINE)

    ax.axhline(1.0, color=BAD, lw=1.2, ls=(0, (4, 3)), zorder=2)
    ax.text(-0.40, 1.09, "100%: one weight is the whole update, and the first point "
            "sits on it", ha="left", va="bottom", fontsize=9, color=BAD)

    ax.fill_between(x, low, high, color=_mix(ACCENT, 0.86), zorder=1)
    ax.plot(x, equal, color=MUTED, lw=1.6, ls=(0, (5, 4)), zorder=3,
            label="an equal split, 1 / n")
    ax.plot(x, median, color=ACCENT, lw=2.4, marker="o", markersize=6.5,
            markerfacecolor="white", markeredgewidth=1.9, zorder=4,
            label="the largest share actually carried")

    ax.set_xticks(x, [f"{n:,}" for n in sizes], fontsize=10.5)
    ax.set_xlabel(f"{V.members} in the update")
    ax.set_ylabel("largest share one " + V.member + " carries")
    ticks = [t for t in (0.001, 0.01, 0.1, 1.0, 10.0)
             if ax.get_ylim()[0] <= t <= ax.get_ylim()[1]]
    ax.set_yticks(ticks, [f"{t:.0%}" if t >= 0.01 else f"{t * 100:g}%" for t in ticks],
                  fontsize=10)
    ax.minorticks_off()
    ax.legend(fontsize=9.5, loc="upper right", frameon=False, handlelength=2.2,
              borderaxespad=0.6)

    _headline(fig, f"Averaging dilutes one {V.member}'s share. It does not cap it.",
              f"The largest weight any single {V.member} carries in what the division "
              f"returns, beside the equal split an honest average would give. The band "
              f"is the range over twenty four model positions.")
    _takeaway(fig, takeaway or
              "These weights are prediction errors, so they take either sign and can "
              "total nearly zero. One customer often carries several times an equal "
              "split, sometimes more than the whole update. Averaging changes what is "
              "disclosed without capping anybody. Part 6 adds the cap.",
              per_line=88)
    return _done(fig)


def isolation_chain():
    """Figure 13. What colluding members can strip off, and what they cannot.

    The steps mirror the task exactly: the coordinator combines by size, so the
    attacker undoes the size weighting rather than an equal average. It ends on one
    participant's update, which is where the experiment ends. A separate closing panel
    says what that is not, because the earlier version finished on "one customer
    recovered" and the notebook has never shown that.
    """
    _style()
    fig, ax = plt.subplots(figsize=(11.2, 5.6))
    _margins(fig)
    _bare(ax); ax.set_xlim(0, 11.2); ax.set_ylim(-0.86, 4.0)

    steps = [(f"register K−1\n{V.sites}",
              "the victim is the only\nhonest unknown", True),
             ("receive the\nweighted average",
              "ḡ = Σ (nₖ/N) gₖ", False),
             ("undo the weighting,\nsubtract your own",
              "g = (N·ḡ − Σ nₖgₖ) / n_v", False),
             (f"one {V.site}'s update,\nisolated",
              "every number,\nto 15 decimals", True)]
    w, h, y0 = 2.42, 2.15, 1.20
    xs = np.linspace(0.16, 11.04 - w, len(steps))
    for k, (x, (title, sub, hot)) in enumerate(zip(xs, steps)):
        _card(ax, x, y0, w, h, _mix(BAD, 0.93) if hot else "white",
              edge="none" if hot else GRID, lw=1.2, r_px=11)
        cx = x + w / 2
        ax.text(cx, y0 + h - 0.38, str(k + 1), ha="center", va="center",
                fontsize=10, fontweight="bold", color="white",
                bbox=dict(boxstyle="circle,pad=0.30",
                          facecolor=BAD if hot else CHARCOAL,
                          edgecolor="none"), zorder=6)
        ax.text(cx, y0 + h - 1.02, title, ha="center", va="center", fontsize=10,
                fontweight="bold", color=INK, linespacing=1.35)
        ax.text(cx, y0 + 0.42, sub, ha="center", va="center", fontsize=8.4,
                color=INK_2, linespacing=1.5,
                family=MONO if "=" in sub else SANS)
        if k < len(steps) - 1:
            ax.add_patch(FancyArrowPatch(
                (x + w + 0.02, y0 + h / 2), (x + w + 0.32, y0 + h / 2),
                arrowstyle="-|>", mutation_scale=12, lw=1.6, color=BASELINE))

    _card(ax, 0.16, -0.72, 10.88, 1.52, PANEL, r_px=9, z=1)
    ax.text(0.42, 0.42, "⚠", fontsize=13, color=FLAG, va="center")
    ax.text(0.82, 0.50, f"It needs every other contribution to be yours. With fewer "
            f"colluders, the group learns the sum of the honest updates,\nwhich is "
            "still revealing when few honest members remain.",
            fontsize=9.5, color=INK_2, va="center", linespacing=1.5)
    ax.text(0.82, -0.24, f"What it is not: a {V.member}. Step 4 returns one "
            f"{V.site}'s update over its whole book,\nand reading a {V.member} out of "
            f"that needs the update to come from a single {V.record}.",
            fontsize=9.5, fontweight="bold", color=INK, va="center", linespacing=1.5)

    _headline(fig, "Secure aggregation has a threat model.",
              f"It hides individual updates from the coordinator. It does not say who "
              "may contribute one.")
    return _done(fig)


def roster_link(recovered_row, roster, matched_index):
    """The synthetic roster, and the single row that matches."""
    rows = []
    for i, r in enumerate(roster):
        mark = " ←" if i == matched_index else ""
        cls = "flag" if i == matched_index else ""
        rows.append([f"<span class='{cls}'>{r[0]}{mark}</span>"] +
                    [f"<span class='{cls}'>{v}</span>" for v in r[1:]])
    body = ("<h4>Matching what we recovered against a public list</h4>"
            + _table(["appointment", "age", "resting BP"], rows, ["", "n", "n"])
            + "<p class='note'>This roster is <b>synthetic</b>. The research records are already "
              "de-identified, so nothing here re-identifies a real person. The point is the "
              "mechanism: recovered measurements can be matched against an outside list, and "
              "two or three of them are often enough.</p>")
    return _html(body)


# ================================================================ PART 7

def recovery_vs_noise(zs, errors, feature_scale=1.0):
    """Figure 14. Noise, not clipping, is what protects."""
    _style()
    fig, ax = plt.subplots(figsize=(9.6, 5.6))
    _margins(fig, left=0.09, right=0.965)
    top = max(max(errors), feature_scale) * 1.30
    ax.set_xscale("log")
    ax.set_ylim(0, top)

    ax.axhspan(0, feature_scale, color=_mix(BAD, 0.955), zorder=0)
    ax.axhspan(feature_scale, top, color=_mix(GOOD, 0.955), zorder=0)
    ax.axhline(feature_scale, color=BASELINE, lw=1.2)
    ax.text(zs[-1], feature_scale * 0.08, "recovered record still close to "
            f"the real {V.member}", fontsize=9.5, color="#a13636", va="bottom",
            ha="right")
    ax.text(zs[0], feature_scale * 1.07, "recovery is mostly noise — the "
            "attack is broken", fontsize=9.5, color=GOOD_TEXT, va="bottom")
    ax.text(zs[0], feature_scale - top * 0.02,
            "the size of a typical measurement", fontsize=9, color=MUTED,
            ha="left", va="top")

    ax.plot(zs, errors, color=CHARCOAL, lw=2.4, solid_capstyle="round", zorder=4)
    ax.scatter(zs, errors, s=62, color=CHARCOAL, zorder=5,
               edgecolor="white", linewidth=1.5)
    ax.set_xticks(zs, [f"{z:g}" for z in zs])
    ax.minorticks_off()
    ax.set_xlabel("noise added to the update")
    ax.set_ylabel("error in the recovered record")

    _headline(fig, "Noise breaks the attack. Clipping alone does not.",
              "Clipping scales the top and bottom of the attacker's ratio by the "
              "same number, so it cancels.")
    return _done(fig)


def accuracy_vs_eps(per_site_eps, accuracies, noise_labels, sizes=None,
                    score_fmt="{:.0%} accurate"):
    """Figure 15. The same model, four different promises.

    per_site_eps: dict {site: [eps at each noise level, strongest first]}
    accuracies:   global accuracy at each noise level
    noise_labels: what to call each noise level, strongest noise first
    sizes:        optional {site: n}, used to name the worst-off site in the takeaway
    score_fmt:    how to write the utility number. Accuracy is meaningless under heavy
                  class imbalance, so an imbalanced scenario passes "AUC {:.3f}".
    """
    _style()
    names = list(per_site_eps)
    n_levels = len(noise_labels)
    fig, ax = plt.subplots(figsize=(10.2, 6.4))
    _margins(fig, takeaway=True, left=0.155, right=0.90)

    lo = min(min(v) for v in per_site_eps.values())
    hi = max(max(v) for v in per_site_eps.values())
    L = np.log10
    ax.set_xlim(L(lo) - 0.28, L(hi) + 0.55)
    ax.set_ylim(-0.6, len(names) - 0.4 + 0.95)

    ticks = [t for t in (0.1, 0.2, 0.5, 1, 2, 5, 10, 20)
             if lo / 1.6 <= t <= hi * 2.2]
    ax.set_xticks([L(t) for t in ticks], [f"{t:g}" for t in ticks])
    ax.set_yticks([])
    ax.grid(axis="x"); ax.set_axisbelow(True)
    ax.spines["bottom"].set_color(BASELINE)

    ys = np.arange(len(names))[::-1]
    for yi, name in zip(ys, names):
        eps = per_site_eps[name]
        c = SITE_COLOUR.get(name, INK)
        xs = [L(e) for e in eps]
        ax.plot(xs, [yi] * len(xs), color=_mix(c, 0.55), lw=1.6, zorder=2)
        for k, x in enumerate(xs):
            fill = _mix(c, (0.55, 0.28, 0.0)[k] if n_levels == 3
                        else 0.55 * (1 - k / max(n_levels - 1, 1)))
            ax.scatter([x], [yi], s=130, color=fill, zorder=4,
                       edgecolor="white", linewidth=1.6)
        ax.text(xs[-1] + 0.09, yi, f"ε {eps[-1]:.1f}", va="center", fontsize=12.5,
                fontweight="bold", color=INK)
    _named_rows(ax, ys, [_short(n) for n in names],
                [SITE_COLOUR.get(n, INK) for n in names])

    # The settings, labelled once above the top row. Two settings can land almost on
    # top of each other for a large participant, so labels that would collide step up.
    top_eps = [L(e) for e in per_site_eps[names[0]]]
    span = max(ax.get_xlim()[1] - ax.get_xlim()[0], 1e-9)
    level, last_x = 0, -1e9
    for x, lab, acc in zip(top_eps, noise_labels, accuracies):
        level = level + 1 if (x - last_x) / span < 0.16 else 0
        last_x = x
        lift = 0.46 + 0.40 * level
        ax.plot([x, x], [ys[0] + 0.22, ys[0] + lift], color=BASELINE, lw=1.1)
        ax.text(x, ys[0] + lift + 0.10, f"{lab}\n{score_fmt.format(acc)}",
                ha="center", va="bottom", fontsize=9, color=INK_2, linespacing=1.35)

    ax.set_xlabel(f"privacy loss ε for this {V.site}'s {V.members} — "
                  "further left is a stronger promise")

    _headline(fig, "The same noise buys very different guarantees.",
              f"Each row is one {V.site}. The dots are noise settings shared by "
              "the whole federation, strongest noise on the left.")
    if sizes:
        smallest = min(sizes, key=lambda k: sizes[k])
        who = f"A {sizes[smallest]}-{V.member} {V.site}"
    else:
        who = f"The smallest {V.site}"
    _takeaway(fig, f"{who} cannot subsample, so each of its {V.members} is a "
                   "larger share of the whole and its guarantee is the weakest.")
    return _done(fig)


def defence_matrix():
    """Figure 16. Which control stops which threat, and what it leaves open."""
    rows = [
        ["An outsider reads the traffic", "authenticated encryption",
         "the coordinator still sees every update"],
        ["The coordinator reads one update", "secure aggregation",
         "nothing about <i>who</i> the other contributors are"],
        [f"One {V.member}'s contribution is unbounded", f"{V.member}-gradient clipping",
         "<span class='bad'>no privacy on its own</span>"],
        ["Inference from an aggregate", "calibrated noise, ε",
         f"costs accuracy, and costs the smallest {V.site} most"],
        ["Fake participants", "identity and admission control",
         "organisational, not arithmetic"],
        [f"One {V.site} submits a huge update", f"{V.site}-update clipping",
         "fails once an attacker holds enough identities"],
    ]
    body = ("<h4>Six threats, six controls, six limits</h4>"
            + _table(["threat", "control", "what it does not do"], rows)
            + "<p class='note'>Two of these are clipping, and they are different operations "
              "on different objects. Only the second one can be enforced by the coordinator.</p>")
    return _html(body)


def clipping_comparison():
    """The two clippings, and who can enforce which."""
    rows = [
        ["applied to", f"one {V.member}'s contribution", f"one {V.site}'s whole update"],
        ["runs where", f"inside the {V.site}'s own software", "at the coordinator"],
        ["who can enforce it", "<span class='bad'>nobody: it is a promise</span>",
         "<span class='good'>the coordinator, every time</span>"],
        ["what it buys", "a bound that makes the noise size meaningful",
         "a bound on how far one participant can move the model"],
    ]
    body = ("<h4>Two things called clipping</h4>"
            + _table(["", f"{V.member}-gradient clipping", f"{V.site}-update clipping"], rows)
            + f"<p class='note'>A {V.site} that intends to misbehave will not run your privacy "
              "code faithfully. That is the difference between a promise and a control.</p>")
    return _html(body)


def privacy_sweep(zs, auc, worst, per_bank, baseline, takeaway=None, metric="AUC"):
    """Part 6. What the noise dial costs, and what it buys, on one x axis.

    zs        the noise multiplier at each setting
    auc       global score at each setting, and `worst` the weakest participant's
    per_bank  {name: [epsilon at each setting]}, largest participant first
    baseline  the score with no privacy at all

    Replaced a three row table and a three level bar chart, which between them showed
    two banks at three settings and could not show the shape. The shape is the finding:
    the first units of privacy are nearly free and the last are ruinous, and the gap
    between the largest and smallest participant barely moves while that happens.

    The sweep stops at z = 16 on purpose. Past there `fedcore.epsilon` bottoms out at
    log(1/delta)/(alpha - 1), which with delta = 1e-5 and orders up to 64 is 0.183, so
    the flattening would be the accountant's search grid rather than the mechanism.
    """
    _style()
    hh = 6.3
    fig, axes = plt.subplots(1, 2, figsize=(11.4, hh))
    _margins(fig, takeaway=True, left=0.075, right=0.975, wspace=0.26)
    fig.subplots_adjust(top=1 - 1.42 / hh, bottom=(TAKE_IN + 0.52) / hh)
    x = np.arange(len(zs), dtype=float)
    labels = [f"{z:g}" for z in zs]

    ax = axes[0]
    _bare(ax)
    ax.set_xlim(-0.4, len(zs) - 0.6)
    ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)
    lo, hi = min(worst) , max(list(auc) + [baseline])
    pad = (hi - lo) * 0.10
    ax.set_ylim(lo - pad, hi + pad * 1.6)
    ax.axhline(baseline, color=POOLED_COLOUR, lw=1.4, ls=(0, (5, 4)), zorder=2)
    ax.text(len(zs) - 0.65, baseline + pad * 0.14, "no privacy at all", ha="right",
            fontsize=8.8, color=INK_2)
    ticks = np.arange(np.floor(lo * 20) / 20, hi + 0.05, 0.05)
    ax.set_yticks(ticks, [f"{t:.2f}" for t in ticks], fontsize=9.5)
    ax.plot(x, auc, color=ACCENT, lw=2.4, marker="o", markersize=6,
            markerfacecolor="white", markeredgewidth=1.8, zorder=4,
            label="the shared model")
    # not a named participant, so deliberately not a participant colour. Grey rather
    # than any hue, because every hue in this figure already belongs to a bank on the
    # right and this line is a floor rather than an identity
    ax.plot(x, worst, color=MUTED, lw=2.2, marker="o", markersize=5.5,
            markerfacecolor="white", markeredgewidth=1.7, zorder=4,
            label=f"the weakest {V.site}")
    ax.set_xticks(x, labels, fontsize=9.5)
    ax.set_xlabel("noise multiplier z")
    ax.set_ylabel(f"{metric} on the fresh cohort")
    ax.set_title("what it costs", fontsize=11, loc="left", color=INK_2, pad=8)
    ax.legend(fontsize=9, loc="lower left", frameon=False, handlelength=1.8)

    ax = axes[1]
    _bare(ax)
    ax.set_xlim(-0.4, len(zs) - 0.6)
    ax.set_yscale("log")
    ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)
    ax.axhline(1.0, color=MUTED, lw=1.2, ls=(0, (2, 3)), zorder=2)
    ax.text(-0.34, 1.06, "ε = 1", ha="left", fontsize=8.8, color=MUTED)
    for k, (name, eps) in enumerate(per_bank.items()):
        ax.plot(x, eps, color=SITE_COLOUR.get(name, OKABE[k % 4]), lw=2.2,
                marker="o", markersize=5.5,
                markerfacecolor="white", markeredgewidth=1.7, zorder=4,
                label=name)
    eticks = [t for t in (0.2, 0.5, 1, 2, 5, 10, 20, 50)
              if min(min(v) for v in per_bank.values()) * 0.9 <= t
              <= max(max(v) for v in per_bank.values()) * 1.1]
    ax.set_yticks(eticks, [f"{t:g}" for t in eticks], fontsize=9.5)
    ax.minorticks_off()
    ax.set_xticks(x, labels, fontsize=9.5)
    ax.set_xlabel("noise multiplier z")
    ax.set_ylabel("ε promised to one of its customers")
    ax.set_title("what it buys", fontsize=11, loc="left", color=INK_2, pad=8)
    ax.legend(fontsize=8.6, loc="upper right", frameon=False, handlelength=1.8,
              labelspacing=0.35)

    _headline(fig, "The first units of privacy are nearly free. The last are ruinous.",
              "One dial, both consequences. Thirteen exchanges of five local steps, "
              "five runs averaged.")
    _takeaway(fig, takeaway or
              "Going from no privacy to a moderate setting costs almost nothing. "
              "Pushing the smallest participant under one costs five times as much "
              "again, and the distance between largest and smallest barely closes "
              "while it happens.", per_line=88)
    return _done(fig)


def configuration_table(rows, eps_sites=("largest", "smallest"), note=None):
    """The three deployment configurations, measured.

    `eps_sites` names the two participants whose ε is tabulated, so the header follows
    the scenario. `note` is the reading of the table, which is scenario-specific.
    """
    note = note or ("All three cost the same to communicate. The whole price of "
                    "privacy here is accuracy, and it is not linear.")
    body = ("<h4>Three ways to run the same federation</h4>"
            + _table(["", "configuration", "noise", "global", f"worst {V.site}",
                      f"ε {eps_sites[0]}", f"ε {eps_sites[1]}"], rows,
                     ["", "", "", "n", "n", "n", "n"])
            + f"<p class='note'>{note}</p>")
    return _html(body)


# ================================================================ PART 8

def bytes_ladder(entries):
    """Figure 17. The cost of scale, from eleven parameters to seven billion.

    entries: list of (label, parameters, bytes_per_exchange[, colour])
    """
    _style()
    fig, ax = plt.subplots(figsize=(9.8, 5.6))
    _margins(fig, left=0.27, right=0.955)
    ys = np.arange(len(entries))[::-1]
    vals = [np.log10(e[2]) for e in entries]

    ax.set_xlim(0, 13.4)
    ax.set_ylim(-0.6, len(entries) - 0.4)
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks([0, 3, 6, 9, 12], ["1 B", "1 kB", "1 MB", "1 GB", "1 TB"])
    ax.grid(axis="x"); ax.set_axisbelow(True)

    for y, e, v in zip(ys, entries, vals):
        colour = e[3] if len(e) > 3 and e[3] else CHARCOAL
        _pill(ax, 0, v, y, 0.56, colour)
        ax.text(v + 0.18, y, _human_bytes(e[2]), va="center", fontsize=12.5,
                fontweight="bold", color=INK)
    tr = blended_transform_factory(ax.transAxes, ax.transData)
    for y, e in zip(ys, entries):
        ax.text(-0.02, y, e[0], transform=tr, ha="right", va="center",
                fontsize=10.5, color=INK)
    ax.set_xlabel("bytes on the network per exchange  (log scale)")

    _headline(fig, "The procedure does not change when the model grows. The bill does.",
              f"Same {V.sites}, same averaging, nine orders of magnitude apart.")
    return _done(fig)


def _human_bytes(b):
    for unit, size in (("TB", 1e12), ("GB", 1e9), ("MB", 1e6), ("kB", 1e3)):
        if b >= size:
            return f"{b/size:,.1f} {unit}"
    return f"{b:,.0f} B"


#: What one parameter costs on the wire under each payload scheme.
PAYLOADS = [("everything", 8.0, "8 bytes a number"),
            ("top 25% of changes", 8.0 * 0.25 + 0.25, "a quarter of them, plus indices"),
            ("one bit per number", 1 / 8, "direction only, one shared magnitude")]

#: Model sizes the wire bill is quoted at. Only the first is trainable on this machine.
MODELS = [("this notebook", 14), ("a small neural network", 1_900),
          ("ResNet-18", 11_700_000), ("a 7-billion-parameter model", 7e9)]


def dial_effects(panels, reference, noise=0.002, metric="AUC"):
    """Part 8. What the three cost dials do to the model, on one shared axis.

    panels: list of (title, x_labels, series) where series is a list of
            (label, [values], note). One panel per dial.

    The shared y-axis is the argument. Two of these dials move the bill by six orders
    of magnitude and the model by less than the seed-to-seed noise, and that only reads
    as flat if the flat panels sit on the same scale as the one that is not.
    """
    _style()
    n = len(panels)
    h = 6.4
    fig, axes = plt.subplots(1, n, figsize=(3.85 * n, h), sharey=True)
    _margins(fig, takeaway=True, left=0.085, right=0.975, wspace=0.14)
    fig.subplots_adjust(top=1 - 1.62 / h)          # clear of a two-line brief
    axes = np.atleast_1d(axes)

    allv = [v for _, _, ss in panels for _, vals, _ in ss for v in vals]
    lo, hi = min(allv + [reference]), max(allv + [reference])
    pad = max((hi - lo) * 0.14, noise * 2.4)

    for pi, (ax, (title, xs, series)) in enumerate(zip(axes, panels)):
        _bare(ax)
        ax.set_xlim(-0.55, len(xs) - 0.45)
        ax.set_ylim(lo - pad, hi + pad)
        ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)
        ax.axhspan(reference - noise, reference + noise, color=_mix(ACCENT, 0.90),
                   zorder=1)
        ax.axhline(reference, color=POOLED_COLOUR, lw=1.4, ls=(0, (5, 4)), zorder=2)
        for k, (label, vals, note) in enumerate(series):
            col = OKABE[k % 4]
            ax.plot(range(len(vals)), vals, lw=2.2, color=col, marker="o",
                    markersize=6, markerfacecolor="white", markeredgewidth=1.8,
                    zorder=4, label=label)
            if note:
                i, txt = note
                ax.annotate(txt, (i, vals[i]), xytext=(9, 4),
                            textcoords="offset points", fontsize=8.4, color=BAD,
                            linespacing=1.4)
        ax.set_xticks(range(len(xs)), xs, fontsize=9.5)
        ax.set_title(title, fontsize=11, loc="left", color=INK_2, pad=8)
        if any(lab for lab, _, _ in series):
            ax.legend(fontsize=9, loc="lower right", frameon=False,
                      handlelength=1.8, borderaxespad=0.5)
    step = 0.005 if (hi - lo) > 0.012 else 0.002
    ticks = np.arange(np.floor((lo - pad) / step) * step,
                      hi + pad + step, step)
    axes[0].set_yticks(ticks, [f"{t:.3f}" for t in ticks], fontsize=9.5)
    axes[0].set_ylabel(f"{metric} on the fresh cohort")
    axes[0].annotate("pooled training, and\nthe seed-to-seed spread",
                     (0, reference), xytext=(4, 26), textcoords="offset points",
                     ha="left", fontsize=8.4, color=INK_2, linespacing=1.4)

    _headline(fig, "Two of the three dials do nothing to the model.",
              f"Every point is the same fourteen-parameter model, trained to the same "
              f"exchange budget. The band is pooled training plus or minus the "
              f"seed-to-seed spread.")
    _takeaway(fig, "Federation size and how many speak move the bill by six orders of "
                   "magnitude and the model by less than the noise. Local work is the "
                   "one dial that changes what you get, and mostly because it changes "
                   "how much training has happened.")
    return _done(fig)


def compression_conditions(steps_panel, speakers_panel, reference, metric="AUC"):
    """Part 7. When compression is free, and when it is not.

    Each panel is (x_labels, [(scheme, [values])...]). The lines converge in the first
    panel and plateau in the second, which is the whole finding: quantisation throws
    away magnitude and keeps direction, so it costs least exactly when the direction is
    the part that carried the information.
    """
    _style()
    hh = 6.4
    fig, axes = plt.subplots(1, 2, figsize=(11.8, hh), sharey=True)
    _margins(fig, takeaway=True, left=0.115, right=0.975, wspace=0.30)
    # the takeaway band is four lines here, so lift the axes clear of it or it
    # paints over the tick labels
    fig.subplots_adjust(top=1 - 1.86 / hh, bottom=(TAKE_IN + 0.62) / hh)

    allv = [v for xs, ss in (steps_panel, speakers_panel) for _, vals in ss for v in vals]
    lo, hi = min(allv + [reference]), max(allv + [reference])
    pad = (hi - lo) * 0.16

    titles = ("as each participant works longer", "as more participants speak")
    for pi, (ax, (xs, series), title) in enumerate(
            zip(axes, (steps_panel, speakers_panel), titles)):
        _bare(ax)
        ax.set_xlim(-0.5, len(xs) - 0.5)
        ax.set_ylim(lo - pad, hi + pad)
        ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)
        ax.axhline(reference, color=POOLED_COLOUR, lw=1.4, ls=(0, (5, 4)), zorder=2)
        for k, (scheme, vals) in enumerate(series):
            col = OKABE[k % 4]
            ax.plot(range(len(vals)), vals, lw=2.4, color=col, marker="o",
                    markersize=6, markerfacecolor="white", markeredgewidth=1.8,
                    zorder=4, label=scheme if pi == 0 else None)
        ax.set_xticks(range(len(xs)), xs, fontsize=9.5)
        ax.set_title(title, fontsize=11, loc="left", color=INK_2, pad=8)
    step = 0.01 if (hi - lo) > 0.03 else 0.005
    ticks = np.arange(np.floor((lo - pad) / step) * step, hi + pad + step, step)
    axes[0].set_yticks(ticks, [f"{t:.3f}" for t in ticks], fontsize=9.5)
    axes[0].set_ylabel(f"{metric} on the fresh cohort")
    axes[0].set_xlabel("local steps each")
    axes[1].set_xlabel("participants speaking each exchange")
    axes[1].annotate("pooled", (0, reference), xytext=(6, -14),
                     textcoords="offset points", fontsize=8.4, color=INK_2)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=9.5, ncols=len(labels), loc="upper left",
               bbox_to_anchor=(0.115, 1 - 1.12 / hh), frameon=False,
               handlelength=1.8, columnspacing=2.0)

    _headline(fig, "One bit is expensive on small updates and free on large ones.",
              "The same schemes as before, now against the other two dials. The dashed "
              "line is pooled training.")
    _takeaway(fig, "Quantisation keeps the direction and throws away the size, so it "
                   "costs most when the update was one small step and nothing once it "
                   "is fifty. Sparsification is close to free throughout. Compression "
                   "is not a fixed discount, it is a bargain whose price depends on "
                   "the rest of the configuration.")
    return _done(fig)


def budget_explorer(records_total=10_000, models=None, payloads=None,
                    fleets=(4, 500, 1000), steps=(1, 5, 20, 50)):
    """Part 8's explorer: what one exchange costs, on four dials.

    Everything it reports is exact arithmetic, which is the reason it reports what it
    does and not more. Bytes on the wire are 2 x active x parameters x bytes-per-
    parameter. Local work is E x the records a participant holds. Both hold at any
    model size.

    It deliberately does not report a run total, and that is not an oversight. A run
    total needs the number of exchanges to convergence, which this notebook measured
    on a fourteen-parameter model in part 4. Multiplying that count by a seven-billion-
    parameter model's byte cost would invent a run length for a model nobody here has
    trained. The same objection retired the accuracy column.

    Rendered as HTML with a little arithmetic in the page, because a slider has to
    compute and a pre-rendered image cannot. The maths is four multiplications.
    """
    _EXPLORER_N[0] += 1
    gid = f"fvb{_EXPLORER_N[0]}"
    models = models or MODELS
    payloads = payloads or PAYLOADS

    def btn_row(name, opts, default, fmt=str):
        return "".join(
            f"<input class='bx' type='radio' name='{gid}{name}' id='{gid}{name}{i}'"
            f"{' checked' if v == default else ''} data-v='{v}'>"
            f"<label class='bx-b' for='{gid}{name}{i}'>{fmt(v)}</label>"
            for i, v in enumerate(opts))

    rows = "".join(
        f"<tr><td>{lab}<span class='sub'>{p:,.0f} parameters</span></td>"
        + f"<td class='n work-col' data-p='{p}'>-</td>"
        + "".join(f"<td class='n' data-p='{p}' data-per='{per}'>-</td>"
                  for _, per, _ in payloads) + "</tr>"
        for lab, p in models)
    heads = ("<th class='n work-col'>local work<span class='sub'>gradient terms, "
             "per speaker</span></th>"
             + "".join(f"<th class='n'>{lab}<span class='sub'>{note}</span></th>"
                       for lab, _, note in payloads))
    group = (f"<tr class='grp'><td></td><th class='work-col'>WHAT EACH SPEAKER "
             f"COMPUTES</th><th colspan='{len(payloads)}'>WHAT CROSSES THE WIRE"
             "</th></tr>")

    style = """<style>
.fvb input.bx{position:absolute;opacity:0;width:0;height:0;}
.fvb .bx-b{display:inline-block;padding:.3rem .7rem;margin:.15rem .25rem .15rem 0;
  border:1.5px solid #c3c2b7;border-radius:.5rem;cursor:pointer;font-size:.86rem;
  font-weight:600;transition:background .12s;}
.fvb .bx-b:hover{background:#f2f1ec;}
.fvb input.bx:checked+.bx-b{background:#4a3aa7;border-color:#4a3aa7;color:#fff!important;}
.fvb .dial{margin:.55rem 0;}
.fvb .dial>b{display:inline-block;min-width:9.5rem;font-size:.8rem;
  text-transform:uppercase;letter-spacing:.07em;color:#6b6a65;}
.fvb input[type=range]{vertical-align:middle;width:19rem;accent-color:#4a3aa7;}
.fvb .live{font-weight:700;color:#4a3aa7;margin-left:.6rem;}
.fvb table{border-collapse:collapse;width:100%;margin-top:.9rem;font-size:.92rem;}
.fvb th,.fvb td{padding:.45rem .65rem;text-align:left;}
.fvb thead th{font-size:.72rem;letter-spacing:.06em;text-transform:uppercase;
  color:#6b6a65;border-bottom:1px solid #c3c2b7;vertical-align:bottom;}
.fvb td.n,.fvb th.n{text-align:right;font-variant-numeric:tabular-nums;}
.fvb td.n{font-family:ui-monospace,Menlo,monospace;font-weight:700;}
.fvb tbody td{border-bottom:1px solid #eceae4;}
.fvb tr.grp th{font-size:.66rem;letter-spacing:.09em;color:#898781;
  border-bottom:none;padding-bottom:0;text-align:center;font-weight:600;}
.fvb .work-col{border-right:1px solid #e1e0d9;}
.fvb .sub{display:block;font-size:.7rem;font-weight:400;color:#898781;
  text-transform:none;letter-spacing:0;}
.fvb .work{margin-top:.7rem;padding:.7rem .9rem;background:#f6f5f2;border-radius:.6rem;
  font-size:.9rem;}
.fvb .calc{font-family:ui-monospace,Menlo,monospace;font-size:.86rem;margin:.35rem 0 0;}
.fvb .calc span{display:inline-block;min-width:7.5rem;font-family:inherit;
  color:#6b6a65;font-weight:600;}
.fvb .why{font-size:.74rem;color:#898781;margin:.05rem 0 .3rem 7.5rem;}
</style>"""

    body = f"""{style}<div class='fv wide'><div class='fvb' id='{gid}'>
<h4>What one exchange costs</h4>
<div class='dial'><b>the federation</b>{btn_row('F', fleets, fleets[0], lambda v: f'{v:,}')}</div>
<div class='dial'><b>speaking</b>
  <input type='range' id='{gid}A' min='1' max='{fleets[0]}' value='{fleets[0]}'>
  <span class='live' id='{gid}AL'></span></div>
<div class='dial'><b>local steps each</b>{btn_row('E', steps, steps[1])}</div>
<table><thead>{group}<tr><th>model size</th>{heads}</tr></thead><tbody>{rows}</tbody></table>
<div class='work' id='{gid}W'></div>
<p class='note'>Every number here is arithmetic: bytes are
2 &times; speakers &times; parameters &times; bytes-per-parameter, and local work is the
steps each participant takes over the records it holds. There is no run total and no
accuracy, because both would need a number of exchanges measured on a model this
machine can actually train, and only the top row qualifies.</p>
</div></div>
<script>(function(){{
 var R={gid!r},T=document.getElementById(R);
 function pick(n){{var e=T.querySelector("input[name='"+R+n+"']:checked");
   return e?parseFloat(e.dataset.v):0;}}
 function hb(b){{var u=['B','kB','MB','GB','TB','PB'],i=0;
   while(b>=1000&&i<u.length-1){{b/=1000;i++;}}
   return (b>=100?b.toFixed(0):b.toFixed(1))+' '+u[i];}}
 function hn(v){{var u=['','k','M','G','T','P','E'],i=0;
   while(v>=1000&&i<u.length-1){{v/=1000;i++;}}
   return (v>=100||i===0?v.toFixed(0):v.toFixed(1))+(u[i]?' '+u[i]:'');}}
 function draw(){{
   var F=pick('F'),E=pick('E'),s=document.getElementById(R+'A');
   if(+s.max!==F){{var was=+s.value/(+s.max||1);s.max=F;
     s.value=Math.max(1,Math.round(F*(was||1)));}}
   var A=+s.value;
   document.getElementById(R+'AL').textContent=
     A.toLocaleString()+' of '+F.toLocaleString()+
     '  ('+(100*A/F).toFixed(A/F<0.1?1:0)+'%)';
   var per0={records_total}/F;
   T.querySelectorAll('td.n').forEach(function(td){{
     td.textContent = td.classList.contains('work-col')
       ? hn(E*per0*(+td.dataset.p))
       : hb(2*A*(+td.dataset.p)*(+td.dataset.per));}});
   var per={records_total}/F,
       cells=T.querySelectorAll('tbody tr:first-child td.n:not(.work-col)'),
       P=+cells[0].dataset.p, alt=[];
   for(var i=1;i<cells.length;i++){{
     var q=+cells[i].dataset.per;
     alt.push('&times; '+(q>=1?q:q.toFixed(3))+' B &rarr; <b>'+hb(2*A*P*q)+'</b>');
   }}
   document.getElementById(R+'W').innerHTML=
     "<b>Where the top row comes from</b>"
    +"<div class='calc'><span>on the wire</span>"
    +"2 &times; "+A.toLocaleString()+" &times; "+P.toLocaleString()
    +" &times; 8 B = <b>"+hb(2*A*P*8)+"</b></div>"
    +"<div class='why'>both directions &times; speaking &times; parameters "
    +"&times; bytes a number</div>"
    +"<div class='calc'><span>other columns</span>"+alt.join('&nbsp;&nbsp; ')+"</div>"
    +"<div class='calc'><span>local work</span>"+E+" &times; "
    +Math.round(per).toLocaleString()+" &times; "+P.toLocaleString()
    +" = <b>"+hn(E*per*P)+"</b> gradient terms, per speaker</div>"
    +"<div class='why'>local steps &times; the records that speaker holds "
    +"&times; parameters</div>";
 }}
 T.addEventListener('input',draw);T.addEventListener('change',draw);draw();
}})();</script>"""
    if HTML is None:
        print("[budget_explorer]")
        return None
    return HTML(_CSS + body)


def fleet_scaling(rows, per_round_label="sampled per round"):
    """Part 8. What happens to the bill when the federation itself grows.


    rows: (label, n_participants, sampled, bytes_if_all, bytes_if_sampled, auc).
    The point is the gap between the two byte columns: past a certain size you stop
    waiting for everyone, not because they are unreliable but because the bill for
    hearing from all of them stops being payable.
    """
    _style()
    n = len(rows)
    h = 0.86 * n + 4.9
    fig, ax = plt.subplots(figsize=(11.4, h))
    _margins(fig, takeaway=True, left=0.245, right=0.80)
    fig.subplots_adjust(bottom=(TAKE_IN + 0.72) / h)
    _bare(ax)
    top = max(r[3] for r in rows)
    ax.set_xscale("log")
    ax.set_xlim(max(1, min(r[4] for r in rows) * 0.45), top * 2.4)
    ax.set_ylim(n - 0.35, -1.30)
    tr = blended_transform_factory(ax.transAxes, ax.transData)

    ax.text(-0.028, -1.05, "THE FEDERATION", transform=tr, ha="right",
            fontsize=8.8, fontweight="bold", color=INK_2)
    ax.text(0.0, -1.05, "BYTES ON THE WIRE PER EXCHANGE", transform=tr,
            fontsize=8.8, fontweight="bold", color=INK_2)

    for i, (label, k, sampled, b_all, b_s, auc) in enumerate(rows):
        ax.text(-0.028, i - 0.12, label, transform=tr, ha="right", va="center",
                fontsize=10.5, color=INK)
        if auc is not None:
            ax.text(-0.028, i + 0.20, f"AUC {auc:.3f}", transform=tr, ha="right",
                    va="center", fontsize=9, color=MUTED)
        _pill(ax, ax.get_xlim()[0], b_all, i - 0.14, 0.30, _mix(NEUTRAL, 0.15), r_px=4)
        _pill(ax, ax.get_xlim()[0], b_s, i + 0.16, 0.30, _mix(ACCENT, 0.30), r_px=4)
        ax.text(b_all * 1.14, i - 0.14, f"{_human_bytes(b_all)}, everyone speaks",
                va="center", fontsize=9, color=MUTED)
        ax.text(b_s * 1.14, i + 0.16,
                f"{_human_bytes(b_s)}, {sampled} of {k:,}", va="center",
                fontsize=9.5, fontweight="bold", color=ACCENT)
    ax.set_xlabel("bytes per exchange, log scale")

    _headline(fig, "A bigger federation is not a bigger conversation.",
              f"Grey is every {V.site} speaking every exchange. Violet is a sample of "
              "them, which is what a system this size actually does.")
    _takeaway(fig, "At four you can wait for everyone. At five hundred the bill for "
                   "doing so is the thing that stops you, so you stop asking, and the "
                   "model barely notices.")
    return _done(fig)


def payload_tradeoff(rows, full_bytes, baseline_auc):
    """Part 8. Sending less of each update, and what it costs.


    rows: (label, bytes, auc, note). One row per compression scheme, the first being
    no compression. Two bars per row: what it saves and what it gives up, so the
    trade is visible in one read rather than inferred across two tables.
    """
    _style()
    n = len(rows)
    fig, ax = plt.subplots(figsize=(11.4, 0.86 * n + 4.3))
    _margins(fig, takeaway=True, left=0.245, right=0.975)
    _bare(ax)
    ax.set_xlim(0, 1.34); ax.set_ylim(n - 0.35, -1.35)
    tr = blended_transform_factory(ax.transAxes, ax.transData)

    ax.text(-0.026, -1.10, "WHAT EACH UPDATE CARRIES", transform=tr, ha="right",
            fontsize=8.8, fontweight="bold", color=INK_2)
    ax.text(0.0, -1.10, "SHARE OF THE FULL PAYLOAD", fontsize=8.8,
            fontweight="bold", color=INK_2)
    ax.text(1.06, -1.10, "WHAT IT SCORES", ha="right", fontsize=8.8,
            fontweight="bold", color=INK_2)

    for i, (label, b, auc, note) in enumerate(rows):
        frac = b / full_bytes
        drop = baseline_auc - auc
        ax.text(-0.026, i, label, transform=tr, ha="right", va="center",
                fontsize=10.5, color=INK)
        _pill(ax, 0, max(frac * 0.72, 0.008), i, 0.44,
              _mix(ACCENT, 0.28 + 0.5 * (1 - frac)), r_px=5)
        ax.text(max(frac * 0.72, 0.008) + 0.016, i,
                f"{b:,.0f} bytes" + ("" if frac > 0.99 else f"   {1/frac:.0f}× less"),
                va="center", fontsize=9.5, color=INK_2)
        ax.text(1.06, i, f"{auc:.3f}", transform=tr, ha="right", va="center",
                fontsize=11, fontweight="bold",
                color=INK if drop < 0.005 else BAD)
        if note:
            ax.text(1.06, i + 0.30, note, transform=tr, ha="right", va="center",
                    fontsize=8.4, color=MUTED)
    ax.set_xticks([])

    _headline(fig, "Say less, and say it in fewer bits.",
              f"Every scheme sends the same {V.records}' worth of learning. They "
              "differ in how much of the update survives the trip.")
    _takeaway(fig, "Compression is the third budget line, and it is bought with "
                   "accuracy. Keeping the largest handful of changes costs almost "
                   "nothing. Sending one bit per number costs real quality.")
    return _done(fig)


def model_comparison(rows, metric="accuracy"):
    body = ("<h4>The same federation, a bigger model</h4>"
            + _table(["model", "parameters", "bytes per exchange", metric],
                     rows, ["", "n", "n", "n"])
            + "<p class='note'>Not one line of the federated procedure changed between these "
              "rows. That is why the same machinery works for a model a billion times larger, "
              "and why the only thing that grew is the bill.</p>")
    return _html(body)


# ================================================================ PART 9

def options_table(rows, metric="accuracy"):
    """Figure 18. Nobody wins on all four columns."""
    body = ("<h4>The board's options</h4>"
            # the worst-scoring participant and the weakest guarantee are not the
            # same participant, so the two columns must not share the word "worst"
            + _table(["option", f"global {metric}", f"lowest {metric}", "communication",
                      "ε promised to any customer"], rows, ["", "n", "n", "n", "n"])
            + "<p class='note'>No option wins every column. Choosing between them is a "
              "management act, and the conditions attached to the choice are where the "
              "earlier parts pay off.</p>")
    return _html(body)


DECISIONS = {"approve": "Approve as proposed",
             "approve_with_conditions": "Approve with conditions",
             "reject": "Reject"}

def CONDITIONS():
    """Worded from the scenario, so the same nine conditions read naturally either way."""
    return {
        "worst_site": f"a minimum accuracy at the weakest {V.site}",
        "privacy": f"a privacy target, stated per {V.site} rather than on average",
        "identity": f"verified {V.site} identities and controlled admission",
        "budget": "a communication budget",
        "audit": "monitoring and an audit trail",
        "scale": "a review the moment participants, parameters or payload move beyond "
                 "what was measured",
    }


def _eps_sentence(eps_gap):
    if not eps_gap:
        return "the same mechanism buys very different guarantees at different sizes, and"
    a, ea, b, eb = eps_gap
    return f"the same mechanism gave {a} ε = {ea:.1f} and {b} ε = {eb:.1f};"


def decision(choice, conditions, eps_gap=None):
    """Task 9. The choice is not graded. The conditions are.

    `eps_gap` is an optional (biggest_name, eps, smallest_name, eps) tuple so the
    worked answer quotes this scenario's measured privacy gap.
    """
    COND = CONDITIONS()
    good = {"identity", "privacy", "scale"}
    picked = set(conditions)
    hits = picked & good
    if choice not in DECISIONS:
        return _html(f"<h4>Unknown decision {choice!r}</h4><p>Pick one of: "
                     + ", ".join(f"<code>{k}</code>" for k in DECISIONS) + "</p>")
    lines = [f"<p><b>{DECISIONS[choice]}</b>, with {len(picked)} condition"
             f"{'s' if len(picked) != 1 else ''}:</p><ul>"]
    for c in conditions:
        lines.append(f"<li>{COND.get(c, c)}</li>")
    lines.append("</ul>")
    verdict = ("<p class='good'>The conditions this project argues for are in your "
               "list.</p>" if len(hits) >= 2 else
               "<p class='flag'>Worth revisiting: which conditions can the coordinator "
               "actually enforce, and which are promises?</p>")
    exemplar = (
        "<details><summary>What this project would argue</summary>"
        "<p>Any decision is defensible if the conditions are. Three of them are hard to "
        "argue against on this evidence. <b>Verified identities</b>, because nothing measured "
        "here constrains who may contribute an update, and no amount of noise fixes that. "
        f"<b>A privacy target stated per {V.site}</b>, because {_eps_sentence(eps_gap)} an "
        f"average would have hidden the {V.site} that carries the risk. And <b>a review on "
        "scale</b>, because every number above was measured at one configuration and none of "
        "them was measured at a larger one.</p>"
        "<p>The harder question is the one with no technical answer: the federation loses "
        "little measurable by excluding its smallest member, and its smallest member loses "
        "the most. Someone has to decide that.</p></details>")
    return _html("<h4>Your decision</h4>" + "".join(lines) + verdict + exemplar)


# ------------------------------------------------------------------- task feedback

def check(*results):
    """Mark the blanks a reader has just filled in, one line per blank.

    Each result is `(label, passed, when_right, when_wrong)`. Printed rather than
    drawn, because a task cell is the one place the reader has just run their own
    code: the verdict has to appear under it in Colab, in JupyterLab and in a saved
    copy, and plain text is the only thing that does all three.

    The wrong-answer line is where the teaching is. It says which blank to look at
    and what the value should have been, never just that something is wrong.
    """
    rows = [(str(label), bool(ok), right, wrong) for label, ok, right, wrong in results]
    width = max(len(r[0]) for r in rows)
    rule = "─" * min(78, max(46, width + 62))
    print(rule)
    for label, ok, right, wrong in rows:
        print(f"{label:<{width}}  {'✓' if ok else '✗'}  {right if ok else wrong}")
    print(rule)
    n_ok = sum(r[1] for r in rows)
    if n_ok == len(rows):
        print("all correct — carry on." if len(rows) > 1 else "correct — carry on.")
    else:
        print(f"{n_ok} of {len(rows)} right. Fix the ✗ above, then run this cell again."
              if len(rows) > 1 else
              "Not right yet. Read the line above, then run this cell again.")


# ----------------------------------------------------------------- the playground

#: The three missions the notebook embeds, and what each one asks for. `build_playground.py`
#: writes the matching files into `missions/`, one mission to a file, each opening straight
#: into it with no mission map to get lost in.
MISSIONS = {
    "build": ("Build the machine",
              "Put the five steps of a federated round in order, then run the loop once."),
    "heist": ("Heist",
              "Turn a dial until one bank's update hands back a real customer."),
    "privacy": ("Now stop yourself",
                "Break the attack and keep the shared model useful, 0.84 or better."),
}


def _mission_file(name):
    """Where the mission page landed, whichever way the reader got the exercise.

    A clone already has it at the repo root, one level up from this file. Colab has
    nothing until it is fetched, and each page is about a megabyte, so it is fetched on
    first use rather than at setup: a reader who stops at part 2 never pays for it. Cell
    0.1 sets `FDD_MISSIONS` to say where from; without it there is nothing to try.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    for d in ("missions", os.path.join("..", "missions"),
              os.path.join(here, "..", "missions"), os.path.join(here, "missions")):
        path = os.path.join(d, f"{name}.html")
        if os.path.exists(path):
            return path
    base = os.environ.get("FDD_MISSIONS", "")
    if not base:
        return None
    try:
        os.makedirs("missions", exist_ok=True)
        path = os.path.join("missions", f"{name}.html")
        print(f"  fetching the {name} mission, about a megabyte, once ...")
        urllib.request.urlretrieve(f"{base}/{name}.html", path)
        return path
    except Exception as exc:                       # offline, or the course moved the files
        print(f"  could not fetch the {name} mission: {exc}")
        return None


def mission(name, height=780):
    """Open one playground mission inside the notebook, at the point it belongs.

    The page is inlined into an iframe rather than linked. A notebook cell cannot serve
    a file to its own output — Colab renders outputs from a sandboxed origin that cannot
    see the kernel's filesystem — and `srcdoc` is the one mechanism that behaves the same
    in Colab, in JupyterLab and in a saved copy. The iframe is also what keeps the
    playground's stylesheet off the notebook's own page.

    The mission runs entirely in the browser on the same records this notebook trains on,
    so nothing is sent anywhere and the numbers on screen are the numbers in the cells
    above.
    """
    if name not in MISSIONS:
        raise KeyError(f"{name!r} is not a mission. Available: {sorted(MISSIONS)}")
    title, ask = MISSIONS[name]
    path = _mission_file(name)
    if path is None or HTML is None:
        print(f"[mission: {title}] {ask}")
        if path is None:
            print(f"  missions/{name}.html is not here, and it could not be fetched.")
            print("  Run cell 0.1 again, or open the mission from the repository yourself.")
        return None
    with open(path, encoding="utf-8") as fh:
        page = fh.read()
    doc = _escape(page, quote=True)
    return HTML(
        f"<div style='margin:0 0 6px;font:600 13px/1.4 system-ui,-apple-system,sans-serif;"
        f"color:#52514e'>🎮 {title} — {ask}</div>"
        f"<iframe srcdoc=\"{doc}\" title=\"{title}\" loading=\"lazy\" "
        f"style='width:100%;height:{height}px;border:1px solid #e1e0d9;border-radius:10px;"
        f"background:#fff'></iframe>"
        f"<div style='margin:6px 0 0;font:400 12px/1.5 system-ui,-apple-system,sans-serif;"
        f"color:#898781'>The mission scrolls inside its own frame. When the card says "
        f"<b>Mission achieved</b>, press <b>Back to the notebook</b> and carry on below.</div>")


def wire_formats(delta, k, byte_costs, labels=None):
    """The three shapes one update can take on the wire, and what each costs to send.

    Same update in all three panels, because that is the point: the bank trains the same
    way and sends the same fourteen numbers' worth of change. What differs is how much of
    it survives the trip. `delta` is a real update from the notebook rather than a drawn
    one, so the bars are the sizes the reader has just computed.
    """
    _style()
    d = np.asarray(delta, float)
    n = len(d)
    keep = set(np.argsort(np.abs(d))[-k:].tolist())
    names = labels or ("Full update", f"Top {k} changes", "One-bit signs")
    tags = ("all magnitudes", "largest magnitudes", "directions only")
    sent = (f"{n} of {n} numerical changes",
            f"{k} values and their positions",
            f"{n} directions, no magnitudes")
    icons = ("▦", str(k), "±")

    W, H = 12.6, 7.9
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0, 1, 1])
    _bare(ax)
    ax.set_xlim(0, W); ax.set_ylim(H, 0)
    top, bot = 1.62, 0.98                       # room for the headline and the takeaway
    gap, side = 0.30, 0.40
    cw = (W - 2 * side - 2 * gap) / 3
    ch = 2.82
    top_abs = np.abs(d).max() or 1.0

    for i in range(3):
        x0 = side + i * (cw + gap)
        _card(ax, x0, top, cw, ch, PANEL, r_px=12, z=1)
        _cell(ax, x0 + 0.24, top + 0.26, 0.52, 0.52, _mix(ACCENT, 0.86), gap_px=0, r_px=8, z=2)
        ax.text(x0 + 0.50, top + 0.58, icons[i], ha="center", va="center",
                fontsize=14.9, color=ACCENT, zorder=3)
        ax.text(x0 + 0.92, top + 0.46, names[i], fontsize=14.4, fontweight="bold",
                color=INK, va="center", zorder=3)
        ax.text(x0 + 0.92, top + 0.74, tags[i], fontsize=10.8, color=MUTED,
                va="center", zorder=3)
        ax.plot([x0 + 0.24, x0 + cw - 0.24], [top + 1.02, top + 1.02],
                color="#e2e0d8", lw=1.0, zorder=2)
        ax.text(x0 + 0.24, top + 1.28, f"the same {n}-number update", fontsize=10.1,
                color=MUTED, va="center", zorder=3)
        ax.text(x0 + 0.24, top + 1.58, sent[i], fontsize=11.3, fontweight="bold",
                color=ACCENT, va="center", zorder=3)

        # the sparkline: one mark per parameter, on a shared baseline
        base = top + 2.28
        x_lo, x_hi = x0 + 0.34, x0 + cw - 0.34
        xs = np.linspace(x_lo, x_hi, n)
        ax.plot([x_lo - 0.06, x_hi + 0.06], [base, base], color=BASELINE, lw=1.0, zorder=2)
        for j, (x, v) in enumerate(zip(xs, d)):
            up = v > 0
            if i == 2:                                       # every direction, one length
                tip = base - 0.30 if up else base + 0.30
                ax.plot([x, x], [base, tip], color=ACCENT, lw=1.6,
                        solid_capstyle="round", zorder=4)
                ax.scatter([x], [tip], marker="^" if up else "v", s=26,
                           color=ACCENT, zorder=5, linewidths=0)
            elif i == 0 or j in keep:                        # a real magnitude, to scale
                h = max(0.07, abs(v) / top_abs * 0.36)
                _cell(ax, x - 0.055, base - h if up else base, 0.11, h,
                      ACCENT, gap_px=0, r_px=2, z=4)
            else:                                            # dropped: a flat placeholder
                _cell(ax, x - 0.055, base - 0.03, 0.11, 0.06,
                      NEUTRAL, gap_px=0, r_px=2, z=3)

    # ------------------------------------------------- what each one costs to send
    by = top + ch + 0.40
    bh = 1.94
    _card(ax, side, by, W - 2 * side, bh, PANEL, r_px=12, z=1)
    ax.text(side + 0.30, by + 0.38, "EFFECT ON NETWORK TRAFFIC", fontsize=9.9,
            color=ACCENT, fontweight="bold", va="center", zorder=3)
    ax.text(W - side - 0.30, by + 0.38, "less detail makes a smaller upload",
            fontsize=14.4, fontweight="bold", color=INK, ha="right", va="center", zorder=3)
    widest = max(byte_costs)
    track_lo, track_hi = side + 2.30, W - side - 3.05
    for i, cost in enumerate(byte_costs):
        y = by + 0.82 + i * 0.44
        ax.text(side + 0.30, y, names[i], fontsize=11.7, fontweight="bold",
                color=INK, va="center", zorder=3)
        _pill(ax, track_lo, track_hi, y, 0.16, TRACK, r_px=5, zorder=2)
        _pill(ax, track_lo, track_lo + (track_hi - track_lo) * cost / widest, y, 0.16,
              ACCENT, r_px=5, zorder=3)
        ax.text(track_hi + 0.22, y, f"{cost:,.0f} bytes", fontsize=12.2,
                fontweight="bold", color=ACCENT, va="center", zorder=3)
        ax.text(W - side - 0.30, y, tags[i], fontsize=10.8, color=MUTED,
                ha="right", va="center", zorder=3)

    _headline(fig, "One update can travel in three forms.",
              f"A {V.site} trains the same way whichever one it uses. What changes is how much "
              "of the update survives the trip, not the records behind it.")
    _takeaway(fig, f"The same {n} numbers, sent three ways, for "
                   f"{byte_costs[0]:,.0f}, {byte_costs[1]:,.0f} and {byte_costs[2]:,.0f} bytes. "
                   "What none of that says is what the smaller two cost in accuracy, which is "
                   "the last thing this part measures.")
    return _done(fig)


def _density(ax, x0, x1, y, n, colour, drawn_cap=208):
    """One glyph per parameter while that is possible, then a suggestion of the rest.

    Returns the caption for what was actually drawn, because a figure that shows 208 marks
    for 1,900 numbers has to say so rather than let a reader count them.
    """
    w, h = x1 - x0, 0.62
    if n <= 32:
        cw = min(0.13, w / (n * 1.55))
        gap = (w - n * cw) / (n - 1) if n > 1 else 0
        for i in range(int(n)):
            _cell(ax, x0 + i * (cw + gap), y - 0.15, cw, 0.30, colour,
                  gap_px=0, r_px=2, z=4)
        return "every one drawn"
    if n <= 40_000:
        cols, rows = 26, 8
        cw, ch = w / cols * 0.72, h / rows * 0.66
        for r in range(rows):
            for c in range(cols):
                _cell(ax, x0 + c * w / cols, y - h / 2 + r * h / rows, cw, ch,
                      colour, gap_px=0, r_px=1, z=4)
        return f"{cols * rows} of {n:,.0f} drawn"
    _cell(ax, x0, y - h / 2, w, h, colour, gap_px=0, r_px=4, z=4)
    return "too many to draw"


def model_widths(entries, cards=3):
    """The same message with more numbers in it, and what that does to the bill.

    entries: (label, parameters, bytes_per_exchange[, colour]). The first `cards` are drawn
    as cards, because the point of the top row is that widening a model changes the size of
    one message and nothing else about the procedure. The ladder underneath carries every
    entry on a log axis: the range is nine orders of magnitude and no linear bar survives it.
    """
    _style()
    shown = entries[:cards]
    W, H = 12.6, 8.6
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0, 1, 1]); _bare(ax)
    ax.set_xlim(0, W); ax.set_ylim(H, 0)
    top, gap, side = 1.66, 0.30, 0.40
    cw = (W - 2 * side - (len(shown) - 1) * gap) / len(shown)
    ch = 2.86

    for i, e in enumerate(shown):
        label, params = e[0], e[1]
        x0 = side + i * (cw + gap)
        _card(ax, x0, top, cw, ch, PANEL, r_px=12, z=1)
        chip = (f"{params:,.0f}" if params < 1000 else
                f"{params/1e6:,.1f}M" if params >= 1e6 else f"{params/1e3:,.1f}k")
        _cell(ax, x0 + 0.24, top + 0.26, 0.78, 0.52, _mix(ACCENT, 0.86),
              gap_px=0, r_px=8, z=2)
        ax.text(x0 + 0.63, top + 0.53, chip, ha="center", va="center",
                fontsize=13.2, fontweight="bold", color=ACCENT, zorder=3)
        human = (f"{params:,.0f}" if params < 1e6 else
                 f"{params/1e9:,.0f} billion" if params >= 1e9 else f"{params/1e6:,.1f} million")
        ax.text(x0 + 1.18, top + 0.42, label.split("\u2014")[0].strip(), fontsize=12.8,
                fontweight="bold", color=INK, va="center", zorder=3)
        ax.text(x0 + 1.18, top + 0.70, f"{human} numbers per message",
                fontsize=10.0, color=MUTED, va="center", zorder=3)
        ax.plot([x0 + 0.24, x0 + cw - 0.24], [top + 1.02, top + 1.02],
                color="#e2e0d8", lw=1.0, zorder=2)
        ax.text(x0 + 0.24, top + 1.30, "the same procedure, every time",
                fontsize=10.1, color=MUTED, va="center", zorder=3)
        ax.text(x0 + 0.24, top + 1.60, _human_bytes(e[2]) + " per exchange",
                fontsize=11.3, fontweight="bold", color=ACCENT, va="center", zorder=3)
        note = _density(ax, x0 + 0.30, x0 + cw - 0.30, top + 2.26, params, ACCENT)
        ax.text(x0 + cw / 2, top + 2.70, note, fontsize=9.7, color=MUTED,
                ha="center", va="center", zorder=5)

    # ------------------------------------------------- the bill, where linear gives up
    by, bh = top + ch + 0.40, 2.46
    _card(ax, side, by, W - 2 * side, bh, PANEL, r_px=12, z=1)
    ax.text(side + 0.30, by + 0.38, "WHAT ONE EXCHANGE COSTS", fontsize=9.9,
            color=ACCENT, fontweight="bold", va="center", zorder=3)
    ax.text(W - side - 0.30, by + 0.38, "each step up is a thousand times the last",
            fontsize=14.4, fontweight="bold", color=INK, ha="right", va="center", zorder=3)
    ax.text(side + 0.30, by + 0.64, "bar length is a log scale, so equal steps are equal "
            "multiples", fontsize=9.7, color=MUTED, va="center", zorder=3)
    lo, hi = side + 3.30, W - side - 2.20
    top_log = np.log10(max(e[2] for e in entries))
    for i, e in enumerate(entries):
        y = by + 1.10 + i * 0.36
        ax.text(side + 0.30, y, e[0], fontsize=11.5, color=INK, va="center", zorder=3)
        _pill(ax, lo, hi, y, 0.15, TRACK, r_px=5, zorder=2)
        col = e[3] if len(e) > 3 and e[3] else ACCENT
        _pill(ax, lo, lo + (hi - lo) * np.log10(e[2]) / top_log, y, 0.15, col,
              r_px=5, zorder=3)
        ax.text(hi + 0.20, y, _human_bytes(e[2]), fontsize=12.0, fontweight="bold",
                color=col, va="center", zorder=3)
    _headline(fig, "The procedure does not change. The message does.",
              f"Same {V.sites}, same averaging, same accountant. All that grows is how many "
              "numbers each exchange has to carry.")
    _takeaway(fig, f"From {_human_bytes(entries[0][2])} to {_human_bytes(entries[-1][2])} an "
                   "exchange, before anyone has decided how many exchanges there will be. "
                   "Nothing in the method noticed.")
    return _done(fig)


def federation_widths(rows):
    """The same procedure with more participants, and who actually speaks.

    rows: (label, participants, sampled, bytes_if_all, bytes_if_sampled, auc), which is
    what the cell above measures. The cards draw the enrolled against the sampled, because
    the whole answer to horizontal scale is that those two stop being the same number. The
    panel prices what that saves, and carries the score so the saving cannot be mistaken
    for a shortcut.
    """
    _style()
    W, H = 12.6, 8.8
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0, 1, 1]); _bare(ax)
    ax.set_xlim(0, W); ax.set_ylim(H, 0)
    top, gap, side = 1.66, 0.30, 0.40
    cw = (W - 2 * side - (len(rows) - 1) * gap) / len(rows)
    ch = 2.86

    for i, (label, n, k, b_all, b_some, auc) in enumerate(rows):
        x0 = side + i * (cw + gap)
        share = k / n
        _card(ax, x0, top, cw, ch, PANEL, r_px=12, z=1)
        chip = f"{n:,.0f}" if n < 1000 else f"{n/1000:,.0f}k"
        _cell(ax, x0 + 0.24, top + 0.26, 0.78, 0.52, _mix(ACCENT, 0.86),
              gap_px=0, r_px=8, z=2)
        ax.text(x0 + 0.63, top + 0.53, chip, ha="center", va="center",
                fontsize=13.2, fontweight="bold", color=ACCENT, zorder=3)
        ax.text(x0 + 1.18, top + 0.42, f"{n:,} enrolled", fontsize=12.8,
                fontweight="bold", color=INK, va="center", zorder=3)
        ax.text(x0 + 1.18, top + 0.70, "every one waited for" if share == 1
                else f"{share:.0%} speak each exchange",
                fontsize=10.0, color=MUTED, va="center", zorder=3)
        ax.plot([x0 + 0.24, x0 + cw - 0.24], [top + 1.02, top + 1.02],
                color="#e2e0d8", lw=1.0, zorder=2)
        ax.text(x0 + 0.24, top + 1.30, "who is asked in one exchange",
                fontsize=10.1, color=MUTED, va="center", zorder=3)
        ax.text(x0 + 0.24, top + 1.60, f"{k:,} of {n:,}", fontsize=11.3,
                fontweight="bold", color=ACCENT, va="center", zorder=3)

        # one dot per participant while that fits, then a fair sample of them
        x_lo, x_hi = x0 + 0.32, x0 + cw - 0.32
        cy, band = top + 2.28, 0.60
        if n <= 24:
            cols, rows_n = int(n), 1
        else:
            cols, rows_n = 25, 8
        drawn = cols * rows_n
        lit = max(1, int(round(share * drawn)))
        dx = (x_hi - x_lo) / cols
        dy = band / rows_n
        r = min(dx, dy) * 0.30 if n > 24 else 0.075
        # Spread the sampled ones through the grid: a round draws from everywhere, and a
        # lit corner reads as though the same few always answer. A plain stride lands them
        # all in one column whenever it divides the column count, so this walks the golden
        # ratio instead, which is the standard way to scatter without a generator.
        on_set = set()
        for j in range(drawn):
            if len(on_set) >= lit:
                break
            on_set.add(int(j * 0.6180339887498949 * drawn) % drawn)
        for j in range(drawn):                       # top up if the walk collided
            if len(on_set) >= lit:
                break
            on_set.add(j)
        for j in range(drawn):
            cx = x_lo + (j % cols) * dx + dx / 2
            yy = cy - band / 2 + (j // cols) * dy + dy / 2
            on = j in on_set
            ax.add_patch(plt.Circle((cx, yy), r, facecolor=ACCENT if on else NEUTRAL,
                                    edgecolor="none", zorder=4 if on else 3))
        ax.text(x0 + cw / 2, top + 2.72,
                "every one drawn" if drawn >= n else f"{drawn} of {n:,} drawn, to scale",
                fontsize=9.7, color=MUTED, ha="center", va="center", zorder=5)

    # ------------------------------------------------- what the sampling saves
    by, bh = top + ch + 0.40, 2.62
    _card(ax, side, by, W - 2 * side, bh, PANEL, r_px=12, z=1)
    ax.text(side + 0.30, by + 0.38, "WHAT ONE EXCHANGE COSTS", fontsize=9.9,
            color=ACCENT, fontweight="bold", va="center", zorder=3)
    ax.text(W - side - 0.30, by + 0.38, "the bill follows who speaks, not who joined",
            fontsize=14.4, fontweight="bold", color=INK, ha="right", va="center", zorder=3)
    key = by + 0.70
    for kx, colour, text in ((side + 0.30, ACCENT, "sent: only the sampled speak"),
                             (side + 3.40, _mix(ACCENT, 0.62),
                              "what waiting for every enrolled one would cost")):
        _cell(ax, kx, key - 0.09, 0.26, 0.18, colour, gap_px=0, r_px=4, z=3)
        ax.text(kx + 0.36, key, text, fontsize=9.9, color=INK_2, va="center", zorder=3)
    ax.text(W - side - 0.30, key, "bar length is a log scale", fontsize=9.4,
            color=MUTED, ha="right", va="center", zorder=3)
    lo, hi = side + 2.70, W - side - 4.35
    # Log lengths, because the two quantities on each row are fifty times apart and a
    # linear bar renders the one that matters as a sliver against the one that does not.
    widest = np.log10(max(r[3] for r in rows))
    span = lambda v: (hi - lo) * np.log10(v) / widest
    for i, (label, n, k, b_all, b_some, auc) in enumerate(rows):
        y = by + 1.26 + i * 0.42
        ax.text(side + 0.30, y, f"{n:,} enrolled", fontsize=11.5, color=INK,
                va="center", zorder=3)
        _pill(ax, lo, hi, y, 0.17, TRACK, r_px=5, zorder=2)
        _pill(ax, lo, lo + span(b_all), y, 0.17, _mix(ACCENT, 0.62), r_px=5, zorder=3)
        _pill(ax, lo, lo + span(b_some), y, 0.17, ACCENT, r_px=5, zorder=4)
        ax.text(hi + 0.20, y, _human_bytes(b_some), fontsize=12.0, fontweight="bold",
                color=ACCENT, va="center", zorder=3)
        if b_all > b_some:
            ax.text(hi + 1.05, y, f"of {_human_bytes(b_all)}", fontsize=11.0, color=MUTED,
                    va="center", zorder=3)
        ax.text(W - side - 0.30, y, f"AUC {auc:.3f}", fontsize=11.7, color=INK_2,
                ha="right", va="center", zorder=3)

    _headline(fig, "The procedure does not change. The guest list does.",
              "Same averaging, same local steps, same number of exchanges. What changes is "
              "how many of the enrolled are asked to answer any one of them.")
    _takeaway(fig, f"{rows[-1][2]:,} of {rows[-1][1]:,} speak and the model still lands at "
                   f"{rows[-1][5]:.3f}. Sampling is not a compromise a large federation "
                   "tolerates; it is how one is designed to run.")
    return _done(fig)
