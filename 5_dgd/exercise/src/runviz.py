"""Figures, explorers and the builder.

Three colours carry meaning and nothing else does. Compute is what you want,
communication is what you pay, memory is what must fit. Idle time is hatched
grey, and anything that does not fit or is not allowed is a hatch, a cross and
the word, in ink.
"""
import io
import math
import re
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
from matplotlib.lines import Line2D
from IPython.display import HTML, display

import brief as b
import runcore as rc

# ------------------------------------------------------------------- the ink
GROUND = "#ffffff"
PANEL = "#f6f4ef"
BAND = "#efece4"
HAIR = "#e2ded4"
HAIR_2 = "#cdc7ba"
INK = "#1c1b19"
INK_2 = "#57544e"
MUTED = "#8d8880"

COMPUTE = "#478c58"
COMMS = "#3068a4"
MEMORY = "#ad6330"
IDLE = "#b9b3a8"

SOFT = {COMPUTE: "#e4efe6", COMMS: "#e0e8f2", MEMORY: "#f5e8dd"}

# Only fonts matplotlib bundles, so a figure looks the same here, in Colab and
# on a machine that has never installed anything.
SERIF = "STIXGeneral"
SANS = "DejaVu Sans"
MONO = "DejaVu Sans Mono"

# The widgets render in the browser, so they may ask for more.
# ------------------------------------------------------------ the figure theme
# Every figure is 980 css px wide and draws from one type scale. Two things make
# that true, and both are easy to undo by accident:
#
#   1. savefig.bbox is NOT "tight". Tight crops each plot to its own content, so
#      every one lands at a different width on the page (they ranged 1008 to
#      1259) and its text scales with it. Plots are pinned to FIG_W instead.
#   2. A plot measures type in points on a 735pt canvas; a hand-drawn figure
#      measures it in units on a 980-unit canvas. The same apparent size is
#      therefore svg x 0.735, which is why the two columns below differ.
#
#   role                  svg    matplotlib    css widgets    on the page
#   the conclusion         25       18.5           22            25 px
#   a big number           17       12.5           17            17 px
#   a heading            16.5       12.5           16            16 px
#   the takeaway band      16         12         15.5            16 px
#   body, rows, notes    15.5       11.5         14.5            15 px
#   ticks and units      13.5         10           13            13 px
#
# Nothing goes below 13.5 in a hand-drawn figure or 10 in a plot. These are read
# on a projector, and anything smaller disappears from the back of the room.

WEB_SERIF = "Charter, 'Source Serif 4', Georgia, serif"
WEB_SANS = "system-ui, -apple-system, 'Segoe UI', Helvetica, sans-serif"
WEB_MONO = "'SF Mono', Menlo, 'DejaVu Sans Mono', monospace"


def use_house_style():
    # Element ids are otherwise random, so the same figure differs byte for byte
    # every time it is drawn. A fixed salt makes a rebuilt notebook diffable.
    matplotlib.rcParams["svg.hashsalt"] = "we7-distributed-training"
    matplotlib.rcParams.update({
        "figure.facecolor": GROUND, "axes.facecolor": GROUND, "savefig.facecolor": GROUND,
        "font.family": "sans-serif", "font.sans-serif": [SANS],
        "font.serif": [SERIF], "font.monospace": [MONO], "font.size": 11.5,
        "mathtext.fontset": "stix",
        "text.color": INK, "axes.labelcolor": INK_2, "axes.edgecolor": HAIR_2,
        "xtick.color": MUTED, "ytick.color": MUTED,
        "xtick.labelsize": 10, "ytick.labelsize": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": 0.9, "lines.linewidth": 2.0, "lines.solid_capstyle": "round",
        # NOT "tight": tight crops each plot to its own content, so every one
        # lands at a different width on the page and its text scales with it.
        "figure.dpi": 110, "savefig.bbox": None, "legend.frameon": False,
    })


use_house_style()

# A plot is 980 css px wide, the same as a hand-drawn one: 980 / 96 inches.
FIG_PX = 980
FIG_W = FIG_PX / 96


def _title(fig, conclusion, sub=None, x=0.012, y=0.985):
    """Titles state the conclusion. The subtitle says what you are looking at."""
    fig.text(x, y, conclusion, ha="left", va="top", color=INK,
             fontfamily="serif", fontsize=18.5, fontweight="bold")
    if sub:
        fig.text(x, y - 0.075, sub, ha="left", va="top", color=INK_2, fontsize=11.5)


BAND_TOP = 0.145          # nothing on a full-canvas figure may go below this


def _takeaway(fig, text, y=0.012, width=96):
    """One band per figure. It wraps, and the band grows to hold what it wraps to."""
    import textwrap
    lines = textwrap.wrap(text, width=width)
    line_h = 0.040
    h = 0.048 + line_h * len(lines)
    fig.patches.append(FancyBboxPatch(
        (0.012, y), 0.976, h, transform=fig.transFigure,
        boxstyle="round,pad=0.004,rounding_size=0.012",
        facecolor=BAND, edgecolor=HAIR, linewidth=0.9, zorder=0))
    first = y + h - 0.024 - line_h / 2
    for i, line in enumerate(lines):
        fig.text(0.5, first - i * line_h, line, ha="center", va="center", color=INK,
                 fontsize=11.5, fontweight="bold")


def _done(fig):
    """Render it here rather than handing the figure to the notebook.

    The inline backend crops every figure to its own content and ignores
    savefig.bbox while doing it, which is how the plots ended up anywhere
    between 984 and 1,183 px wide while every drawn figure was exactly 980.
    Rendering it ourselves pins the width, and closing it here means the cell
    shows the figure once rather than twice.
    """
    buf = io.StringIO()
    fig.savefig(buf, format="svg", bbox_inches=None, facecolor=GROUND)
    plt.close(fig)
    svg = buf.getvalue()
    svg = svg[svg.index("<svg"):]
    svg = re.sub(r'(<svg[^>]*?)width="[0-9.]+pt"\s+height="[0-9.]+pt"',
                 r'\1width="100%"', svg, count=1)
    return HTML(f'<div style="max-width:{FIG_PX}px">{svg}</div>')


def _tidy(ax):
    ax.tick_params(length=0, pad=6)
    for s in ax.spines.values():
        s.set_color(HAIR_2)
    ax.set_axisbelow(True)


def _mono(t):
    t.set_fontfamily("monospace")
    return t


# ------------------------------------------------------- svg diagram helpers
BAND_H = 42               # the takeaway band on an svg figure
_SVG_WRAP = 108           # characters before the takeaway wraps


def _svg(width, content_bottom, conclusion, sub=None, takeaway=None, body=""):
    """One diagram. The height is derived from where the drawing ends, plus the band.

    Passing where the content stops, rather than a total height, is what stops a
    figure from ever being drawn underneath its own takeaway.
    """
    import textwrap
    pad = 34
    lines = textwrap.wrap(takeaway, _SVG_WRAP) if takeaway else []
    band_h = BAND_H + 20 * (len(lines) - 1) if lines else 0
    height = content_bottom + (band_h + 22 if lines else 14)
    head = (f'<text x="{pad}" y="36" font-family="{WEB_SERIF}" font-size="25" font-weight="700" '
            f'fill="{INK}">{conclusion}</text>')
    if sub:
        head += f'<text x="{pad}" y="61" font-size="16" fill="{INK_2}">{sub}</text>'
    band = ""
    if lines:
        by = content_bottom + 14
        band = (f'<rect x="{pad-8}" y="{by}" width="{width-2*pad+16}" height="{band_h}" rx="9" '
                f'fill="{BAND}" stroke="{HAIR}"/>')
        first = by + band_h / 2 - 10 * (len(lines) - 1) + 5
        for i, line in enumerate(lines):
            band += (f'<text x="{width/2}" y="{first + i*20:.0f}" text-anchor="middle" '
                     f'font-size="16" font-weight="700" fill="{INK}">{line}</text>')
    return HTML(
        f'<svg viewBox="0 0 {width} {height}" width="100%" style="max-width:{width}px" '
        f'xmlns="http://www.w3.org/2000/svg" font-family="{WEB_SANS}">'
        f'<rect width="{width}" height="{height}" fill="{GROUND}"/>{head}{body}{band}</svg>')


def _wrap(text, width):
    """Break a note into lines the caller draws one under the other."""
    import textwrap
    return textwrap.wrap(text, width)


def _num(v, places=2):
    """Two decimals unless the number is whole, so 1/3 never prints as 0.333333."""
    return f"{v:.0f}" if abs(v - round(v)) < 1e-9 else f"{v:.{places}f}"


def _box(x, y, w, h, fill=GROUND, stroke=HAIR_2, r=8, width=1.6, dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return (f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{r}" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="{width}"{d}/>')


def _label(x, y, text, size=13, colour=INK_2, anchor="middle", weight=None, mono=False):
    f = f' font-family="{WEB_MONO}"' if mono else ""
    w = f' font-weight="{weight}"' if weight else ""
    return (f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" font-size="{size}"{f}{w} '
            f'fill="{colour}">{text}</text>')


def _arrow(x1, y1, x2, y2, colour=HAIR_2, width=1.6, head=6):
    import math
    a = math.atan2(y2 - y1, x2 - x1)
    hx, hy = x2 - head * math.cos(a), y2 - head * math.sin(a)
    p1 = (hx - head * 0.6 * math.sin(a), hy + head * 0.6 * math.cos(a))
    p2 = (hx + head * 0.6 * math.sin(a), hy - head * 0.6 * math.cos(a))
    return (f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{hx:.1f}" y2="{hy:.1f}" stroke="{colour}" '
            f'stroke-width="{width}" stroke-linecap="round"/>'
            f'<polygon points="{x2:.1f},{y2:.1f} {p1[0]:.1f},{p1[1]:.1f} {p2[0]:.1f},{p2[1]:.1f}" '
            f'fill="{colour}"/>')


def _hatch(uid, colour="#d8d3c9", width=1.6, gap=7):
    """Idle time. Light enough that anything drawn on top of it still reads."""
    return (f'<defs><pattern id="{uid}" width="{gap}" height="{gap}" patternTransform="rotate(45)" '
            f'patternUnits="userSpaceOnUse"><line x1="0" y1="0" x2="0" y2="{gap}" stroke="{colour}" '
            f'stroke-width="{width}"/></pattern></defs>')


# ============================================================ reference one
def efficiency_curve():
    """Part 3, cell 37. The plot family: conclusion title, direct labels, one band."""
    buffer_gb = rc.whole_model_buffer_gb()
    one = rc.one_machine_step_seconds()
    link = b.LINK_ACROSS / b.GB

    def eff(m):
        comp = one / m
        comm = 2 * (m - 1) * (buffer_gb / m) / link
        return comp / (comp + comm)

    machines = rc.whole_boxes()
    values = [eff(m) * 100 for m in machines]
    crossing = max(m for m in machines if eff(m) >= b.EFFICIENCY_TARGET)

    fig, ax = plt.subplots(figsize=(FIG_W, 5.0))
    fig.subplots_adjust(top=0.80, bottom=0.24, left=0.075, right=0.985)

    ax.axhspan(b.EFFICIENCY_TARGET * 100, 100, color=SOFT[COMPUTE], zorder=0)
    ax.axhline(b.EFFICIENCY_TARGET * 100, color=COMPUTE, lw=1.4, ls=(0, (5, 3)), zorder=2)
    ax.plot(machines, values, color=COMMS, zorder=3)
    ax.plot([crossing], [eff(crossing) * 100], "o", ms=9, color=COMMS,
            markeredgecolor=GROUND, markeredgewidth=2, zorder=4)
    ax.plot([b.MACHINES], [eff(b.MACHINES) * 100], "o", ms=9, color=INK,
            markeredgecolor=GROUND, markeredgewidth=2, zorder=4)

    _mono(ax.annotate(f"{crossing} machines", (crossing, eff(crossing) * 100),
                      textcoords="offset points", xytext=(12, 12),
                      color=COMMS, fontsize=11.5, fontweight="bold"))
    _mono(ax.annotate(f"all {b.MACHINES:,}\n{eff(b.MACHINES)*100:.0f}%", (b.MACHINES, eff(b.MACHINES) * 100),
                      textcoords="offset points", xytext=(-14, -30), ha="right",
                      color=INK, fontsize=11.5, fontweight="bold"))
    ax.text(machines[1], b.EFFICIENCY_TARGET * 100 + 2.4,
            f"the {b.EFFICIENCY_TARGET:.0%} the brief asks for", color=COMPUTE, fontsize=10)

    ax.set_xscale("log", base=2)
    ax.set_xticks([8, 32, 128, 512, 2048])
    ax.set_xticklabels(["8", "32", "128", "512", "2,048"])
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.set_xlabel("machines, every one holding the whole model")
    ax.set_ylabel("share of each step spent on arithmetic")
    ax.set_ylim(0, 104)
    ax.set_xlim(7, 2600)
    _tidy(ax)

    _title(fig, f"This way of working can use {crossing} machines. We own {b.MACHINES:,}.",
           "Each machine still holds the whole model, so every one of them agrees over the same "
           f"{buffer_gb:.0f} GB every step.")
    _takeaway(fig, "The bill for agreeing stops growing. The work it is compared against does not stop shrinking.")
    return _done(fig)


# ============================================================ reference two
def what_one_machine_holds():
    """Part 4, cell 43. The diagram family: hand-authored SVG, no plot underneath."""
    items = [
        ("the weights", 2, "narrow", MEMORY),
        ("their direction of travel", 2, "narrow", MEMORY),
        ("a wider copy of the weights", 4, "wide", MEMORY),
        ("a record of how each one has been moving", 4, "wide", MEMORY),
        ("and how much that has varied", 4, "wide", MEMORY),
    ]
    total = sum(n for _, n, _, _ in items)
    W, H, pad = 980, 348, 34
    span = W - 2 * pad
    unit = span / total
    whole = b.PARAMETERS * total / b.GB

    parts = [f'<svg viewBox="0 0 {W} {H}" width="100%" style="max-width:{FIG_PX}px" '
             f'xmlns="http://www.w3.org/2000/svg" font-family="{WEB_SANS}">',
             f'<rect width="{W}" height="{H}" fill="{GROUND}"/>',
             f'<text x="{pad}" y="34" font-family="{WEB_SERIF}" font-size="25" '
             f'font-weight="700" fill="{INK}">Two bytes to store a parameter. Sixteen to improve one.</text>',
             f'<text x="{pad}" y="58" font-size="15.5" fill="{INK_2}">Training keeps five arrays, '
             f'each one holding a number for every parameter. Only the first array is the model.</text>']

    x, y, bar_h = pad, 116, 62
    for label, n, kind, colour in items:
        w = n * unit
        fill = SOFT[colour] if kind == "narrow" else GROUND
        parts.append(f'<rect x="{x:.1f}" y="{y}" width="{w - 3:.1f}" height="{bar_h}" rx="7" '
                     f'fill="{fill}" stroke="{colour}" stroke-width="1.6"/>')
        parts.append(f'<text x="{x + w/2 - 1.5:.1f}" y="{y + bar_h/2 + 7}" text-anchor="middle" '
                     f'font-family="{WEB_MONO}" font-size="22" font-weight="600" '
                     f'fill="{colour}">{n}</text>')
        wrapped = label.split(" ")
        line1 = " ".join(wrapped[:3])
        line2 = " ".join(wrapped[3:])
        parts.append(f'<text x="{x + w/2 - 1.5:.1f}" y="{y + bar_h + 22}" text-anchor="middle" '
                     f'font-size="13.5" fill="{INK_2}">{line1}</text>')
        if line2:
            parts.append(f'<text x="{x + w/2 - 1.5:.1f}" y="{y + bar_h + 38}" text-anchor="middle" '
                         f'font-size="13.5" fill="{INK_2}">{line2}</text>')
        parts.append(f'<text x="{x + w/2 - 1.5:.1f}" y="{y + bar_h + 58}" text-anchor="middle" '
                     f'font-family="{WEB_MONO}" font-size="13.5" '
                     f'fill="{MUTED}">{b.PARAMETERS * n / b.GB:,.0f} GB</text>')
        x += w

    # The takeaway is drawn, not only stated: one block is the model, four are the cost.
    group_y = y - 16
    first_w = items[0][1] * unit - 3
    rest_x = pad + first_w + 3
    rest_w = W - pad - rest_x - 3
    for gx, gw, label, colour in [(pad, first_w, "the model", INK),
                                  (rest_x, rest_w, "what it takes to improve it", MUTED)]:
        parts.append(f'<path d="M{gx:.1f} {group_y+8} L{gx:.1f} {group_y} L{gx+gw:.1f} {group_y} '
                     f'L{gx+gw:.1f} {group_y+8}" fill="none" stroke="{HAIR_2}" stroke-width="1.2"/>')
        parts.append(f'<text x="{gx + gw/2:.1f}" y="{group_y - 7}" text-anchor="middle" '
                     f'font-size="13.5" fill="{colour}">{label}</text>')

    brace_y = y + bar_h + 76
    parts.append(f'<path d="M{pad} {brace_y} L{pad} {brace_y+7} L{W-pad-3} {brace_y+7} L{W-pad-3} {brace_y}" '
                 f'fill="none" stroke="{HAIR_2}" stroke-width="1.4"/>')
    parts.append(f'<text x="{W/2}" y="{brace_y + 30}" text-anchor="middle" '
                 f'font-family="{WEB_MONO}" font-size="16.5" fill="{INK}">'
                 f'2 + 2 + 4 + 4 + 4 = {total} bytes a parameter, {whole:,.0f} GB for this model</text>')

    band_y = H - 44
    parts.append(f'<rect x="{pad-8}" y="{band_y}" width="{W-2*pad+16}" height="34" rx="9" '
                 f'fill="{BAND}" stroke="{HAIR}"/>')
    model_bytes = items[0][1]
    parts.append(f'<text x="{W/2}" y="{band_y + 22}" text-anchor="middle" font-size="15.5" '
                 f'font-weight="700" fill="{INK}">Training a model needs about '
                 f'{(total-model_bytes)//model_bytes} times the memory of storing one.</text>')
    parts.append('</svg>')
    return HTML("".join(parts))


def show(figure_or_html):
    if isinstance(figure_or_html, HTML):
        display(figure_or_html)
    else:
        plt.show()


# ==================================================================== part 0
def the_brief():
    """Cell 0.3. The problem in two sentences, and the two things nobody can change.

    Deliberately not a list of every given number. The batch, the link speeds, the
    targets and the failure rate all arrive in the part that first needs them.
    """
    W, pad = 980, 34
    body = ""
    for i, line in enumerate([
            f"Train a {b.PARAMETERS/1e9:.0f} billion parameter model on "
            f"{b.TOKEN_BUDGET/1e12:.0f} trillion tokens.",
            f"You have {b.MACHINES:,} machines, and {b.DEADLINE_DAYS} days."]):
        body += _label(pad, 96 + i * 30, line, 22, INK, anchor="start", weight=600)
    body += _label(pad, 180, f"You have also been asked to keep the machines working, rather than "
                   f"waiting, for at least {b.EFFICIENCY_TARGET:.0%} of the time.",
                   15.5, INK_2, anchor="start")

    cards = [("the model", [
                  f"{b.PARAMETERS/1e9:.0f} billion parameters",
                  f"{b.LAYERS} layers, {b.HIDDEN:,} wide",
                  f"{b.HEADS} attention heads, {b.KEY_VALUE_HEADS} carrying keys and values"]),
             (f"one machine, and you own {b.MACHINES:,}", [
                  f"{b.MEMORY_GB} GB of memory",
                  f"{b.PEAK_FLOPS/1e12:,.0f} TFLOP/s at peak",
                  f"{b.SUSTAINED:.0%} of that in a run that really happens"])]
    cw = (W - 2 * pad - 26) / 2
    for i, (name, facts) in enumerate(cards):
        x = pad + i * (cw + 26)
        body += _box(x, 210, cw, 132, fill=PANEL, stroke=HAIR, r=10)
        body += _label(x + 20, 236, name, 13.5, MUTED, anchor="start")
        for j, fact in enumerate(facts):
            body += _label(x + 20, 268 + j * 26, fact, 16.5, INK, anchor="start")
    return _svg(W, 358, "One run, and the two things nobody can change.",
                "Nothing in this notebook is measured. Every number in it is worked out from "
                "numbers like these.",
                "The model is fixed and the machine is fixed. Everything between them is a "
                "decision, and that is what the rest of this is about.", body)


# ==================================================================== part 1
def size_of_the_job():
    """Cell 12. A run is a fixed quantity of arithmetic."""
    W, pad = 980, 34
    years = rc.years_on_one_machine()
    peak_years = rc.years_on_one_machine(b.PEAK_FLOPS)
    terms = [("6", "operations for every\nparameter, every token"),
             (f"{b.PARAMETERS/1e9:.0f} billion", "parameters"),
             (f"{b.TOKEN_BUDGET/1e12:.0f} trillion", "tokens")]
    body, x, y, w = "", pad, 96, 176
    for value, caption in terms:
        body += _box(x, y, w, 62, fill=SOFT[COMPUTE], stroke=COMPUTE)
        body += _label(x + w / 2, y + 38, value, 22, COMPUTE, weight=600, mono=True)
        for i, line in enumerate(caption.split("\n")):
            body += _label(x + w / 2, y + 82 + i * 17, line, 13.5)
        x += w
        if caption != terms[-1][1]:
            body += _label(x + 13, y + 40, "x", 19, MUTED)
            x += 26
    body += _label(x + 22, y + 40, "=", 19, MUTED)
    body += _box(x + 44, y, 214, 62, fill=GROUND, stroke=INK)
    body += _label(x + 151, y + 38, f"{rc.whole_job_flop():.1e}", 22, INK, weight=600, mono=True)
    body += _label(x + 151, y + 80, "operations, the whole run", 13.5)

    bar_y, bar_x, bar_w = 236, pad, W - 2 * pad
    share = peak_years / years
    body += _label(bar_x, bar_y - 14, f"the same job on one machine, {years:.0f} years end to end",
                   15.5, INK, anchor="start", weight=600)
    body += _box(bar_x, bar_y, bar_w * share, 38, fill=COMPUTE, stroke=COMPUTE, r=7)
    body += _box(bar_x + bar_w * share, bar_y, bar_w * (1 - share), 38,
                 fill="url(#hatch_size)", stroke=HAIR_2, r=7)
    body = _hatch("hatch_size") + body
    body += _label(bar_x + bar_w * share / 2, bar_y + 25, f"{peak_years:.0f} years",
                   17, GROUND, weight=600, mono=True)
    body += _label(bar_x + bar_w * (share + (1 - share) / 2), bar_y + 25,
                   f"{years - peak_years:.0f} years", 16, INK, weight=600, mono=True)
    body += _label(bar_x + bar_w * share / 2, bar_y + 58,
                   "arithmetic, at the rate on the datasheet", 15.5, COMPUTE)
    body += _label(bar_x + bar_w * (share + (1 - share) / 2), bar_y + 58,
                   "the machine not keeping up with itself", 13.5, INK_2)
    return _svg(W, bar_y + 74, f"One machine would take {years:.0f} years.",
                "The job is fixed before any hardware is named. What the hardware decides is how long it takes.",
                f"Only {b.SUSTAINED:.0%} of that time is the machine doing arithmetic at the rate it "
                f"was sold at. The rest is it waiting on itself.", body)


# ==================================================================== part 2
def splitting_the_batch():
    """Cell 2.1. The same eight examples, handed out two ways, side by side.

    The machines are drawn and the numbers they hold sit inside them, so the
    comparison is one glance rather than a memory of the panel above. It stops at
    the failure: the weighting that repairs it is the task underneath.
    """
    W, pad = 980, 34
    eight = rc.EIGHT
    cw, ch, cgap = 40, 34, 5

    def chips(x, y, values, fill, colour):
        out = ""
        for i, v in enumerate(values):
            cx = x + i * (cw + cgap)
            out += _box(cx, y, cw, ch, fill=fill, stroke=fill, r=6, width=1)
            out += _label(cx + cw / 2, y + 23, f"{v:+d}", 16.5, colour, weight=600, mono=True)
        return out

    # ------------------------------------------------ what we want back, once
    body = _box(pad, 92, W - 2 * pad, 48, fill=PANEL, stroke=HAIR, r=9)
    body += _label(pad + 16, 121, "all eight together", 15.5, MUTED, anchor="start")
    body += chips(pad + 150, 99, eight, GROUND, INK)
    body += _label(668, 121, "their average", 15.5, MUTED, anchor="end")
    body += _label(682, 122, f"{rc.whole_batch_mean():.1f}", 22, INK, anchor="start",
                   weight=700, mono=True)
    body += _label(728, 121, "the answer we want back", 15.5, INK_2, anchor="start")

    # ------------------------------------------------------- the two splits
    py, pw, gap = 160, (W - 2 * pad - 22) / 2, 22
    panels = [(pad, rc.EVEN_SPLIT, "two examples each",
               "what each machine holds, and what it answers",
               COMPUTE, SOFT[COMPUTE], "the answer we wanted, exactly", 1.6),
              (pad + pw + gap, rc.RAGGED_SPLIT, "three, two, one and two",
               "a scheduler gave the fast machine a little more",
               MEMORY, SOFT[MEMORY], "seven percent off, and nothing says so", 2.2)]
    truth = rc.whole_batch_mean()
    for px, split, head, sub, colour, soft, note, edge in panels:
        means = rc.local_means(split)
        plain = sum(means) / len(means)
        body += _box(px, py, pw, 310, fill=GROUND, stroke=colour if edge > 2 else HAIR_2,
                     r=10, width=edge)
        body += _label(px + 18, py + 26, head, 16.5, colour, anchor="start", weight=600)
        body += _label(px + 18, py + 46, sub, 13.5, MUTED, anchor="start")
        for i, (a_, c_) in enumerate(split):
            ry = py + 56 + i * 44
            body += (f'<line x1="{px+18}" y1="{ry-5}" x2="{px+pw-18}" y2="{ry-5}" '
                     f'stroke="{HAIR}" stroke-width="1"/>')
            body += _label(px + 18, ry + 23, f"machine {i+1}", 15.5, MUTED, anchor="start")
            body += chips(px + 96, ry, eight[a_:c_], soft, colour)
            shown = f"{means[i]:.0f}" if split is rc.EVEN_SPLIT else f"{means[i]:.2f}"
            body += _label(px + pw - 18, ry + 23, shown, 17, colour, anchor="end",
                           weight=600, mono=True)
        body += (f'<line x1="{px+18}" y1="{py+231}" x2="{px+pw-18}" y2="{py+231}" '
                 f'stroke="{HAIR}" stroke-width="1"/>')
        body += _box(px + 18, py + 244, pw - 36, 42, fill=soft, stroke=soft, r=8)
        body += _label(px + 34, py + 270, "average of the four answers", 15.5, colour, anchor="start")
        total = f"{truth:.1f}" if abs(plain - truth) < 1e-9 else _num(plain, 4)
        body += _label(px + pw - 34, py + 271, total, 22, colour, anchor="end",
                       weight=700, mono=True)
        body += _label(px + pw / 2, py + 304, note, 15.5, colour, weight=600)

    return _svg(W, 478, "Same eight examples. One split works and one does not.",
                "Each number is one example's opinion about which way the model should move.",
                "Averaging the machines only works when they hold the same number of examples, "
                "which is exactly why the mistake stays invisible.", body)


# ==================================================================== part 3
def hub_and_ring():
    """Cell 3.1. The same four machines wired two ways, and what each costs.

    Replaces a cost curve that never drew a hub and a ring that spent most of its
    width on a schedule grid. Both halves of Task 2 are on the page: the buffer cut
    into one slice per machine, and two laps that are each one hand-off short.
    """
    W, pad, n = 980, 34, 4
    buf = rc.whole_model_buffer_gb()
    hub_load = lambda m: (m - 1) * buf
    ring_load = lambda m: 2 * (m - 1) * buf / m
    slice_gb = buf / n
    big = b.MACHINES

    def node(cx, cy, r, label, fill, colour, size=16, weight=600):
        return (f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r}" fill="{fill}"/>'
                + _label(cx, cy + size * 0.36, label, size, colour, weight=weight, mono=True))

    def head(x, y, angle, colour):
        return (f'<polygon points="0,0 -9,-4.2 -9,4.2" fill="{colour}" '
                f'transform="translate({x:.1f},{y:.1f}) rotate({angle:.1f})"/>')

    py, pw, gap = 92, (W - 2 * pad - 22) / 2, 22
    lx, rx = pad, pad + pw + gap
    body = ""

    # ------------------------------------------------------------- the hub
    body += _box(lx, py, pw, 330, fill=GROUND, stroke=MEMORY, r=10, width=2.2)
    body += _label(lx + 18, py + 26, "a hub", 16.5, MEMORY, anchor="start", weight=600)
    body += _label(lx + 18, py + 46, "everyone sends everything to machine 1", 13.5, MUTED,
                   anchor="start")
    dx0, dy0 = lx + 18, py + 60                       # where the drawing starts
    hx, hy = dx0 + 204, dy0 + 88                      # the collecting machine
    sats = [(dx0 + 54, dy0 + 36, "2", -10, -14), (dx0 + 354, dy0 + 36, "3", 10, -14),
            (dx0 + 204, dy0 + 174, "4", 40, 2)]
    for sx, sy, name, ox, oy in sats:
        vx, vy = hx - sx, hy - sy
        d = (vx * vx + vy * vy) ** 0.5
        ux, uy = vx / d, vy / d
        x1, y1 = sx + 21 * ux, sy + 21 * uy
        x2, y2 = hx - 27 * ux, hy - 27 * uy
        body += (f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2-9*ux:.1f}" y2="{y2-9*uy:.1f}" '
                 f'stroke="{MEMORY}" stroke-width="1.6"/>')
        body += head(x2, y2, math.degrees(math.atan2(uy, ux)), MEMORY)
        body += _label((x1 + x2) / 2 + ox, (y1 + y2) / 2 + oy, f"{buf:.0f} GB", 13.5, MUTED,
                       mono=True)
    for sx, sy, name, _, _ in sats:
        body += node(sx, sy, 21, name, SOFT[MEMORY], MEMORY)
    body += node(hx, hy, 27, "1", MEMORY, GROUND, size=19, weight=700)

    # ------------------------------------------------------------ the ring
    body += _box(rx, py, pw, 330, fill=GROUND, stroke=HAIR_2, r=10)
    body += _label(rx + 18, py + 26, "a ring", 16.5, COMMS, anchor="start", weight=600)
    body += _label(rx + 18, py + 46, "everyone sends one slice to their neighbour", 13.5, MUTED,
                   anchor="start")
    cx, cy, R = rx + 112, py + 160, 64
    at = lambda t: (cx + R * math.cos(math.radians(t)), cy + R * math.sin(math.radians(t)))
    for k in range(n):
        t0, t1 = -70 + 90 * k, -24 + 90 * k
        (ax, ay), (bx, by) = at(t0), at(t1)
        body += (f'<path d="M {ax:.1f} {ay:.1f} A {R} {R} 0 0 1 {bx:.1f} {by:.1f}" fill="none" '
                 f'stroke="{COMMS}" stroke-width="1.6"/>') + head(bx, by, t1 + 90, COMMS)
    for k in range(n):
        nx, ny = at(-90 + 90 * k)
        body += node(nx, ny, 19, f"{k+1}", SOFT[COMMS], COMMS)
    body += _label(cx, cy - 4, "nothing", 13.5, MUTED)
    body += _label(cx, cy + 12, "in the middle", 13.5, MUTED)

    kx, kw = rx + 206, pw - 224                       # the column beside the ring
    body += _label(kx, py + 72, f"the {buf:.0f} GB, cut into four slices", 13.5, MUTED,
                   anchor="start")
    for k in range(n):
        chip = (kw - 3 * 4) / n
        body += _box(kx + k * (chip + 4), py + 80, chip, 24, fill=SOFT[COMMS],
                     stroke=SOFT[COMMS], r=5, width=1)
        body += _label(kx + k * (chip + 4) + chip / 2, py + 96, f"{slice_gb:.0f}", 13.5, COMMS,
                       weight=600, mono=True)
    for k, (lap, what) in enumerate([("lap 1", "adds"), ("lap 2", "copies")]):
        ly = py + 132 + k * 48
        body += _label(kx, ly, f"{lap} \u00b7 {what}", 15.5, INK, anchor="start", weight=600)
        body += _label(kx, ly + 18, f"{n-1} hand-offs, one short of a lap", 13.5, MUTED,
                       anchor="start")

    # ------------------------------------- the same question, answered twice
    for px, colour, who, load, note in [
            (lx, MEMORY, "machine 1 takes in", hub_load(n),
             "three whole buffers, one from each of the others"),
            (rx, COMMS, "every machine takes in", ring_load(n),
             f"{2*(n-1)} hand-offs of one slice, and nobody is special")]:
        body += (f'<line x1="{px+18}" y1="{py+270}" x2="{px+pw-18}" y2="{py+270}" '
                 f'stroke="{HAIR}" stroke-width="1"/>')
        body += _label(px + 18, py + 292, who, 15.5, INK_2, anchor="start")
        body += _label(px + pw - 18, py + 293, f"{load:.0f} GB", 23, colour, anchor="end",
                       weight=700, mono=True)
        body += _label(px + 18, py + 312, note, 15.5, MUTED, anchor="start")

    # ------------------------------- and the same question, all the way up
    x0, x1, ytop, ybot = 150, 860, 486, 648
    lo, hi = 100, 1e6                                 # four clean decades of GB
    span, steps = math.log10(hi / lo), int(math.log2(big))
    px_of = lambda m: x0 + (math.log2(m) - 1) / (steps - 1) * (x1 - x0)
    py_of = lambda v: ybot - (math.log10(v) - math.log10(lo)) / span * (ybot - ytop)

    body += _label(pad, 442, f"and the same question, from two machines up to the {big:,} "
                   "in the brief", 16, INK, anchor="start", weight=600)
    body += _label(pad, 461, "GB through the busiest machine, every step", 13.5, MUTED,
                   anchor="start")
    for d in range(5):
        v = lo * 10 ** d
        gy = py_of(v)
        body += (f'<line x1="{x0}" y1="{gy:.1f}" x2="{x1}" y2="{gy:.1f}" stroke="{HAIR}" '
                 f'stroke-width="1"/>')
        body += _label(x0 - 12, gy + 4, f"{v:,.0f}" if v < 1e6 else "1,000,000", 13.5, MUTED,
                       anchor="end", mono=True)
    counts = [2 ** k for k in range(1, steps + 1)]
    for m in counts[::2] + [counts[-1]]:
        body += _label(px_of(m), ybot + 22, f"{m:,}", 13.5, MUTED, mono=True)
    body += (f'<line x1="{px_of(n):.1f}" y1="{ytop}" x2="{px_of(n):.1f}" y2="{ybot}" '
             f'stroke="{HAIR_2}" stroke-width="1" stroke-dasharray="3 4"/>')
    body += _label(px_of(n), ytop + 16, "the four above", 13.5, MUTED)

    for load, colour, name in [(hub_load, MEMORY, "the hub"), (ring_load, COMMS, "the ring")]:
        pts = " ".join(f"{px_of(m):.1f},{py_of(load(m)):.1f}" for m in counts)
        body += (f'<polyline points="{pts}" fill="none" stroke="{colour}" stroke-width="2.4" '
                 f'stroke-linejoin="round"/>')
        for m in counts:
            body += (f'<circle cx="{px_of(m):.1f}" cy="{py_of(load(m)):.1f}" r="3.4" '
                     f'fill="{colour}" stroke="{GROUND}" stroke-width="1.4"/>')
        end = py_of(load(big))
        body += _label(x1 + 14, end - 4, name, 15.5, colour, anchor="start", weight=600)
        shown = (f"{load(big)/1000:,.0f} TB" if load(big) >= 1000 else f"{load(big):.0f} GB")
        body += _label(x1 + 14, end + 14, shown, 15.5, colour, anchor="start", weight=700, mono=True)

    return _svg(W, 686, "Four machines, two ways to wire them.",
                f"Every step, all four have to end up holding the same {buf:.0f} GB. "
                "Each circle is one machine.",
                "What decides a design is the volume through its busiest machine, and only one "
                "of these two stops growing.", body)


# ==================================================================== part 4
def what_fits():
    """Part 4, cell 4.2. Three models, two things you might do with them, one machine.

    Merges the model comparison and the storing-against-training figure, which were
    both answering the same question and both quoting the brief's 1,120 GB. The
    eight billion row is the lesson: it fits one way and not the other.
    """
    W, pad = 980, 34
    label_w, gap = 266, 14
    col_w = (W - 2 * pad - label_w - 2 * gap) / 2
    cols = [(pad + label_w + gap, "just to hold it", b.WEIGHT_BYTES, "the weights, and nothing else"),
            (pad + label_w + 2 * gap + col_w, "training it", b.BYTES_PER_PARAMETER,
             "all five arrays, every step")]
    models = [(b.PARAMETERS, "the model in the brief"), (8e9, "an 8 billion model"),
              (1e9, "a 1 billion model")]

    body = ""
    for cx, name, per, note in cols:
        body += _label(cx + col_w / 2, 108, name, 16, INK, weight=600)
        body += _label(cx + col_w / 2, 127, f"{per} bytes a parameter", 13.5, MUTED, mono=True)
        body += _label(cx + col_w / 2, 145, note, 13.5, MUTED)

    top = 162
    for r, (n, name) in enumerate(models):
        ry = top + r * 78
        if r == 0:                                   # the one the plan depends on
            body += _box(pad - 8, ry - 8, W - 2 * pad + 16, 70, fill=PANEL, stroke=PANEL, r=9)
        body += _label(pad + label_w, ry + 26, name, 16, INK, anchor="end", weight=600)
        if r == 0:
            body += _label(pad + label_w, ry + 45, f"{n/1e9:.0f} billion parameters", 13.5, MUTED,
                           anchor="end")
        for cx, _, per, _ in cols:
            need = n * per / b.GB
            fits = need <= b.MEMORY_GB
            body += _box(cx, ry, col_w, 54, fill=SOFT[MEMORY] if fits else GROUND,
                         stroke=MEMORY, r=7, width=1.4, dash=None if fits else "5 4")
            body += _label(cx + 20, ry + 33, f"{need:,.0f} GB", 19, MEMORY, anchor="start",
                           weight=600, mono=True)
            body += _label(cx + col_w - 20, ry + 33,
                           "fits" if fits else f"{need/b.MEMORY_GB:.1f} times too big",
                           15.5, INK_2 if fits else MEMORY, anchor="end",
                           weight=None if fits else 600)

    y = top + len(models) * 78 + 6
    body += (f'<line x1="{pad}" y1="{y}" x2="{W-pad}" y2="{y}" stroke="{HAIR_2}" '
             f'stroke-width="1"/>')
    body += _label(pad + label_w, y + 30, "the largest model that fits", 15.5, INK, anchor="end",
                   weight=600)
    body += _label(pad + label_w, y + 48, f"in one machine's {b.MEMORY_GB} GB", 13.5, MUTED,
                   anchor="end")
    for cx, _, per, _ in cols:
        wall = b.MEMORY_GB * b.GB / per
        body += _label(cx + col_w / 2, y + 38, f"{wall/1e9:.1f} billion", 23, MEMORY,
                       weight=700, mono=True)

    return _svg(W, y + 62, "The wall is not a property of the machine.",
                f"One machine holds {b.MEMORY_GB} GB. What fits in it depends on the model, and on "
                "which of these two things you are doing with it.",
                "Storing a model and training it are different questions, and only one of them is "
                "what this notebook is pricing.", body)


# ==================================================================== part 5
def two_ways_to_slice(machines=4):
    """Cell 55. The shapes are the explanation, so the shapes get drawn."""
    W, pad = 980, 34
    tw, th, gap, gutter = 330, 26, 40, 100
    row_gap = 12
    panels = [
        (pad + gutter, "cut so each machine makes part of the layer's output", MEMORY, False),
        (pad + gutter + tw + gap + gutter, "cut so each machine uses part of the input", COMMS, True),
    ]
    body = _hatch("hatch_slice", colour="#cfd8e6", width=1.4, gap=6)
    in_y, rows_y = 130, 202
    bottom = rows_y + machines * (th + row_gap) + 76

    def track(x, y, fill=GROUND, stroke=HAIR_2, width=1.2):
        return _box(x, y, tw, th, fill=fill, stroke=stroke, r=5, width=width)

    for x0, title, colour, summed in panels:
        soft = SOFT[colour]
        body += _label(x0 - gutter, 98, title, 16.5, INK, anchor="start", weight=600)

        # what goes in
        body += _label(x0 - gutter, in_y + 18, "what goes in", 13.5, MUTED, anchor="start")
        if summed:
            for m in range(machines):
                qx = x0 + m * tw / machines
                body += _box(qx + 1, in_y, tw / machines - 2, th, fill=soft, stroke=colour,
                             r=4, width=1.2)
                body += _label(qx + tw / machines / 2, in_y + 18, f"{m+1}", 13.5, colour, weight=600)
        else:
            body += track(x0, in_y, fill=soft, stroke=colour)
            body += _label(x0 + tw / 2, in_y + 18, "all of it, to every machine", 13.5, colour)

        # what each machine gets back
        body += _label(x0 - gutter, rows_y - 14, "what each machine hands back", 13.5, MUTED, anchor="start")
        for m in range(machines):
            y = rows_y + m * (th + row_gap)
            body += track(x0, y, fill=GROUND, stroke=HAIR)
            if summed:
                body += _box(x0, y, tw, th, fill="url(#hatch_slice)", stroke=colour, r=5, width=1.4)
            else:
                qx = x0 + m * tw / machines
                body += _box(qx + 1, y, tw / machines - 2, th, fill=soft, stroke=colour,
                             r=4, width=1.4)
            body += _label(x0 - 12, y + 18, f"machine {m+1}", 13.5, MUTED, anchor="end")
            if summed and m < machines - 1:
                body += _label(x0 + tw + 16, y + th / 2 + 12, "+", 19, colour, weight=600)

        last = rows_y + (machines - 1) * (th + row_gap) + th
        body += (f'<line x1="{x0}" y1="{last+16}" x2="{x0+tw}" y2="{last+16}" '
                 f'stroke="{HAIR_2}" stroke-width="1.4"/>')
        body += track(x0, last + 26, fill=colour, stroke=colour, width=1.6)
        body += _label(x0 + tw / 2, last + 44, "the layer's output", 15.5, GROUND, weight=600)
        note = ("four full-width results, so they must be added up"
                if summed else "four different pieces, so they simply line up")
        body += _label(x0 + tw / 2, last + 70, note, 15.5, colour, weight=600)

    divider = pad + gutter + tw + gap / 2
    body += (f'<line x1="{divider}" y1="92" x2="{divider}" y2="{bottom-14}" '
             f'stroke="{HAIR}" stroke-width="1"/>')
    return _svg(W, bottom, "Two ways to cut one layer, and the shapes tell you which is free.",
                "Both spread the same arithmetic over four machines. Look at what each one hands back.",
                f"Only the second needs a round of agreement, and it needs two going forward and "
                f"two coming back, in every one of the {b.LAYERS} layers, every step.", body)


def the_schedule(stages=4, micro=(1, 16), animate=True):
    """Cell 59. The animated one. Motion is the explanation here.

    The blocks are drawn complete. The animation only fades them in, so the
    figure still reads with motion disabled, in a static export, or under
    prefers-reduced-motion.
    """
    W, pad = 980, 34
    row_h, gap = 32, 9
    body = _hatch("hatch_sched")
    css = ["@keyframes sched_wash{0%{opacity:0}22%{opacity:.55}100%{opacity:0}}",
           "@media (prefers-reduced-motion:reduce){[id^=sched_]{animation:none}}"]
    y = 106
    for panel, m in enumerate(micro):
        slots = m + stages - 1
        idle = (stages - 1) / slots
        span = W - 2 * pad - 118
        sw = span / slots
        body += _label(pad, y - 14, f"{m} micro-batch{'es' if m > 1 else ''} in flight, {slots} "
                       f"slots, and {stages - 1} of them are a wait", 16, INK, anchor="start",
                       weight=600)
        for stage in range(stages):
            ry = y + stage * (row_h + gap)
            body += _label(pad, ry + 21, f"machine {stage+1}", 13.5, MUTED, anchor="start")
            body += _box(pad + 92, ry, span, row_h, fill="url(#hatch_sched)", stroke=HAIR,
                         r=5, width=1)
            for k in range(m):
                bx = pad + 92 + (stage + k) * sw
                body += (f'<rect x="{bx:.1f}" y="{ry}" width="{sw-2:.1f}" height="{row_h}" '
                         f'rx="4" fill="{SOFT[COMPUTE]}" stroke="{COMPUTE}" stroke-width="1.2"/>')
                if animate:
                    uid = f"sched_{panel}_{stage}_{k}"
                    delay = (stage + k) * 0.13
                    css.append(f"#{uid}{{opacity:0;animation:sched_wash 2.6s {delay:.2f}s infinite}}")
                    body += (f'<rect id="{uid}" x="{bx:.1f}" y="{ry}" width="{sw-2:.1f}" '
                             f'height="{row_h}" rx="4" fill="{COMPUTE}"/>')
        y += stages * (row_h + gap) + 12
        body += _box(W - pad - 214, y, 214, 32, fill=BAND, stroke=HAIR, r=8)
        body += _label(W - pad - 107, y + 21, f"{idle*100:.1f}% of the machine waiting",
                       15.5, INK, weight=600)
        y += 76
    style_block = f"<style>{''.join(css)}</style>" if animate else ""
    return _svg(W, y - 44, "Feed it more micro-batches and the waiting shrinks.",
                f"{stages} machines, each holding {b.LAYERS//stages} consecutive layers and handing "
                f"its work to the next. Hatching is a machine with nothing to do.",
                "It never reaches zero, and the number of micro-batches is capped by the batch "
                "you wanted.",
                style_block + body)


def what_stays_resident(copies=None, cut=32):
    """Cell 64. Named by what they stop storing, then by the name you will hear."""
    copies = copies or b.REFERENCE_DP
    levels = [
        ("nothing shared", "every machine keeps all of it", "1x"),
        ("share what the optimiser remembers", "the third of the bill nobody reads twice", "1x"),
        ("also share the gradients", "each machine keeps the slice it will apply", "1x"),
        ("also share the weights themselves", "gathered when needed, dropped again", "1.5x"),
    ]
    per_p = [_resident(i, copies) for i in range(4)]
    gb = [p * b.PARAMETERS / cut / b.GB for p in per_p]

    fig, ax = plt.subplots(figsize=(FIG_W, 4.8))
    fig.subplots_adjust(top=0.77, bottom=0.19, left=0.335, right=0.94)
    ypos = list(range(4))[::-1]
    ax.barh(ypos, gb, height=0.55, color=SOFT[MEMORY], edgecolor=MEMORY, linewidth=1.6, zorder=3)
    for i, y in enumerate(ypos):
        _mono(ax.text(gb[i] + 0.7, y, f"{gb[i]:.2f} GB", va="center", color=MEMORY,
                      fontsize=11.5, fontweight="bold"))
        ax.text(47, y, levels[i][2], va="center", ha="center", color=INK_2, fontsize=11.5,
                fontfamily="monospace")
        ax.text(-0.4, y - 0.33, levels[i][1], va="center", ha="right", color=MUTED, fontsize=10)
    ax.text(47, 3.62, "data moved each step", va="center", ha="center", color=MUTED, fontsize=10)
    ax.text(0, 3.62, "memory held on one machine", va="center", ha="left", color=MUTED, fontsize=10)
    ax.set_yticks(ypos)
    ax.set_yticklabels([l[0] for l in levels], fontsize=11.5)
    ax.set_xlim(0, 52)
    ax.set_xticks([])
    ax.spines["bottom"].set_visible(False)
    _tidy(ax)
    _title(fig, "Same arithmetic, same data, same result. Only what is resident changed.",
           f"The model is already cut {cut} ways, and what remains is shared across the {copies} copies.")
    _takeaway(fig, "The last row saves the most memory and moves the most data. Nothing here is "
              "free, and this one is paid for in bandwidth.")
    return _done(fig)


def _resident(level, copies):
    """What a machine still holds per parameter. Mirrors the function you write."""
    return [16, 4, 2, 0][level] + [0, 12, 14, 16][level] / copies


# ==================================================================== part 6
def _config_columns(fig, results, labels, subtitles, notes=None):
    """Name, configuration, outcome, then the six gates that explain it."""
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    n = len(results)
    left, gap = (0.355, 0.03) if n < 3 else (0.300, 0.022)
    colw = (0.978 - left - (n - 1) * gap) / n
    notes = notes or [""] * n
    rows = rc.GATE_NAMES
    top, rh = 0.600, 0.066
    for c, (res, label, sub, note) in enumerate(zip(results, labels, subtitles, notes)):
        x = left + c * (colw + gap)
        ax.add_patch(FancyBboxPatch((x, top - len(rows) * rh - 0.012), colw, len(rows) * rh + 0.268,
                                    boxstyle="round,pad=0.005,rounding_size=0.016",
                                    facecolor=PANEL if c else GROUND, edgecolor=HAIR, lw=1.1))
        ax.text(x + colw / 2, top + 0.205, label, ha="center", fontsize=16, fontweight="bold",
                color=INK, family="serif")
        ax.text(x + colw / 2, top + 0.152, sub, ha="center", fontsize=11.5, color=MUTED,
                family="monospace")
        if note:
            ax.text(x + colw / 2, top + 0.118, note, ha="center", fontsize=10, color=MUTED)
        ax.text(x + colw / 2, top + 0.080, f"{res['readouts']['days']:.1f} days", ha="center",
                fontsize=21.5, fontweight="bold", color=INK, family="monospace")
        for r, gate in enumerate(rows):
            y = top - r * rh
            ok = res["gates"][gate]
            if not ok:
                ax.add_patch(FancyBboxPatch((x + 0.012, y - 0.027), colw - 0.024, 0.054,
                                            boxstyle="round,pad=0.002,rounding_size=0.010",
                                            facecolor=GROUND, edgecolor=INK, lw=1.5, zorder=4))
            ax.text(x + colw / 2, y, res["short"][gate], ha="center", va="center",
                    fontsize=11.5, family="monospace", zorder=5,
                    color=INK if not ok else INK_2, fontweight="bold" if not ok else "normal")
            if not ok:
                ax.text(x + colw - 0.028, y, "✗", ha="center", va="center", fontsize=11.5,
                        color=INK, zorder=5)
    ax.text(left - 0.020, top + 0.152, "across x along x copies", ha="right", va="center",
            fontsize=10, color=MUTED, family="monospace")
    for r, gate in enumerate(rows):
        ax.text(left - 0.020, top - r * rh, gate, ha="right", va="center", fontsize=11.5, color=INK)
    return ax


def _needs(namespace, *functions):
    """The named functions that are not written yet, in the order asked for.

    Parts 6 and 7 all draw from scored configurations, so any of them can be
    reached before the arithmetic behind them exists. The builder already says
    "not yet" in that case; these say which function is missing rather than
    raising out of a format string.
    """
    return [f for f in functions if not callable((namespace or {}).get(f))]


def _not_yet(missing, what):
    """A card in place of a figure, naming what still has to be written."""
    W, pad = 980, 34
    body = _box(pad, 92, W - 2 * pad, 74, fill=PANEL, stroke=HAIR_2, dash="6 5")
    body += _label(W / 2, 124, "Not yet: " + ", ".join(missing), 16.5, INK, weight=600, mono=True)
    body += _label(W / 2, 148, f"Write {'them' if len(missing) > 1 else 'it'} in the cell above, "
                   f"run it, then run this one again.", 15.5, INK_2)
    return _svg(W, 178, what, "This card is drawn from a configuration scored with your own "
                "functions, and one of them is still blank.", None, body)


def three_configurations(namespace):
    """Cell 6.2. The whole decision in one table: two that differ by one dial,
    and one that solves memory outright and loses on everything else.

    Merges the A-against-B table and the it-fits-and-fails cards. Both were
    readouts of a configuration, and the third column is worth more beside the
    other two than alone: it uses a sixty fourth of the memory and still misses
    the date by three weeks.
    """
    missing = _needs(namespace, "bytes_sent_per_machine", "idle_share", "model_state_gb",
                     "resident_bytes_per_parameter", "shape_divides",
                     "hides_behind_the_arithmetic")
    if missing:
        return _not_yet(missing, "The configurations, once there is something to score.")
    a = rc.evaluate(8, 4, b.REFERENCE_DP, namespace)
    c = rc.evaluate(4, 8, b.REFERENCE_DP, namespace)
    d = rc.evaluate(1, 1, b.MACHINES, namespace, sharing_level=3)
    fig = plt.figure(figsize=(FIG_W, 6.8))
    _config_columns(fig, [a, c, d], ["A", "B", "C"],
                    [f"8 x 4 x {b.REFERENCE_DP}", f"4 x 8 x {b.REFERENCE_DP}",
                     f"1 x 1 x {b.MACHINES}"],
                    ["", "", "every copy shared out"])
    gap = c["readouts"]["days"] - a["readouts"]["days"]
    _title(fig, "One passes, one misses a target, and one is not a plan at all.",
           "A and B cut the model the same 32 ways and hold the same 35 GB. C cuts nothing and "
           "shares everything instead.")
    _takeaway(fig, f"A finishes {gap:.1f} days sooner than B, and C has the memory problem "
                   f"comprehensively solved while failing three gates that are not about memory.")
    return _done(fig)


def does_the_plan_notice(namespace, speed=0.8, factor=2):
    """Part 7, cell 7.1. Two ordinary changes, and the plan answers only one.

    Merges the slow-machine bars and the longer-context table. They were the same
    move made twice, and side by side the contrast is the finding: the same
    arithmetic is right about one of these and silent about the other.
    """
    missing = _needs(namespace, "bytes_sent_per_machine", "idle_share")
    if missing:
        return _not_yet(missing, "Two changes, and whether the plan notices them.")
    W, pad = 980, 34
    pw, gap = (W - 2 * pad - 22) / 2, 22
    lx, rx = pad, pad + pw + gap

    slow = []
    for name, tp, pp in [("A", 8, 4), ("B", 4, 8)]:
        r = rc.evaluate(tp, pp, b.REFERENCE_DP, namespace)["readouts"]
        step = r["arithmetic seconds"] / speed / (1 - r["idle share"]) + r["agreeing seconds"]
        late = b.STEPS * step / b.SECONDS_A_DAY
        slow.append((f"configuration {name}", f"{r['days']:.1f} d", f"{late:.1f} d",
                     "past the date" if late > b.DEADLINE_DAYS else "still early",
                     late > b.DEADLINE_DAYS))
    before = rc.evaluate(8, 4, b.REFERENCE_DP, namespace)["readouts"]
    step_s = f"{before['step seconds']:.2f} s"
    wide = [("the context", f"{b.CONTEXT:,}", f"{b.CONTEXT*factor:,}", "", False),
            ("copies of the model", f"{b.REFERENCE_DP}",
             f"{b.MACHINES // (8 * 4 * factor)}", "", False),
            ("a step costs", step_s, step_s, "unchanged", True)]

    body = ""
    panels = [(lx, COMMS, "the plan notices", f"one machine runs at {speed:.0%} of the others",
               slow, "a synchronous run moves at the pace of its slowest member, and the "
               "links are unaffected, so only the arithmetic is charged", 1.6),
              (rx, MEMORY, "the plan does not notice",
               f"the board asks for {factor} times the context", wide,
               "the machines had to come from somewhere, and our model charges a flat rate for "
               "every token, so the work that grows with the context is not in it anywhere", 2.2)]
    for px, colour, head, sub, rows, note, edge in panels:
        body += _box(px, 92, pw, 286, fill=GROUND, stroke=colour if edge > 2 else HAIR_2,
                     r=10, width=edge)
        body += _label(px + 18, 118, head, 16.5, colour, anchor="start", weight=600)
        body += _label(px + 18, 138, sub, 13.5, MUTED, anchor="start")
        for i, (name, was, now, verdict, flag) in enumerate(rows):
            ry = 176 + i * 40
            body += _label(px + 18, ry, name, 15.5, INK_2, anchor="start")
            body += _label(px + 200, ry, was, 15.5, MUTED, anchor="end", mono=True)
            body += _label(px + 218, ry, "\u2192", 15.5, HAIR_2)
            body += _label(px + 236, ry, now, 17, colour, anchor="start", weight=600, mono=True)
            if verdict:
                body += _label(px + pw - 18, ry, verdict, 13.5,
                               colour if flag else MUTED, anchor="end",
                               weight=600 if flag else None)
        body += (f'<line x1="{px+18}" y1="{302}" x2="{px+pw-18}" y2="{302}" '
                 f'stroke="{HAIR}" stroke-width="1"/>')
        for j, line in enumerate(_wrap(note, 58)):
            body += _label(px + 18, 326 + j * 20, line, 13.5, MUTED, anchor="start")

    return _svg(W, 396, "Change one thing, and the plan answers one of these and not the other.",
                "Both are ordinary things that happen to a run that takes weeks. Only one of them "
                "moves a number the plan reports.",
                "The same arithmetic is right about the first and silent about the second, and "
                "nothing in its output tells you which you are looking at.", body)


def the_run_stops():
    """Cell 81. The one place the notebook uses an assumed failure rate."""
    import numpy as np
    write = b.CHECKPOINT_WRITE_SECONDS
    between = b.HOURS_BETWEEN_STOPS * 3600
    intervals = np.linspace(120, 7200, 400)
    overhead = write / intervals + intervals / (2 * between)
    best, best_overhead = rc.checkpoint_optimum()

    fig, axes = plt.subplots(1, 2, figsize=(FIG_W, 4.5), gridspec_kw={"width_ratios": [2, 1]})
    fig.subplots_adjust(top=0.76, bottom=0.26, left=0.075, right=0.975, wspace=0.28)
    ax = axes[0]
    writing = 100 * write / intervals
    losing = 100 * intervals / (2 * between)
    minutes = intervals / 60
    ax.fill_between(minutes, 0, writing, color=SOFT[MEMORY], edgecolor=MEMORY, lw=1.4, zorder=2)
    ax.fill_between(minutes, writing, writing + losing, facecolor="#faf9f6", edgecolor="#d8d3c9",
                    lw=1.2, hatch="/", zorder=2)
    ax.plot(minutes, 100 * overhead, color=COMMS, lw=2.6, zorder=4)
    ax.plot([best / 60], [100 * best_overhead], "o", ms=10, color=COMMS,
            markeredgecolor=GROUND, markeredgewidth=2.2, zorder=5)
    _mono(ax.annotate(f"{best/60:.0f} min, {100*best_overhead:.2f}%", (best / 60, 100 * best_overhead),
                      textcoords="offset points", xytext=(16, -26), ha="left", color=COMMS,
                      fontsize=11.5, fontweight="bold", zorder=6))
    ax.text(92, 0.62, "what writing costs", color=MEMORY, fontsize=11.5, ha="center", zorder=6)
    ax.text(92, 2.15, "what you expect to lose", color=INK_2, fontsize=11.5, ha="center", zorder=6)
    ax.text(58, 5.15, "the two together", color=COMMS, fontsize=11.5, fontweight="bold", zorder=6)
    ax.set_xlabel("minutes between checkpoints")
    ax.set_ylabel("share of the run spent on this")
    ax.set_ylim(0, 6)
    ax.set_xlim(2, 120)
    ax.set_yticks([0, 2, 4, 6])
    ax.set_yticklabels(["0%", "2%", "4%", "6%"])
    _tidy(ax)

    ax2 = axes[1]
    hours = [3, 6, 12, 24, 48]
    bests = [rc.checkpoint_optimum(hours_between_stops=h)[0] / 60 for h in hours]
    ax2.plot(hours, bests, color=INK, marker="o", ms=7, markeredgecolor=GROUND, markeredgewidth=1.6)
    ax2.plot([b.HOURS_BETWEEN_STOPS], [best / 60], "o", ms=10, color=COMMS,
             markeredgecolor=GROUND, markeredgewidth=2, zorder=4)
    ax2.set_xscale("log", base=2)
    ax2.set_xticks(hours)
    ax2.set_xticklabels([str(h) for h in hours])
    ax2.set_xlabel("hours between stops, assumed")
    ax2.set_ylabel("cheapest interval, minutes")
    _tidy(ax2)
    _title(fig, "The cheapest interval is where the two costs are equal.",
           f"Writing takes {write:.1f} seconds. A stop is assumed every {b.HOURS_BETWEEN_STOPS} hours, "
           f"and that assumption is not measured anywhere in this notebook.")
    _takeaway(fig, "The panel on the right is how much of the answer that one assumption is carrying.")
    return _done(fig)


# ============================================================ widget scaffold
_UID = [0]


def _uid(prefix="w"):
    _UID[0] += 1
    return f"{prefix}{_UID[0]}"


_WIDGET_CSS = """
#UID{{font-family:{sans};color:{ink};max-width:980px;font-size:16px;line-height:1.5}}
#UID *{{box-sizing:border-box}}
#UID .card{{border:1px solid {hair};border-radius:12px;background:{ground};padding:20px 22px}}
#UID .head{{font-family:{serif};font-size:20px;font-weight:700;margin:0 0 4px}}
#UID .sub{{color:{ink2};font-size:14.5px;margin:0 0 16px}}
#UID .mono{{font-family:{mono};font-variant-numeric:tabular-nums}}
#UID .muted{{color:{muted}}}
#UID .band{{background:{band};border:1px solid {hair};border-radius:9px;padding:12px 16px;
  font-size:14.5px;font-weight:700;text-align:center;margin-top:16px}}
#UID button{{font:inherit;cursor:pointer}}
"""


def _widget(uid, body, css="", script=""):
    style = (_WIDGET_CSS + css).replace("UID", uid).format(
        sans=WEB_SANS, serif=WEB_SERIF, mono=WEB_MONO, ink=INK, ink2=INK_2, muted=MUTED,
        hair=HAIR, hair2=HAIR_2, ground=GROUND, panel=PANEL, band=BAND,
        compute=COMPUTE, comms=COMMS, memory=MEMORY, idle=IDLE,
        soft_compute=SOFT[COMPUTE], soft_comms=SOFT[COMMS], soft_memory=SOFT[MEMORY])
    js = f"<script>(function(){{const R=document.getElementById('{uid}');{script}}})();</script>" if script else ""
    return HTML(f"<style>{style}</style><div id='{uid}'>{body}</div>{js}")


def quick_check(key):
    """One multiple choice question. The explanation teaches; it does not only mark.

    Deliberately no JavaScript. A notebook's saved outputs never get their
    scripts run, so a check built on click handlers is dead for anyone reading
    the solutions rather than executing them. Hidden radios and sibling
    selectors do the same job out of the box, everywhere.

    It also stays live after the first answer. Locking it bought nothing, since
    the explanation appears on that first click regardless, and a panel that
    stops responding reads as broken rather than as answered.
    """
    import quizzes
    question, options, answer, because = quizzes.QUESTIONS[key]
    uid = _uid("qc")
    picks = "".join(f"<input type='radio' name='{uid}pick' id='{uid}_{i}' class='pick'>"
                    for i in range(len(options)))
    opts = "".join(
        f"<label for='{uid}_{i}' class='opt'><span class='dot'></span>"
        f"<span class='txt'>{o}</span></label>" for i, o in enumerate(options))
    body = (f"<div class='card'>{picks}<div class='head'>{question}</div>"
            f"<div class='opts'>{opts}</div>"
            f"<div class='why'><div class='verdict'><span class='yes'>Yes.</span>"
            f"<span class='no'>Not this one.</span></div>"
            f"{because if because.startswith('<') else '<p>' + because + '</p>'}</div></div>")

    # one rule per option, because which one was picked has to be visible in CSS alone
    per_option = ""
    for i in range(len(options)):
        if i == answer:
            per_option += f"#UID #UID_{i}:checked ~ .why .yes{{{{display:inline}}}}"
        else:
            per_option += (f"#UID #UID_{i}:checked ~ .why .no{{{{display:inline}}}}"
                           f"#UID #UID_{i}:checked ~ .opts .opt:nth-of-type({i+1})"
                           f"{{{{border-color:{{ink}};border-width:2px}}}}"
                           f"#UID #UID_{i}:checked ~ .opts .opt:nth-of-type({i+1}) .dot"
                           f"{{{{border-color:{{ink}}}}}}")
    css = """
#UID .pick{{position:absolute;width:1px;height:1px;opacity:0;margin:0}}
#UID .opts{{display:flex;flex-direction:column;gap:8px;margin-top:14px}}
#UID .opt{{display:flex;gap:12px;align-items:flex-start;text-align:left;width:100%;
  background:{ground};border:1px solid {hair};border-radius:10px;padding:11px 14px;
  font-size:15.5px;color:{ink};transition:border-color .12s;cursor:pointer}}
#UID .opt:hover{{border-color:{hair2}}}
#UID .pick:focus-visible + .opt,#UID .pick:focus-visible ~ .opts .opt{{outline:none}}
#UID .dot{{width:15px;height:15px;border:1.6px solid {hair2};border-radius:50%;flex:0 0 auto;
  margin-top:3px}}
#UID .pick:checked ~ .opts .opt:nth-of-type(ANS){{border-color:{compute};background:{soft_compute}}}
#UID .pick:checked ~ .opts .opt:nth-of-type(ANS) .dot{{border-color:{compute};
  background:{compute};box-shadow:inset 0 0 0 3px {soft_compute}}}
#UID .why{{display:none;margin-top:16px;border-top:1px solid {hair};padding-top:14px;
  font-size:15.5px;color:{ink2}}}
#UID .pick:checked ~ .why{{display:block}}
#UID .why p{{margin:6px 0 0}}
#UID .worked{{margin:14px 0 4px;border:1px solid {hair};border-radius:9px;overflow:hidden}}
#UID .wrow{{display:grid;grid-template-columns:1fr auto 132px;gap:14px;align-items:baseline;
  padding:9px 14px;border-bottom:1px solid {hair};font-size:14.5px}}
#UID .wrow:last-child{{border-bottom:none}}
#UID .wrow:first-child{{background:{panel}}}
#UID .wrow b{{font-family:{mono};font-size:14.5px;color:{ink};white-space:nowrap}}
#UID .wrow i{{font-style:normal;color:{muted};font-size:13.5px;text-align:right}}
#UID .verdict{{font-weight:700;color:{ink}}}
#UID .verdict .yes,#UID .verdict .no{{display:none}}
""".replace("ANS", str(answer + 1)) + per_option
    return _widget(uid, body, css)


# =================================================================== explorers
_SLIDER_CSS = """
#UID .ctl{{display:flex;flex-direction:column;gap:14px;margin:4px 0 18px}}
#UID .ctl label{{display:grid;grid-template-columns:200px 1fr 118px;gap:14px;align-items:center;
  font-size:14.5px;color:{ink2}}}
#UID input[type=range]{{width:100%;accent-color:{comms};height:22px}}
#UID .now{{font-family:{mono};font-size:15px;font-weight:600;color:{ink};text-align:right}}
#UID .out{{display:flex;gap:10px;flex-wrap:wrap;margin-top:4px}}
#UID .stat{{flex:1 1 150px;border:1px solid {hair};border-radius:10px;padding:12px 14px;
  background:{panel}}}
#UID .stat .k{{font-size:13px;color:{muted};min-height:30px;line-height:1.35}}
#UID .stat .v{{font-family:{mono};font-size:22px;font-weight:600;margin-top:4px}}
#UID .bar{{display:flex;height:34px;border-radius:8px;overflow:hidden;border:1px solid {hair};
  margin-top:16px;background:{panel}}}
#UID .bar div{{display:flex;align-items:center;justify-content:center;font-size:13.5px;
  font-weight:600;color:{ground};white-space:nowrap;transition:width .12s}}
"""


def sustained_share_lab():
    """Cell 15. The rate you buy against the rate you get."""
    uid = _uid("share")
    # the same arithmetic the script does, so the widget reads correctly before it runs
    _r = b.PEAK_FLOPS * b.SUSTAINED
    _rate = (rc.whole_job_flop() / _r / 86400 / 365.25,
             rc.whole_job_flop() / _r / b.MACHINES / 86400)
    body = f"""<div class='card'>
      <div class='head'>What the machine is sold at, and what the run keeps</div>
      <div class='sub'>Everything else in the brief is fixed. Move only this.</div>
      <div class='ctl'><label><span>share of the peak rate a real run sustains</span>
        <input type='range' min='5' max='100' step='5' value='{int(b.SUSTAINED*100)}'>
        <span class='now'>{b.SUSTAINED*100:.0f}%</span></label></div>
      <div class='out'>
        <div class='stat'><div class='k'>on one machine</div>
          <div class='v' data-k='years'>{_rate[0]:.0f} years</div></div>
        <div class='stat'><div class='k'>on all {b.MACHINES:,}, if splitting were perfect</div>
          <div class='v' data-k='days'>{_rate[1]:.1f} days</div></div>
        <div class='stat'><div class='k'>arithmetic you paid for and never see</div>
          <div class='v' data-k='lost'>{100-b.SUSTAINED*100:.0f}%</div></div>
      </div>
      <div class='bar'><div class='got' style='width:{b.SUSTAINED*100:.0f}%'>what the run gets</div>
        <div class='gone' style='width:{100-b.SUSTAINED*100:.0f}%'>what the invoice covered</div></div>
      <div class='band' data-k='note'>The share is a property of the run, not of the machine. Only a
        benchmark tells you it.</div></div>"""
    css = _SLIDER_CSS + """
#UID .got{{background:{compute}}}
#UID .gone{{background:{idle};color:{ink}}}
"""
    script = f"""
const WHOLE={rc.whole_job_flop()}, PEAK={b.PEAK_FLOPS}, M={b.MACHINES}, DAY=86400;
const r=R.querySelector('input'), out=k=>R.querySelector(`[data-k=${{k}}]`);
function draw(){{
  const share=+r.value/100, rate=PEAK*share;
  const years=WHOLE/rate/DAY/365.25, days=WHOLE/rate/M/DAY;
  R.querySelector('.now').textContent=(share*100).toFixed(0)+'%';
  out('years').textContent=years.toFixed(0)+' years';
  out('days').textContent=days.toFixed(1)+' days';
  out('lost').textContent=(100-share*100).toFixed(0)+'%';
  R.querySelector('.got').style.width=(share*100)+'%';
  R.querySelector('.gone').style.width=(100-share*100)+'%';
  R.querySelector('.got').textContent=share>0.18?'what the run gets':'';
  R.querySelector('.gone').textContent=share<0.85?'what the invoice covered':'';
  out('note').textContent = share>=0.99
    ? 'Nobody reaches this. It is the number on the datasheet.'
    : 'The share is a property of the run, not of the machine. Only a benchmark tells you it.';
}}
r.addEventListener('input',draw); draw();
"""
    return _widget(uid, body, css, script)


def scaling_lab():
    """Cell 39. Machines, link speed, and how much of the agreeing hides."""
    uid = _uid("scale")
    # the same arithmetic the script does, at the dials' own starting positions
    _LINKS = [10, 25, 50, 100, 900]
    _m0, _link0, _hide0 = 2 ** 11, _LINKS[1], 0.0

    def _eff(m, link, hide):
        comp = rc.one_machine_step_seconds() / m
        comm = 2 * (m - 1) * (rc.whole_model_buffer_gb() / m) / link * (1 - hide)
        return comp / (comp + comm), comp, comm

    _d = _eff(_m0, _link0, _hide0)
    _best = max((p for p in range(8, b.MACHINES + 1, 8)
                 if _eff(p, _link0, _hide0)[0] >= b.EFFICIENCY_TARGET), default=8)
    _note = (f"At this link speed the target holds up to {_best:,} machines, of the "
             f"{b.MACHINES:,} you own." if _eff(8, _link0, _hide0)[0] >= b.EFFICIENCY_TARGET
             else "At this link speed the target is out of reach at any size.")
    body = f"""<div class='card'>
      <div class='head'>Where adding machines stops paying</div>
      <div class='sub'>Every machine still holds the whole model, so they all agree over
        {rc.whole_model_buffer_gb():.0f} GB every step.</div>
      <div class='ctl'>
        <label><span>machines</span>
          <input data-c='m' type='range' min='3' max='11' step='1' value='11'>
          <span class='now' data-n='m'>{_m0:,}</span></label>
        <label><span>what each link carries</span>
          <input data-c='b' type='range' min='0' max='4' step='1' value='1'>
          <span class='now' data-n='b'>{_link0} GB/s</span></label>
        <label><span>agreeing hidden behind the arithmetic</span>
          <input data-c='o' type='range' min='0' max='90' step='10' value='0'>
          <span class='now' data-n='o'>{_hide0*100:.0f}%</span></label>
      </div>
      <div class='out'>
        <div class='stat'><div class='k'>arithmetic, per machine per step</div>
          <div class='v' data-k='comp'>{_d[1]:.2f} s</div></div>
        <div class='stat'><div class='k'>agreeing, per machine per step</div>
          <div class='v' data-k='comm'>{_d[2]:.2f} s</div></div>
        <div class='stat'><div class='k'>share of the step on arithmetic</div>
          <div class='v' data-k='eff'>{_d[0]*100:.1f}%</div></div>
      </div>
      <div class='bar'><div class='c1' style='width:{_d[0]*100:.1f}%'>{'arithmetic' if _d[0] > 0.2 else ''}</div>
        <div class='c2' style='width:{100-_d[0]*100:.1f}%'>{'agreeing' if _d[0] < 0.8 else ''}</div></div>
      <div class='band' data-k='note'>{_note}</div></div>"""
    css = _SLIDER_CSS + """
#UID .c1{{background:{compute}}}
#UID .c2{{background:{comms}}}
"""
    script = f"""
const ONE={rc.one_machine_step_seconds()}, BUF={rc.whole_model_buffer_gb()},
      TARGET={b.EFFICIENCY_TARGET}, OWNED={b.MACHINES};
const LINKS=[10,25,50,100,900];
const get=c=>R.querySelector(`[data-c=${{c}}]`), out=k=>R.querySelector(`[data-k=${{k}}]`);
function eff(m,link,hide){{
  const comp=ONE/m, comm=2*(m-1)*(BUF/m)/link*(1-hide);
  return [comp/(comp+comm), comp, comm];
}}
function draw(){{
  const m=Math.pow(2,+get('m').value), link=LINKS[+get('b').value], hide=+get('o').value/100;
  const [e,comp,comm]=eff(m,link,hide);
  R.querySelector('[data-n=m]').textContent=m.toLocaleString();
  R.querySelector('[data-n=b]').textContent=link+' GB/s';
  R.querySelector('[data-n=o]').textContent=(hide*100).toFixed(0)+'%';
  out('comp').textContent=comp.toFixed(2)+' s';
  out('comm').textContent=comm.toFixed(2)+' s';
  out('eff').textContent=(e*100).toFixed(1)+'%';
  R.querySelector('.c1').style.width=(e*100)+'%';
  R.querySelector('.c2').style.width=(100-e*100)+'%';
  R.querySelector('.c1').textContent=e>0.2?'arithmetic':'';
  R.querySelector('.c2').textContent=e<0.8?'agreeing':'';
  let best=8; for(let p=8;p<=OWNED;p+=8){{ if(eff(p,link,hide)[0]>=TARGET) best=p; }}
  out('note').textContent = eff(8,link,hide)[0]<TARGET
    ? 'At this link speed the target is out of reach at any size.'
    : `At this link speed the target holds up to ${{best.toLocaleString()}} machines, of the ${{OWNED.toLocaleString()}} you own.`;
}}
R.querySelectorAll('input').forEach(i=>i.addEventListener('input',draw)); draw();
"""
    return _widget(uid, body, css, script)


# ==================================================================== builder
_TP_CHOICES = [1, 2, 3, 4, 6, 8, 16]
_PP_CHOICES = [1, 2, 4, 8, 16, 32]


_BUILDER_CSS = """
#UID .dials{{display:flex;gap:26px;flex-wrap:wrap;margin:6px 0 20px;align-items:flex-start}}
#UID .dname{{font-size:13.5px;color:{muted};margin-bottom:7px}}
#UID .opts{{display:flex;gap:5px}}
#UID .pill{{border:1px solid {hair};background:{ground};border-radius:8px;
  padding:7px 12px;font-family:{mono};font-size:15px;color:{ink2};cursor:pointer;
  display:inline-block;line-height:1.15}}
#UID .pill:hover{{border-color:{hair2}}}
#UID .pill.on{{background:{comms};border-color:{comms};color:{ground};font-weight:600}}
#UID .derived{{font-size:22px;font-weight:600;padding:4px 0}}
#UID table.gates{{width:100%;border-collapse:collapse;font-size:15px}}
#UID .gates td{{padding:9px 8px;border-bottom:1px solid {hair};vertical-align:middle}}
#UID .gates tr:last-child td{{border-bottom:none}}
#UID .gn{{width:22px;font-family:{mono};font-size:13px;color:{hair2}}}
#UID .gname{{width:230px}}
#UID .gmark{{width:88px;font-family:{mono};font-size:13.5px;font-weight:600}}
#UID .gwhy{{color:{ink2};font-size:14.5px}}
#UID tr.pass .gmark{{color:{compute}}}
#UID tr.fail .gmark{{color:{ink}}}
#UID tr.fail{{background:{panel}}}
#UID tr.wait .gmark,#UID tr.wait .gwhy{{color:{hair2}}}
#UID .verdict{{margin-top:16px;border-radius:9px;padding:14px 18px;font-size:16px;
  font-weight:700;border:1px solid {hair};background:{band}}}
#UID .verdict.good{{background:{soft_compute};border-color:{compute};color:{compute}}}
#UID .verdict small{{display:block;font-weight:400;color:{ink2};margin-top:4px;font-size:14px}}
"""


def builder(namespace=None):
    """Cells 6 and 70. Two dials, six gates, and every gate is a function you wrote.

    Scored with the functions defined in the notebook, not with a reference copy.
    A gate whose function is missing says so rather than inventing an answer.

    No JavaScript. Every combination of the two dials is scored here and written
    out, and hidden radios pick which one is on show. A panel driven by click
    handlers is dead in a notebook's saved outputs, and dials that do nothing
    when pressed read as broken rather than as static.
    """
    uid = _uid("build")
    ns = namespace or {}
    combos = [(tp, pp) for tp in _TP_CHOICES for pp in _PP_CHOICES]
    scored = {}
    for tp, pp in combos:
        if b.MACHINES % (tp * pp):
            scored[(tp, pp)] = None
            continue
        dp = b.MACHINES // (tp * pp)
        r = rc.evaluate(tp, pp, dp, ns)
        scored[(tp, pp)] = {"dp": dp,
                            "gates": [(g, r["gates"][g], r["why"][g]) for g in rc.GATE_NAMES],
                            "verdict": rc.verdict(r)}

    def panel(cfg):
        """The gate table and the verdict for one configuration."""
        rows = ""
        for i, name in enumerate(rc.GATE_NAMES):
            if not cfg:
                klass, mark, why = "wait", "not yet", ""
            else:
                ok, why = cfg["gates"][i][1], cfg["gates"][i][2]
                klass = "wait" if ok is None else "pass" if ok else "fail"
                mark = "not yet" if ok is None else "ok" if ok else "not met"
            rows += (f"<tr class='{klass}'><td class='gn'>{i+1}</td><td class='gname'>{name}</td>"
                     f"<td class='gmark'>{mark}</td><td class='gwhy'>{why}</td></tr>")
        if not cfg:
            v_class, v_text = "verdict", ("These two do not divide the cluster.<small>Pick a pair "
                                          f"whose product goes into {b.MACHINES:,}.</small>")
        else:
            waiting = [g[0] for g in cfg["gates"] if g[1] is None]
            failed = [g[0] for g in cfg["gates"] if g[1] is False]
            if waiting:
                note = "Still to write: " + "; ".join(waiting) + "."
            elif failed:
                note = "What stops it: " + "; ".join(failed) + "."
            else:
                note = ("Every gate passes, which means it is worth benchmarking. It does not "
                        "mean it is worth starting.")
            v_class = "verdict" + (" good" if cfg["verdict"] == "meets the brief" else "")
            v_text = cfg["verdict"][0].upper() + cfg["verdict"][1:] + f"<small>{note}</small>"
        return (f"<table class='gates'><tbody>{rows}</tbody></table>"
                f"<div class='{v_class}'>{v_text}</div>")

    head = 'Which configuration do we run?'
    sub = f'Two dials. The copies are whatever is left of the {b.MACHINES:,} machines.'

    # every radio comes before every label and panel, so a pair of :checked
    # inputs can select what is on show through sibling combinators alone
    picks = ""
    for key, choices, default in [("tp", _TP_CHOICES, 8), ("pp", _PP_CHOICES, 4)]:
        picks += "".join(
            f"<input type='radio' class='pick' name='{uid}{key}' id='{uid}{key}{c}'"
            f"{' checked' if c == default else ''}>" for c in choices)
    dials = ""
    for name, key, choices in [("across a layer", "tp", _TP_CHOICES),
                               ("along the depth", "pp", _PP_CHOICES)]:
        opts = "".join(f"<label class='pill' for='{uid}{key}{c}'>{c}</label>" for c in choices)
        dials += (f"<div class='dial'><div class='dname'>{name}</div>"
                  f"<div class='opts'>{opts}</div></div>")
    derived = "".join(
        f"<span class='derived mono d{tp}_{pp}'>"
        f"{f'{scored[(tp, pp)]["dp"]:,}' if scored[(tp, pp)] else 'does not divide'}</span>"
        for tp, pp in combos)
    panels = "".join(f"<div class='p p{tp}_{pp}'>{panel(scored[(tp, pp)])}</div>"
                     for tp, pp in combos)
    body = (f"<div class='card'>{picks}<div class='head'>{head}</div><div class='sub'>{sub}</div>"
            f"<div class='dials'>{dials}<div class='dial'>"
            f"<div class='dname'>copies, derived</div>{derived}</div></div>{panels}</div>")

    show = "".join(
        f"#UID #UID{'tp'}{tp}:checked ~ #UID{'pp'}{pp}:checked ~ .dials .d{tp}_{pp},"
        f"#UID #UID{'tp'}{tp}:checked ~ #UID{'pp'}{pp}:checked ~ .p{tp}_{pp}"
        f"{{{{display:block}}}}" for tp, pp in combos)
    on = "".join(
        f"#UID #UID{key}{c}:checked ~ .dials label[for='UID{key}{c}']"
        f"{{{{background:{{comms}};border-color:{{comms}};color:{{ground}};font-weight:600}}}}"
        for key, choices in [("tp", _TP_CHOICES), ("pp", _PP_CHOICES)] for c in choices)
    return _widget(uid, body, css=_BUILDER_CSS + """
#UID .pick{{position:absolute;width:1px;height:1px;opacity:0;margin:0}}
#UID .derived,#UID .p{{display:none}}
""" + show + on)

