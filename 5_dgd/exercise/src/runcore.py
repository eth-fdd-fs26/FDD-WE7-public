"""The machinery behind the figures and the builder.

Nothing here needs reading to work through the notebook. The functions you write
yourself are the ones the builder calls; this file only arranges them.
"""
import numpy as np

import brief as b

# ---------------------------------------------------------------- part 2 data
EIGHT = [3, -3, 1, 3, 6, 2, 5, 7]
EVEN_SPLIT = [(0, 2), (2, 4), (4, 6), (6, 8)]
RAGGED_SPLIT = [(0, 3), (3, 5), (5, 6), (6, 8)]


def slices(split):
    return [EIGHT[a:c] for a, c in split]


def local_means(split):
    return [sum(s) / len(s) for s in slices(split)]


def sizes(split):
    return [len(s) for s in slices(split)]


def whole_batch_mean():
    return sum(EIGHT) / len(EIGHT)


def toy_gradients(machines=4, width=12, seed=0):
    """One direction per machine from a small model, for the corroborating check."""
    rng = np.random.default_rng(seed)
    rows = rng.normal(0, 1, size=(machines * 5, width))
    counts = [7, 5, 3, 5][:machines]
    out, start = [], 0
    for n in counts:
        out.append(rows[start:start + n].mean(axis=0))
        start += n
    return out, counts, rows[:sum(counts)].mean(axis=0)


# ------------------------------------------------------------ derived numbers
def whole_job_flop():
    return 6 * b.PARAMETERS * b.TOKEN_BUDGET


def step_flop(batch_tokens=None):
    return 6 * b.PARAMETERS * (batch_tokens or b.GLOBAL_BATCH)


def years_on_one_machine(rate=None):
    return whole_job_flop() / (rate or b.RATE_FLOPS) / b.SECONDS_A_DAY / 365.25


def one_machine_step_seconds():
    return step_flop() / b.RATE_FLOPS


def whole_model_buffer_gb():
    return 2 * b.PARAMETERS / b.GB


def buffer_after_cut_gb(tp, pp):
    return 2 * b.PARAMETERS / (tp * pp) / b.GB


def global_batch(context, seq_per_micro, microbatches, copies):
    return context * seq_per_micro * microbatches * copies


def microbatches_for(copies, context=None, seq_per_micro=None, target=None):
    """Hold the batch contract when the copy count moves."""
    context = context or b.CONTEXT
    seq_per_micro = seq_per_micro or b.SEQ_PER_MICRO
    target = target or b.GLOBAL_BATCH
    return max(1, round(target / (context * seq_per_micro * copies)))


def checkpoint_optimum(write_seconds=None, hours_between_stops=None):
    """Writing costs time every interval; stopping costs the work since the last one."""
    write = write_seconds or b.CHECKPOINT_WRITE_SECONDS
    between = (hours_between_stops or b.HOURS_BETWEEN_STOPS) * 3600
    interval = (2 * write * between) ** 0.5
    overhead = write / interval + interval / (2 * between)
    return interval, overhead


# ------------------------------------------------------------------ the gates
GATE_NAMES = [
    "the product is valid",
    "the cut is legal",
    "the model state fits",
    "the batch contract holds",
    "efficiency meets the target",
    "it meets the date",
]

_NEEDED = {
    "the cut is legal": ("shape_divides", "hides_behind_the_arithmetic"),
    "the model state fits": ("model_state_gb", "resident_bytes_per_parameter"),
    "efficiency meets the target": ("bytes_sent_per_machine", "idle_share"),
    "it meets the date": ("bytes_sent_per_machine", "idle_share"),
}


def _missing(namespace, gate):
    return [n for n in _NEEDED.get(gate, ()) if not callable(namespace.get(n))]


def evaluate(tp, pp, dp, namespace, sharing_level=0):
    """Score one configuration using the functions defined in `namespace`.

    A gate whose function has not been written yet reports that, rather than
    quietly falling back to an answer the student has not produced.
    """
    micro = microbatches_for(dp)
    cut = tp * pp
    result = {
        "tp": tp, "pp": pp, "dp": dp, "microbatches": micro,
        "gates": {}, "why": {}, "short": {}, "readouts": {},
    }

    def gate(name, ok, why="", short=""):
        result["gates"][name] = ok
        result["why"][name] = why
        result["short"][name] = short or why

    gate("the product is valid", tp * pp * dp == b.MACHINES,
         f"{tp} x {pp} x {dp} = {tp*pp*dp:,} of {b.MACHINES:,} machines",
         f"{tp*pp*dp:,} machines")

    miss = _missing(namespace, "the cut is legal")
    if miss:
        gate("the cut is legal", None, f"waiting for {' and '.join(miss)}")
        ratio = None
    else:
        ratio = namespace["hides_behind_the_arithmetic"](
            tp, b.HIDDEN, b.PEAK_FLOPS, b.LINK_IN_A_BOX)
        divides = namespace["shape_divides"](tp, b.HIDDEN, b.HEADS, b.KEY_VALUE_HEADS)
        in_a_box = tp <= b.MACHINES_A_BOX
        why = ("the shape does not divide" if not divides else
               "the cut leaves its box" if not in_a_box else
               f"ratio {ratio:.3f}, the moving hides")
        gate("the cut is legal", bool(divides and in_a_box and ratio <= 1), why,
             f"ratio {ratio:.3f}")
    result["readouts"]["overlap ratio"] = ratio

    miss = _missing(namespace, "the model state fits")
    if miss:
        gate("the model state fits", None, f"waiting for {' and '.join(miss)}")
        state = None
    else:
        per_parameter = namespace["resident_bytes_per_parameter"](sharing_level, dp)
        state = namespace["model_state_gb"](b.PARAMETERS / cut, per_parameter)
        gate("the model state fits", state <= b.MEMORY_GB,
             f"{state:.1f} GB of {b.MEMORY_GB} GB, persistent model state only",
             f"{state:.1f} of {b.MEMORY_GB} GB")
    result["readouts"]["model state GB"] = state

    batch = global_batch(b.CONTEXT, b.SEQ_PER_MICRO, micro, dp)
    gate("the batch contract holds", batch == b.GLOBAL_BATCH,
         f"{batch:,.0f} tokens against the {b.GLOBAL_BATCH:,.0f} asked for",
         f"{batch/1e6:.2f}M tokens")
    result["readouts"]["global batch"] = batch

    miss = _missing(namespace, "efficiency meets the target")
    if miss:
        for name in ("efficiency meets the target", "it meets the date"):
            gate(name, None, f"waiting for {' and '.join(miss)}")
        result["readouts"].update({"idle share": None, "efficiency": None, "days": None})
        return result

    idle = namespace["idle_share"](pp, micro)
    arithmetic = step_flop() / b.RATE_FLOPS / b.MACHINES
    agreeing = (namespace["bytes_sent_per_machine"](dp, buffer_after_cut_gb(tp, pp))
                / (b.LINK_ACROSS / b.GB))
    step = arithmetic / (1 - idle) + agreeing
    efficiency = arithmetic / step
    days = b.STEPS * step / b.SECONDS_A_DAY

    gate("efficiency meets the target", efficiency >= b.EFFICIENCY_TARGET,
         f"{efficiency:.1%} against a {b.EFFICIENCY_TARGET:.0%} target", f"{efficiency:.1%}")
    margin = b.DEADLINE_DAYS - days
    gate("it meets the date", days <= b.DEADLINE_DAYS,
         f"{days:.2f} days against {b.DEADLINE_DAYS}",
         f"{margin:.1f} days early" if margin >= 0 else f"{-margin:.1f} days late")
    result["readouts"].update({
        "idle share": idle, "efficiency": efficiency, "days": days,
        "arithmetic seconds": arithmetic, "agreeing seconds": agreeing, "step seconds": step,
    })
    return result


def verdict(result):
    """Valid is the arithmetic and the training problem. Meets the brief adds the targets."""
    g = result["gates"]
    structural = ["the product is valid", "the cut is legal", "the batch contract holds"]
    targets = ["the model state fits", "efficiency meets the target", "it meets the date"]
    if any(g.get(k) is None for k in g):
        return "not scored yet"
    if not all(g[k] for k in structural):
        return "not valid"
    return "meets the brief" if all(g[k] for k in targets) else "valid, misses the brief"


def whole_boxes(step=None):
    step = step or b.MACHINES_A_BOX
    return list(range(step, b.MACHINES + 1, step))
