"""The quick checks.

Every number in an explanation is computed from the brief, so a change there
moves the explanations with it. The gates check the answers against the same
functions the notebook uses, so nothing here rests on intuition.

Each entry is (question, options, index of the answer, what the explanation adds).
"""
import brief as b
import runcore as rc


def _worked(*rows):
    """A compact block of arithmetic. Three columns: what, the sum, what it means."""
    out = "<div class='worked'>"
    for what, sum_, means in rows:
        out += f"<div class='wrow'><span>{what}</span><b>{sum_}</b><i>{means}</i></div>"
    return out + "</div>"


def _p(text):
    return f"<p>{text}</p>"


# ------------------------------------------------------- the numbers they use
_STATE = b.PARAMETERS * b.BYTES_PER_PARAMETER / b.GB
_OVER = _STATE / b.MEMORY_GB
_RAGGED = rc.local_means(rc.RAGGED_SPLIT)
_SIZES = rc.sizes(rc.RAGGED_SPLIT)
_PLAIN = sum(_RAGGED) / 4
_WEIGHTED = sum(n * m for n, m in zip(_SIZES, _RAGGED)) / sum(_SIZES)
_TRUE = rc.whole_batch_mean()
_ONE_STEP = rc.one_machine_step_seconds()
_BUFFER = rc.whole_model_buffer_gb()
_LINK = b.LINK_ACROSS / b.GB


def _dp_step(machines):
    """A step with every machine holding the whole model, as in part 3."""
    arithmetic = _ONE_STEP / machines
    agreeing = 2 * (machines - 1) * (_BUFFER / machines) / _LINK
    return arithmetic, agreeing


_HALF, _FULL = _dp_step(b.MACHINES // 2), _dp_step(b.MACHINES)
_SPEEDUP = sum(_HALF) / sum(_FULL)
_SHARED_OPT = 4 + 12 / b.REFERENCE_DP
_RATIO_BOX = 3 / (2 * b.HIDDEN) * b.PEAK_FLOPS / b.LINK_IN_A_BOX
_RATIO_FABRIC = 3 / (2 * b.HIDDEN) * b.PEAK_FLOPS / b.LINK_ACROSS
_CUT_BUFFER = 2 * b.PARAMETERS / 32 / b.GB


QUESTIONS = {

    "faster_chip": (
        "A vendor offers the same machine with double the peak rate. How much faster does your run "
        "finish?",
        ["Twice as fast, since the arithmetic is the whole job.",
         "Less than twice, and nobody can tell you how much without measuring.",
         "About 35 percent faster, because that is the sustained share.",
         "No faster, because the links have not changed."],
        1,
        _p("The time a run takes is the operations divided by the peak rate times the sustained "
           "share. Doubling the peak rate only halves the time if the share survives, and the share "
           "is the part the vendor is not selling you. A faster chip usually has better arithmetic "
           "units and the same memory bandwidth, the same links and the same path the data has to "
           "travel, so whatever was already the bottleneck has not moved.")
        + _worked(
            ("the machine you have",
             f"1 PFLOP/s &times; {b.SUSTAINED:.0%} = {b.SUSTAINED:.2f}", "effective"),
            ("double the peak, share holds",
             f"2 &times; {b.SUSTAINED:.0%} = {2*b.SUSTAINED:.2f}", "twice as fast"),
            ("double the peak, memory caps the share at 20%",
             f"2 &times; 20% = {2*0.20:.2f}",
             f"{100*(2*0.20)/b.SUSTAINED - 100:.0f} percent faster"))
        + _p("Same chip and the same claim on the datasheet, and the honest answer is somewhere "
             "between those last two. Only running a small piece of your own workload says where."),
    ),

    "bigger_slice": (
        "One machine finishes early, so the scheduler gives it a bigger slice of the next batch. "
        "What breaks?",
        ["Nothing. More work on a faster machine is exactly what you want.",
         "The machines stop agreeing, because they now hold different amounts.",
         "The plain average stops being the right way to combine them.",
         "The run slows down to the pace of the slowest machine."],
        2,
        _p("Each machine's average has to count in proportion to how many examples it holds. With "
           "even slices that happens to be a plain average, which is exactly why the mistake stays "
           "invisible until the slices are not even. These are the four machines from this part, "
           f"holding {', '.join(str(n) for n in _SIZES[:-1])} and {_SIZES[-1]} examples:")
        + _worked(
            ("what the four of them answered",
             "&nbsp;&nbsp;".join(f"{m:.2f}" for m in _RAGGED), "one each"),
            ("a plain average of those four",
             f"&divide; 4 = {_PLAIN:.4f}",
             f"{100*(_PLAIN-_TRUE)/_TRUE:.2f} percent off"),
            ("each one weighted by what it holds",
             f"&divide; {sum(_SIZES)} = {_WEIGHTED:.4f}", "exactly right"))
        + _p("Nothing about that division is strange, and a scheduler making a sensible decision "
             "produces it. The cluster then steadily optimises something slightly different from "
             "what it was asked to, and no output it produces would ever show you."),
    ),

    "why_slower": (
        "You double the size of the cluster and the run gets only 1.3 times faster. Which term did "
        "that, and what would you measure to confirm it?",
        ["The arithmetic, and you would measure the sustained share.",
         "The agreeing, and you would measure the share of a step spent doing it.",
         "Both equally, and you would measure the total step time.",
         "Neither. That is what doubling a cluster normally gives you."],
        1,
        _p("The arithmetic term always divides cleanly, because the work per machine is the work "
           "divided by the machines. When doubling stops paying it is never the arithmetic that "
           "failed. On the brief's own numbers, going from "
           f"{b.MACHINES//2:,} machines to {b.MACHINES:,} does exactly this:")
        + _worked(
            (f"{b.MACHINES//2:,} machines",
             f"{_HALF[0]:.2f} + {_HALF[1]:.2f} = {sum(_HALF):.2f} s", "a step"),
            (f"{b.MACHINES:,} machines",
             f"{_FULL[0]:.2f} + {_FULL[1]:.2f} = {sum(_FULL):.2f} s", "a step"),
            ("so the run gets",
             f"{sum(_HALF):.2f} &divide; {sum(_FULL):.2f} = {_SPEEDUP:.2f}&times;",
             "faster, not 2"))
        + _p("The arithmetic halved exactly as it should. The agreeing did not move at all, so it "
             "went from half of a step to two thirds of one. The number worth taking to the meeting "
             "is the share of a step spent agreeing, because that is the one that grew."),
    ),

    "inference_or_training": (
        "A model's weights are 2 bytes a parameter. Why can the same model need many times that "
        "amount of memory to train?",
        ["Because every parameter becomes larger once training starts.",
         "Because training keeps extra arrays alongside the weights, one entry for every parameter.",
         "Because the training data has to sit in the machine's memory alongside the model.",
         "Because training and inference use different architectures."],
        1,
        _p("A parameter is the same two bytes before a step and after it. What changes is how many "
           "arrays with one entry per parameter the machine is holding at once: the weights, their "
           "direction of travel, the wider copy, and the optimiser's two records.")
        + _worked(
            ("just to hold the model",
             f"{b.PARAMETERS/1e9:.0f} billion &times; 2 bytes = {b.PARAMETERS*2/b.GB:,.0f} GB",
             "the weights alone"),
            ("to train the same model",
             f"{b.PARAMETERS/1e9:.0f} billion &times; {b.BYTES_PER_PARAMETER} bytes "
             f"= {_STATE:,.0f} GB", "five arrays, not one"),
            (f"so one {b.MEMORY_GB} GB machine holds",
             f"{b.MEMORY_GB*b.GB/2/1e9:.1f} billion to run, "
             f"{b.MEMORY_GB*b.GB/b.BYTES_PER_PARAMETER/1e9:.1f} to train",
             f"{b.BYTES_PER_PARAMETER/2:.0f} times fewer"))
        + _p("The other three describe things that do not happen. The architecture is identical, "
             "since inference runs the model that training produced. The data arrives in slices "
             "and is nowhere near the size of the model state. And nothing about a parameter grows: "
             "the arrays around it are what grew. That last point is also why storing the weights "
             f"more narrowly buys less than people expect. Twelve of the {b.BYTES_PER_PARAMETER} "
             "bytes are in the other three arrays, and halving the first one does not move them."),
    ),

    "across_boxes": (
        "A vendor proposes spreading each layer's arithmetic across four boxes rather than four "
        "machines inside one. What do you ask?",
        ["How many machines that would need in total.",
         "What it costs per machine-hour.",
         "What the link between the boxes carries per second.",
         "How much sooner the run would finish."],
        2,
        _p("The cut moves a fixed amount of data every time it is used, and whether that hides "
           "behind the arithmetic depends on one thing only. The proposal changes that one thing "
           "and leaves everything else alone:")
        + _worked(
            ("four machines inside one box",
             f"{b.LINK_IN_A_BOX/b.GB:.0f} GB/s, ratio {_RATIO_BOX:.3f}", "under 1, so it hides"),
            ("the same four, one to a box",
             f"{b.LINK_ACROSS/b.GB:.0f} GB/s, ratio {_RATIO_FABRIC:.2f}",
             "over 1, so it does not"),
            ("what actually changed",
             f"a link {b.LINK_IN_A_BOX/b.LINK_ACROSS:.0f}&times; slower",
             "same cut, same model"))
        + _p("The other three questions are all reasonable and none of them decides it. This is one "
             "of the few places in the notebook where a proposal can be turned down before anybody "
             "measures anything."),
    ),

    "wrapup": (
        "One of these is false. Which?",
        ["Splitting a batch across machines gives exactly the same answer, if the slices are "
         "weighted by size.",
         "Adding machines barely changes what each one has to send, and steadily raises the share "
         "of a step spent sending it.",
         "A configuration that passes every gate in this notebook is safe to start.",
         "Cutting the model across machines also shrinks what the copies have to agree over."],
        2,
        _p("The other three you derived yourself, and each one has a number behind it:")
        + _worked(
            ("splitting is exact",
             f"weighted {_WEIGHTED:.4f} against a true {_TRUE:.4f}", "part 2"),
            ("the agreeing stops shrinking",
             f"{_FULL[0]:.2f} s of arithmetic, {_FULL[1]:.2f} s of agreeing", "part 3"),
            ("cutting the model shrinks the buffer",
             f"{_BUFFER:.0f} GB &rarr; {_CUT_BUFFER:.2f} GB", "part 6"))
        + _p("The false one is the third. Every time in this notebook is modelled from the brief, "
             "and the brief leaves out what the forward pass holds, the working buffers, and the "
             "fixed overhead on every machine. Passing the gates means a plan is worth "
             "benchmarking. It has never meant it is worth starting, and the last part is entirely "
             "about the distance between those two."),
    ),
}
