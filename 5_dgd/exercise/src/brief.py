"""The run you have been asked to schedule.

Every number the notebook prints comes from here. Change one and everything
downstream moves with it.
"""

# The model
PARAMETERS      = 70e9
LAYERS          = 80
HIDDEN          = 8192
HEADS           = 64
KEY_VALUE_HEADS = 8

# The job
TOKEN_BUDGET  = 2e12
CONTEXT       = 8192
SEQ_PER_MICRO = 1
MICROBATCHES  = 16
REFERENCE_DP  = 64

# The cluster
MACHINES       = 2048
MACHINES_A_BOX = 8
MEMORY_GB      = 96
PEAK_FLOPS     = 1000e12
SUSTAINED      = 0.35
LINK_IN_A_BOX  = 900e9
LINK_ACROSS    = 25e9

# The bill: 2 weights + 2 gradients + 4 master weights + 4 + 4 optimiser
WEIGHT_BYTES        = 2      # one parameter, stored
BYTES_PER_PARAMETER = 16     # one parameter, and the four arrays training keeps beside it

# What management has asked for
DEADLINE_DAYS     = 25
EFFICIENCY_TARGET = 0.75

# Stated planning assumptions. Neither is measured anywhere in this notebook.
CHECKPOINT_WRITE_SECONDS = 11.2
HOURS_BETWEEN_STOPS      = 12

# Units
GB      = 1e9
SECONDS_A_DAY = 86400

GLOBAL_BATCH = CONTEXT * SEQ_PER_MICRO * MICROBATCHES * REFERENCE_DP
RATE_FLOPS   = PEAK_FLOPS * SUSTAINED
STEPS        = TOKEN_BUDGET / GLOBAL_BATCH


def as_rows():
    """The brief, in the order the opening card prints it."""
    return [
        ("the model",     f"{PARAMETERS/1e9:.0f} billion parameters, {LAYERS} layers, "
                          f"{HIDDEN:,} wide, {HEADS} heads of which {KEY_VALUE_HEADS} carry keys and values"),
        ("the job",       f"{TOKEN_BUDGET/1e12:.0f} trillion tokens, in batches of "
                          f"{GLOBAL_BATCH:,.0f} ({CONTEXT:,} x {SEQ_PER_MICRO} x {MICROBATCHES} x {REFERENCE_DP})"),
        ("the cluster",   f"{MACHINES:,} machines in boxes of {MACHINES_A_BOX}, {MEMORY_GB} GB each"),
        ("each machine",  f"{PEAK_FLOPS/1e12:.0f} TFLOP/s at peak, {SUSTAINED:.0%} of it sustained"),
        ("the links",     f"{LINK_IN_A_BOX/GB:.0f} GB/s inside a box, {LINK_ACROSS/GB:.0f} GB/s between boxes"),
        ("the bill",      f"{BYTES_PER_PARAMETER} bytes for every parameter"),
        ("the ask",       f"finished within {DEADLINE_DAYS} days, with at least "
                          f"{EFFICIENCY_TARGET:.0%} of every step spent on arithmetic"),
    ]
