"""A cluster that is not there.

Nothing in this file talks to a machine. It is a shell with four commands in it,
enough to see what a box holds and to start one process for each machine in it.

What is real: the shape of a command and what it tells you. What is not: the
hardware behind it. Nothing here reaches a driver, a card or a network.
"""

import os

GPUS = 8
MODEL = "NVIDIA H100 96GB"
TOTAL_MIB = 98304                 # the brief's 96 GB, as a machine would report it


# -------------------------------------------------------------- the machines
def nvidia_smi(used_mib=0):
    """What the box reports. Eight machines, and nothing of this is real."""
    a, bw, c = 33, 21, 22                      # the three columns of the real thing
    def row(l, m, r, fill=" "):
        return "|" + f"{l:{fill}<{a}}" + "|" + f"{m:{fill}<{bw}}" + "|" + f"{r:{fill}<{c}}" + "|"
    edge = "+" + "-" * (a + bw + c + 2) + "+"
    print(edge)
    print("|" + f"{' NVIDIA-SMI 550.54.15      Driver Version: 550.54.15      CUDA Version: 12.4':<{a + bw + c + 2}}" + "|")
    print(row("", "", "", "-"))
    print(row(" GPU  Name            Persist-M", " Bus-Id        Disp.A", "        Memory-Usage "))
    print(row("", "", "", "="))
    for g in range(GPUS):
        print(row(f"   {g}  {MODEL:<18}On", f" 00000000:{0x1B + g:02X}:00.0 Off",
                  f" {used_mib:>6}MiB / {TOTAL_MIB}MiB"))
    print(edge)
    total = TOTAL_MIB // 1024
    print(f"\n{GPUS} machines, {total} GB each, which nvidia-smi calls GPUs. "
          + ("Nothing is running on any of them." if not used_mib else ""))


# ----------------------------------------------------------------- the console
PROMPT = "fdd@box-01:~$"


def console(command):
    """A shell with four commands in it, none of which reach anything real."""
    command = str(command).strip()
    print(f"{PROMPT} {command}")
    head = command.split()[0] if command else ""

    if head == "nvidia-smi":
        nvidia_smi()
    elif head == "torchrun":
        n = GPUS
        for part in command.split():
            if part.startswith("--nproc-per-node"):
                n = int(part.split("=")[-1]) if "=" in part else GPUS
        print(f"[torchrun] starting {n} processes on {GPUS} machines\n")
        for r in range(n):
            print(f"   [rank {r}]  LOCAL_RANK={r}  RANK={r}  WORLD_SIZE={n}"
                  + ("  ->  machine " + str(r) if r < GPUS else "  ->  no machine to run on"))
        if n == GPUS:
            print(f"\n✓ {n} processes, one for each machine, each knowing which one it is")
        else:
            print(f"\n✗ {n} processes for {GPUS} machines, so they cannot map one to one")
    elif head == "sinfo":
        print("PARTITION  NODES  MACHINES/NODE  STATE")
        print(f"gpu-small  {b_boxes():>5}  {GPUS:>13}  idle")
    elif head == "hostname":
        print("box-01")
    elif not head:
        print()
    else:
        print(f"bash: {head}: command not found")
        print("✗ nothing listed. The command that shows what a box holds starts with nvidia.")


def b_boxes():
    import brief
    return brief.MACHINES // brief.MACHINES_A_BOX
