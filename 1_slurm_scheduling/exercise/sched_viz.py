"""Presentation, simulation & quiz helpers for the WE7 notebook 03:
"You are now the cluster scheduler".

Same idea as WE4's `intro_viz`, `pg_viz` and `ac_viz`: every HTML/CSS
illustration, interactive widget and quiz *answer key* lives here, out of the
notebook, so the teaching cells stay about the *idea*.

Nothing here needs a GPU, a network, or Slurm. The whole cluster is a handful
of integers.

    import sched_viz as sv
    sv.cluster_map()
    sv.timeline(schedule)

Students are told not to read this file. (You are, presumably, not a student.)
"""
import json as _json
import copy as _copy
import random as _random

from IPython.display import HTML, display

# ===========================================================================
#  §0  Palette: kept in one place so every widget colours jobs the same way
# ===========================================================================
# Every widget paints its own white surface, so it must paint its own text
# colour too: in a dark-themed notebook the inherited colour is near-white and
# would vanish against that surface.
TEXT = "#24262b"
INK = "#2b2d6b"
ACCENT = "#764ba2"
ACCENT2 = "#667eea"
GREEN = "#2e9e7a"
RED = "#c0554e"
AMBER = "#e0a500"
GREY = "#9aa0b5"

# One colour per job name, reused by every timeline in the notebook so that a
# rectangle keeps its identity from Part 1 to Part 7.
JOB_COLOR = {
    "J1": "#4a5bd0", "J2": "#c9548f", "J3": "#e0a500", "J4": "#2e9e7a",
    "J5": "#3fa7c4", "J6": "#8d6ec9", "J7": "#c0554e", "J8": "#6b8e23",
    "Q1": "#4a5bd0", "Q2": "#e0a500", "Q3": "#c9548f", "Q4": "#8d6ec9",
    "Q5": "#3fa7c4",
}
_PALETTE = ["#4a5bd0", "#c9548f", "#e0a500", "#2e9e7a", "#3fa7c4",
            "#8d6ec9", "#c0554e", "#6b8e23", "#d2731f", "#5b7fb4"]

GROUPS = ["bio", "clim", "robo", "econ"]
GROUP_NAME = {
    "bio": "Biomedical Imaging",
    "clim": "Climate Modelling",
    "robo": "Robotics",
    "econ": "Economics",
}
GROUP_COLOR = {"bio": "#4a5bd0", "clim": "#2e9e7a", "robo": "#c9548f",
               "econ": "#e0a500"}
# Target shares, set by the administrators. Biomedical Imaging paid for two of
# the four nodes; Climate brought a national grant; Robotics and Economics
# joined later and pay a smaller subscription.
TARGET_SHARE = {"bio": 0.40, "clim": 0.30, "robo": 0.20, "econ": 0.10}


def color_of(name):
    if name in JOB_COLOR:
        return JOB_COLOR[name]
    return _PALETTE[abs(hash(name)) % len(_PALETTE)]


# ===========================================================================
#  §1  The cluster and the workloads
# ===========================================================================
NODE_SPECS = [
    {"name": "node01", "cores": 8, "mem": 64, "gpus": 0},
    {"name": "node02", "cores": 8, "mem": 64, "gpus": 0},
    {"name": "node03", "cores": 8, "mem": 64, "gpus": 2},
    {"name": "node04", "cores": 8, "mem": 64, "gpus": 2},
]
N_NODES = len(NODE_SPECS)


def new_cluster():
    """A fresh cluster with nothing running on it."""
    return [{"name": n["name"], "cores": n["cores"], "mem": n["mem"],
             "gpus": n["gpus"], "free_cores": n["cores"],
             "free_mem": n["mem"], "free_gpus": n["gpus"]}
            for n in NODE_SPECS]


def make_job(name, nodes, requested_time, actual_duration=None, group="bio",
             cores=8, mem=64, gpus=0, submit_time=0, qos=1.0):
    """One job, spelled out. Every field is visible; nothing is computed later."""
    if actual_duration is None:
        actual_duration = requested_time
    return {"name": name, "group": group, "nodes": nodes, "cores": cores,
            "mem": mem, "gpus": gpus, "requested_time": requested_time,
            "actual_duration": actual_duration, "submit_time": submit_time,
            "qos": qos}


def five_jobs():
    """The lecture's running example: four nodes, five whole-node jobs, all at t = 0.

    Listed in priority order. `actual_duration` equals `requested_time` here:
    every user declared honestly. Part 3 is where that stops being true.
    """
    return [
        make_job("J1", nodes=1, requested_time=3, group="bio"),
        make_job("J2", nodes=4, requested_time=2, group="clim"),
        make_job("J3", nodes=3, requested_time=4, group="bio"),
        make_job("J4", nodes=3, requested_time=3, group="robo"),
        make_job("J5", nodes=1, requested_time=2, group="econ"),
    ]


def workload(n=30, seed=7, arrivals=0, honest=True):
    """A pile of jobs that nobody would place by hand.

    arrivals = 0   -> everything is submitted at t = 0 (a closed workload)
    arrivals = H   -> submissions are spread over the first H hours
    honest = False -> some users pad their wall time and a few under-declare
    """
    rng = _random.Random(seed)
    jobs = []
    for i in range(n):
        g = rng.choices(GROUPS, weights=[35, 30, 25, 10])[0]
        nodes = rng.choice([1, 1, 1, 1, 2, 2, 3])
        cores = rng.choice([2, 4, 4, 8, 8])
        gpus = 1 if (cores >= 4 and rng.random() < 0.25) else 0
        if gpus:                      # only two of the four nodes have GPUs at all
            nodes = min(nodes, sum(1 for x in NODE_SPECS if x["gpus"] >= gpus))
        req = rng.choice([1, 2, 2, 3, 3, 4, 6])
        act = req
        if not honest:
            r = rng.random()
            if r < 0.30:                    # padded: asked for far more than needed
                act = max(1, int(round(req * rng.choice([0.4, 0.5, 0.6]))))
            elif r < 0.40:                  # under-declared: will hit the limit
                act = req + rng.choice([1, 2, 3])
        jobs.append(make_job("j%02d" % (i + 1), nodes=nodes, requested_time=req,
                             actual_duration=act, group=g, cores=cores,
                             mem=cores * 8, gpus=gpus,
                             submit_time=(rng.randrange(arrivals) if arrivals else 0),
                             qos=rng.choice([1.0, 1.0, 1.0, 2.0])))
    jobs.sort(key=lambda j: (j["submit_time"], j["name"]))
    return jobs


def starving_workload(hours=24, per_hour=3, seed=1):
    """One wide job, and a stream of narrow ones that never stops arriving.

    The wide job is `W`: three nodes for six hours, from the group that has
    been consuming most. Everything else is one node for an hour or two, from
    the other groups, with a QoS the administrators granted them. This is the
    shape of workload on which a queue without ageing quietly stops serving
    somebody.
    """
    rng = _random.Random(seed)
    jobs = [make_job("W", nodes=3, requested_time=6, group="bio", cores=8,
                     mem=64, submit_time=0, qos=1.0)]
    k = 0
    for t in range(hours):
        for _ in range(per_hour):
            k += 1
            jobs.append(make_job("s%02d" % k, nodes=1,
                                 requested_time=rng.choice([1, 1, 2]),
                                 group=rng.choice(["clim", "robo", "econ"]),
                                 cores=8, mem=64, submit_time=t, qos=2.0))
    return jobs


# ===========================================================================
#  §2  The engine
#      This is exactly the loop the notebook writes by hand in Part 2, with
#      priority ordering and backfilling bolted on for Parts 4 and 5.
# ===========================================================================
def _fits(node, job):
    return (node["free_cores"] >= job["cores"] and
            node["free_mem"] >= job["mem"] and
            node["free_gpus"] >= job["gpus"])


def find_nodes(cluster, job):
    """The names of `job["nodes"]` nodes with enough free resources, or None.

    GPU-free nodes are tried first, so a plain CPU job does not squat on a GPU
    node that a GPU job will need later.
    """
    candidates = sorted([n for n in cluster if _fits(n, job)],
                        key=lambda n: (n["gpus"], n["name"]))
    if len(candidates) < job["nodes"]:
        return None
    return [n["name"] for n in candidates[:job["nodes"]]]


def allocate(cluster, names, job):
    for n in cluster:
        if n["name"] in names:
            n["free_cores"] -= job["cores"]
            n["free_mem"] -= job["mem"]
            n["free_gpus"] -= job["gpus"]


def release(cluster, names, job):
    for n in cluster:
        if n["name"] in names:
            n["free_cores"] += job["cores"]
            n["free_mem"] += job["mem"]
            n["free_gpus"] += job["gpus"]


def _earliest_start(cluster, running, job, now):
    """The first moment `job` could have everything it asked for. This is the
    number Slurm writes down when it makes a reservation."""
    times = [now] + sorted(set(r["end"] for r in running))
    for t in times:
        probe = _copy.deepcopy(cluster)
        for r in running:
            if r["end"] <= t:
                release(probe, r["nodes"], r["job"])
        if find_nodes(probe, job) is not None:
            return t
    return None


def priority_of(job, now, weights, share_factor, age_scale=24.0):
    """The toy score of Part 4. Not Slurm's real formula, but the same shape."""
    age = min(1.0, (now - job["submit_time"]) / float(age_scale))
    fair = share_factor.get(job["group"], 0.5)
    qos = min(1.0, job["qos"] / 2.0)
    return (weights.get("age", 0.0) * age +
            weights.get("fair_share", 0.0) * fair +
            weights.get("qos", 0.0) * qos)


def score_at(jobs, schedule, name, when, weights, age_scale=24.0):
    """The Part 4 score of one job at one particular moment.

    The fair-share factor is rebuilt from the node-hours every group had been
    charged by then, exactly the way ``run_schedule(dynamic_fairshare=True)``
    does it, so the number is the one the scheduler actually saw at that moment.

    Args:
        jobs: the job list handed to run_schedule; carries submit_time and qos.
        schedule: what run_schedule returned for those jobs.
        name: the job to score, e.g. "W".
        when: the moment to score it at, in hours.
        weights: the same {"age": .., "fair_share": .., "qos": ..} dict.
        age_scale: hours of waiting at which the age term reaches its maximum.

    Returns:
        The score, as a float.
    """
    charged = dict((g, 0.0) for g in TARGET_SHARE)
    for s in schedule:
        ran_for = min(s["end"], when) - s["start"]
        if ran_for > 0:
            charged[s["group"]] = charged.get(s["group"], 0.0) + \
                s["n_nodes"] * (s["cores"] / NODE_CORES) * ran_for
    job = [j for j in jobs if j["name"] == name][0]
    return priority_of(job, when, weights, share_factors(charged), age_scale)


def run_schedule(jobs, backfill=False, weights=None, share_factor=None,
                 age_scale=24.0, trace=False, dynamic_fairshare=False,
                 reserve_depth=1):
    """Simulate the cluster hour by hour and return where every job landed.

    weights = None      -> FIFO: jobs are considered in submission order
    weights = {...}     -> priority order, recomputed at every scheduling round
    backfill            -> lower-priority jobs may start early if they finish
                           before the reservation made for the blocked job
    dynamic_fairshare   -> recompute F = 2**(-U/S) from the node-hours charged
                           so far, at every round, instead of holding it fixed
    reserve_depth       -> how many blocked jobs get a protected start time.
                           1 is what the lecture describes. Higher numbers
                           protect more big jobs and leave fewer holes to fill.
                           (Each reservation is computed against what is running
                           now, which is a simplification of what Slurm does.)
    """
    jobs = [dict(j) for j in jobs]
    cluster = new_cluster()
    pending = sorted(jobs, key=lambda j: (j["submit_time"], jobs.index(j)))
    running, done, log = [], [], []
    now = min(j["submit_time"] for j in jobs)
    share_factor = dict(share_factor or {})

    while pending or running:
        # --- 1. give back the resources of everything that has finished ------
        for r in list(running):
            if r["end"] <= now:
                release(cluster, r["nodes"], r["job"])
                running.remove(r)
                done.append(r)

        # --- 2. who is allowed to be considered right now --------------------
        queue = [j for j in pending if j["submit_time"] <= now]
        if dynamic_fairshare:
            charged = dict((g, 0.0) for g in TARGET_SHARE)
            for r in done + running:
                j2 = r["job"]
                charged[j2["group"]] = charged.get(j2["group"], 0.0) + \
                    j2["nodes"] * (j2["cores"] / NODE_CORES) * \
                    (min(r["end"], now) - r["start"])
            share_factor = share_factors(charged)
        if weights is not None:
            queue.sort(key=lambda j: -priority_of(j, now, weights, share_factor,
                                                  age_scale))

        reserved_start, reserved_job, n_reserved = None, None, 0
        for job in list(queue):
            names = find_nodes(cluster, job)
            if names is not None and (reserved_start is None or
                                      now + job["requested_time"] <= reserved_start):
                run_time = min(job["requested_time"], job["actual_duration"])
                rec = {"job": job, "name": job["name"], "nodes": names,
                       "start": now, "end": now + run_time,
                       "killed": job["actual_duration"] > job["requested_time"],
                       "backfilled": reserved_start is not None}
                allocate(cluster, names, job)
                running.append(rec)
                pending.remove(job)
                if trace:
                    log.append(("start", now, job["name"], names,
                                rec["backfilled"], reserved_start))
            elif n_reserved < reserve_depth:
                # This job does not fit: Slurm reserves for it the earliest
                # start it can guarantee, and protects that promise.
                t = _earliest_start(cluster, running, job, now)
                if t is not None:
                    reserved_start = t if reserved_start is None else min(reserved_start, t)
                    reserved_job = reserved_job or job["name"]
                    n_reserved += 1
                if trace:
                    log.append(("reserve", now, job["name"], t))
                if not backfill:
                    break

        # --- 3. jump the clock to the next thing that can possibly change ----
        future = [r["end"] for r in running]
        future += [j["submit_time"] for j in pending if j["submit_time"] > now]
        if not future:
            if not pending:          # everything has finished, we are done
                break
            raise ValueError(
                "%s can never run on this cluster: it asks for %d nodes with "
                "%d cores / %d GB / %d GPU each, and no set of nodes is that big. "
                "Slurm rejects such a job at submission time (PartitionNodeLimit)."
                % (pending[0]["name"], pending[0]["nodes"], pending[0]["cores"],
                   pending[0]["mem"], pending[0]["gpus"]))
        now = min(future)

    for r in running:
        done.append(r)
    schedule = []
    for r in done:
        j = r["job"]
        schedule.append({
            "name": j["name"], "group": j["group"], "nodes": r["nodes"],
            "n_nodes": j["nodes"], "cores": j["cores"], "gpus": j["gpus"],
            "start": r["start"], "end": r["end"],
            "submit_time": j["submit_time"],
            "requested_time": j["requested_time"],
            "actual_duration": j["actual_duration"],
            "killed": r["killed"], "backfilled": r.get("backfilled", False),
            "waiting": r["start"] - j["submit_time"],
            "turnaround": r["end"] - j["submit_time"],
        })
    schedule.sort(key=lambda s: (s["start"], s["name"]))
    return (schedule, log) if trace else schedule


# ===========================================================================
#  §2b  Metrics: every one of them is a two-line formula, on purpose
# ===========================================================================
NODE_CORES = float(NODE_SPECS[0]["cores"])


def node_hours(s, t0=None, t1=None):
    """Node-hours a job occupies inside the window [t0, t1].

    A job that takes 2 of a node's 8 cores holds a quarter of that node, so it
    is charged a quarter of a node-hour per hour. Whole-node jobs (everything
    in Parts 1 to 5) are charged exactly what you would count by hand.
    """
    a = s["start"] if t0 is None else max(s["start"], t0)
    b = s["end"] if t1 is None else min(s["end"], t1)
    if b <= a:
        return 0.0
    return s["n_nodes"] * (s["cores"] / NODE_CORES) * (b - a)


def metrics(schedule, n_nodes=N_NODES, window=24):
    first_start = min(s["start"] for s in schedule)
    last_end = max(s["end"] for s in schedule)
    makespan = last_end - first_start

    busy = sum(node_hours(s) for s in schedule)
    total = float(n_nodes * makespan)
    gpu_busy = sum(s["n_nodes"] * s["gpus"] * (s["end"] - s["start"])
                   for s in schedule)
    gpu_total = float(sum(n["gpus"] for n in NODE_SPECS) * makespan)
    per_group = group_waiting(schedule)

    return {
        "makespan": makespan,
        "busy_node_hours": busy,
        "total_node_hours": total,
        "idle_node_hours": total - busy,
        "utilisation": busy / total if total else 0.0,
        "gpu_utilisation": gpu_busy / gpu_total if gpu_total else 0.0,
        "avg_waiting": sum(s["waiting"] for s in schedule) / float(len(schedule)),
        "max_waiting": max(s["waiting"] for s in schedule),
        "worst_group_waiting": max(per_group.values()) if per_group else 0.0,
        "avg_turnaround": sum(s["turnaround"] for s in schedule) / float(len(schedule)),
        "killed": sum(1 for s in schedule if s["killed"]),
        "n_jobs": len(schedule),
        "throughput": len(schedule) / float(makespan) if makespan else 0.0,
        "fairness": fairness(schedule, window=window),
    }


def group_waiting(schedule):
    """Average waiting time per research group."""
    tot, n = {}, {}
    for s in schedule:
        tot[s["group"]] = tot.get(s["group"], 0.0) + s["waiting"]
        n[s["group"]] = n.get(s["group"], 0) + 1
    return dict((g, tot[g] / n[g]) for g in tot)


def delivered_share(schedule, window=None):
    """Share of the node-hours *delivered so far* that went to each group.

    With a window, this is what the groups got in the first `window` hours,
    which is exactly what the ordering of the queue decides. Without one, it is
    the whole run, and every policy that finishes every job gives the same
    answer.
    """
    t0 = min(s["start"] for s in schedule)
    t1 = None if window is None else t0 + window
    used = dict((g, 0.0) for g in GROUPS)
    for s in schedule:
        used[s["group"]] = used.get(s["group"], 0.0) + node_hours(s, t0, t1)
    tot = sum(used.values()) or 1.0
    return dict((g, used[g] / tot) for g in used)


def fairness(schedule, target=None, window=24):
    """1.0 when the node-hours delivered in the first `window` hours matched the
    target shares exactly, 0.0 at the worst possible mismatch.

    This is one minus the total-variation distance between what the groups were
    served and what the administrators said they should be served.
    """
    target = target or TARGET_SHARE
    u = delivered_share(schedule, window=window)
    dev = sum(abs(u.get(g, 0.0) - target.get(g, 0.0)) for g in target)
    return 1.0 - dev / 2.0


def share_factors(usage, target=None, decay_periods=0.0, half_life=1.0):
    """F = 2 ** (-U / S), with optional decay applied to the raw usage first."""
    target = target or TARGET_SHARE
    decayed = dict((g, usage.get(g, 0.0) * (0.5 ** (decay_periods / float(half_life))))
                   for g in target)
    tot = sum(decayed.values()) or 1.0
    return dict((g, 2.0 ** (-(decayed[g] / tot) / target[g])) for g in target)


# ===========================================================================
#  §3  Generic renderers  (shared with WE4's intro_viz / pg_viz / ac_viz)
# ===========================================================================
_FONT = "system-ui,Segoe UI,Roboto,sans-serif"


_UID_N = [0]


def _uid(tag, key):
    """Unique per call, so two copies of a widget on one page never collide."""
    _UID_N[0] += 1
    return "%s_%d_%d" % (tag, abs(hash(str(key))) % 10 ** 6, _UID_N[0])


def _card(inner, maxw=880):
    display(HTML(
        '<div style="font-family:%s;border:1px solid #e6e8ee;border-radius:14px;'
        'padding:18px;max-width:%dpx;background:#fff;color:%s">%s</div>'
        % (_FONT, maxw, TEXT, inner)))


def _mc_render(question, setup, options, answer_index, reveal):
    """The setup is rendered ABOVE the question.

    You cannot ask someone something and then tell them what you are talking
    about: the situation has to be in place before the question lands.
    """
    # `reveal` is either one string for the whole question, or one string per
    # option — a wrong answer is only useful if it says why *that* answer is wrong.
    per_option = not isinstance(reveal, str)
    if per_option and len(reveal) != len(options):
        raise ValueError("need one explanation per option (%d options, %d given)"
                         % (len(options), len(reveal)))
    data = {"opts": list(options), "ans": int(answer_index),
            "reveal": None if per_option else reveal,
            "why": list(reveal) if per_option else None}
    # An empty setup means the question stands on its own, with no preamble box.
    _setup_html = ('<div class="mc-setup">%s</div>' % setup) if setup.strip() else ""
    uid = _uid("mc", (setup, tuple(options), answer_index))
    tmpl = r'''
<style>
#__UID__{font-family:__FONT__;border:1px solid #e6e8ee;border-radius:14px;padding:16px;max-width:800px;background:#fff;color:#24262b}
#__UID__ .mc-setup{color:#555;font-size:13px;line-height:1.6;background:#f7f8fc;border-radius:9px;padding:11px 13px;margin-bottom:12px}
#__UID__ .mc-setup b.mc-lab{display:block;font-size:10px;font-weight:800;letter-spacing:.07em;text-transform:uppercase;color:#8a8fa3;margin-bottom:5px}
#__UID__ .mc-head{font-weight:800;font-size:15px;margin-bottom:12px;line-height:1.45}
#__UID__ .mc-opt{display:flex;align-items:flex-start;gap:10px;border:1px solid #e2e5ef;border-radius:10px;padding:10px 12px;margin-bottom:8px;cursor:pointer;font-size:13.5px;line-height:1.5;transition:.12s}
#__UID__ .mc-opt:hover{border-color:#764ba2;background:#faf9ff}
#__UID__ .mc-dot{width:16px;height:16px;border-radius:50%;border:2px solid #c2c7da;flex:0 0 auto;margin-top:2px}
#__UID__ .mc-opt code{background:#f3f0ff;border-radius:5px;padding:1px 5px;font-size:12.5px}
#__UID__ .mc-opt.sel{border-color:#764ba2;background:#f1edff}
#__UID__ .mc-opt.sel .mc-dot{background:#764ba2;border-color:#764ba2}
#__UID__ .mc-opt.ok{border-color:#46b46e;background:#e7f7ec}
#__UID__ .mc-opt.no{border-color:#e07a7a;background:#fdecec}
#__UID__ .mc-btn{cursor:pointer;border:none;border-radius:8px;padding:9px 18px;font-size:13.5px;font-weight:700;color:#fff;background:linear-gradient(135deg,#667eea,#764ba2);margin-top:6px}
#__UID__ .mc-rev{font-size:13px;color:#2c2350;margin-top:10px;min-height:18px;line-height:1.6}
#__UID__ .mc-again{margin-top:8px;font-size:12.5px;color:#8a8fa3;font-weight:600}
</style>
<div id="__UID__">
  __SETUP__
  <div class="mc-head">__Q__</div>
  <div class="mc-list"></div>
  <button class="mc-btn">Check my answer</button>
  <div class="mc-rev"></div>
</div>
<script>
(function(){
  const D=__DATA__, root=document.getElementById("__UID__");
  let idx=D.opts.map((_,i)=>i);
  for(let i=idx.length-1;i>0;i--){const j=Math.floor(Math.random()*(i+1));[idx[i],idx[j]]=[idx[j],idx[i]];}
  const list=root.querySelector(".mc-list");
  idx.forEach(orig=>{
    const o=document.createElement("div"); o.className="mc-opt"; o.dataset.i=orig;
    o.innerHTML='<span class="mc-dot"></span>'+D.opts[orig];
    list.appendChild(o);
  });
  const opts=list.querySelectorAll(".mc-opt"); let sel=null;
  opts.forEach(o=>o.addEventListener("click",()=>{
    sel=+o.dataset.i; opts.forEach(x=>x.classList.remove("sel","ok","no")); o.classList.add("sel");
    root.querySelector(".mc-rev").textContent="";
  }));
  root.querySelector(".mc-btn").addEventListener("click",()=>{
    if(sel===null){root.querySelector(".mc-rev").textContent="Pick an option first!";return;}
    opts.forEach(o=>{const i=+o.dataset.i; o.classList.remove("sel");
      // Only ever mark the option that was actually chosen. Highlighting the
      // correct one after a wrong guess hands over the answer for free.
      if(sel===D.ans){ if(i===D.ans)o.classList.add("ok"); }
      else if(i===sel)o.classList.add("no");});
    let msg;
    if(sel===D.ans){
      msg="✅ <b>Correct.</b> "+(D.why?D.why[D.ans]:D.reveal);
    } else if(D.why){
      msg="❌ <b>Not quite.</b> "+D.why[sel]
         +'<div class="mc-again">Pick another option and check again.</div>';
    } else {
      msg='❌ <b>Not quite.</b> <span class="mc-again">Pick another option and check again.</span>';
    }
    root.querySelector(".mc-rev").innerHTML=msg;
  });
})();
</script>'''
    display(HTML(tmpl.replace("__UID__", uid).replace("__FONT__", _FONT)
                 .replace("__SETUP__", _setup_html).replace("__Q__", question)
                 .replace("__DATA__", _json.dumps(data))))


def _tf_render(title, statements,
               prompt="Click every statement you think is TRUE, then check."):
    items = [{"t": t, "ok": bool(v)} for t, v in statements]
    uid = _uid("tf", (title, tuple(t for t, _ in statements)))
    tmpl = r'''
<style>
#__UID__{font-family:__FONT__;border:1px solid #e6e8ee;border-radius:14px;padding:16px;max-width:800px;background:#fff;color:#24262b}
#__UID__ .tf-head{font-weight:800;font-size:15px;margin-bottom:4px}
#__UID__ .tf-sub{color:#666;font-size:12.5px;margin-bottom:12px}
#__UID__ .tf-opt{display:flex;align-items:center;gap:10px;border:1px solid #e2e5ef;border-radius:10px;padding:9px 12px;margin-bottom:7px;cursor:pointer;font-size:13.5px;line-height:1.5}
#__UID__ .tf-opt:hover{border-color:#764ba2;background:#faf9ff}
#__UID__ .tf-box{width:16px;height:16px;border-radius:4px;border:2px solid #c2c7da;flex:0 0 auto}
#__UID__ .tf-opt.sel{border-color:#764ba2;background:#f1edff}
#__UID__ .tf-opt.sel .tf-box{background:#764ba2;border-color:#764ba2}
#__UID__ .tf-opt.ok{border-color:#46b46e;background:#e7f7ec}
#__UID__ .tf-opt.no{border-color:#e07a7a;background:#fdecec}
#__UID__ .tf-btn{cursor:pointer;border:none;border-radius:8px;padding:9px 18px;font-size:13.5px;font-weight:700;color:#fff;background:linear-gradient(135deg,#667eea,#764ba2);margin-top:6px}
#__UID__ .tf-status{font-size:13px;font-weight:700;color:#3b2d6b;margin-top:10px;min-height:18px}
</style>
<div id="__UID__">
  <div class="tf-head">__TITLE__</div>
  <div class="tf-sub">__PROMPT__</div>
  <div class="tf-list"></div>
  <button class="tf-btn">Check</button>
  <div class="tf-status"></div>
</div>
<script>
(function(){
  let DATA=__DATA__.slice();
  for(let i=DATA.length-1;i>0;i--){const j=Math.floor(Math.random()*(i+1));[DATA[i],DATA[j]]=[DATA[j],DATA[i]];}
  const root=document.getElementById("__UID__"), list=root.querySelector(".tf-list");
  DATA.forEach((d,i)=>{
    const row=document.createElement("div"); row.className="tf-opt"; row.dataset.i=i;
    row.innerHTML='<span class="tf-box"></span>'+d.t;
    row.addEventListener("click",()=>{row.classList.remove("ok","no");row.classList.toggle("sel");});
    list.appendChild(row);
  });
  root.querySelector(".tf-btn").addEventListener("click",()=>{
    let right=0; const rows=list.querySelectorAll(".tf-opt");
    rows.forEach(r=>{
      const d=DATA[+r.dataset.i], picked=r.classList.contains("sel");
      r.classList.remove("ok","no");
      if(picked===d.ok)right++; else r.classList.add("no");
      if(d.ok)r.classList.add("ok");
    });
    root.querySelector(".tf-status").textContent =
      right+" / "+DATA.length+" correct"+(right===DATA.length?" 🎉":". Green = actually true.");
  });
})();
</script>'''
    display(HTML(tmpl.replace("__UID__", uid).replace("__FONT__", _FONT)
                 .replace("__TITLE__", title).replace("__PROMPT__", prompt)
                 .replace("__DATA__", _json.dumps(items))))


def _nq_render(title, questions,
               prompt="Work each number out, type it in, then check."):
    """questions: list of (question_html, answer_number, tolerance, reveal)."""
    items = [{"q": q, "a": float(a), "tol": float(tol), "rev": rev}
             for q, a, tol, rev in questions]
    uid = _uid("nq", (title, tuple(q for q, _, _, _ in questions)))
    tmpl = r'''
<style>
#__UID__{font-family:__FONT__;border:1px solid #e6e8ee;border-radius:14px;padding:16px;max-width:800px;background:#fff;color:#24262b}
#__UID__ .nq-head{font-weight:800;font-size:15px;margin-bottom:4px}
#__UID__ .nq-sub{color:#666;font-size:12.5px;margin-bottom:12px}
#__UID__ .nq-row{border:1px solid #e2e5ef;border-radius:10px;padding:10px 12px;margin-bottom:8px;font-size:13.5px;line-height:1.55}
#__UID__ .nq-row.ok{border-color:#46b46e;background:#e7f7ec}
#__UID__ .nq-row.no{border-color:#e07a7a;background:#fdecec}
#__UID__ input{width:110px;padding:5px 8px;border:1px solid #c2c7da;border-radius:7px;font-size:13px;margin-top:6px;background:#fff;color:#24262b}
#__UID__ .nq-rev{font-size:12.5px;color:#3b2d6b;margin-top:6px;display:none}
#__UID__ .nq-btn{cursor:pointer;border:none;border-radius:8px;padding:9px 18px;font-size:13.5px;font-weight:700;color:#fff;background:linear-gradient(135deg,#667eea,#764ba2);margin-top:6px}
#__UID__ .nq-status{font-size:13px;font-weight:700;color:#3b2d6b;margin-top:10px;min-height:18px}
</style>
<div id="__UID__">
  <div class="nq-head">__TITLE__</div>
  <div class="nq-sub">__PROMPT__</div>
  <div class="nq-list"></div>
  <button class="nq-btn">Check</button>
  <div class="nq-status"></div>
</div>
<script>
(function(){
  const D=__DATA__, root=document.getElementById("__UID__"), list=root.querySelector(".nq-list");
  D.forEach((d,i)=>{
    const row=document.createElement("div"); row.className="nq-row"; row.dataset.i=i;
    row.innerHTML=d.q+'<br><input type="text" placeholder="your answer">'
      +'<div class="nq-rev">'+d.rev+'</div>';
    list.appendChild(row);
  });
  root.querySelector(".nq-btn").addEventListener("click",()=>{
    let right=0;
    list.querySelectorAll(".nq-row").forEach(r=>{
      const d=D[+r.dataset.i], v=parseFloat(r.querySelector("input").value.replace(",","."));
      r.classList.remove("ok","no");
      const good=!isNaN(v)&&Math.abs(v-d.a)<=d.tol;
      r.classList.add(good?"ok":"no"); if(good)right++;
      r.querySelector(".nq-rev").style.display=good?"none":"block";
    });
    root.querySelector(".nq-status").textContent=
      right+" / "+D.length+" correct"+(right===D.length?" 🎉":". The hints under the red ones should help.");
  });
})();
</script>'''
    display(HTML(tmpl.replace("__UID__", uid).replace("__FONT__", _FONT)
                 .replace("__TITLE__", title).replace("__PROMPT__", prompt)
                 .replace("__DATA__", _json.dumps(items))))


# ===========================================================================
#  §4  The timeline: the picture the whole notebook is built around.
#      Every job is a rectangle: as many rows tall as it asked for nodes,
#      as many columns wide as it asked for hours.
# ===========================================================================
_ROW_H = 38
_GUTTER = 92


def _lanes(schedule, node_names):
    """Where inside each node row does every job sit?

    Several jobs can share a node, so a node row is `cores` slots tall and each
    job takes as many slots as it asked for cores. This walks the schedule in
    start order and gives every job the lowest band of slots that is free on
    all of its nodes for its whole run. Whole-node jobs always land on band 0
    and fill the row, so Parts 1 to 5 look exactly as they did before.
    """
    total = NODE_CORES
    taken = dict((n, []) for n in node_names)      # (lo, hi, start, end) per node
    out = {}
    for s in sorted(schedule, key=lambda x: (x["start"], x["name"])):
        need = min(total, max(1, s["cores"]))
        nodes = [n for n in s["nodes"] if n in taken]
        lo = 0
        while lo + need <= total:
            clash = any(not (b[3] <= s["start"] or b[2] >= s["end"])
                        and not (b[1] <= lo or b[0] >= lo + need)
                        for n in nodes for b in taken[n])
            if not clash:
                break
            lo += 1
        for n in nodes:
            taken[n].append((lo, lo + need, s["start"], s["end"]))
        out[s["name"]] = (lo / total, need / total)
    return out


def _runs(indices):
    """[0,1,2,4] -> [(0,3), (4,1)]: consecutive rows become one rectangle."""
    out, i = [], 0
    idx = sorted(indices)
    while i < len(idx):
        j = i
        while j + 1 < len(idx) and idx[j + 1] == idx[j] + 1:
            j += 1
        out.append((idx[i], idx[j] - idx[i] + 1))
        i = j + 1
    return out


def timeline_html(schedule, hours=None, px=54, title=None, note=None,
                  reservation=None, show_declared=False, dim=(), shade_idle=False,
                  node_names=None, overlays=(), ghosts=()):
    """One Gantt chart as an HTML string (so other widgets can embed it)."""
    names = node_names or [n["name"] for n in NODE_SPECS]
    gpu_of = dict((n["name"], n["gpus"]) for n in NODE_SPECS)
    row_of = dict((n, i) for i, n in enumerate(names))
    if hours is None:
        hours = max([s["end"] for s in schedule] + [1])
        if show_declared:
            hours = max(hours, max(s["start"] + s["requested_time"] for s in schedule))
        if reservation:
            hours = max(hours, reservation[1] + 1)
    W = int(hours * px)
    H = len(names) * _ROW_H

    parts = []
    if title:
        parts.append('<div style="font-weight:800;font-size:15px;color:%s;'
                     'margin-bottom:8px">%s</div>' % (INK, title))

    # --- hour ruler --------------------------------------------------------
    stride = 1
    while stride * px < 26:
        stride += 1
    ruler = ""
    for h in range(0, int(hours) + 1, stride):
        ruler += ('<div style="position:absolute;left:%dpx;top:0;font-size:11px;'
                  'color:#8a8fa3;transform:translateX(-50%%)">%dh</div>'
                  % (h * px, h))
    parts.append('<div style="position:relative;margin-left:%dpx;height:16px;'
                 'width:%dpx">%s</div>' % (_GUTTER, W, ruler))

    # --- the grid ----------------------------------------------------------
    rows = ""
    for i, n in enumerate(names):
        badge = ('<span style="background:#efe9fb;color:%s;border-radius:4px;'
                 'padding:0 4px;font-size:9.5px;margin-left:4px">%dgpu</span>'
                 % (ACCENT, gpu_of.get(n, 0))) if gpu_of.get(n, 0) else ""
        rows += ('<div style="position:absolute;left:0;top:%dpx;width:%dpx;'
                 'height:%dpx;line-height:%dpx;font-size:11.5px;color:#555;'
                 'font-weight:600">%s%s</div>'
                 % (i * _ROW_H, _GUTTER - 8, _ROW_H, _ROW_H, n, badge))
        rows += ('<div style="position:absolute;left:%dpx;top:%dpx;width:%dpx;'
                 'height:%dpx;background:%s;border-top:1px solid #eceef5"></div>'
                 % (_GUTTER, i * _ROW_H, W, _ROW_H,
                    "#fdf6f4" if shade_idle else "#f7f8fc"))
    grid = ""
    for h in range(0, int(hours) + 1, stride):
        grid += ('<div style="position:absolute;left:%dpx;top:0;width:1px;'
                 'height:%dpx;background:%s"></div>'
                 % (_GUTTER + h * px, H, "#e2e5ef" if h else "#c2c7da"))

    # --- one rectangle per job (merged over consecutive node rows) ---------
    lanes = _lanes(schedule, names)
    blocks = ""
    for s in schedule:
        col = color_of(s["name"])
        faded = s["name"] in dim
        idxs = [row_of[n] for n in s["nodes"] if n in row_of]
        band_lo, band_h = lanes.get(s["name"], (0.0, 1.0))
        # A job holding only part of a node sits in the same band on each of its
        # nodes, which is not one rectangle, so only whole-node jobs are merged.
        pieces = _runs(idxs) if band_h >= 1.0 else [(r, 1) for r in sorted(idxs)]
        for (r0, span) in pieces:
            x = _GUTTER + s["start"] * px
            w = max(4, (s["end"] - s["start"]) * px)
            top = (r0 + band_lo) * _ROW_H + 2
            hgt = max(7, (span - 1 + band_h) * _ROW_H - 4)
            roomy = w > 34 and hgt > 26
            sub = ("<div style='font-weight:500;font-size:10px;opacity:.85'>%dn · %dh</div>"
                   % (s["n_nodes"], s["end"] - s["start"])) if roomy else ""
            first = (r0, span) == pieces[0]
            name = s["name"] if (w > 24 and hgt > 12 and first) else ""
            extra = ""
            # A killed job is outlined in red rather than labelled: the picture
            # stays readable at any size, and the tooltip says what happened.
            ring = "inset 0 0 0 1px rgba(255,255,255,.45)"
            if s.get("killed"):
                ring = "inset 0 0 0 2px " + RED
            elif s.get("backfilled"):
                extra = ('<div style="position:absolute;right:2px;top:0;'
                         'font-size:%dpx">⤴</div>' % (11 if roomy else 9))
            blocks += (
                '<div title="%s: %d node(s) × %d cores on %s, %dh→%dh (declared %dh)%s" '
                'style="position:absolute;left:%dpx;top:%dpx;'
                'width:%dpx;height:%dpx;background:%s;opacity:%s;border-radius:%dpx;'
                'box-shadow:%s;color:#fff;'
                'display:flex;flex-direction:column;align-items:center;'
                'justify-content:center;font-size:%dpx;font-weight:800;line-height:1.1;'
                'overflow:hidden">%s<div>%s</div>%s</div>'
                % (s["name"], s["n_nodes"], s["cores"], "+".join(s["nodes"]),
                   s["start"], s["end"], s["requested_time"],
                   " - killed at its declared limit" if s.get("killed") else "",
                   x, top, w - 3, hgt, col,
                   "0.28" if faded else "1", 7 if hgt > 20 else 4, ring,
                   12 if hgt > 26 else 9, extra, name, sub))
            if show_declared and s["start"] + s["requested_time"] > s["end"]:
                dx = _GUTTER + s["end"] * px
                dw = (s["start"] + s["requested_time"] - s["end"]) * px
                blocks += (
                    '<div title="%s: declared %dh, ran %dh" style="position:absolute;'
                    'left:%dpx;top:%dpx;width:%dpx;height:%dpx;border:2px dashed %s;'
                    'border-left:none;border-radius:0 6px 6px 0;color:%s;font-size:9px;'
                    'display:flex;align-items:center;justify-content:center;'
                    'background:repeating-linear-gradient(45deg,rgba(0,0,0,.05) 0 5px,'
                    'transparent 5px 10px)">%s</div>'
                    % (s["name"], s["requested_time"], s["end"] - s["start"],
                       dx, top, max(3, dw - 3), hgt, col, col,
                       "declared" if (dw > 60 and hgt > 20) else ""))

    # --- highlighted windows, and candidate rectangles being tested -------
    for o in overlays:
        r0, span = min(o["rows"]), len(o["rows"])
        blocks += (
            '<div style="position:absolute;left:%dpx;top:%dpx;width:%dpx;height:%dpx;'
            'border:2px dashed %s;border-radius:8px;background:%s;color:%s;font-size:11px;'
            'font-weight:800;display:flex;align-items:center;justify-content:center;'
            'text-align:center;line-height:1.35">%s</div>'
            % (_GUTTER + o["t0"] * px + 2, r0 * _ROW_H + 3,
               (o["t1"] - o["t0"]) * px - 6, span * _ROW_H - 7,
               o.get("color", AMBER), o.get("fill", "rgba(224,165,0,.10)"),
               o.get("color", AMBER), o.get("label", "")))
    for g in ghosts:
        r0, span = min(g["rows"]), len(g["rows"])
        col = color_of(g["name"])
        mark = {"yes": "✔", "no": "✘"}.get(g.get("verdict", ""), "?")
        mcol = GREEN if g.get("verdict") == "yes" else RED if g.get("verdict") == "no" else GREY
        blocks += (
            '<div style="position:absolute;left:%dpx;top:%dpx;width:%dpx;height:%dpx;'
            'border:2px dashed %s;border-radius:8px;background:%s22;color:%s;font-size:12px;'
            'font-weight:800;display:flex;flex-direction:column;align-items:center;'
            'justify-content:center;opacity:.95">%s'
            '<div style="font-weight:600;font-size:10px">%s</div></div>'
            '<div style="position:absolute;left:%dpx;top:%dpx;font-size:15px;color:%s;'
            'font-weight:800">%s</div>'
            % (_GUTTER + g["start"] * px + 2, r0 * _ROW_H + 3,
               g["hours"] * px - 6, span * _ROW_H - 7, col, col, col,
               g["name"], g.get("sub", ""),
               _GUTTER + (g["start"] + g["hours"]) * px + 4, r0 * _ROW_H + 6, mcol, mark))

    # --- the reservation marker -------------------------------------------
    if reservation:
        rjob, rt = reservation
        blocks += (
            '<div style="position:absolute;left:%dpx;top:-4px;width:3px;height:%dpx;'
            'background:%s"></div>'
            '<div style="position:absolute;left:%dpx;top:%dpx;font-size:11px;'
            'font-weight:800;color:%s;background:#fff;border:1px solid %s;'
            'border-radius:6px;padding:2px 6px;white-space:nowrap">🔒 reserved for %s at %dh</div>'
            % (_GUTTER + rt * px, H + 8, RED,
               _GUTTER + rt * px + 6, H + 10, RED, RED, rjob, rt))

    pad = 34 if reservation else 6
    parts.append('<div style="position:relative;width:%dpx;height:%dpx;'
                 'margin-bottom:%dpx">%s%s%s</div>'
                 % (_GUTTER + W + 10, H + pad, 4, rows, grid, blocks))
    if note:
        parts.append('<div style="font-size:12.5px;color:#444;line-height:1.6;'
                     'background:#f6f7fb;border-radius:8px;padding:9px 12px;'
                     'margin-top:6px;max-width:%dpx">%s</div>' % (_GUTTER + W, note))
    return ('<div style="font-family:%s;border:1px solid #e6e8ee;border-radius:14px;'
            'padding:16px;background:#fff;color:%s;overflow-x:auto;display:inline-block">'
            '%s</div>' % (_FONT, TEXT, "".join(parts)))


def timeline(schedule, **kw):
    """Draw a schedule. Call it after every step: that is what it is for."""
    display(HTML(timeline_html(schedule, **kw)))


def metric_strip(m, keys=None, title=None):
    """The four numbers of Part 1, side by side."""
    keys = keys or ["makespan", "avg_waiting", "avg_turnaround", "utilisation"]
    label = {"makespan": "makespan", "avg_waiting": "avg waiting",
             "avg_turnaround": "avg turnaround", "utilisation": "utilisation",
             "idle_node_hours": "idle node-hours", "busy_node_hours": "busy node-hours",
             "gpu_utilisation": "GPU utilisation", "fairness": "share equity",
             "throughput": "throughput", "max_waiting": "worst waiting",
             "killed": "jobs killed", "worst_group_waiting": "worst group waits"}
    who = {"makespan": "cluster", "utilisation": "cluster", "idle_node_hours": "cluster",
           "busy_node_hours": "cluster", "gpu_utilisation": "cluster",
           "throughput": "cluster",
           "avg_waiting": "user", "avg_turnaround": "user", "max_waiting": "user",
           "fairness": "groups", "worst_group_waiting": "groups", "killed": "user"}
    cells = ""
    for k in keys:
        v = m[k]
        if k in ("utilisation", "gpu_utilisation", "fairness"):
            txt = "%.1f%%" % (100 * v)
        elif isinstance(v, float):
            txt = "%.2f h" % v if "wait" in k or "turn" in k else "%.2f" % v
        else:
            txt = "%d h" % v if k == "makespan" else str(v)
        tag = who.get(k, "")
        col = ACCENT if tag == "cluster" else GREEN if tag == "user" else AMBER
        cells += ('<div style="flex:1;min-width:120px;border:1px solid #e6e8ee;'
                  'border-top:3px solid %s;border-radius:10px;padding:10px 12px">'
                  '<div style="font-size:10.5px;color:#8a8fa3;text-transform:uppercase;'
                  'letter-spacing:.04em">%s</div>'
                  '<div style="font-size:20px;font-weight:800;color:%s;margin-top:2px">%s</div>'
                  '<div style="font-size:10.5px;color:%s;font-weight:600">%s</div></div>'
                  % (col, label.get(k, k), INK, txt, col, tag))
    head = ('<div style="font-weight:800;font-size:14px;color:%s;margin-bottom:8px">%s</div>'
            % (INK, title)) if title else ""
    display(HTML('<div style="font-family:%s;max-width:900px;background:#fff;color:%s;'
                 'border:1px solid #e6e8ee;border-radius:14px;padding:14px">%s'
                 '<div style="display:flex;gap:10px;flex-wrap:wrap">%s</div></div>'
                 % (_FONT, TEXT, head, cells)))


# ===========================================================================
#  §4b  The opening — what a node is made of, and where the hour is going
# ===========================================================================
# Nothing in this notebook assumes "core", "memory" or "GPU" is already a
# familiar word. This is where each one is defined, once, in plain language.
_NODE_PARTS = {
    "cores": {
        "col": "#667eea",
        "title": "A core is one worker",
        "body":
            "It follows <b>one stream of instructions at a time</b>. Eight cores means this "
            "computer can genuinely do eight things at once &mdash; not take turns very quickly, "
            "actually at the same time."
            "<br><br>When your job asks for 4 cores, four of these workers are handed to you and "
            "<b>nobody else can use them</b> until your job finishes."
            "<br><br><span style='color:#8a8fa3'>For scale: a laptop has roughly 4 to 10. Each "
            "node here has 8, and there are four nodes &mdash; 32 in the whole cluster.</span>"},
    "mem": {
        "col": "#2e9e7a",
        "title": "Memory is the desk you spread the work out on",
        "body":
            "Everything the job is actively using has to fit on it <b>at the same time</b> &mdash; the "
            "data, the model, the intermediate results."
            "<br><br>Ask for 32 GB and 32 GB is set aside for you alone. Ask for <b>too little</b> "
            "and the job is killed the moment the data no longer fits: no warning, no partial "
            "result, the hours already spent are gone."
            "<br><br><span style='color:#8a8fa3'>For scale: a laptop has 8 to 32 GB. Each node "
            "here has 64 GB.</span>"},
    "gpus": {
        "col": "#c9548f",
        "title": "A GPU is a specialist chip",
        "body":
            "A core does one thing after another, very fast. A GPU does <b>thousands of small "
            "identical calculations simultaneously</b> &mdash; which is precisely what training a "
            "neural network is made of. The same work can run tens of times faster on one."
            "<br><br>They are the <b>scarcest thing</b> in any cluster, and a job that asks for a "
            "GPU can only run on a node that has a free one."
            "<br><br><span style='color:#8a8fa3'>Here: node01 and node02 have none at all. node03 "
            "and node04 have two each &mdash; 4 in the whole cluster.</span>"},
}


def what_is_a_node():
    """One node, opened up: cores, memory and GPUs, defined in plain language.

    All four nodes are selectable, so that "only some nodes have a GPU" is
    something you see happen rather than something you are told.
    """
    data = {"parts": _NODE_PARTS, "nodes": NODE_SPECS}
    uid = _uid("wn", "what_is_a_node")
    tmpl = r"""
<style>
#__UID__{font-family:__FONT__;border:1px solid #e6e8ee;border-radius:14px;padding:18px;max-width:900px;background:#fff;color:#24262b}
#__UID__ .wn-head{font-weight:800;font-size:15px;color:#2b2d6b}
#__UID__ .wn-sub{font-size:12.5px;color:#666;margin:4px 0 14px;line-height:1.55}
#__UID__ .wn-wrap{display:flex;gap:14px;flex-wrap:wrap;align-items:stretch}
#__UID__ .wn-node{flex:1 1 300px;border:1px solid #e2e5ef;border-radius:12px;padding:12px 13px}
#__UID__ .wn-nname{font-weight:800;font-size:12.5px;color:#2b2d6b;display:flex;justify-content:space-between;align-items:center;margin-bottom:9px}
#__UID__ .wn-tag{font-size:9.5px;background:#f0f1f5;color:#8a8fa3;border-radius:4px;padding:2px 6px;font-weight:700;letter-spacing:.03em}
#__UID__ .wn-row{border:1.5px solid #e8eaf2;border-radius:9px;padding:8px 9px;margin-bottom:6px;cursor:pointer;transition:.13s;background:#fff}
#__UID__ .wn-row:hover{border-color:#c3b0e0;background:#faf9ff;box-shadow:0 1px 5px rgba(118,75,162,.10)}
#__UID__ .wn-row.on{border-color:#764ba2;background:#f7f3ff}
#__UID__ .wn-lab{font-size:10.5px;font-weight:800;letter-spacing:.05em;text-transform:uppercase;margin-bottom:6px;display:flex;align-items:center;gap:7px}
#__UID__ .wn-dot{width:13px;height:13px;border-radius:50%;border:2px solid #c2c7da;flex:0 0 auto;transition:.13s;box-sizing:border-box}
#__UID__ .wn-row:hover .wn-dot{border-color:#a98fd0}
#__UID__ .wn-amt{margin-left:auto}
#__UID__ .wn-hint{font-size:10px;font-weight:700;color:#a3a8ba;letter-spacing:.03em;text-transform:none}
#__UID__ .wn-row.on .wn-hint{display:none}
#__UID__ .wn-cells{display:flex;gap:3px}
#__UID__ .wn-c{height:19px;border-radius:4px;flex:1 1 0}
#__UID__ .wn-none{height:19px;border:1.5px dashed #dcdfe9;border-radius:4px;color:#a3a8ba;font-size:10px;font-weight:700;letter-spacing:.04em;display:flex;align-items:center;justify-content:center;width:100%}
#__UID__ .wn-panel{flex:1 1 320px;background:#f7f8fc;border-radius:12px;padding:14px 15px;font-size:12.5px;line-height:1.62;color:#333}
#__UID__ .wn-pt{font-weight:800;font-size:13.5px;margin-bottom:7px;line-height:1.4}
#__UID__ .wn-pick{font-size:11.5px;color:#666;margin:13px 0 6px}
#__UID__ .wn-mini{display:flex;gap:6px;flex-wrap:wrap}
#__UID__ .wn-m{border:1.5px solid #e2e5ef;border-radius:8px;padding:6px 10px;font-size:11px;color:#666;cursor:pointer;transition:.13s}
#__UID__ .wn-m:hover{background:#faf9ff}
#__UID__ .wn-m.on{border-color:#764ba2;background:#faf9ff;color:#2b2d6b;font-weight:700}
#__UID__ .wn-foot{margin-top:13px;font-size:12.5px;color:#333;background:#f3f0ff;border-radius:9px;padding:11px 13px;line-height:1.6}
</style>
<div id="__UID__">
  <div class="wn-head">&#128269; What is actually inside one node</div>
  <div class="wn-sub">A <b>node</b> is one computer in the rack. Your cluster has four of them.
    Click any of the <b>three coloured rows</b> to find out what it is &mdash; then click a
    different node underneath to see how the four differ.</div>
  <div class="wn-wrap">
    <div class="wn-node">
      <div class="wn-nname"><span class="wn-nt"></span><span class="wn-tag">ONE NODE</span></div>
      <div class="wn-rows"></div>
    </div>
    <div class="wn-panel"></div>
  </div>
  <div class="wn-pick">Your cluster is <b>four</b> of these &mdash; click one:</div>
  <div class="wn-mini"></div>
  <div class="wn-foot">Every job in this notebook is a request for <b>some cores</b>,
    <b>some memory</b>, <b>sometimes a GPU</b> &mdash; and <b>how long</b> it needs them.
    Those are the only four numbers in the whole hour.</div>
</div>
<script>
(function(){
  const D=__DATA__, root=document.getElementById("__UID__");
  let node=2, part="cores";
  const mini=root.querySelector(".wn-mini");
  D.nodes.forEach((n,i)=>{
    const el=document.createElement("span");
    el.className="wn-m"; el.dataset.i=i;
    el.innerHTML=n.name+" &middot; "+n.cores+" cores &middot; "+n.mem+" GB &middot; "
      +(n.gpus?'<b style="color:#c9548f">'+n.gpus+" GPU</b>":'<b style="color:#9aa0b5">no GPU</b>');
    el.addEventListener("click",()=>{node=i;draw();});
    mini.appendChild(el);
  });
  function rowHTML(key,label,amount,count,col){
    let cells;
    if(count>0){
      cells="";
      for(let i=0;i<count;i++)
        cells+='<span class="wn-c" style="background:'+col+'"></span>';
    } else {
      cells='<div class="wn-none">this node has no GPU</div>';
    }
    const on=part===key;
    const dot='<span class="wn-dot"'+(on?' style="background:'+col+';border-color:'+col+'"':'')+'></span>';
    return '<div class="wn-row'+(on?" on":"")+'" data-k="'+key+'">'
      +'<div class="wn-lab" style="color:'+(count>0?col:"#a3a8ba")+'">'
      +dot+'<span>'+label+'</span>'
      +'<span class="wn-hint">click to see what this is</span>'
      +'<span class="wn-amt">'+amount+'</span></div>'
      +'<div class="wn-cells">'+cells+'</div></div>';
  }
  function draw(){
    const n=D.nodes[node];
    root.querySelector(".wn-nt").textContent=n.name;
    root.querySelector(".wn-rows").innerHTML=
        rowHTML("cores","cores",n.cores,n.cores,"#667eea")
      + rowHTML("mem","memory",n.mem+" GB",8,"#2e9e7a")
      + rowHTML("gpus","GPUs",n.gpus?n.gpus:"none",n.gpus,"#c9548f");
    root.querySelectorAll(".wn-row").forEach(r=>
      r.addEventListener("click",()=>{part=r.dataset.k;draw();}));
    root.querySelectorAll(".wn-m").forEach((m,i)=>m.classList.toggle("on",i===node));
    const d=D.parts[part];
    root.querySelector(".wn-panel").innerHTML=
      '<div class="wn-pt" style="color:'+d.col+'">'+d.title+'</div>'+d.body;
  }
  draw();
})();
</script>"""
    display(HTML(tmpl.replace("__UID__", uid).replace("__FONT__", _FONT)
                 .replace("__DATA__", _json.dumps(data))))


# ===========================================================================
#  §5  Part 0: the cluster map, and packing it by hand
# ===========================================================================
# What is already running when you take over the cluster on Monday morning.
RUNNING_NOW = [
    {"name": "A", "node": "node01", "cores": 4, "mem": 32, "gpus": 0, "group": "bio"},
    {"name": "B", "node": "node03", "cores": 6, "mem": 48, "gpus": 2, "group": "clim"},
]
# What five researchers have just asked for.
QUEUE_NOW = [
    {"name": "Q1", "nodes": 1, "cores": 4, "mem": 32, "gpus": 0, "group": "bio",
     "why": "one MRI volume, no GPU"},
    {"name": "Q2", "nodes": 2, "cores": 8, "mem": 64, "gpus": 0, "group": "clim",
     "why": "two whole nodes, an ocean grid split in half"},
    {"name": "Q3", "nodes": 1, "cores": 4, "mem": 32, "gpus": 2, "group": "robo",
     "why": "policy training, needs both GPUs of one node"},
    {"name": "Q4", "nodes": 3, "cores": 8, "mem": 64, "gpus": 0, "group": "econ",
     "why": "three whole nodes, a big Monte-Carlo sweep"},
    {"name": "Q5", "nodes": 1, "cores": 2, "mem": 16, "gpus": 0, "group": "econ",
     "why": "a small calibration run"},
]


def cluster_map(interactive=True):
    """The cluster as it is right now: what each node has, and what is left.

    Click a queued job to hand it its resources. Everything it is given is
    taken out of the map immediately and stays out until the job finishes.
    """
    data = {"nodes": NODE_SPECS, "running": RUNNING_NOW, "queue": QUEUE_NOW,
            "interactive": bool(interactive),
            "gcol": GROUP_COLOR, "jcol": dict((q["name"], color_of(q["name"]))
                                              for q in QUEUE_NOW)}
    uid = _uid("cm", "cluster_map")
    tmpl = r'''
<style>
#__UID__{font-family:__FONT__;border:1px solid #e6e8ee;border-radius:14px;padding:16px;max-width:900px;background:#fff;color:#24262b}
#__UID__ .cm-head{font-weight:800;font-size:15px;color:#2b2d6b}
#__UID__ .cm-sub{font-size:12.5px;color:#666;margin:3px 0 13px;line-height:1.55}
#__UID__ .cm-grid{display:flex;gap:10px;flex-wrap:wrap;margin-bottom:14px}
#__UID__ .cm-node{flex:1;min-width:190px;border:1px solid #e2e5ef;border-radius:11px;padding:10px 12px}
#__UID__ .cm-nname{font-weight:800;font-size:13px;color:#2b2d6b;display:flex;justify-content:space-between;align-items:center}
#__UID__ .cm-badge{font-size:9.5px;background:#efe9fb;color:#764ba2;border-radius:4px;padding:1px 5px;font-weight:700}
#__UID__ .cm-res{margin-top:8px;font-size:11px;color:#666}
#__UID__ .cm-bar{height:15px;border-radius:5px;background:#eef0f6;position:relative;overflow:hidden;margin:2px 0 6px;display:flex}
#__UID__ .cm-seg{height:100%;border-right:1px solid #fff;transition:width .18s}
#__UID__ .cm-free{font-size:10.5px;color:#2e9e7a;font-weight:700}
#__UID__ .cm-qhead{font-weight:800;font-size:13px;color:#2b2d6b;margin-bottom:6px}
#__UID__ .cm-job{display:flex;align-items:center;gap:10px;border:1px solid #e2e5ef;border-radius:10px;padding:8px 11px;margin-bottom:7px;font-size:12.5px;cursor:pointer;transition:.12s}
#__UID__ .cm-job:hover{border-color:#764ba2;background:#faf9ff}
#__UID__ .cm-job.no{opacity:.55;cursor:not-allowed;background:#fdf3f3;border-color:#f0d6d6}
#__UID__ .cm-job.placed{opacity:.45;background:#f3f7f4;border-color:#cfe4d6;cursor:default}
#__UID__ .cm-chip{width:22px;height:22px;border-radius:6px;flex:0 0 auto}
#__UID__ .cm-fit{margin-left:auto;font-size:11px;font-weight:800}
#__UID__ .cm-btn{cursor:pointer;border:none;border-radius:8px;padding:8px 16px;font-size:13px;font-weight:700;color:#fff;background:linear-gradient(135deg,#667eea,#764ba2)}
#__UID__ .cm-note{background:#f6f7fb;border-radius:8px;padding:10px 12px;font-size:12.5px;color:#333;line-height:1.6;margin-top:10px}
</style>
<div id="__UID__">
  <div class="cm-head">🖥️ Your cluster, right now</div>
  <div class="cm-sub">Four nodes. Whatever a job is allocated, it <b>keeps</b> until it finishes:
    the bar below is not a load average, it is ownership. A node can hold several jobs at once,
    as long as its cores, memory and GPUs are enough for all of them.<br>
    <span style="color:#8a8fa3">In every bar the coloured part is already handed out and the grey
    part is free. Two jobs are running: <b style="color:#4a5bd0">A</b> (Biomedical Imaging, on
    node01) and <b style="color:#2e9e7a">B</b> (Climate, on node03).</span></div>
  <div class="cm-grid"></div>
  <div class="cm-qhead">📥 Jobs waiting in the queue: click one to start it now</div>
  <div class="cm-list"></div>
  <div class="cm-note"></div>
  <button class="cm-btn" style="margin-top:10px">↺ Give everything back</button>
</div>
<script>
(function(){
  const D=__DATA__, root=document.getElementById("__UID__");
  let placed={};
  function freeOf(){
    const f={};
    D.nodes.forEach(n=>f[n.name]={cores:n.cores,mem:n.mem,gpus:n.gpus,jobs:[]});
    D.running.forEach(r=>{const x=f[r.node];x.cores-=r.cores;x.mem-=r.mem;x.gpus-=r.gpus;
      x.jobs.push({name:r.name,cores:r.cores,mem:r.mem,gpus:r.gpus,col:D.gcol[r.group]});});
    Object.keys(placed).forEach(k=>{
      const q=D.queue.find(j=>j.name===k);
      placed[k].forEach(nm=>{const x=f[nm];x.cores-=q.cores;x.mem-=q.mem;x.gpus-=q.gpus;
        x.jobs.push({name:q.name,cores:q.cores,mem:q.mem,gpus:q.gpus,col:D.jcol[q.name]});});
    });
    return f;
  }
  function ask(q){
    // Spelled out rather than "2 nodes x 4 cores": cores on different nodes
    // cannot share memory, so where they sit is part of the request.
    const per=q.cores+' CPU core'+(q.cores>1?'s':'')+' and '+q.mem+' GB of memory (RAM)'
      +(q.gpus?' and '+q.gpus+' GPU'+(q.gpus>1?'s':''):'');
    return q.nodes>1 ? '<b>'+q.nodes+' nodes</b>, each with '+per : per;
  }
  function fitNodes(q,f){
    const ok=D.nodes.filter(n=>f[n.name].cores>=q.cores&&f[n.name].mem>=q.mem&&f[n.name].gpus>=q.gpus)
      .sort((a,b)=>(a.gpus-b.gpus)||(a.name<b.name?-1:1));
    return ok.length>=q.nodes?ok.slice(0,q.nodes).map(n=>n.name):null;
  }
  function draw(){
    const f=freeOf(), grid=root.querySelector(".cm-grid"); grid.innerHTML="";
    let idleCores=0, totCores=0;
    D.nodes.forEach(n=>{
      const x=f[n.name]; idleCores+=x.cores; totCores+=n.cores;
      let segs="";
      x.jobs.forEach(j=>{segs+='<span class="cm-seg" title="'+j.name+': '+j.cores+' cores" style="width:'
        +(100*j.cores/n.cores)+'%;background:'+j.col+'"></span>';});
      let msegs="";
      x.jobs.forEach(j=>{msegs+='<span class="cm-seg" style="width:'+(100*j.mem/n.mem)+'%;background:'+j.col+'"></span>';});
      let gsegs="";
      if(n.gpus>0){x.jobs.forEach(j=>{if(j.gpus)gsegs+='<span class="cm-seg" style="width:'
        +(100*j.gpus/n.gpus)+'%;background:'+j.col+'"></span>';});}
      grid.insertAdjacentHTML("beforeend",
        '<div class="cm-node"><div class="cm-nname">'+n.name
        +(n.gpus?'<span class="cm-badge">'+n.gpus+' GPU</span>':'<span class="cm-badge" style="background:#f0f1f5;color:#8a8fa3">CPU only</span>')
        +'</div>'
        +'<div class="cm-res">cores <span class="cm-free">'+x.cores+' / '+n.cores+' free</span></div>'
        +'<div class="cm-bar">'+segs+'</div>'
        +'<div class="cm-res">memory <span class="cm-free">'+x.mem+' / '+n.mem+' GB free</span></div>'
        +'<div class="cm-bar">'+msegs+'</div>'
        +(n.gpus?'<div class="cm-res">GPUs <span class="cm-free">'+x.gpus+' / '+n.gpus+' free</span></div>'
          +'<div class="cm-bar">'+gsegs+'</div>':'')
        +'</div>');
    });
    const list=root.querySelector(".cm-list"); list.innerHTML="";
    let anyFit=false, blocked=[];
    D.queue.forEach(q=>{
      const done=!!placed[q.name], nodes=done?null:fitNodes(q,f);
      if(!done&&nodes)anyFit=true; if(!done&&!nodes)blocked.push(q.name);
      const cls=done?"placed":(nodes?"":"no");
      const verdict=done?'<span class="cm-fit" style="color:#2e9e7a">running on '+placed[q.name].join(", ")+'</span>'
        :(nodes?'<span class="cm-fit" style="color:#2e9e7a">fits ✔</span>'
               :'<span class="cm-fit" style="color:#c0554e">does not fit ✘</span>');
      const row=document.createElement("div"); row.className="cm-job "+cls; row.dataset.n=q.name;
      row.innerHTML='<span class="cm-chip" style="background:'+D.jcol[q.name]+'"></span>'
        +'<span><b>'+q.name+'</b> &middot; '+ask(q)
        +'<br><span style="color:#8a8fa3;font-size:11px">'+q.why+'</span></span>'+verdict;
      if(D.interactive&&!done&&nodes)row.addEventListener("click",()=>{placed[q.name]=nodes;draw();});
      list.appendChild(row);
    });
    let msg;
    if(Object.keys(placed).length===0)
      msg="Two jobs are already running, so parts of node01 and node03 are gone. Read the queue: "
        +(D.queue.length-blocked.length)+" of these "+D.queue.length+" fit right now, "
        +blocked.length+" ("+blocked.join(", ")+") does not. Click one and watch the map lose "
        +"exactly the cores, memory and GPUs it asked for.";
    else if(!anyFit)
      msg="<b>Nothing else fits.</b> "+blocked.length+" job"+(blocked.length>1?"s":"")+" ("
        +blocked.join(", ")+") cannot start"
        +(idleCores>0
          ? " even though <b>"+idleCores+" of "+totCores+" cores are still idle</b>: what is free "
            +"is scattered across nodes in the wrong shapes. <b>Every choice you made left "
            +"something unused.</b>"
          : ". This time nothing is being wasted: <b>every core in the cluster is working</b>. "
            +"The jobs left over simply have to wait their turn, which is not the same thing as "
            +"the cluster sitting idle.")
        +" Press \u21ba and see what would have happened if you had started the jobs in a "
        +"different order.";
    else
      msg="Placed so far: <b>"+Object.keys(placed).join(", ")+"</b>. Idle cores: <b>"+idleCores
        +" / "+totCores+"</b>."+(blocked.length?" Still stuck: "+blocked.join(", ")+".":"")
        +" Keep going, and notice which job you have just made impossible.";
    root.querySelector(".cm-note").innerHTML=msg;
  }
  root.querySelector(".cm-btn").addEventListener("click",()=>{placed={};draw();});
  draw();
})();
</script>'''
    display(HTML(tmpl.replace("__UID__", uid).replace("__FONT__", _FONT)
                 .replace("__DATA__", _json.dumps(data))))


# ===========================================================================
#  §6  Part 1: the shape of a job, then place the five yourself
# ===========================================================================
def _cell_grid(px, row_h):
    """The 1 node x 1 hour cells, as a background, so a rectangle is countable."""
    return ("background-image:repeating-linear-gradient(to right,"
            "rgba(255,255,255,.38) 0 1px,transparent 1px %dpx),"
            "repeating-linear-gradient(to bottom,rgba(255,255,255,.38) 0 1px,"
            "transparent 1px %dpx)" % (px, row_h))


def _shape(job, px, row_h, sub=True):
    """One job as the rectangle it is: nodes tall, declared hours wide."""
    w = job["requested_time"] * px
    h = job["nodes"] * row_h
    inner = ('<div style="font-size:13px;font-weight:800">%s</div>' % job["name"])
    if sub and h >= 30:
        inner += ('<div style="font-size:10.5px;font-weight:600;opacity:.9">'
                  '%dn × %dh</div>' % (job["nodes"], job["requested_time"]))
    return ('<div style="width:%dpx;height:%dpx;background:%s;%s;border-radius:7px;'
            'box-shadow:inset 0 0 0 1px rgba(255,255,255,.45);color:#fff;display:flex;'
            'flex-direction:column;align-items:center;justify-content:center;'
            'line-height:1.15">%s</div>'
            % (w, h, color_of(job["name"]), _cell_grid(px, row_h), inner))


def job_shapes(jobs=None, px=44, focus="J3"):
    """The mental image the whole notebook runs on, drawn instead of described.

    Left: one job with its two dimensions quoted. Right: the whole queue at the
    same scale as `hand_scheduler`'s board, so the shapes the reader is about to
    drag are the shapes shown here.
    """
    jobs = jobs or five_jobs()
    row_h = _ROW_H
    pick = next((j for j in jobs if j["name"] == focus), jobs[0])
    fw, fh = pick["requested_time"] * px, pick["nodes"] * row_h

    # --- left panel: the rule, quoted on one rectangle ---------------------
    arrow = ('<div style="position:relative;height:14px;width:%dpx;margin-bottom:4px">'
             '<div style="position:absolute;top:6px;left:0;width:%dpx;height:2px;'
             'background:%s"></div>'
             '<div style="position:absolute;top:1px;left:-1px;color:%s;font-size:11px;'
             'line-height:1">◀</div>'
             '<div style="position:absolute;top:1px;right:-1px;color:%s;font-size:11px;'
             'line-height:1">▶</div></div>' % (fw, fw, GREY, GREY, GREY))
    side = ('<div style="position:relative;width:16px;height:%dpx;margin-right:6px">'
            '<div style="position:absolute;left:8px;top:0;width:2px;height:%dpx;'
            'background:%s"></div>'
            '<div style="position:absolute;left:3px;top:-6px;color:%s;font-size:11px">▲</div>'
            '<div style="position:absolute;left:3px;bottom:-6px;color:%s;font-size:11px">▼</div>'
            '</div>' % (fh, fh, GREY, GREY, GREY))
    left = (
        '<div>'
        '<div style="font-size:11px;color:#8a8fa3;text-transform:uppercase;'
        'letter-spacing:.04em;margin-bottom:9px">How to read a rectangle</div>'
        '<div style="margin-left:22px;font-size:11.5px;color:%s;font-weight:700">'
        '%d h wide: the time it declared</div>'
        '<div style="margin-left:22px">%s</div>'
        '<div style="display:flex;align-items:center">%s%s'
        '<div style="margin-left:9px;font-size:11.5px;color:%s;font-weight:700;'
        'white-space:nowrap;line-height:1.4">%d nodes tall: what it asked for</div></div>'
        '<div style="margin-left:22px;margin-top:9px;font-size:12px;color:%s">'
        '<b style="color:%s">%s</b> · %d nodes × %d h = <b>%d node-hours</b></div>'
        '</div>'
        % (INK, pick["requested_time"], arrow, side, _shape(pick, px, row_h, sub=False),
           INK, pick["nodes"], TEXT, color_of(pick["name"]), pick["name"],
           pick["nodes"], pick["requested_time"],
           pick["nodes"] * pick["requested_time"]))

    # --- right panel: the queue, at the board's own scale -------------------
    tallest = max(j["nodes"] for j in jobs) * row_h
    tiles = ""
    for j in jobs:
        tiles += ('<div style="display:flex;flex-direction:column;'
                  'justify-content:flex-end;height:%dpx">%s'
                  '<div style="font-size:10.5px;color:#8a8fa3;margin-top:5px;'
                  'text-align:center">%dn × %dh<br>%d node-h</div></div>'
                  % (tallest, _shape(j, px, row_h), j["nodes"], j["requested_time"],
                     j["nodes"] * j["requested_time"]))
    right = ('<div>'
             '<div style="font-size:11px;color:#8a8fa3;text-transform:uppercase;'
             'letter-spacing:.04em;margin-bottom:9px">The five jobs, to scale</div>'
             '<div style="display:flex;gap:12px;align-items:flex-end;flex-wrap:wrap">'
             '%s</div></div>' % tiles)

    total = sum(j["nodes"] * j["requested_time"] for j in jobs)
    _card(
        '<div style="font-weight:800;font-size:15px;color:%s">🧱 Every job is a rectangle</div>'
        '<div style="font-size:12.5px;color:#666;margin:3px 0 14px;line-height:1.55">'
        'As many rows <b>tall</b> as the nodes it asked for, as many columns <b>wide</b> as the '
        'hours it declared.</div>'
        '<div style="display:flex;gap:34px;flex-wrap:wrap;align-items:flex-start">%s%s</div>'
        '<div style="background:#f6f7fb;border-radius:8px;padding:10px 12px;font-size:12.5px;'
        'color:#333;line-height:1.6;margin-top:14px">One cell is <b>one node for one hour</b>. '
        'These five are <b>%d node-hours</b> of work, and the grid they have to be packed into '
        'is only <b>%d rows</b> tall, with time running to the right, and no two rectangles '
        'ever overlapping.</div>'
        % (INK, left, right, total, N_NODES), maxw=940)


def hand_scheduler(hours=13, enforce_order=True):
    """Place J1…J5 on the timeline by hand and watch the four metrics move.

    Click the job at the head of the queue, then click the cell where its
    top-left corner should go. A job occupies as many node rows as it asked
    for and as many hour columns as it declared.
    """
    jobs = [{"name": j["name"], "nodes": j["nodes"], "hours": j["requested_time"],
             "group": j["group"], "col": color_of(j["name"])} for j in five_jobs()]
    data = {"jobs": jobs, "nodes": [n["name"] for n in NODE_SPECS],
            "gpus": [n["gpus"] for n in NODE_SPECS], "hours": hours,
            "px": 44, "rowh": _ROW_H, "gutter": 96, "order": bool(enforce_order),
            "fifo": [{"name": s["name"], "start": s["start"],
                      "rows": sorted([n["name"] for n in NODE_SPECS].index(x)
                                     for x in s["nodes"])}
                     for s in run_schedule(five_jobs(), backfill=False)]}
    uid = _uid("hs", "hand_scheduler")
    tmpl = r'''
<style>
#__UID__{font-family:__FONT__;border:1px solid #e6e8ee;border-radius:14px;padding:16px;max-width:940px;background:#fff;color:#24262b}
#__UID__ .hs-head{font-weight:800;font-size:15px;color:#2b2d6b}
#__UID__ .hs-sub{font-size:12.5px;color:#666;margin:3px 0 12px;line-height:1.55}
#__UID__ .hs-q{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:12px;align-items:flex-start}
#__UID__ .hs-card{position:relative;height:52px;border:1px solid #e2e5ef;border-radius:9px;padding:6px 9px;font-size:11.5px;cursor:pointer;min-width:80px;transition:.12s}
#__UID__ .hs-card:hover{border-color:#764ba2}
#__UID__ .hs-card.sel{border-color:#764ba2;background:#f1edff;box-shadow:0 0 0 2px #e6dcff}
#__UID__ .hs-card.done{opacity:.4;cursor:default}
#__UID__ .hs-card.locked{opacity:.4;cursor:not-allowed}
#__UID__ .hs-rect{position:absolute;left:9px;bottom:7px;border-radius:3px}
#__UID__ .hs-board{position:relative;overflow-x:auto}
#__UID__ .hs-cell{position:absolute;border:1px solid #eceef5;background:#f7f8fc;cursor:pointer}
#__UID__ .hs-cell:hover{background:#efe9fb}
#__UID__ .hs-blk{position:absolute;border-radius:7px;color:#fff;font-weight:800;font-size:12.5px;display:flex;flex-direction:column;align-items:center;justify-content:center;box-shadow:inset 0 0 0 1px rgba(255,255,255,.35)}
#__UID__ .hs-m{display:flex;gap:9px;flex-wrap:wrap;margin-top:12px}
#__UID__ .hs-box{flex:1;min-width:118px;border:1px solid #e6e8ee;border-top:3px solid #764ba2;border-radius:10px;padding:8px 11px}
#__UID__ .hs-box.user{border-top-color:#2e9e7a}
#__UID__ .hs-k{font-size:10px;color:#8a8fa3;text-transform:uppercase;letter-spacing:.04em}
#__UID__ .hs-v{font-size:19px;font-weight:800;color:#2b2d6b}
#__UID__ .hs-w{font-size:10px;font-weight:700}
#__UID__ .hs-btn{cursor:pointer;border:none;border-radius:8px;padding:7px 14px;font-size:12.5px;font-weight:700;color:#fff;background:linear-gradient(135deg,#667eea,#764ba2);margin:10px 6px 0 0}
#__UID__ .hs-btn.gh{background:#fff;color:#764ba2;border:1px solid #d6cdf0}
#__UID__ .hs-note{background:#f6f7fb;border-radius:8px;padding:10px 12px;font-size:12.5px;color:#333;line-height:1.6;margin-top:11px}
</style>
<div id="__UID__">
  <div class="hs-head">🧱 Place the five rectangles</div>
  <div class="hs-sub">Every job is a <b>rectangle</b>: as many rows tall as the nodes it asked for,
    as many columns wide as the hours it declared. Pick the job at the head of the queue, then click
    the cell where its <b>top-left corner</b> goes. Rectangles may not overlap, and the four
    numbers underneath update on every placement.</div>
  <div class="hs-q"></div>
  <div class="hs-board"></div>
  <div class="hs-m"></div>
  <button class="hs-btn">↺ Start over</button>
  <button class="hs-btn gh">👀 Show what plain FIFO does</button>
  <div class="hs-note"></div>
</div>
<script>
(function(){
  const D=__DATA__, root=document.getElementById("__UID__");
  const PX=D.px, RH=D.rowh, GU=D.gutter, NR=D.nodes.length, NH=D.hours;
  let placed={}, sel=null;
  const nextJob=()=>D.jobs.find(j=>!placed[j.name]);
  function occupied(){
    const g=[]; for(let r=0;r<NR;r++){g.push(new Array(NH).fill(null));}
    Object.keys(placed).forEach(n=>{const p=placed[n], j=D.jobs.find(x=>x.name===n);
      p.rows.forEach(r=>{for(let h=p.start;h<p.start+j.hours;h++) if(h<NH) g[r][h]=n;});});
    return g;
  }
  function drawQueue(){
    const q=root.querySelector(".hs-q"); q.innerHTML="";
    D.jobs.forEach(j=>{
      const done=!!placed[j.name];
      const locked=D.order&&!done&&nextJob()&&nextJob().name!==j.name;
      const cls=done?"done":(locked?"locked":(sel===j.name?"sel":""));
      const d=document.createElement("div"); d.className="hs-card "+cls;
      d.innerHTML='<b>'+j.name+'</b> · '+j.nodes+'n × '+j.hours+'h'
        +'<div class="hs-rect" style="width:'+(j.hours*9)+'px;height:'+(j.nodes*7)+'px;background:'+j.col+'"></div>';
      if(!done&&!locked)d.addEventListener("click",()=>{sel=j.name;draw();});
      q.appendChild(d);
    });
  }
  function drawBoard(){
    const b=root.querySelector(".hs-board"); b.innerHTML="";
    b.style.height=(NR*RH+26)+"px"; b.style.width=(GU+NH*PX+14)+"px";
    for(let h=0;h<=NH;h++) b.insertAdjacentHTML("beforeend",
      '<div style="position:absolute;left:'+(GU+h*PX)+'px;top:0;font-size:10.5px;color:#8a8fa3;transform:translateX(-50%)">'+h+'h</div>');
    const g=occupied();
    for(let r=0;r<NR;r++){
      b.insertAdjacentHTML("beforeend",
        '<div style="position:absolute;left:0;top:'+(16+r*RH)+'px;width:'+(GU-8)+'px;height:'+RH
        +'px;line-height:'+RH+'px;font-size:11.5px;font-weight:600;color:#555">'+D.nodes[r]
        +(D.gpus[r]?' <span style="font-size:9px;background:#efe9fb;color:#764ba2;border-radius:4px;padding:0 3px">gpu</span>':'')+'</div>');
      for(let h=0;h<NH;h++){
        const c=document.createElement("div"); c.className="hs-cell";
        c.style.cssText+="left:"+(GU+h*PX)+"px;top:"+(16+r*RH)+"px;width:"+PX+"px;height:"+RH+"px";
        c.addEventListener("click",()=>place(r,h));
        b.appendChild(c);
      }
    }
    Object.keys(placed).forEach(n=>{
      const p=placed[n], j=D.jobs.find(x=>x.name===n);
      const r0=Math.min.apply(null,p.rows), span=p.rows.length;
      b.insertAdjacentHTML("beforeend",
        '<div class="hs-blk" style="left:'+(GU+p.start*PX+1)+'px;top:'+(17+r0*RH)+'px;width:'
        +(j.hours*PX-3)+'px;height:'+(span*RH-3)+'px;background:'+j.col+'">'+j.name
        +'<div style="font-weight:500;font-size:9.5px;opacity:.85">'+p.start+'h → '+(p.start+j.hours)+'h</div></div>');
    });
  }
  function place(r,h){
    const j=sel?D.jobs.find(x=>x.name===sel):nextJob();
    if(!j||placed[j.name]){msg("Pick a job from the queue first.");return;}
    if(D.order&&nextJob().name!==j.name){msg("FIFO: "+nextJob().name+" is at the head of the queue.");return;}
    if(r+j.nodes>NR){msg(j.name+" needs "+j.nodes+" node rows, so it does not fit starting at "+D.nodes[r]+".");return;}
    if(h+j.hours>NH){msg("That would run past the right-hand edge of the board.");return;}
    const g=occupied();
    for(let rr=r;rr<r+j.nodes;rr++) for(let hh=h;hh<h+j.hours;hh++)
      if(g[rr][hh]){msg("That rectangle would overlap "+g[rr][hh]+". A job owns its nodes outright: nothing else may touch them.");return;}
    const rows=[]; for(let rr=r;rr<r+j.nodes;rr++) rows.push(rr);
    placed[j.name]={start:h,rows:rows}; sel=null; draw();
  }
  function metrics(){
    const names=Object.keys(placed); if(!names.length) return null;
    let busy=0,last=0,wait=0,turn=0;
    names.forEach(n=>{const p=placed[n], j=D.jobs.find(x=>x.name===n);
      busy+=j.nodes*j.hours; last=Math.max(last,p.start+j.hours);
      wait+=p.start; turn+=p.start+j.hours;});
    return {n:names.length,makespan:last,busy:busy,total:NR*last,
            wait:wait/names.length,turn:turn/names.length,
            util:busy/(NR*last),idle:NR*last-busy};
  }
  function drawMetrics(){
    const m=metrics(), box=root.querySelector(".hs-m");
    const cell=(k,v,w,cls)=>'<div class="hs-box '+cls+'"><div class="hs-k">'+k+'</div>'
      +'<div class="hs-v">'+v+'</div><div class="hs-w" style="color:'
      +(cls==="user"?"#2e9e7a":"#764ba2")+'">'+w+'</div></div>';
    if(!m){box.innerHTML=cell("makespan","–","cluster","")+cell("utilisation","–","cluster","")
      +cell("idle node-hours","–","cluster","")+cell("avg waiting","–","user","user")
      +cell("avg turnaround","–","user","user");return;}
    box.innerHTML=cell("makespan",m.makespan+" h","cluster","")
      +cell("utilisation",(100*m.util).toFixed(1)+"%","cluster","")
      +cell("idle node-hours",m.idle,"cluster","")
      +cell("avg waiting",m.wait.toFixed(2)+" h","user","user")
      +cell("avg turnaround",m.turn.toFixed(2)+" h","user","user");
  }
  function msg(t){root.querySelector(".hs-note").innerHTML=t;}
  function draw(){
    drawQueue();drawBoard();drawMetrics();
    const m=metrics();
    if(Object.keys(placed).length===D.jobs.length&&m)
      msg("All five placed. Makespan <b>"+m.makespan+" h</b>, utilisation <b>"
        +(100*m.util).toFixed(1)+"%</b>, <b>"+m.idle+"</b> idle node-hours, that is "+m.idle
        +" hours of machine that four research groups paid for and nobody used. "
        +"Try again in a different order and see how low you can get the idle count.");
    else if(!root.querySelector(".hs-note").innerHTML)
      msg("Start with <b>"+nextJob().name+"</b>, the head of the queue.");
  }
  root.querySelector(".hs-btn").addEventListener("click",()=>{placed={};sel=null;msg("");draw();});
  root.querySelectorAll(".hs-btn")[1].addEventListener("click",()=>{
    placed={}; D.fifo.forEach(f=>{placed[f.name]={start:f.start,rows:f.rows};});
    sel=null; draw();
    msg("This is plain FIFO: consider the queue in order, and start a job the moment its nodes are "
      +"free. J2 needs all four nodes, so from 0h to 3h three nodes stand idle waiting for it, "
      +"even though J5 would have fitted in that gap. <b>Hold on to that hole.</b>");
  });
  draw();
})();
</script>'''
    display(HTML(tmpl.replace("__UID__", uid).replace("__FONT__", _FONT)
                 .replace("__DATA__", _json.dumps(data))))


# ===========================================================================
#  §7  Part 3: the number you have to declare
# ===========================================================================
def walltime_slider(real_hours=3, nodes=3, start_declared=3, max_declared=8):
    """One job, one slider: `--time`. The real work never changes.

    Below the limit the job is killed. Above it, the rectangle grows into the
    timeline while the work inside it stays the same size.
    """
    data = {"real": real_hours, "nodes": nodes, "start": start_declared,
            "maxd": max_declared, "rows": [n["name"] for n in NODE_SPECS],
            "col": color_of("J4"), "px": 62, "rowh": _ROW_H, "gutter": 82}
    uid = _uid("wt", "walltime_slider")
    tmpl = r'''
<style>
#__UID__{font-family:__FONT__;border:1px solid #e6e8ee;border-radius:14px;padding:16px;max-width:900px;background:#fff;color:#24262b}
#__UID__ .wt-head{font-weight:800;font-size:15px;color:#2b2d6b}
#__UID__ .wt-sub{font-size:12.5px;color:#666;margin:3px 0 12px;line-height:1.55}
#__UID__ .wt-ctl{display:flex;align-items:center;gap:12px;margin-bottom:12px;flex-wrap:wrap}
#__UID__ input[type=range]{flex:1;min-width:230px;accent-color:#764ba2}
#__UID__ .wt-val{font-size:13.5px;font-weight:800;color:#4a3a86;min-width:150px;font-family:ui-monospace,Menlo,monospace}
#__UID__ .wt-board{position:relative}
#__UID__ .wt-num{display:flex;gap:9px;flex-wrap:wrap;margin-top:12px}
#__UID__ .wt-box{flex:1;min-width:150px;border:1px solid #e6e8ee;border-radius:10px;padding:8px 11px}
#__UID__ .wt-k{font-size:10px;color:#8a8fa3;text-transform:uppercase;letter-spacing:.04em}
#__UID__ .wt-v{font-size:19px;font-weight:800}
#__UID__ .wt-note{background:#f6f7fb;border-radius:8px;padding:10px 12px;font-size:12.5px;color:#333;line-height:1.6;margin-top:11px}
</style>
<div id="__UID__">
  <div class="wt-head">⏱️ The only number the scheduler is allowed to read</div>
  <div class="wt-sub">One job, three nodes, and <b>three hours of real work</b> that never changes.
    Slide <code>--time</code> (the number the user types) and watch the rectangle the scheduler
    plans around.</div>
  <div class="wt-ctl">
    <span class="wt-val"></span>
    <input type="range" min="1" max="__MAXD__" value="__START__" step="1">
  </div>
  <div class="wt-board"></div>
  <div class="wt-num"></div>
  <div class="wt-note"></div>
</div>
<script>
(function(){
  const D=__DATA__, root=document.getElementById("__UID__");
  const PX=D.px, RH=D.rowh, GU=D.gutter, NR=D.rows.length;
  const sl=root.querySelector("input"), board=root.querySelector(".wt-board");
  function draw(){
    const dec=+sl.value, real=D.real, killed=dec<real, done=Math.min(dec,real);
    const NH=D.maxd+1;
    root.querySelector(".wt-val").textContent="#SBATCH --time="+dec+":00:00";
    let h='';
    for(let t=0;t<=NH;t++) h+='<div style="position:absolute;left:'+(GU+t*PX)
      +'px;top:0;font-size:10.5px;color:#8a8fa3;transform:translateX(-50%)">'+t+'h</div>';
    for(let r=0;r<NR;r++){
      h+='<div style="position:absolute;left:0;top:'+(16+r*RH)+'px;width:'+(GU-8)+'px;height:'+RH
        +'px;line-height:'+RH+'px;font-size:11.5px;font-weight:600;color:#555">'+D.rows[r]+'</div>';
      h+='<div style="position:absolute;left:'+GU+'px;top:'+(16+r*RH)+'px;width:'+(NH*PX)
        +'px;height:'+RH+'px;background:#f7f8fc;border-top:1px solid #eceef5"></div>';
    }
    for(let t=0;t<=NH;t++) h+='<div style="position:absolute;left:'+(GU+t*PX)+'px;top:16px;width:1px;height:'
      +(NR*RH)+'px;background:'+(t?"#e2e5ef":"#c2c7da")+'"></div>';
    const top=17+RH, hgt=D.nodes*RH-3;
    // the work that actually happens
    h+='<div style="position:absolute;left:'+(GU+1)+'px;top:'+top+'px;width:'+(done*PX-3)
      +'px;height:'+hgt+'px;background:'+(killed?"#c0554e":D.col)+';border-radius:7px;color:#fff;font-weight:800;'
      +'font-size:13px;display:flex;flex-direction:column;align-items:center;justify-content:center">'
      +(killed?'killed at '+dec+'h':'J4 · real work')
      +'<div style="font-weight:500;font-size:10px;opacity:.9">'+done+' h × '+D.nodes+' nodes</div></div>';
    if(killed){
      h+='<div style="position:absolute;left:'+(GU+dec*PX+1)+'px;top:'+top+'px;width:'+((real-dec)*PX-3)
        +'px;height:'+hgt+'px;border:2px dashed #c0554e;border-radius:7px;color:#c0554e;font-size:11px;'
        +'font-weight:700;display:flex;align-items:center;justify-content:center;text-align:center;'
        +'background:repeating-linear-gradient(45deg,rgba(192,85,78,.08) 0 6px,transparent 6px 12px)">'
        +'work never done</div>';
    } else if(dec>real){
      h+='<div style="position:absolute;left:'+(GU+real*PX+1)+'px;top:'+top+'px;width:'+((dec-real)*PX-3)
        +'px;height:'+hgt+'px;border:2px dashed '+D.col+';border-left:none;border-radius:0 7px 7px 0;color:'
        +D.col+';font-size:11px;font-weight:700;display:flex;align-items:center;justify-content:center;'
        +'background:repeating-linear-gradient(45deg,rgba(0,0,0,.04) 0 6px,transparent 6px 12px)">'
        +'declared, never used</div>';
    }
    board.style.height=(NR*RH+24)+"px"; board.style.width=(GU+NH*PX+16)+"px";
    board.innerHTML=h;
    const box=(k,v,c)=>'<div class="wt-box"><div class="wt-k">'+k+'</div><div class="wt-v" style="color:'
      +c+'">'+v+'</div></div>';
    root.querySelector(".wt-num").innerHTML=
      box("rectangle the scheduler plans", D.nodes+" × "+dec+" h = "+(D.nodes*dec)+" node-h","#764ba2")
     +box("node-hours you are charged", (D.nodes*done)+" node-h", killed?"#c0554e":"#2e9e7a")
     +box("results you keep", killed?"nothing":"everything", killed?"#c0554e":"#2e9e7a");
    let m;
    if(killed)
      m="<b>Under-declared.</b> Slurm does not measure your job, it enforces your number: at "+dec
       +"h it sends the job a signal and then kills it. You are still charged for the "+(D.nodes*done)
       +" node-hours it burnt, and unless the job wrote checkpoints to disk you keep nothing at all. "
       +"The three hours of work simply did not finish.";
    else if(dec===real)
      m="<b>Exactly right, and nobody is ever exactly right.</b> Real jobs vary with the input, the "
       +"node, the network. Declaring the true duration to the hour is luck, not skill, which is why "
       +"every user pads. The question of the rest of this notebook is <i>how much</i> padding costs.";
    else
      m="<b>Over-declared by "+(dec-real)+" h.</b> Nothing is killed and you are only charged for "
       +(D.nodes*done)+" node-hours, so this looks free. It is not: the scheduler now believes your "
       +"job is a <b>"+D.nodes+" × "+dec+"</b> rectangle and will only ever look for a hole that big. "
       +"The hatched part is empty space you carry around with you. Part 5 puts a price on it.";
    root.querySelector(".wt-note").innerHTML=m;
  }
  sl.addEventListener("input",draw); draw();
})();
</script>'''
    display(HTML(tmpl.replace("__UID__", uid).replace("__FONT__", _FONT)
                 .replace("__MAXD__", str(max_declared))
                 .replace("__START__", str(start_declared))
                 .replace("__DATA__", _json.dumps(data))))


# ===========================================================================
#  §8  Part 4: priority, and fair-share as a loop
# ===========================================================================
# Six jobs sitting in the queue at one particular moment. Ages are in hours.
WAITING = [
    {"name": "W1", "what": "3D reconstruction", "group": "bio", "age": 22,
     "qos": "normal", "size": "2n × 4h"},
    {"name": "W2", "what": "ocean run", "group": "clim", "age": 2,
     "qos": "high", "size": "4n × 6h"},
    {"name": "W3", "what": "policy sweep", "group": "robo", "age": 8,
     "qos": "normal", "size": "1n × 2h"},
    {"name": "W4", "what": "risk calibration", "group": "econ", "age": 30,
     "qos": "normal", "size": "1n × 1h"},
    {"name": "W5", "what": "MRI batch", "group": "bio", "age": 1,
     "qos": "high", "size": "3n × 3h"},
    {"name": "W6", "what": "downscaling", "group": "clim", "age": 14,
     "qos": "normal", "size": "2n × 2h"},
]
# What each group has consumed lately, normalised to sum to 1.
USAGE_NOW = {"bio": 0.62, "clim": 0.25, "robo": 0.10, "econ": 0.03}


def priority_mixer():
    """Three weights, one queue. Move a weight and watch the order change.

    Each bar is split into the three contributions, so the order is never a
    mystery: you can see which term put a job where it is.
    """
    data = {"jobs": WAITING, "usage": USAGE_NOW, "target": TARGET_SHARE,
            "gname": GROUP_NAME, "gcol": GROUP_COLOR, "age_scale": 24}
    uid = _uid("pm", "priority_mixer")
    tmpl = r'''
<style>
#__UID__{font-family:__FONT__;border:1px solid #e6e8ee;border-radius:14px;padding:16px;max-width:900px;background:#fff;color:#24262b}
#__UID__ .pm-head{font-weight:800;font-size:15px;color:#2b2d6b}
#__UID__ .pm-sub{font-size:12.5px;color:#666;margin:3px 0 12px;line-height:1.55}
#__UID__ .pm-pre{display:flex;gap:7px;flex-wrap:wrap;margin-bottom:12px}
#__UID__ .pm-p{cursor:pointer;border:1px solid #d6cdf0;background:#fff;color:#764ba2;border-radius:8px;padding:6px 12px;font-size:12px;font-weight:700}
#__UID__ .pm-p.on{background:linear-gradient(135deg,#667eea,#764ba2);color:#fff;border-color:transparent}
#__UID__ .pm-row{display:flex;align-items:center;gap:10px;margin-bottom:7px;font-size:12.5px}
#__UID__ .pm-lab{width:104px;font-weight:700}
#__UID__ .pm-row input{flex:1;min-width:150px;accent-color:#764ba2}
#__UID__ .pm-w{width:44px;text-align:right;font-weight:800;font-family:ui-monospace,Menlo,monospace}
#__UID__ .pm-form{background:#f6f7fb;border-radius:8px;padding:9px 12px;font-size:12.5px;margin:11px 0;font-family:ui-monospace,Menlo,monospace;color:#3b2d6b}
#__UID__ .pm-hdr{display:flex;align-items:center;gap:9px;margin:2px 0 7px;font-size:10.5px;color:#8a8fa3;text-transform:uppercase;letter-spacing:.04em;font-weight:700}
#__UID__ .pm-job{display:flex;align-items:center;gap:9px;margin-bottom:6px;font-size:12px}
#__UID__ .pm-rank{width:20px;font-weight:800;color:#8a8fa3}
#__UID__ .pm-name{width:210px}
#__UID__ .pm-bar{flex:1;height:20px;background:#f0f1f6;border-radius:5px;display:flex;overflow:hidden;min-width:150px}
#__UID__ .pm-seg{height:100%}
#__UID__ .pm-tot{width:46px;text-align:right;font-weight:800;font-family:ui-monospace,Menlo,monospace;color:#2b2d6b}
#__UID__ .pm-leg{font-size:11px;color:#666;margin:9px 0 0}
#__UID__ .pm-note{background:#fdf9f1;border:1px solid #f0e3c8;border-radius:8px;padding:10px 12px;font-size:12.5px;color:#5a4a20;line-height:1.6;margin-top:11px}
</style>
<div id="__UID__">
  <div class="pm-head">⚖️ Six jobs waiting. Who goes first?</div>
  <div class="pm-sub">A toy score, <b>not</b> Slurm's real formula, but the same shape:
    three things you care about, three weights you choose. Each bar is one job's score, split into
    the <b>contribution of each factor</b> &mdash; that factor's value multiplied by its weight.
    Pick a stance, or move the weights yourself, and read the order off the bars.</div>
  <div class="pm-pre">
    <button class="pm-p" data-a="0.5" data-f="0.3" data-q="0.2">Balanced (the default)</button>
    <button class="pm-p" data-a="0.2" data-f="0.7" data-q="0.1">Fairness-oriented</button>
    <button class="pm-p" data-a="0.2" data-f="0.1" data-q="0.7">Urgency-oriented</button>
    <button class="pm-p" data-a="0.0" data-f="0.4" data-q="0.6">Throughput-oriented</button>
  </div>
  <div class="pm-row"><span class="pm-lab" style="color:#4a5bd0">age</span>
    <input type="range" min="0" max="100" value="50"><span class="pm-w"></span></div>
  <div class="pm-row"><span class="pm-lab" style="color:#2e9e7a">fair-share</span>
    <input type="range" min="0" max="100" value="30"><span class="pm-w"></span></div>
  <div class="pm-row"><span class="pm-lab" style="color:#e0a500">QoS</span>
    <input type="range" min="0" max="100" value="20"><span class="pm-w"></span></div>
  <div class="pm-form"></div>
  <div class="pm-hdr"><span style="width:20px"></span><span style="width:210px">job</span>
    <span style="flex:1;min-width:150px">contribution of each factor&nbsp;=&nbsp;value × weight</span>
    <span style="width:46px;text-align:right">score</span></div>
  <div class="pm-list"></div>
  <div class="pm-leg">Bar segments, left to right:
    <b style="color:#4a5bd0">age</b> · <b style="color:#2e9e7a">fair-share</b> ·
    <b style="color:#e0a500">QoS</b>. A segment is <b>not</b> the factor's value: it is that value
    <b>× its weight</b>, so setting a weight to zero makes its segment vanish however good the
    job looks on that factor. Hover a segment for the arithmetic. The whole bar is the score on the
    right, so the longest bar starts soonest.</div>
  <div class="pm-note"></div>
</div>
<script>
(function(){
  const D=__DATA__, root=document.getElementById("__UID__");
  const sl=root.querySelectorAll(".pm-row input"), ws=root.querySelectorAll(".pm-w");
  const COL=["#4a5bd0","#2e9e7a","#e0a500"];
  const F={}; Object.keys(D.target).forEach(g=>{F[g]=Math.pow(2,-(D.usage[g]/D.target[g]));});
  function draw(){
    const w=[+sl[0].value/100,+sl[1].value/100,+sl[2].value/100];
    w.forEach((v,i)=>ws[i].textContent=v.toFixed(2));
    root.querySelector(".pm-form").innerHTML="priority = "+w[0].toFixed(2)+"·age + "
      +w[1].toFixed(2)+"·fair_share + "+w[2].toFixed(2)+"·qos";
    const scored=D.jobs.map(j=>{
      const raw=[Math.min(1,j.age/D.age_scale), F[j.group], (j.qos==="high"?1.0:0.5)];
      const parts=raw.map((v,k)=>w[k]*v);
      return {j:j, raw:raw, parts:parts, tot:parts[0]+parts[1]+parts[2]};
    }).sort((a,b)=>b.tot-a.tot);
    const mx=Math.max.apply(null,scored.map(s=>s.tot))||1;
    let h="";
    scored.forEach((s,i)=>{
      let segs="";
      s.parts.forEach((p,k)=>{segs+='<span class="pm-seg" title="'+["age","fair-share","QoS"][k]
        +': value '+s.raw[k].toFixed(2)+' × weight '+w[k].toFixed(2)+' = '+p.toFixed(3)
        +' of the score" style="width:'+(100*p/mx)+'%;background:'+COL[k]+'"></span>';});
      h+='<div class="pm-job"><span class="pm-rank">'+(i+1)+'</span>'
        +'<span class="pm-name"><b>'+s.j.name+'</b> '+s.j.what
        +'<br><span style="color:'+D.gcol[s.j.group]+';font-size:10.5px;font-weight:700">'
        +D.gname[s.j.group]+'</span><span style="color:#8a8fa3;font-size:10.5px"> · waiting '
        +s.j.age+'h · '+s.j.size+(s.j.qos==="high"?' · QoS high':'')+'</span></span>'
        +'<span class="pm-bar">'+segs+'</span><span class="pm-tot">'+s.tot.toFixed(2)+'</span></div>';
    });
    root.querySelector(".pm-list").innerHTML=h;
    const last=scored[scored.length-1].j, first=scored[0].j;
    let m="<b>"+first.name+"</b> ("+D.gname[first.group]+") goes first and <b>"+last.name
      +"</b> goes last. ";
    if(w[0]===0) m+="You have set the age weight to <b>zero</b>. Nothing about a job now improves "
      +"while it waits: its score today is its score forever. On a real cluster new jobs keep "
      +"arriving, and any job that starts below them stays below them. Watch that happen in the "
      +"next cell.";
    else m+="Because the age weight is <b>"+w[0].toFixed(2)+"</b>, every job climbs on its own as "
      +"it waits: after "+D.age_scale+" h of waiting the age term is at its maximum. That is the "
      +"only term that moves without anybody's permission.";
    root.querySelector(".pm-note").innerHTML=m;
    root.querySelectorAll(".pm-p").forEach(b=>b.classList.toggle("on",
      Math.abs(+b.dataset.a-w[0])<1e-9&&Math.abs(+b.dataset.f-w[1])<1e-9&&Math.abs(+b.dataset.q-w[2])<1e-9));
  }
  sl.forEach(s=>s.addEventListener("input",draw));
  root.querySelectorAll(".pm-p").forEach(b=>b.addEventListener("click",()=>{
    sl[0].value=100*(+b.dataset.a); sl[1].value=100*(+b.dataset.f); sl[2].value=100*(+b.dataset.q); draw();}));
  draw();
})();
</script>'''
    display(HTML(tmpl.replace("__UID__", uid).replace("__FONT__", _FONT)
                 .replace("__DATA__", _json.dumps(data))))


def fairshare_loop(capacity=672, half_life=2):
    """Fair-share is not a number, it is a loop that runs every week.

    Each week the groups ask for work in proportion to the share they were
    given, the scheduler serves them in proportion to their fair-share factor,
    the node-hours are charged, and the factor moves. Old consumption is never
    deleted: it is halved, and halved again.
    """
    data = {"groups": GROUPS, "gname": GROUP_NAME, "gcol": GROUP_COLOR,
            "target": TARGET_SHARE, "start": USAGE_NOW, "cap": capacity,
            "hl": half_life}
    uid = _uid("fs", "fairshare_loop")
    tmpl = r'''
<style>
#__UID__{font-family:__FONT__;border:1px solid #e6e8ee;border-radius:14px;padding:16px;max-width:900px;background:#fff;color:#24262b}
#__UID__ .fs-head{font-weight:800;font-size:15px;color:#2b2d6b}
#__UID__ .fs-sub{font-size:12.5px;color:#666;margin:3px 0 12px;line-height:1.55}
#__UID__ table{border-collapse:collapse;width:100%;font-size:12.5px;margin-bottom:12px;color:#24262b}
#__UID__ th{text-align:left;font-size:10.5px;color:#8a8fa3;text-transform:uppercase;letter-spacing:.04em;padding:0 8px 5px 0;font-weight:700}
#__UID__ td{padding:6px 8px 6px 0;border-top:1px solid #f0f1f6;vertical-align:middle;color:#24262b}
#__UID__ .fs-tog{cursor:pointer;border:1px solid #d6cdf0;background:#fff;color:#764ba2;border-radius:20px;padding:3px 11px;font-size:11px;font-weight:700}
#__UID__ .fs-tog.off{border-color:#e2e5ef;background:#f7f8fc;color:#8a8fa3}
#__UID__ .fs-mini{height:9px;background:#f0f1f6;border-radius:3px;position:relative;width:120px;display:inline-block;vertical-align:middle}
#__UID__ .fs-fill{position:absolute;left:0;top:0;bottom:0;border-radius:3px}
#__UID__ .fs-tick{position:absolute;top:-3px;bottom:-3px;width:2px;background:#2b2d6b}
#__UID__ .fs-btn{cursor:pointer;border:none;border-radius:8px;padding:7px 14px;font-size:12.5px;font-weight:700;color:#fff;background:linear-gradient(135deg,#667eea,#764ba2);margin:0 6px 0 0}
#__UID__ .fs-btn.gh{background:#fff;color:#764ba2;border:1px solid #d6cdf0}
#__UID__ .fs-ctl{display:flex;align-items:center;gap:10px;font-size:12px;margin:10px 0;flex-wrap:wrap}
#__UID__ .fs-ctl input[type=range]{width:150px;accent-color:#764ba2}
#__UID__ .fs-note{background:#f6f7fb;border-radius:8px;padding:10px 12px;font-size:12.5px;color:#333;line-height:1.6;margin-top:11px}
</style>
<div id="__UID__">
  <div class="fs-head">🔁 Fair-share, week by week</div>
  <div class="fs-sub">Each group was given a <b>target share S</b> by you, the administrators.
    <b>U</b> is what it has actually consumed lately, as a fraction of everything consumed.
    The fair-share factor is <b>F = 2<sup>−U/S</sup></b>, so a group sitting exactly on its target
    has F = 0.5, and every group starts drifting back towards that. Switch a group to
    <b>Quiet</b> and run a few weeks.</div>
  <table><thead><tr><th>group</th><th>target S</th><th>consumed U</th><th>F = 2<sup>−U/S</sup></th>
    <th>this week</th><th>submitting?</th></tr></thead><tbody class="fs-body"></tbody></table>
  <div class="fs-ctl">
    <span>half-life of old consumption: <b class="fs-hl"></b></span>
    <input type="range" min="1" max="8" value="__HL__" step="1">
    <button class="fs-btn">▶ Run one week</button>
    <button class="fs-btn">⏭ Run four weeks</button>
    <button class="fs-btn gh">↺ Back to week 0</button>
  </div>
  <div style="font-size:11px;color:#8a8fa3;margin-top:4px">fair-share factor F, one point per week</div>
  <svg class="fs-chart" width="860" height="150"></svg>
  <div class="fs-note"></div>
</div>
<script>
(function(){
  const D=__DATA__, root=document.getElementById("__UID__");
  const G=D.groups; let raw={}, active={}, week=0, hist=[];
  const sl=root.querySelector("input[type=range]");
  function reset(){
    week=0; hist=[]; raw={}; G.forEach(g=>{raw[g]=D.start[g]*D.cap*2; active[g]=true;});
    record(); draw();
  }
  function factors(){
    const tot=G.reduce((s,g)=>s+raw[g],0)||1, U={}, F={};
    G.forEach(g=>{U[g]=raw[g]/tot; F[g]=Math.pow(2,-U[g]/D.target[g]);});
    return {U:U,F:F};
  }
  function record(){const f=factors(); hist.push({w:week,F:Object.assign({},f.F),U:Object.assign({},f.U)});}
  function step(){
    const hl=+sl.value, f=factors(), served={};
    // demand in proportion to the share you were given, served in proportion to F
    let w=0; G.forEach(g=>{if(active[g])w+=D.target[g]*f.F[g];});
    G.forEach(g=>{served[g]=(active[g]&&w>0)?D.cap*D.target[g]*f.F[g]/w:0;});
    G.forEach(g=>{raw[g]=raw[g]*Math.pow(0.5,1/hl)+served[g];});
    week++; record(); return served;
  }
  function draw(served){
    const f=factors(), body=root.querySelector(".fs-body"); let h="";
    G.forEach(g=>{
      const U=f.U[g], S=D.target[g];
      h+='<tr><td><b style="color:'+D.gcol[g]+'">'+D.gname[g]+'</b></td>'
        +'<td>'+(100*S).toFixed(0)+'%</td>'
        +'<td style="white-space:nowrap">'+(100*U).toFixed(1)+'% <span class="fs-mini"><span class="fs-fill" style="width:'
        +Math.min(100,100*U/0.7)+'%;background:'+D.gcol[g]+'"></span>'
        +'<span class="fs-tick" style="left:'+(100*S/0.7)+'%"></span></span></td>'
        +'<td><b style="color:'+(f.F[g]>0.4995?"#2e9e7a":"#c0554e")+'">'+f.F[g].toFixed(3)+'</b></td>'
        +'<td>'+(served?Math.round(served[g])+" node-h":"–")+'</td>'
        +'<td><button class="fs-tog'+(active[g]?"":" off")+'" data-g="'+g+'">'
        +(active[g]?"Active":"Quiet")+'</button></td></tr>';
    });
    body.innerHTML=h;
    body.querySelectorAll(".fs-tog").forEach(b=>b.addEventListener("click",()=>{
      active[b.dataset.g]=!active[b.dataset.g]; draw(served);}));
    root.querySelector(".fs-hl").textContent=sl.value+" week"+(sl.value>1?"s":"");
    chart();
    let m="<b>Week "+week+".</b> The black tick in each bar is that group's target. ";
    if(week===0) m+="Biomedical Imaging is far above its 40% target, so its F is the lowest in the "
      +"table and its jobs will lose almost every tie. Economics has barely used the cluster, so its "
      +"F is close to 1. Run a week.";
    else {
      const q=G.filter(g=>!active[g]);
      m+=q.length?("<b>"+q.map(g=>D.gname[g]).join(", ")+"</b> submitted nothing this week, so "
        +"nothing was charged, but the old consumption is still there. It was multiplied by "
        +Math.pow(0.5,1/(+sl.value)).toFixed(3)+", and it will be again next week. Nothing is ever "
        +"set back to zero: it just halves every "+sl.value+" week"+(sl.value>1?"s":"")+" until it "
        +"stops mattering.")
        :("Every group is submitting. Watch U crawl towards the tick and F towards 0.500: that is "
          +"the equilibrium, the point where a group is consuming exactly the share it was given.");
    }
    root.querySelector(".fs-note").innerHTML=m;
  }
  function chart(){
    const svg=root.querySelector(".fs-chart"), W=860,H=150,L=34,B=22;
    const n=Math.max(6,hist.length-1);
    let h='<line x1="'+L+'" y1="'+(H-B)+'" x2="'+(W-6)+'" y2="'+(H-B)+'" stroke="#c2c7da"/>'
      +'<line x1="'+L+'" y1="6" x2="'+L+'" y2="'+(H-B)+'" stroke="#c2c7da"/>';
    [0,0.5,1].forEach(v=>{const y=6+(1-v)*(H-B-6);
      h+='<line x1="'+L+'" y1="'+y+'" x2="'+(W-6)+'" y2="'+y+'" stroke="'+(v===0.5?"#2b2d6b":"#eceef5")
        +'" stroke-dasharray="'+(v===0.5?"4 3":"0")+'"/>'
        +'<text x="4" y="'+(y+4)+'" font-size="10" fill="#8a8fa3">'+v.toFixed(1)+'</text>';});
    h+='<text x="'+(W-8)+'" y="'+(6+0.5*(H-B-6)-5)+'" font-size="10" fill="#2b2d6b" '
      +'text-anchor="end">F = 0.5 · exactly on target</text>';
    G.forEach(g=>{
      const pts=hist.map(p=>{const x=L+(p.w/n)*(W-L-10), y=6+(1-p.F[g])*(H-B-6);return x+","+y;});
      h+='<polyline fill="none" stroke="'+D.gcol[g]+'" stroke-width="2.2" points="'+pts.join(" ")+'"/>';
      if(hist.length){const p=hist[hist.length-1], x=L+(p.w/n)*(W-L-10), y=6+(1-p.F[g])*(H-B-6);
        h+='<circle cx="'+x+'" cy="'+y+'" r="3.2" fill="'+D.gcol[g]+'"/>';}
    });
    hist.forEach(p=>{const x=L+(p.w/n)*(W-L-10);
      h+='<text x="'+x+'" y="'+(H-8)+'" font-size="9.5" fill="#8a8fa3" text-anchor="middle">'+p.w+'</text>';});
    svg.innerHTML=h;
  }
  const btns=root.querySelectorAll(".fs-btn");
  btns[0].addEventListener("click",()=>draw(step()));
  btns[1].addEventListener("click",()=>{let s;for(let i=0;i<4;i++)s=step();draw(s);});
  btns[2].addEventListener("click",reset);
  sl.addEventListener("input",()=>draw());
  reset();
})();
</script>'''
    display(HTML(tmpl.replace("__UID__", uid).replace("__FONT__", _FONT)
                 .replace("__HL__", str(half_life))
                 .replace("__DATA__", _json.dumps(data))))


# ===========================================================================
#  §9  Part 5: reservation and backfilling, one step at a time
# ===========================================================================
def _place(name, start, end, rows, **kw):
    d = {"name": name, "start": start, "end": end,
         "nodes": [NODE_SPECS[r]["name"] for r in rows],
         "n_nodes": len(rows), "cores": 8, "gpus": 0,
         "requested_time": end - start, "killed": False, "backfilled": False}
    d.update(kw)
    return d


def backfill_walkthrough():
    """The reservation argument, drawn on the timeline the way the lecture draws it."""
    R = [0, 1, 2, 3]
    frames = []

    def add(sched, caption, **kw):
        frames.append({"html": timeline_html(sched, hours=10, px=54, **kw),
                       "cap": caption})

    add([], "<b>0h.</b> Four nodes, nothing running, and five jobs in the queue in priority order: "
            "J1 (1 node, 3h), J2 (4 nodes, 2h), J3 (3 nodes, 4h), J4 (3 nodes, 3h), J5 (1 node, 2h).")
    add([_place("J1", 0, 3, [0])],
        "<b>J1 is the head of the queue and it fits.</b> One node for three hours, starting now. "
        "node02, node03 and node04 are free.")
    add([_place("J1", 0, 3, [0])],
        "<b>J2 is next, and it does not fit.</b> It asked for all four nodes; three are free. "
        "Under strict priority the story ends here: nothing else may start, and those three nodes "
        "stand idle for three hours.",
        overlays=[{"t0": 0, "t1": 3, "rows": [1, 2, 3], "color": GREY,
                   "fill": "rgba(154,160,181,.10)", "label": "3 nodes idle"}])
    add([_place("J1", 0, 3, [0])],
        "<b>Reservation.</b> Instead of just refusing J2, Slurm works out the earliest moment it "
        "could give J2 everything it asked for. J1 releases node01 at 3h, so that moment is "
        "<b>3h</b>, and Slurm writes it down. From now on no later decision is allowed to push "
        "J2 past 3h. Without that written-down promise, a big job could be overtaken by a stream "
        "of small ones for ever.",
        reservation=("J2", 3))
    add([_place("J1", 0, 3, [0])],
        "<b>Now there is a shape to fill.</b> Three nodes, from now until the reservation: a "
        "window <b>3 nodes tall and 3 hours wide</b>. Anything that fits inside it can run without "
        "touching J2's promise.",
        reservation=("J2", 3),
        overlays=[{"t0": 0, "t1": 3, "rows": [1, 2, 3],
                   "label": "the hole: 3 nodes × 3 h"}])
    add([_place("J1", 0, 3, [0])],
        "<b>Test J3.</b> Nodes: 3 asked, 3 free. Fine. Time: <code>finish_time = 0 + 4 = 4</code>, "
        "and <code>4 &lt;= 3</code> is false. J3 would still be running when J2 is due to start, "
        "so J3 is refused. Note which number was used: J3's <b>declared</b> 4 hours.",
        reservation=("J2", 3),
        ghosts=[{"name": "J3", "start": 0, "hours": 4, "rows": [1, 2, 3],
                 "sub": "ends 4h > 3h", "verdict": "no"}])
    add([_place("J1", 0, 3, [0])],
        "<b>Test J4.</b> Nodes: 3 asked, 3 free. Time: <code>finish_time = 0 + 3 = 3</code>, and "
        "<code>3 &lt;= 3</code> is true. J4 starts <b>now</b>, ahead of J3, which has higher "
        "priority. That is backfilling.",
        reservation=("J2", 3),
        ghosts=[{"name": "J4", "start": 0, "hours": 3, "rows": [1, 2, 3],
                 "sub": "ends 3h ≤ 3h", "verdict": "yes"}])
    add([_place("J1", 0, 3, [0]), _place("J4", 0, 3, [1, 2, 3], backfilled=True)],
        "<b>J4 is in.</b> J5 is tested next: it only wants one node for two hours, which would fit "
        "in time, but there is no free node left. J5 waits.",
        reservation=("J2", 3))
    add([_place("J1", 0, 3, [0]), _place("J4", 0, 3, [1, 2, 3], backfilled=True),
         _place("J2", 3, 5, R)],
        "<b>3h: the promise is kept.</b> J1 and J4 finish together, and J2 starts at exactly the "
        "moment that was written down. It would have started at 3h with strict priority too: "
        "<b>backfilling cost J2 nothing.</b>")
    add(run_schedule(five_jobs(), backfill=True),
        "<b>The finished schedule.</b> Makespan 9h instead of 12h, 2 idle node-hours instead of 14, "
        "utilisation 94.4% instead of 70.8%. J2 started at the same moment either way. The job that "
        "moved is J3, overtaken in the queue by a job with lower priority, though not actually "
        "delayed. Backfilling protects the reserved job, and reshuffles everyone else.")

    data = {"frames": frames}
    uid = _uid("bw", "backfill_walkthrough")
    tmpl = r'''
<style>
#__UID__{font-family:__FONT__;border:1px solid #e6e8ee;border-radius:14px;padding:16px;max-width:900px;background:#fff;color:#24262b}
#__UID__ .bw-head{font-weight:800;font-size:15px;color:#2b2d6b;margin-bottom:9px}
#__UID__ .bw-nav{display:flex;align-items:center;gap:9px;margin-bottom:11px}
#__UID__ .bw-btn{cursor:pointer;border:none;border-radius:8px;padding:7px 15px;font-size:13px;font-weight:700;color:#fff;background:linear-gradient(135deg,#667eea,#764ba2)}
#__UID__ .bw-btn:disabled{opacity:.35;cursor:default}
#__UID__ .bw-dots{display:flex;gap:5px;margin-left:6px}
#__UID__ .bw-dot{width:9px;height:9px;border-radius:50%;background:#e2e5ef;cursor:pointer}
#__UID__ .bw-dot.on{background:#764ba2}
#__UID__ .bw-cap{background:#f6f7fb;border-radius:8px;padding:11px 13px;font-size:12.5px;color:#333;line-height:1.65;margin-top:11px;min-height:56px}
#__UID__ .bw-cap code{background:#efe9fb;border-radius:4px;padding:1px 5px;font-size:12px}
</style>
<div id="__UID__">
  <div class="bw-head">🔒 One reservation, one hole, one test</div>
  <div class="bw-nav">
    <button class="bw-btn" data-d="-1">◀ back</button>
    <button class="bw-btn" data-d="1">next ▶</button>
    <span style="font-size:12px;color:#8a8fa3" class="bw-n"></span>
    <span class="bw-dots"></span>
  </div>
  <div class="bw-view"></div>
  <div class="bw-cap"></div>
</div>
<script>
(function(){
  const D=__DATA__, root=document.getElementById("__UID__"); let i=0;
  const dots=root.querySelector(".bw-dots");
  D.frames.forEach((_,k)=>{const d=document.createElement("span"); d.className="bw-dot";
    d.addEventListener("click",()=>{i=k;draw();}); dots.appendChild(d);});
  function draw(){
    root.querySelector(".bw-view").innerHTML=D.frames[i].html;
    root.querySelector(".bw-cap").innerHTML=D.frames[i].cap;
    root.querySelector(".bw-n").textContent="step "+(i+1)+" / "+D.frames.length;
    dots.querySelectorAll(".bw-dot").forEach((d,k)=>d.classList.toggle("on",k===i));
    root.querySelectorAll(".bw-btn").forEach(b=>{
      const d=+b.dataset.d; b.disabled=(i+d<0)||(i+d>=D.frames.length);});
  }
  root.querySelectorAll(".bw-btn").forEach(b=>b.addEventListener("click",()=>{i+=+b.dataset.d;draw();}));
  draw();
})();
</script>'''
    display(HTML(tmpl.replace("__UID__", uid).replace("__FONT__", _FONT)
                 .replace("__DATA__", _json.dumps(data))))


# ===========================================================================
#  §10  Part 6: two policies, one workload
# ===========================================================================
# Higher is better for some of these and worse for others. Say which, once.
_BETTER = {"makespan": "low", "avg_waiting": "low", "avg_turnaround": "low",
           "max_waiting": "low", "worst_group_waiting": "low", "killed": "low",
           "idle_node_hours": "low", "utilisation": "high", "gpu_utilisation": "high",
           "fairness": "high", "throughput": "high", "busy_node_hours": "high"}
_AUDIENCE = {"makespan": "the cluster", "utilisation": "the cluster",
             "gpu_utilisation": "the cluster", "idle_node_hours": "the cluster",
             "throughput": "the cluster", "avg_waiting": "the user",
             "avg_turnaround": "the user", "max_waiting": "the user",
             "killed": "the user", "fairness": "the four groups",
             "worst_group_waiting": "the four groups"}
_LABEL = {"makespan": "makespan", "utilisation": "utilisation",
          "gpu_utilisation": "GPU utilisation", "idle_node_hours": "idle node-hours",
          "throughput": "throughput (jobs/h)", "avg_waiting": "average waiting",
          "avg_turnaround": "average turnaround", "max_waiting": "worst waiting",
          "worst_group_waiting": "worst group's average wait",
          "fairness": "share equity", "killed": "jobs killed at the limit"}


def _fmt(k, v):
    if k in ("utilisation", "gpu_utilisation", "fairness"):
        return "%.1f%%" % (100 * v)
    if k in ("makespan", "killed"):
        return "%d" % v if k == "killed" else "%d h" % v
    if k == "throughput":
        return "%.2f" % v
    if k == "idle_node_hours":
        return "%.0f" % v
    return "%.2f h" % v


def compare_card(items, keys=None, title="Two policies, one workload"):
    """items: [(label, metrics dict), ...]. Arrows mark who won each row."""
    keys = keys or ["makespan", "throughput", "utilisation", "gpu_utilisation",
                    "avg_waiting", "avg_turnaround", "fairness",
                    "worst_group_waiting"]
    head = "".join('<th style="text-align:right;padding:0 10px 6px">%s</th>' % lab
                   for lab, _ in items)
    rows = ""
    for k in keys:
        vals = [m[k] for _, m in items]
        best = min(vals) if _BETTER[k] == "low" else max(vals)
        cells = ""
        for v in vals:
            win = abs(v - best) < 1e-12
            cells += ('<td style="text-align:right;padding:6px 10px;font-weight:%s;'
                      'color:%s;font-variant-numeric:tabular-nums">%s%s</td>'
                      % ("800" if win else "500", GREEN if win else "#555",
                         _fmt(k, v), " ●" if win else ""))
        rows += ('<tr><td style="padding:6px 10px 6px 0">%s</td>%s'
                 '<td style="padding:6px 0;font-size:11px;color:#8a8fa3">matters to %s</td></tr>'
                 % (_LABEL[k], cells, _AUDIENCE[k]))
    _card('<div style="font-weight:800;font-size:15px;color:%s;margin-bottom:9px">%s</div>'
          '<table style="border-collapse:collapse;width:100%%;font-size:12.5px;color:%s">'
          '<thead><tr><th style="text-align:left;padding:0 0 6px;font-size:10.5px;'
          'color:#8a8fa3;text-transform:uppercase;letter-spacing:.04em"></th>%s'
          '<th></th></tr></thead><tbody>%s</tbody></table>'
          '<div style="font-size:11.5px;color:#666;margin-top:9px">● marks the better number '
          'in that row. Read down a column and no policy wins everything: that is the point.</div>'
          % (INK, title, TEXT, head, rows), maxw=760)


def group_card(schedule, window=24, title="What each group actually got"):
    """Per-group waiting times and delivered share against the agreed target."""
    got = delivered_share(schedule, window=window)
    wait = group_waiting(schedule)
    n = {}
    for s in schedule:
        n[s["group"]] = n.get(s["group"], 0) + 1
    rows = ""
    for g in GROUPS:
        u, t = got.get(g, 0.0), TARGET_SHARE[g]
        rows += (
            '<tr><td style="padding:7px 10px 7px 0"><b style="color:%s">%s</b>'
            '<div style="font-size:10.5px;color:#8a8fa3">%d jobs</div></td>'
            '<td style="padding:7px 10px;text-align:right">%.0f%%</td>'
            '<td style="padding:7px 10px;width:180px">'
            '<div style="position:relative;height:11px;background:#f0f1f6;border-radius:3px">'
            '<div style="position:absolute;left:0;top:0;bottom:0;width:%.1f%%;background:%s;'
            'border-radius:3px"></div>'
            '<div style="position:absolute;top:-3px;bottom:-3px;left:%.1f%%;width:2px;'
            'background:%s"></div></div>'
            '<div style="font-size:10.5px;color:#666;margin-top:3px;white-space:nowrap">'
            '%.1f%% delivered <span style="color:#8a8fa3">· tick = target</span></div></td>'
            '<td style="padding:7px 0;text-align:right;font-weight:700;color:%s">%.1f h</td></tr>'
            % (GROUP_COLOR[g], GROUP_NAME[g], n.get(g, 0), 100 * t,
               min(100, 100 * u / 0.5), GROUP_COLOR[g], 100 * t / 0.5, INK,
               100 * u, RED if wait.get(g, 0) > 1.5 * (sum(wait.values()) / len(wait))
               else INK, wait.get(g, 0.0)))
    _card('<div style="font-weight:800;font-size:15px;color:%s;margin-bottom:4px">%s</div>'
          '<div style="font-size:12px;color:#666;margin-bottom:9px">Share of the node-hours '
          'delivered in the first %d hours, against the target the administrators agreed, '
          'and how long that group waited on average.</div>'
          '<table style="border-collapse:collapse;width:100%%;font-size:12.5px;color:%s">'
          '<thead><tr><th style="text-align:left;font-size:10.5px;color:#8a8fa3;'
          'text-transform:uppercase;padding-bottom:5px">group</th>'
          '<th style="text-align:right;font-size:10.5px;color:#8a8fa3;text-transform:uppercase;'
          'padding-bottom:5px">target</th>'
          '<th style="text-align:left;font-size:10.5px;color:#8a8fa3;text-transform:uppercase;'
          'padding:0 10px 5px">delivered</th>'
          '<th style="text-align:right;font-size:10.5px;color:#8a8fa3;text-transform:uppercase;'
          'padding-bottom:5px">avg wait</th></tr></thead><tbody>%s</tbody></table>'
          % (INK, title, window, TEXT, rows), maxw=680)


# ===========================================================================
#  §11  Part 7: design your scheduler
# ===========================================================================
_LEADERBOARD = []

DEFAULT_CONFIG = {
    "w_age": 0.5,
    "w_fair_share": 0.3,
    "w_qos": 0.2,
    "backfill": True,
    "reserve_depth": 1,
    "fair_share_live": True,
}


def evaluate(jobs, config, name="my policy", window=24, quiet=False):
    """Run one configuration and add it to the session's leaderboard.

    The config may be spelled either way round: the flat "w_age" /
    "w_fair_share" / "w_qos" / "fair_share_live" keys, or the
    weights={...} / dynamic_fairshare spelling that run_schedule takes
    in Parts 4-6. Both mean the same thing.

    A key that is neither raises, rather than being silently ignored and
    quietly evaluating the default policy under the student's own name.
    """
    cfg = dict(DEFAULT_CONFIG)
    config = dict(config)
    weights = config.pop("weights", None) or {}
    for k in ("age", "fair_share", "qos"):
        if k in weights:
            config["w_" + k] = weights[k]
    if "dynamic_fairshare" in config:
        config["fair_share_live"] = config.pop("dynamic_fairshare")
    unknown = sorted(k for k in config if k not in DEFAULT_CONFIG)
    if unknown:
        raise TypeError(
            "sv.evaluate: unknown setting(s) %s.\nThe knobs are: %s "
            "(or weights={...} and dynamic_fairshare, as in Parts 4-6)."
            % (", ".join(unknown), ", ".join(sorted(DEFAULT_CONFIG))))
    cfg.update(config)
    schedule = run_schedule(
        jobs,
        backfill=cfg["backfill"],
        weights={"age": cfg["w_age"], "fair_share": cfg["w_fair_share"],
                 "qos": cfg["w_qos"]},
        dynamic_fairshare=cfg["fair_share_live"],
        reserve_depth=cfg["reserve_depth"])
    m = metrics(schedule, window=window)
    _LEADERBOARD[:] = [e for e in _LEADERBOARD if e["name"] != name]
    _LEADERBOARD.append({"name": name, "cfg": cfg, "m": m, "schedule": schedule})
    if not quiet:
        scorecard()
    return schedule, m


def clear_leaderboard():
    _LEADERBOARD[:] = []


def scorecard():
    """Every configuration tried so far, on the four things the groups care about."""
    if not _LEADERBOARD:
        _card("Nothing evaluated yet: call <code>sv.evaluate(...)</code> first.")
        return
    axes = [("performance", ["makespan", "throughput"], ACCENT),
            ("user experience", ["avg_waiting", "avg_turnaround"], GREEN),
            ("infrastructure", ["utilisation", "gpu_utilisation"], ACCENT2),
            ("equity", ["fairness", "worst_group_waiting"], AMBER)]
    keys = [k for _, ks, _ in axes for k in ks]
    best = {}
    for k in keys:
        vals = [e["m"][k] for e in _LEADERBOARD]
        best[k] = min(vals) if _BETTER[k] == "low" else max(vals)
    head = "".join('<th style="text-align:right;padding:0 10px 7px;font-size:12px">%s'
                   '<div style="font-size:10px;color:#8a8fa3;font-weight:500;'
                   'white-space:nowrap">age %.2f · fair %.2f · qos %.2f<br>'
                   'backfill %s · depth %d</div></th>'
                   % (e["name"], e["cfg"]["w_age"], e["cfg"]["w_fair_share"],
                      e["cfg"]["w_qos"], "on" if e["cfg"]["backfill"] else "off",
                      e["cfg"]["reserve_depth"])
                   for e in _LEADERBOARD)
    rows = ""
    for axis, ks, col in axes:
        rows += ('<tr><td colspan="%d" style="padding:10px 0 3px;font-size:10.5px;'
                 'text-transform:uppercase;letter-spacing:.04em;font-weight:800;color:%s">'
                 '%s</td></tr>' % (len(_LEADERBOARD) + 1, col, axis))
        for k in ks:
            cells = ""
            for e in _LEADERBOARD:
                v = e["m"][k]
                win = abs(v - best[k]) < 1e-12
                cells += ('<td style="text-align:right;padding:5px 10px;font-weight:%s;'
                          'color:%s;font-variant-numeric:tabular-nums">%s%s</td>'
                          % ("800" if win else "500", GREEN if win else "#555",
                             _fmt(k, v), " ●" if win else ""))
            rows += ('<tr><td style="padding:5px 10px 5px 0;color:#444">%s</td>%s</tr>'
                     % (_LABEL[k], cells))
    _card('<div style="font-weight:800;font-size:15px;color:%s;margin-bottom:9px">'
          '🏁 Your policies, side by side</div>'
          '<table style="border-collapse:collapse;width:100%%;font-size:12.5px;color:%s">'
          '<thead><tr><th></th>%s</tr></thead><tbody>%s</tbody></table>'
          '<div style="font-size:11.5px;color:#666;margin-top:10px">There is no column that wins '
          'every row. Pick the one you are willing to defend to the four groups, and be ready to '
          'name who pays for it.</div>' % (INK, TEXT, head, rows), maxw=860)


# ===========================================================================
#  §12  The closing bridge: what you will actually type and read
# ===========================================================================
_SQUEUE = [
    ("JOBID   PARTITION   NAME       USER    ST   TIME  NODES  NODELIST(REASON)", None),
    ("4812301  normal     mri_recon  hkell    R   1:42      2  eu[0117-0118]",
     "Running. Two nodes, one hour forty-two so far. <b>TIME is elapsed time, not the limit</b>."),
    ("4812344  normal     ocean_9k   dweiss   R   0:07      4  eu[0203-0206]",
     "Running on four nodes. This is the wide rectangle everyone else has to fit around."),
    ("4812350  normal     calib_b    lmarti  PD   0:00      1  (Priority)",
     "Pending. Nothing is wrong with it: <b>other jobs simply scored higher</b>. "
     "Age, fair-share and QoS: Part 4."),
    ("4812351  normal     calib_c    lmarti  PD   0:00      1  (Priority)",
     "The same again. Two jobs from one user, both outranked."),
    ("4812352  normal     sweep_big  rjonas  PD   0:00      3  (Resources)",
     "Pending for a different reason: it is <b>top of the queue</b> and the machines are not free. "
     "This is the job that gets a reservation, and the hole in front of it is what backfilling "
     "fills: Part 5."),
    ("4812355  gpu        rl_train   akoch    R   3:20      1  eu-g3-004",
     "Running on a GPU node. Asking for a GPU narrows you to the nodes that have one."),
    ("4812361  normal     tiny_post  hkell   PD   0:00      1  (Resources)",
     "A one-node job, waiting behind a reservation. If its <code>--time</code> is short enough to "
     "finish before that reservation, it starts early. If not, it sits here."),
    ("4812370  normal     sweep_xl   rjonas  PD   0:00      3  (QOSMaxJobsPerUserLimit)",
     "Blocked by <b>policy, not by capacity</b>. The nodes may well be free; this user has already "
     "hit a limit their association allows."),
    ("4812377  normal     nightly    svc_ci  PD   0:00      2  (AssocGrpCPUMinutesLimit)",
     "The same family: the group has spent its allocation for the period. Fair-share pushed the "
     "priority down; this limit stops the job outright."),
    ("4812380  normal     recon_v2   hkell   PD   0:00      2  (Dependency)",
     "Waiting for another job to finish first. Nothing to do with the scheduler's opinion of it."),
]


def squeue_card():
    """Ten lines of `squeue`, annotated with the parts of this notebook."""
    rows = ""
    for line, note in _SQUEUE:
        if note is None:
            rows += ('<div style="font-family:ui-monospace,Menlo,monospace;font-size:11.5px;'
                     'color:#8a8fa3;padding:3px 0;border-bottom:1px solid #e2e5ef;'
                     'white-space:pre">%s</div>' % line)
        else:
            rows += ('<div style="padding:7px 0;border-bottom:1px solid #f2f3f8">'
                     '<div style="font-family:ui-monospace,Menlo,monospace;font-size:11.5px;'
                     'color:#2b2d6b;white-space:pre;overflow-x:auto">%s</div>'
                     '<div style="font-size:11.5px;color:#555;margin-top:3px;line-height:1.55">'
                     '↳ %s</div></div>' % (line, note))
    _card('<div style="font-weight:800;font-size:15px;color:%s;margin-bottom:3px">'
          '📋 <code>squeue</code>: the same ideas, on a real cluster</div>'
          '<div style="font-size:12px;color:#666;margin-bottom:9px">The reason in brackets is the '
          'scheduler telling you which part of this notebook you are standing in.</div>%s'
          % (INK, rows), maxw=880)


def sbatch_card():
    """One job script: two lines draw the rectangle, three more say what fits inside it."""
    lines = [
        ("#!/bin/bash", "a <b>Bash</b> script; #SBATCH lines are comments Slurm reads first"),
        ("#SBATCH --job-name=mri_recon", "a label, so this job is not just a number in the queue"),
        ("#SBATCH --nodes=2", "<b>how tall</b> the rectangle is: 2 whole nodes"),
        ("#SBATCH --ntasks-per-node=8", "8 separate processes started on each node. Not a "
                                        "size: how many pieces the job is split into."),
        ("#SBATCH --mem=64G", "64 GB reserved on each node, whether the job uses it or not"),
        ("#SBATCH --time=04:00:00", "<b>how wide</b> the rectangle is: killed at 4 hours, "
                                    "whatever it is doing"),
        ("#SBATCH --gpus-per-node=1", "1 GPU reserved on each node, and only nodes that have "
                                      "one qualify at all"),
        ("", None),
        ("srun python reconstruct.py --input $SCRATCH/scan_042",
         "the job itself, launched by <code>srun</code> on the nodes just reserved"),
    ]
    rows = ""
    for code, note in lines:
        rows += ('<div style="display:flex;flex-wrap:wrap;row-gap:2px;column-gap:14px;'
                 'align-items:baseline;padding:2px 0">'
                 '<div style="font-family:ui-monospace,Menlo,monospace;font-size:12px;'
                 'color:%s;flex:0 0 auto;white-space:pre">%s</div>'
                 '<div style="font-size:11.5px;color:#666;line-height:1.5;flex:1 1 260px;'
                 'min-width:200px">%s</div></div>'
                 % (ACCENT if note else "#555", code or "&nbsp;", note or ""))
    _card('<div style="font-weight:800;font-size:15px;color:%s;margin-bottom:9px">'
          '📝 Two lines draw the rectangle, three more say what fits inside it</div>%s'
          '<div style="font-size:12px;color:#444;margin-top:10px;background:#f6f7fb;'
          'border-radius:8px;padding:10px 12px;line-height:1.6"><code>--nodes</code> and '
          '<code>--time</code> are the same rectangle from Part 1: 2 nodes tall, 4 hours wide. '
          '<code>--ntasks-per-node</code>, <code>--mem</code> and <code>--gpus-per-node</code> '
          'are not extra dimensions of it. They say what has to fit inside the shape you already '
          'declared.</div>' % (INK, rows), maxw=820)


# ===========================================================================
#  §12b  Per-part bridges: the same idea, as a command you can actually type
# ===========================================================================
# One card per part. A line beginning with "$ " is something you type; every
# other line is output. Nothing here runs in the notebook - there is no Slurm
# on Colab - so the output is realistic rather than real.
_SLURM_DOCS = "https://slurm.schedmd.com/"

_SLURM_REF = {
    "part0": (
        "Part 0 &middot; what you have",
        [("$ sinfo", "everything the cluster has, grouped by partition and state"),
         ("PARTITION  AVAIL  TIMELIMIT  NODES  STATE  NODELIST", None),
         ("normal*       up 1-00:00:00      1    mix  node01",
          "<code>mix</code>: partly allocated. This is the half-used node in the map above."),
         ("normal*       up 1-00:00:00      1   idle  node02", None),
         ("gpu           up 1-00:00:00      2   idle  node[03-04]",
          "The GPU nodes are their own partition, which is one way a site keeps CPU jobs off "
          "them."),
         ("$ scontrol show node node03", "one node, in full"),
         ("CfgTRES=cpu=8,mem=64G,gres/gpu=1", "what the node has: <code>sv.NODE_SPECS</code>"),
         ("AllocTRES=cpu=4,mem=32G",
          "what jobs have taken out of it. The difference is your <code>free_cores</code>.")],
        "sinfo", "sinfo.html"),

    "part1": (
        "Part 1 &middot; the four numbers, on a real job",
        [("$ sacct -j 4812352", "what happened to a job that has already finished"),
         ("JobID    JobName    Partition  Account  AllocCPUS  State      ExitCode", None),
         ("4812352  sweep_big  normal     bio             24  COMPLETED  0:0",
          "Useful, but there is nothing here to subtract: no submit time, no start time."),
         ("$ sacct -j 4812352 --format=Submit,Start,End,Elapsed",
          "<code>--format</code> asks for the columns you actually want"),
         ("Submit               Start                End                  Elapsed", None),
         ("2026-03-02T08:59:41  2026-03-02T12:04:02  2026-03-02T16:04:10  04:00:08",
          "Submit to Start is the <b>waiting time</b> (3 h 04). Submit to End is the "
          "<b>turnaround</b>. Makespan and utilisation are not here at all: they belong to the "
          "cluster, not to any one job.")],
        "sacct", "sacct.html"),

    "part2": (
        "Part 2 &middot; your five steps, as two configuration lines",
        [("$ scontrol show config",
          "the whole configuration, a few hundred lines. Two of them are the loop you wrote."),
         ("SchedulerType    = sched/backfill", "Part 5, and the default nearly everywhere."),
         ("SelectType       = select/cons_tres",
          "<b>Cons</b>umable <b>tr</b>ackable <b>res</b>ources: cores, memory and GPUs counted "
          "separately on every node. That is your step 2 and your step 3."),
         ("$ scontrol show partition normal",
          "what a job is given when its submitter declares nothing"),
         ("DefaultTime=01:00:00  MaxTime=1-00:00:00  DefMemPerCPU=2048",
          "Both the default and the ceiling live on the <b>partition</b>, not on the job.")],
        "Scheduling Configuration Guide", "sched_config.html"),

    "part3": (
        "Part 3 &middot; declared versus real, in three commands",
        [("$ scontrol show job 4812361", "a job while it is still running"),
         ("RunTime=00:12:44  TimeLimit=03:00:00",
          "<code>TimeLimit</code> is <code>requested_time</code>. <code>RunTime</code> is the "
          "clock. Slurm compares those two and nothing else."),
         ("$ sacct -j 4812361", "the same job, afterwards"),
         ("4812361  fit_b  normal  clim  8  TIMEOUT  0:0",
          "<code>TIMEOUT</code> is the whole story: it reached its limit and was killed, and "
          "the node-hours it burnt are still charged to the group."),
         ("$ seff 4812361", "how much of what you asked for you actually used"),
         ("Wall-clock time: 00:41:03 of 03:00:00   Memory Efficiency: 21.44% of 64.00 GB",
          "The measure of your padding. <code>seff</code> ships in "
          "<code>slurm-contribs</code>, so most clusters have it but it is not Slurm proper.")],
        "sbatch, --time", "sbatch.html"),

    "part4": (
        "Part 4 &middot; the priority score, itemised",
        [("$ sprio -j 4812352", "why one pending job scored what it scored"),
         ("JOBID    PRIORITY   AGE  FAIRSHARE  JOBSIZE  PARTITION   QOS", None),
         ("4812352     10423   500       6923     1000       2000     0",
          "Your three terms, sitting next to the ones we left out. Nine factors in the real "
          "formula, same shape as yours."),
         ("$ sshare -U", "your own account: the shares you were given against what you used"),
         ("Account  User   RawShares  NormShares  RawUsage  EffectvUsage  FairShare", None),
         ("bio      hkell        400    0.400000   8123400      0.612000   0.114253",
          "<code>NormShares</code> is <b>S</b>, <code>EffectvUsage</code> is <b>U</b>, "
          "<code>FairShare</code> is <b>F</b>. Section 4.2, computed live."),
         ("$ scontrol show config", "the same file as Part 2. These are the weights."),
         ("PriorityWeightAge=1000  PriorityWeightFairshare=10000  PriorityWeightQOS=5000", None),
         ("PriorityDecayHalfLife=7-00:00:00",
          "Your half-life slider, in <code>slurm.conf</code>. Here: seven days.")],
        "Multifactor Priority Plugin", "priority_multifactor.html"),

    "part5": (
        "Part 5 &middot; the reservation, written down",
        [("$ squeue --start -j 4812352",
          "<code>--start</code> asks the backfiller when it is planning to run this"),
         ("JOBID    NAME       ST  START_TIME           NODES  SCHEDNODES     REASON", None),
         ("4812352  sweep_big  PD  2026-03-02T12:04:00      3  eu[0203-0205]  (Resources)",
          "<code>START_TIME</code> <b>is</b> the reservation, and <code>SCHEDNODES</code> are "
          "the machines being held for it. Everything backfilled has to finish before that "
          "timestamp."),
         ("$ scontrol show config", "one line decides how the holes in front of it get filled"),
         ("SchedulerParameters = bf_window=1440,bf_resolution=60,bf_max_job_test=100",
          "Two limits your simulator does not have: the backfiller looks <b>1440 minutes</b> "
          "ahead and tests only the first <b>100</b> pending jobs. Job 143 may fit the hole "
          "perfectly and never be asked.")],
        "Scheduling Configuration Guide", "sched_config.html"),

    "part6": (
        "Part 6 &middot; who actually got the machine",
        [("$ sshare -a", "every account, not just your own"),
         ("Account   RawShares  NormShares  EffectvUsage  FairShare", None),
         ("bio             400    0.400000      0.389000   0.509622", None),
         ("clim            300    0.300000      0.274000   0.530957", None),
         ("robo            200    0.200000      0.191000   0.515842", None),
         ("econ            100    0.100000      0.146000   0.363493",
          "<code>sv.group_card()</code> on a real week. <code>EffectvUsage</code> against "
          "<code>NormShares</code> is exactly the gap you argued about: Economics is at 14.6% "
          "of a cluster it has 10% of, so its factor is the one being pushed down.")],
        "Classic Fairshare Algorithm", "classic_fair_share.html"),

    "part7": (
        "Part 7 &middot; your configuration is somebody's Monday",
        [("$ sacctmgr show assoc", "administrator territory: the shares themselves"),
         ("Account  User   Shares  MaxJobs  GrpTRESMins",
          "Your <b>target shares</b>, as one integer per group."),
         ("$ sacctmgr show qos", "and the classes an administrator grants"),
         ("Name     Priority  MaxWall   GrpTRES",
          "Your <code>qos</code> column, with limits of its own that the notebook never models."),
         ("(you will not be able to run either of these as a user)",
          "Which is the point. The recommendation you just wrote is these two tables plus the "
          "weights from Part 4, in a file somebody has to sign off on and then defend.")],
        "sacctmgr", "sacctmgr.html"),
}


def slurm_ref(key):
    """The bridge card for one part: what the idea looks like as a command."""
    title, rows, doc_name, doc_page = _SLURM_REF[key]
    body = ""
    for line, note in rows:
        is_cmd = line.startswith("$ ")
        body += ('<div style="font-family:ui-monospace,Menlo,monospace;font-size:11.5px;'
                 'color:%s;font-weight:%s;white-space:pre;overflow-x:auto;padding:%s">%s</div>'
                 % (INK if is_cmd else "#8a8fa3", "700" if is_cmd else "400",
                    "7px 0 1px" if is_cmd else "1px 0", line))
        if note:
            body += ('<div style="font-size:11.5px;color:#555;margin:2px 0 4px;line-height:1.55">'
                     '&#8627; %s</div>' % note)
    _card('<div style="font-weight:800;font-size:15px;color:%s;margin-bottom:3px">'
          '&#128039; On the real cluster: %s</div>'
          '<div style="font-size:12px;color:#666;margin-bottom:9px">None of this runs here; '
          'there is no Slurm on Colab. It is what you type once you have an account on one.</div>'
          '%s<div style="font-size:11px;color:#8a8fa3;margin-top:11px;border-top:1px solid '
          '#f2f3f8;padding-top:8px">Reference: <a href="%s%s" style="color:%s">%s</a></div>'
          % (INK, title, body, _SLURM_DOCS, doc_page, ACCENT2, doc_name), maxw=880)


_CHEATSHEET = [
    ("What is in the cluster?",
     [("sinfo", "every node, its partition and its state"),
      ("scontrol show node &lt;node&gt;", "one node in full: CfgTRES against AllocTRES")]),
    ("Where is my job in the queue?",
     [("squeue -u $USER", "and read the reason in brackets"),
      ("squeue --start -j &lt;id&gt;", "the start time the backfiller is planning on"),
      ("sprio -j &lt;id&gt;", "why it scored what it scored")]),
    ("What did my job actually do?",
     [("sacct -j &lt;id&gt;", "how it ended, and whether it hit TIMEOUT"),
      ("seff &lt;id&gt;", "how much of what you asked for you used")]),
    ("Who has been using the machine?",
     [("sshare -U", "your shares, your usage, your fair-share factor"),
      ("sshare -a", "the same, for every group at once")]),
    ("How is this cluster configured?",
     [("scontrol show config", "the weights, the half-life and the backfill limits"),
      ("scontrol show partition &lt;name&gt;", "the default and maximum wall time you inherit")]),
]


def slurm_cheatsheet():
    """Everything from the eight bridge cards, grouped by the question you are asking."""
    blocks = ""
    for question, lines in _CHEATSHEET:
        rows = ""
        for cmd, note in lines:
            rows += ('<div style="display:flex;gap:12px;align-items:baseline;padding:2px 0;'
                     'flex-wrap:wrap"><div style="font-family:ui-monospace,Menlo,monospace;'
                     'font-size:11.5px;color:%s;min-width:330px;white-space:pre">%s</div>'
                     '<div style="font-size:11px;color:#666;line-height:1.5">%s</div></div>'
                     % (ACCENT, cmd, note))
        blocks += ('<div style="margin-bottom:11px"><div style="font-size:12.5px;font-weight:700;'
                   'color:%s;margin-bottom:3px">%s</div>%s</div>' % (INK, question, rows))
    _card('<div style="font-weight:800;font-size:15px;color:%s;margin-bottom:9px">'
          '&#128203; The whole notebook, as commands</div>%s'
          '<div style="font-size:11.5px;color:#444;background:#f6f7fb;border-radius:8px;'
          'padding:10px 12px;line-height:1.6">Every one of these answers a question you now know '
          'how to ask. The manual pages are at <a href="%s" style="color:%s">slurm.schedmd.com</a>'
          ' - <code>priority_multifactor</code>, <code>classic_fair_share</code> and '
          '<code>sched_config</code> are the three worth reading after this hour.</div>'
          % (INK, blocks, _SLURM_DOCS, ACCENT2), maxw=880)


# ===========================================================================
#  §13  Quiz banks  (options are shuffled at render time, so order means nothing)
# ===========================================================================
_MC_QUIZZES = {
    "why_scheduler": (
        "Why does a shared cluster need a program deciding who runs when?",
        "",
        ["Because two jobs started by hand at the same moment would land on the same node and "
         "fight over its cores and memory, so both run badly or crash",
         "Because the nodes would overheat if too many people used them at once",
         "Because a program can predict how long each job will take and plan around it",
         "Because jobs have to be compiled centrally before they can be run"],
        0,
        ["A job owns what it is allocated. Its cores, its memory and its GPUs are reserved "
         "for it until it finishes, and nobody else can use them. Two people starting jobs by "
         "hand on the same node break that rule. Both jobs compete for the same cores and the "
         "same memory, and both suffer.",
         "Overheating is not the problem. A node is built to run at full load for weeks, and "
         "the cooling is designed for exactly that. The real problem is two jobs being handed "
         "the same cores and the same memory at the same time.",
         "The scheduler predicts nothing. It only knows the time limit the user wrote in the "
         "job script, and that number is often wrong. Part 3 is about what happens then.",
         "Compilation has nothing to do with it. You may compile on the cluster or bring a "
         "program that is already built. What has to be organised is who gets which cores, "
         "and when."]),
    "equity_trap": (
        "A colleague wants to switch backfilling off, and points at the scorecard: it scores the "
        "best equity number on the whole board. What is wrong with the argument?",
        "",
        ["Equity only measures how the delivered hours were split between the groups. With "
         "backfilling off everyone is served much later, so the split can look fair while every "
         "group waits far longer",
         "The equity number is simply computed wrongly when backfilling is off",
         "Backfilling cannot affect equity at all, so that number has to be a coincidence",
         "Switching backfilling off also lowers the fair-share weight, and the two cancel out"],
        0,
        ["Equity is a <b>ratio</b>: it compares what each group received against what it was "
         "promised. A schedule where everybody waits three times longer can split those hours "
         "perfectly. That is why it has to be read next to the waiting time, never on its own.",
         "The number is right. It is the reading that is wrong: equity answers how the hours were "
         "shared out, and says nothing about how long anyone waited for them.",
         "It does affect it. Filling the holes changes which jobs start early, and those jobs "
         "belong to groups, so the delivered shares move.",
         "They are separate settings. Turning backfilling off leaves every weight exactly where "
         "you set it."]),
    "policy_tradeoff": (
        "Policy B raises the fair-share weight from 0.0 to 0.8. What happens to the average "
        "waiting time?",
        "Answer before you run the next cell. The workload is the same thirty jobs either way: "
        "the only thing that changes is the order they are considered in.",
        ["It goes up: serving the group that was under-served means the jobs that were in front "
         "of it now wait longer",
         "It goes down: a fairer order is a better order, so everybody is served sooner",
         "It does not change: reordering the queue moves jobs around, but the same work still "
         "has to run",
         "It goes up for the group that was over-served and stays the same for everybody else"],
        0,
        "Up, from 6.60 h to 9.17 h. This is Part 4 again, in a different costume: <b>ageing did "
        "not create anything, it moved the pain</b>, and neither does fair-share. Every job you "
        "promote is a job somebody else queues behind. The last option is the tempting one: the "
        "bill is not paid only by the group that was over-served. Robotics had nothing to do "
        "with the argument and its average wait goes from 1.2 h to 5.8 h."),
    "walltime": (
        "How does the scheduler know that a job will take three hours?",
        "In Part 2 the simulator moved the clock forward by each job's duration. Where did that "
        "number come from?",
        ["It measured the job the last time this user ran something similar",
         "The user wrote it in the job script, and the scheduler never learns the real duration",
         "It estimates the duration from the number of nodes and cores requested",
         "It watches the job for the first few minutes and extrapolates"],
        1,
        "<code>--time</code> is a declaration, not a measurement. Slurm plans entirely on the "
        "number the user typed, and finds out the truth only when the job either exits or hits the "
        "limit and is killed. Everything else in this notebook follows from that: a job is a "
        "promise about size <b>and</b> duration."),
    "better_for_whom": (
        "Policy A finishes the day three hours earlier and keeps more of the machine busy. "
        "Policy B delivers the shares the four groups agreed on. Which scheduler is better?",
        "Both tables are in front of you and neither of them is wrong.",
        ["The question is incomplete: better depends on who the cluster belongs to and what "
         "those groups were promised",
         "Policy A: more science came out of the machine that week, and that is measurable",
         "Policy B: the four groups signed an agreement, and honouring it is not optional",
         "Policy A, because utilisation is the only one of these numbers that can be measured "
         "objectively"],
        0,
        "The second and third options are both <b>defensible positions</b>, and you will meet "
        "people who hold them. What neither of them is, is a fact you can read off a table. The "
        "number that settles the argument is not in either column: it is who paid for the nodes "
        "and what they were told they would get. Notice also that utilisation is no more "
        "objective than equity: both are exact numbers, and neither tells you what to do."),
    "reservation": (
        "What is a reservation actually protecting against?",
        "J2 could not start, so Slurm worked out the earliest moment it could give J2 all four "
        "nodes, and wrote that moment down.",
        ["Against the big job being overtaken for ever by a stream of smaller jobs that keep "
         "arriving and keep fitting",
         "Against the cluster running out of memory while the big job waits",
         "Against the user cancelling the job before it starts",
         "Against two schedulers making conflicting decisions at the same time"],
        0,
        "Without a written-down start time, every round of scheduling would look at a wide job, "
        "find it does not fit, and let something narrower in instead. The wide job would keep "
        "losing, for ever. The reservation is what makes filling the holes in front of it "
        "<b>safe</b>: that is why reservation and backfilling are one mechanism, not two."),
    "fairshare": (
        "A group has consumed exactly its target share. What is its fair-share factor?",
        "With <b>F = 2<sup>−U/S</sup></b>, where U is the group's normalised consumption and S the "
        "target share the administrators assigned it.",
        ["1.0: nothing is held against a group that is on target",
         "0.5: the exponent works out to exactly −1",
         "0.0: the group has spent its entitlement",
         "It depends on how many jobs the group has in the queue right now"],
        1,
        "U = S gives U/S = 1 and F = 0.5, the equilibrium of the loop. Above target, U/S > 1 and F "
        "falls towards 0; below target it climbs towards 1. Nothing is ever reset: old consumption "
        "just decays, halving every half-life, until it stops mattering."),
    "declared_or_real": (
        "In the backfill test, which of the job's two durations is used?",
        "The test was <code>finish_time = current_time + ???</code>, then "
        "<code>finish_time &lt;= reserved_start</code>.",
        ["<code>actual_duration</code>, because the point is whether the job really finishes in time",
         "<code>requested_time</code>, because it is the only duration the scheduler has",
         "Whichever of the two is smaller, to be safe",
         "The average of the two, since neither is reliable on its own"],
        1,
        "The scheduler cannot see <code>actual_duration</code>: in this notebook that field exists "
        "only to move the simulated clock, and on a real cluster it does not exist until the job "
        "has ended. Backfilling is decided entirely on the declared number, which is why padding "
        "your wall time is what stops your job fitting in a hole."),
}

_TF_QUIZZES = {
    # 4 true, 4 false
    "metrics": ("Waiting, turnaround, makespan, utilisation", [
        ("Turnaround time is never smaller than waiting time for the same job.", True),
        ("Makespan is the span from the first job starting to the last one finishing, so it "
         "describes the whole workload rather than any single job.", True),
        ("Utilisation counts the node-hours the jobs occupied, divided by the node-hours the "
         "cluster had available over the makespan.", True),
        ("A schedule with the lowest average waiting time also has the highest utilisation.", False),
        ("Waiting time and turnaround time describe the cluster's performance rather than one "
         "user's experience.", False),
        ("Two schedules with the same makespan always have the same utilisation.", False),
        ("Idle node-hours are hours of machine that the groups paid for and nobody used.", True),
        ("Shortening the makespan cannot change the utilisation, since the same work gets done "
         "either way.", False),
    ]),
    # 5 true, 5 false
    "rectangle": ("A job as a rectangle", [
        ("Both sides of the rectangle are chosen by the person submitting the job.", True),
        ("A job that finishes early releases its nodes early, even though it declared more time.",
         True),
        ("A job that runs past its declared time is killed, and is still charged for the hours it "
         "burnt.", True),
        ("Slurm measures how long a job normally takes and corrects the declared time.", False),
        ("One node can hold several jobs at the same time, if its cores, memory and GPUs are "
         "enough for all of them.", True),
        ("Declaring more time than you need is free, because you are only charged for what you "
         "use.", False),
        ("Asking for more nodes always makes a job start sooner, since it is more important.",
         False),
        ("A job keeps the resources it was given for its whole run, and nothing else is scheduled "
         "onto them.", True),
        ("A job always occupies whole nodes, so two jobs can never share one.", False),
        ("Time a job declared but did not use is credited back to the group for its next job.",
         False),
    ]),
    # 3 true, 3 false
    "backfilling": ("Reservation and backfilling", [
        ("A backfilled job is allowed to start ahead of jobs with higher priority.", True),
        ("A job qualifies for backfilling only if its <i>declared</i> finish time is no later than "
         "the reserved start.", True),
        ("Backfilling can push the reserved job to a later start than the one it was promised.",
         False),
        ("Backfilling changes the order in which the remaining jobs run.", True),
        ("Backfilling helps most when every job in the queue is the same shape.", False),
        ("Turning backfilling on means the scheduler stops using priorities.", False),
    ]),
}

_NUMBER_QUIZZES = {
    "metrics_math": ("🔢 The four numbers, on the FIFO schedule you just drew", [
        ("All five jobs were submitted at 0h. J4 started at 9h. What was its <b>waiting time</b>, "
         "in hours?", 9.0, 0.01,
         "waiting_time = start_time − submission_time = 9 − 0 = 9."),
        ("J4 ran for 3 hours after starting at 9h. What was its <b>turnaround time</b>?", 12.0, 0.01,
         "turnaround_time = completion_time − submission_time = 12 − 0 = 12. Turnaround always "
         "contains the waiting."),
        ("The five jobs occupy 34 node-hours in total, and the schedule spans 12 hours on 4 nodes. "
         "What is the <b>utilisation</b>, as a percentage?", 70.83, 0.3,
         "34 busy node-hours ÷ (4 nodes × 12 hours = 48 available) = 0.708, so 70.8%. The missing "
         "14 node-hours are the holes in the picture."),
    ]),
    "backfill_math": ("🔢 The backfill test, by hand", [
        ("It is 0h. J2 is reserved to start at 3h. J3 asks for 3 nodes and declares 4 hours. "
         "What is its <code>finish_time</code>?", 4.0, 0.01,
         "finish_time = current_time + requested_time = 0 + 4 = 4. And 4 ≤ 3 is false, so J3 is "
         "refused: it would still be running when J2 is due."),
        ("Same moment, J4: 3 nodes, 3 declared hours. What is its <code>finish_time</code>?",
         3.0, 0.01,
         "0 + 3 = 3, and 3 ≤ 3 is true. Ending exactly on the reservation is allowed."),
        ("J4 really only needs 3 hours, but its user declares 5. What is its "
         "<code>finish_time</code> now?", 5.0, 0.01,
         "0 + 5 = 5. The test uses the <i>declared</i> number, so J4 no longer fits the hole, "
         "even though the work inside it is unchanged."),
    ]),
}


def mc_quiz(key):
    _mc_render(*_MC_QUIZZES[key])


def true_false_quiz(key):
    title, statements = _TF_QUIZZES[key]
    _tf_render(title, statements)


def number_quiz(key):
    title, questions = _NUMBER_QUIZZES[key]
    _nq_render(title, questions)


# ===========================================================================
#  §14  Final boss: timed true/false flash quiz with lives
# ===========================================================================
# Balanced pool (27 true / 27 false), phrased so neither answer is given away
# by the wording: no "always/never" tells, no absurd falses.
_FLASH_POOL = [
    # --- the shared-resource situation -------------------------------------
    ("A job owns the cores and memory it was allocated until it ends.", True),
    ("Several jobs can share one node when its resources cover all of them.", True),
    ("Free cores scattered across different nodes can still fail to fit a job that needs them "
     "on one node.", True),
    ("A cluster with free cores is a cluster with nothing waiting in its queue.", False),
    ("Handing out access in advance is what lets a job assume nobody else will touch its "
     "memory.", False),
    # --- the rectangle -----------------------------------------------------
    ("How many nodes a job gets is decided by the person who submits it.", True),
    ("How long a job is allowed to run is decided by the person who submits it.", True),
    ("Slurm learns a job's real duration from previous runs by the same user.", False),
    ("A job that exits early gives its nodes back before its declared time is up.", True),
    ("Requesting more nodes shortens the time a job waits in the queue.", False),
    # --- metrics -----------------------------------------------------------
    ("Waiting time is the gap between submitting a job and it starting.", True),
    ("Turnaround time is waiting time plus the time the job spent running.", True),
    ("Makespan describes one job's experience of the queue.", False),
    ("Utilisation compares the node-hours used against the node-hours available.", True),
    ("A schedule that finishes sooner leaves fewer idle node-hours behind, whatever else "
     "changes.", False),
    ("Average waiting time is the number a research group notices first.", True),
    ("One number is enough to say whether a schedule was good.", False),
    # --- wall time ---------------------------------------------------------
    ("Declaring less time than a job needs gets the job killed at the limit.", True),
    ("A job killed at its limit is charged for the node-hours it burnt.", True),
    ("Padding the declared wall time costs nothing, since billing follows real usage.", False),
    ("The scheduler plans around the declared duration rather than the real one.", True),
    ("A generous wall time makes a job easier to fit into a gap.", False),
    # --- priority and fair-share -------------------------------------------
    ("Slurm combines several factors into one priority number.", True),
    ("The age factor grows on its own while a job waits.", True),
    ("With the age weight at zero, a job's score stops improving as it waits.", True),
    ("The weights of a priority formula follow from the mathematics of scheduling.", False),
    ("A group's fair-share factor is 0.5 when its consumption sits on its target share.", True),
    ("Fair-share means every group is entitled to the same number of node-hours.", False),
    ("Consumption recorded by fair-share is deleted once the period ends.", False),
    ("Old consumption halves once per half-life until it stops mattering.", True),
    ("A group that submits nothing for a while sees its fair-share factor climb.", True),
    ("A high fair-share factor guarantees the group's next job starts immediately.", False),
    # --- reservation and backfilling ---------------------------------------
    ("A reservation is the earliest start the scheduler can guarantee a blocked job.", True),
    ("A backfilled job may start ahead of jobs that scored higher.", True),
    ("Backfilling delays the job the reservation was made for.", False),
    ("The backfill test compares the declared finish time against the reserved start.", True),
    ("Backfilling leaves the order of the remaining jobs unchanged.", False),
    ("Without reservations a wide job could be overtaken indefinitely.", True),
    ("Backfilling matters most when every queued job has the same shape.", False),
    ("Widening a job's declared time can cost it a slot it would otherwise have had.", True),
    ("Backfilling is a separate scheduler that runs whenever the main one is idle.", False),
    # --- efficiency against fairness ----------------------------------------
    ("A schedule can be better for the cluster and worse for a user at the same time.", True),
    ("Raising the fair-share weight is enough on its own to change who gets served.", False),
    # --- reading the queue --------------------------------------------------
    ("A job pending with reason (Priority) is waiting because other jobs scored higher.", True),
    ("A job pending with reason (Priority) is waiting because no node is free.", False),
    ("A job pending with reason (Resources) scored high enough but the machines are busy.", True),
    ("A job blocked by a QoS limit is blocked by capacity rather than by policy.", False),
    ("A memory request is checked before the job starts, not once it is running.", True),
    ("A job asking for a GPU can be placed on any node of the cluster.", False),
    ("Utilisation counts a node as busy while jobs are queued for it.", False),
    ("Raising the QoS weight moves every job forward in the queue.", False),
    ("A job's priority is fixed at submission and does not move afterwards.", False),
    ("Once the set of jobs is fixed, the makespan is fixed with it.", False),
    ("Turnaround time is the cluster's metric and makespan is the user's.", False),
    ("Under strict priority, a job further down the queue may start while the head is blocked.",
     False),
    ("The fair-share factor is computed from how many jobs a group has submitted.", False),
]


def flash_quiz(n_to_pass=10, lives=3, seconds=10):
    """Everything at once, one statement at a time, against the clock."""
    data = {"pool": [{"t": t, "ok": bool(v)} for t, v in _FLASH_POOL],
            "need": n_to_pass, "lives": lives, "secs": seconds}
    uid = _uid("fq", "flash_quiz")
    tmpl = r'''
<style>
#__UID__{font-family:__FONT__;border:1px solid #e6e8ee;border-radius:14px;padding:18px;max-width:760px;background:#fff;color:#24262b;text-align:center}
#__UID__ .fq-top{display:flex;justify-content:space-between;align-items:center;font-size:12.5px;font-weight:700;color:#2b2d6b;margin-bottom:12px}
#__UID__ .fq-bar{height:5px;background:#eef0f6;border-radius:3px;overflow:hidden;margin-bottom:16px}
#__UID__ .fq-fill{height:100%;background:linear-gradient(90deg,#667eea,#764ba2);width:100%;transition:width .1s linear}
#__UID__ .fq-stmt{font-size:16px;line-height:1.55;color:#2b2d6b;min-height:70px;display:flex;align-items:center;justify-content:center;padding:0 10px}
#__UID__ .fq-btns{display:flex;gap:12px;justify-content:center;margin-top:14px}
#__UID__ .fq-b{cursor:pointer;border:none;border-radius:10px;padding:11px 34px;font-size:14px;font-weight:800;color:#fff}
#__UID__ .fq-t{background:#2e9e7a}
#__UID__ .fq-f{background:#c0554e}
#__UID__ .fq-go{background:linear-gradient(135deg,#667eea,#764ba2)}
#__UID__ .fq-msg{font-size:13px;color:#555;margin-top:12px;min-height:22px;line-height:1.55}
</style>
<div id="__UID__">
  <div class="fq-top"><span class="fq-score"></span><span class="fq-lives"></span></div>
  <div class="fq-bar"><div class="fq-fill"></div></div>
  <div class="fq-stmt"></div>
  <div class="fq-btns"></div>
  <div class="fq-msg"></div>
</div>
<script>
(function(){
  const D=__DATA__, root=document.getElementById("__UID__");
  let pool=[], i=0, score=0, lives=D.lives, timer=null, left=0, live=false;
  const stmt=root.querySelector(".fq-stmt"), btns=root.querySelector(".fq-btns");
  const msg=root.querySelector(".fq-msg"), fill=root.querySelector(".fq-fill");
  function shuffled(){const a=D.pool.slice();
    for(let k=a.length-1;k>0;k--){const j=Math.floor(Math.random()*(k+1));[a[k],a[j]]=[a[j],a[k]];}
    return a;}
  function head(){
    root.querySelector(".fq-score").textContent="✅ "+score+" / "+D.need;
    root.querySelector(".fq-lives").textContent="❤️".repeat(Math.max(0,lives));
  }
  function stop(){if(timer){clearInterval(timer);timer=null;}}
  function start(){
    pool=shuffled(); i=0; score=0; lives=D.lives; live=true; next();
  }
  function next(){
    stop();
    if(score>=D.need) return finish(true);
    if(lives<=0) return finish(false);
    if(i>=pool.length) return finish(score>=D.need);
    head(); stmt.textContent=pool[i].t;
    btns.innerHTML='<button class="fq-b fq-t">TRUE</button><button class="fq-b fq-f">FALSE</button>';
    btns.querySelector(".fq-t").addEventListener("click",()=>answer(true));
    btns.querySelector(".fq-f").addEventListener("click",()=>answer(false));
    msg.textContent="";
    left=D.secs*1000;
    timer=setInterval(()=>{left-=100; fill.style.width=(100*left/(D.secs*1000))+"%";
      if(left<=0){stop(); lives--; msg.innerHTML="⏱️ Out of time. The answer was <b>"
        +(pool[i].ok?"TRUE":"FALSE")+"</b>."; i++; head(); setTimeout(next,1300);}},100);
  }
  function answer(v){
    stop();
    const ok=(v===pool[i].ok);
    if(ok){score++; msg.innerHTML="✅ Correct.";}
    else{lives--; msg.innerHTML="❌ It was <b>"+(pool[i].ok?"TRUE":"FALSE")+"</b>.";}
    i++; head(); setTimeout(next,ok?600:1400);
  }
  function finish(won){
    live=false; stop(); fill.style.width="100%";
    stmt.innerHTML=won?"🏁 <b>Cleared.</b> "+score+" correct, "+lives+" live"+(lives===1?"":"s")+" left."
                      :"💀 <b>Out of lives</b> at "+score+" / "+D.need+".";
    msg.innerHTML=won?"You can explain a rectangle, a reservation and a decaying fair-share factor. "
                      +"That is the whole hour."
                     :"Scroll back to the part that caught you out, then run this again.";
    btns.innerHTML='<button class="fq-b fq-go">'+(won?"Play again":"Try again")+'</button>';
    btns.querySelector(".fq-go").addEventListener("click",start);
  }
  stmt.innerHTML="<b>3 lives · "+D.secs+" seconds per statement · "+D.need+" correct to clear.</b>";
  msg.textContent="Everything from the whole notebook, in no particular order.";
  btns.innerHTML='<button class="fq-b fq-go">Start</button>';
  btns.querySelector(".fq-go").addEventListener("click",start);
  head(); 
})();
</script>'''
    display(HTML(tmpl.replace("__UID__", uid).replace("__FONT__", _FONT)
                 .replace("__DATA__", _json.dumps(data))))
