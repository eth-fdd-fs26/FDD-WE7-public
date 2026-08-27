"""Presentation, plotting & quiz helpers for the WE7 notebook 02:
"One backbone, many tasks: parameter-efficient fine-tuning".

Same idea as WE7's `sched_viz` and WE3's `dp_viz`: every HTML/CSS
illustration, interactive widget and quiz *answer key* lives here, out of the
notebook, so the teaching cells stay about the *idea*.

Nothing here trains anything. The model, the data, the training loop and every
number that ends up on a dashboard are built in the notebook itself, in plain
sight; this file only draws them.

    import peft_viz as pv
    pv.dashboard(record, baseline=records[0])

Students are told not to read this file. (You are, presumably, not a student.)
"""
import json as _json
import os as _os

import numpy as _np
import matplotlib.pyplot as _plt
from IPython.display import HTML, display

# ===========================================================================
#  §0  Palette: one place, so every picture colours the same thing the same
# ===========================================================================
# Every widget paints its own white surface, so it must paint its own text
# colour too: in a dark-themed notebook the inherited colour is near-white
# and would vanish against that surface.
TEXT = "#24262b"
INK = "#2b2d6b"
ACCENT = "#764ba2"          # storage / checkpoints
ACCENT2 = "#667eea"
FROZEN = "#8b93ab"          # frozen, shared backbone
FROZEN_BG = "#eef0f6"
TRAIN = "#e08a1e"           # trainable, task-specific
TRAIN_BG = "#fdf1de"
FULL = "#c0554e"            # full fine-tuning
GREEN = "#2e9e7a"           # performance
GREY = "#9aa0b5"

METHOD_COLOR = {
    "Full fine-tuning": FULL,
    "BitFit": "#3fa7c4",
    "LoRA": TRAIN,
    "Diff Pruning": GREEN,
}

_FONT = ("-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,'Helvetica Neue',"
         "Arial,sans-serif")

_UID_N = [0]


def _uid(tag, key=""):
    """Unique per call, so two copies of a widget on one page never collide."""
    _UID_N[0] += 1
    return "%s_%d_%d" % (tag, abs(hash(str(key))) % 10 ** 6, _UID_N[0])


def _card(inner, maxw=880):
    display(HTML(
        '<div style="font-family:%s;border:1px solid #e6e8ee;border-radius:14px;'
        'padding:18px;max-width:%dpx;background:#fff;color:%s">%s</div>'
        % (_FONT, maxw, TEXT, inner)))


def _human_bytes(n):
    """Bytes as something a person reads, in the decimal units disks are sold in
    (1 KB = 1000 B), so the arithmetic in the exercises comes out round."""
    n = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1000.0 or unit == "TB":
            if unit == "B":
                return "%d B" % int(round(n))
            return "%.1f %s" % (n, unit)
        n /= 1000.0
    return "%.1f TB" % n


def _num(n):
    """Thousands separator, so 4482 does not read as 4 482 000."""
    return "{:,}".format(int(n))


def _bar(frac, color, height=9, bg="#eef0f6"):
    frac = max(0.0, min(1.0, float(frac)))
    return ('<div style="background:%s;border-radius:5px;height:%dpx;width:100%%">'
            '<div style="background:%s;border-radius:5px;height:%dpx;width:%.1f%%"></div>'
            '</div>' % (bg, height, color, height, 100.0 * frac))


# ===========================================================================
#  §1  Quiz renderers  (verbatim house style: options shuffled at render time)
# ===========================================================================
def _mc_render(title, question, options, answer_index, reveal):
    data = {"opts": list(options), "ans": int(answer_index), "reveal": reveal}
    uid = _uid("mc", (question, tuple(options), answer_index))
    tmpl = r'''
<style>
#__UID__{font-family:__FONT__;border:1px solid #e6e8ee;border-radius:14px;padding:16px;max-width:800px;background:#fff;color:#24262b}
#__UID__ .mc-head{font-weight:800;font-size:15px;margin-bottom:4px}
#__UID__ .mc-q{color:#444;font-size:13.5px;margin-bottom:12px;line-height:1.55}
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
</style>
<div id="__UID__">
  <div class="mc-head">__TITLE__</div>
  <div class="mc-q">__Q__</div>
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
      if(i===D.ans)o.classList.add("ok"); else if(i===sel)o.classList.add("no");});
    root.querySelector(".mc-rev").innerHTML=(sel===D.ans?"✅ Correct. ":"❌ Not quite. ")+D.reveal;
  });
})();
</script>'''
    display(HTML(tmpl.replace("__UID__", uid).replace("__FONT__", _FONT)
                 .replace("__TITLE__", title).replace("__Q__", question)
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
#  §2  Part 0. one model per task, and what that costs to store
# ===========================================================================
_TASK_TINT = ["#4a5bd0", "#c9548f", "#2e9e7a", "#e0a500", "#8d6ec9"]


def _block(w, h, bg, border, top, bottom, tcol="#fff"):
    return ('<div style="width:%dpx;height:%dpx;background:%s;border:1px solid %s;'
            'border-radius:9px;display:flex;flex-direction:column;align-items:center;'
            'justify-content:center;color:%s;font-size:11.5px;line-height:1.35;'
            'text-align:center;padding:2px">'
            '<div style="font-weight:800">%s</div><div style="opacity:.9">%s</div></div>'
            % (w, h, bg, border, tcol, top, bottom))


def one_model_per_task(base_parameter_count=1_000_000_000, bytes_per_param=2,
                       module_fraction=0.005, n_tasks=3):
    """The picture the whole notebook keeps coming back to.

    Left: one complete fine-tuned copy per task.  Right: one shared backbone
    plus a small task module per task.  Storage numbers are written inside the
    blocks, so nobody has to read a bar length.
    """
    full = base_parameter_count * bytes_per_param
    mod = full * module_fraction
    names = ["Task A", "Task B", "Task C", "Task D", "Task E"][:n_tasks]

    left = "".join(
        _block(150, 84, FULL, "#a8443e", n + " · full copy",
               _human_bytes(full) + "<br>every parameter changed")
        for n in names)
    mods = "".join(
        _block(96, 52, _TASK_TINT[i % len(_TASK_TINT)], "#00000022",
               names[i] + " module", _human_bytes(mod))
        for i in range(n_tasks))

    html = (
        '<div style="display:flex;gap:26px;flex-wrap:wrap;align-items:flex-start">'
        '<div>'
        '<div style="font-weight:800;color:%s;margin-bottom:8px">Full fine-tuning</div>'
        '<div style="display:flex;gap:10px">%s</div>'
        '<div style="margin-top:9px;font-size:12px;color:#555">stored in total: '
        '<b style="color:%s">%s</b></div></div>'
        '<div style="width:1px;background:#e6e8ee;align-self:stretch"></div>'
        '<div>'
        '<div style="font-weight:800;color:%s;margin-bottom:8px">PEFT</div>'
        '<div style="display:flex;gap:14px;align-items:center">%s'
        '<div style="display:flex;flex-direction:column;gap:8px">%s</div></div>'
        '<div style="margin-top:9px;font-size:12px;color:#555">stored in total: '
        '<b style="color:%s">%s</b> &nbsp;=&nbsp; one backbone + %d small modules</div>'
        '</div></div>'
        '<div style="margin-top:14px;font-size:12px;color:#444;background:#f6f7fb;'
        'border-radius:8px;padding:10px 12px;line-height:1.6">The backbone is the same '
        'in both pictures and it never disappears: PEFT changes <b>what has to be trained '
        'and what has to be saved for each new task</b>, not how big the pretrained model '
        'is. Numbers here assume a %s-parameter model at %d bytes per parameter and a task '
        'module of %.1f%% of it.</div>'
        % (INK, left, FULL, _human_bytes(full * n_tasks),
           INK,
           _block(150, 84 + 62 * (n_tasks - 1) // 2, FROZEN, "#767d94",
                  "shared backbone", _human_bytes(full) + "<br>frozen, stored once"),
           mods, ACCENT, _human_bytes(full + mod * n_tasks), n_tasks,
           _num(base_parameter_count), bytes_per_param, 100 * module_fraction))
    _card(html, maxw=900)


def storage_scenario(base_parameter_count, number_of_tasks,
                     module_fraction=0.005, bytes_per_param=2):
    """The same comparison as a bill: two totals, the ratio, and the arithmetic."""
    per_copy = base_parameter_count * bytes_per_param
    per_module = per_copy * module_fraction
    full_total = per_copy * number_of_tasks
    peft_total = per_copy + per_module * number_of_tasks
    ratio = full_total / peft_total if peft_total else float("nan")

    rows = [
        ("One full copy per task", full_total, FULL,
         "%d tasks x %s" % (number_of_tasks, _human_bytes(per_copy))),
        ("Shared backbone + one module per task", peft_total, ACCENT,
         "%s once + %d x %s" % (_human_bytes(per_copy), number_of_tasks,
                                _human_bytes(per_module))),
    ]
    body = ""
    for label, total, color, how in rows:
        body += (
            '<div style="margin-bottom:12px">'
            '<div style="display:flex;justify-content:space-between;font-size:13px;'
            'margin-bottom:4px"><b>%s</b><b style="color:%s">%s</b></div>%s'
            '<div style="font-size:11.5px;color:#777;margin-top:3px">%s</div></div>'
            % (label, color, _human_bytes(total),
               _bar(total / max(full_total, 1), color, height=11), how))
    body += ('<div style="font-size:13px;color:%s;background:#f6f7fb;border-radius:8px;'
             'padding:10px 12px"><b>%.1fx less storage</b> for %d tasks, and the gap grows '
             'with every task added, because only the small module is per-task.</div>'
             % (INK, ratio, number_of_tasks))
    _card('<div style="font-weight:800;font-size:15px;color:%s;margin-bottom:12px">'
          '💾 Storing %d adapted versions of a %s-parameter model</div>%s'
          % (INK, number_of_tasks, _num(base_parameter_count), body), maxw=760)


# ===========================================================================
#  §3  The repeated dashboard: drawn again after every method
# ===========================================================================
#  A record is a plain dict built in the notebook, in full view:
#      method, total_params, trainable_params, train_state_bytes, ckpt_bytes,
#      target_acc, source_acc, train_time_s, infer_ms, changes_path, mergeable
# ---------------------------------------------------------------------------

def param_strip(total_params, trainable_params, title="Parameters", note=""):
    """One rectangle cut in proportion: frozen on the left, trainable on the right."""
    frac = trainable_params / float(total_params)
    frozen = total_params - trainable_params
    wtr = max(frac * 100.0, 0.8)        # keep a sliver visible even at 0.1%
    html = (
        '<div style="font-weight:800;font-size:14px;color:%s;margin-bottom:9px">%s</div>'
        '<div style="display:flex;height:34px;border-radius:8px;overflow:hidden;'
        'border:1px solid #dfe3ee">'
        '<div style="width:%.2f%%;background:%s;display:flex;align-items:center;'
        'justify-content:center;color:#fff;font-size:11.5px;font-weight:700">frozen</div>'
        '<div style="width:%.2f%%;background:%s"></div></div>'
        '<div style="display:flex;justify-content:space-between;font-size:12px;'
        'margin-top:7px"><span style="color:%s"><b>%s</b> frozen &amp; shared</span>'
        '<span style="color:%s"><b>%s</b> trainable (%.2f%%)</span></div>%s'
        % (INK, title, 100.0 - wtr, FROZEN, wtr, TRAIN,
           FROZEN, _num(frozen), TRAIN, _num(trainable_params), 100.0 * frac,
           ('<div style="font-size:11.5px;color:#777;margin-top:6px">%s</div>' % note)
           if note else ""))
    _card(html, maxw=720)


def memory_stack(total_params, trainable_params, bytes_per_param=4,
                 title="Training state, the simplified estimate"):
    """Weights + gradients + two optimiser moments, as stacked blocks with numbers."""
    w = total_params * bytes_per_param
    g = trainable_params * bytes_per_param
    o = trainable_params * 2 * bytes_per_param
    tot = w + g + o
    parts = [("parameters (all of them, frozen included)", w, FROZEN),
             ("gradients (trainable only)", g, TRAIN),
             ("optimiser moments, 2 per trainable parameter", o, ACCENT)]
    rows = ""
    for label, val, color in parts:
        rows += (
            '<div style="margin-bottom:9px">'
            '<div style="display:flex;justify-content:space-between;font-size:12.5px;'
            'margin-bottom:3px"><span>%s</span><b style="color:%s">%s</b></div>%s</div>'
            % (label, color, _human_bytes(val), _bar(val / float(tot), color)))
    _card('<div style="font-weight:800;font-size:14px;color:%s;margin-bottom:10px">%s</div>%s'
          '<div style="border-top:1px solid #eceef5;margin-top:6px;padding-top:8px;'
          'font-size:13px;display:flex;justify-content:space-between">'
          '<b>total</b><b style="color:%s">%s</b></div>'
          '<div style="font-size:11.5px;color:#777;margin-top:8px;line-height:1.55">'
          'Activations, temporary buffers and allocator overhead are <b>not</b> counted here. '
          'This is an estimate of the three things whose size follows directly from the '
          'parameter counts, not a measurement of what the framework allocates.</div>'
          % (INK, title, rows, INK, _human_bytes(tot)), maxw=720)


def _cell(label, value, color=INK, sub=""):
    return ('<div style="flex:1 1 130px;min-width:130px">'
            '<div style="font-size:11px;color:#7b8194;text-transform:uppercase;'
            'letter-spacing:.4px">%s</div>'
            '<div style="font-size:19px;font-weight:800;color:%s;margin-top:2px">%s</div>'
            '<div style="font-size:11px;color:#8a90a3;margin-top:1px">%s</div></div>'
            % (label, color, value, sub or "&nbsp;"))


def _rel(value, base):
    if not base:
        return ""
    return "%.1f%% of full fine-tuning" % (100.0 * value / float(base))


def dashboard(rec, baseline=None, title=None):
    """The card drawn after every method. Numbers first, bars second."""
    color = METHOD_COLOR.get(rec["method"], ACCENT)
    b = baseline or {}
    pct = 100.0 * rec["trainable_params"] / float(rec["total_params"])

    top = "".join([
        _cell("total params", _num(rec["total_params"]), FROZEN,
              "everything a forward pass needs"),
        _cell("trainable", _num(rec["trainable_params"]), TRAIN, "%.2f%% of the model" % pct),
        _cell("training state", _human_bytes(rec["train_state_bytes"]), ACCENT2,
              _rel(rec["train_state_bytes"], b.get("train_state_bytes"))),
        _cell("task checkpoint", _human_bytes(rec["ckpt_bytes"]), ACCENT,
              _rel(rec["ckpt_bytes"], b.get("ckpt_bytes"))),
    ])
    bottom = "".join([
        _cell("target accuracy", "%.1f%%" % (100 * rec["target_acc"]), GREEN,
              "the task we adapted to"),
        _cell("source accuracy", "%.1f%%" % (100 * rec["source_acc"]), GREY,
              "the task it was pretrained on"),
        _cell("training time", "%.2f s" % rec["train_time_s"], INK,
              _rel(rec["train_time_s"], b.get("train_time_s"))),
        _cell("inference", "%.3f ms" % rec["infer_ms"], INK,
              _rel(rec["infer_ms"], b.get("infer_ms"))),
    ])

    bars = ""
    for label, key, col in (("trainable parameters", "trainable_params", TRAIN),
                            ("training-state memory", "train_state_bytes", ACCENT2),
                            ("task checkpoint", "ckpt_bytes", ACCENT),
                            ("training time", "train_time_s", GREY)):
        ref = b.get(key)
        if not ref:
            continue
        frac = rec[key] / float(ref)
        # anything above the baseline is drawn in the full-fine-tuning colour, so a
        # bar that fills the row cannot be mistaken for a saving
        shown = col if frac <= 1.0 else FULL
        bars += ('<div style="margin-bottom:7px">'
                 '<div style="display:flex;justify-content:space-between;font-size:12px;'
                 'margin-bottom:3px"><span>%s</span><b style="color:%s">%.1f%%</b></div>%s</div>'
                 % (label, shown, 100 * frac, _bar(frac, shown)))
    if bars:
        bars = ('<div style="margin-top:14px;border-top:1px solid #eceef5;padding-top:12px">'
                '<div style="font-size:11.5px;color:#7b8194;margin-bottom:8px">'
                'relative to <b>full fine-tuning</b> (100%% = the whole bar)</div>%s</div>' % bars)

    tags = ""
    for yes, no, on in (("changes the execution path", "leaves the execution path alone",
                         rec.get("changes_path")),
                        ("mergeable into the base weights", "cannot be merged away",
                         rec.get("mergeable"))):
        tags += ('<span style="font-size:11px;border-radius:20px;padding:3px 10px;'
                 'margin-right:6px;background:%s;color:%s;border:1px solid %s">%s</span>'
                 % (("#fdf1de" if on else "#eef0f6"), (TRAIN if on else "#7b8194"),
                    ("#f0d9b4" if on else "#dfe3ee"), (yes if on else no)))

    _card('<div style="display:flex;align-items:center;gap:10px;margin-bottom:14px">'
          '<div style="width:12px;height:12px;border-radius:3px;background:%s"></div>'
          '<div style="font-weight:800;font-size:16px;color:%s">%s</div></div>'
          '<div style="display:flex;gap:14px;flex-wrap:wrap;margin-bottom:14px">%s</div>'
          '<div style="display:flex;gap:14px;flex-wrap:wrap">%s</div>%s'
          '<div style="margin-top:14px">%s</div>'
          % (color, INK, title or rec["method"], top, bottom, bars, tags), maxw=880)


# ===========================================================================
#  §4  Plots: decision boundaries, sweeps, the Pareto view, the sparse grid
# ===========================================================================
_CMAP_PTS = ["#3f5bd0", "#e0721e"]


def _scatter(ax, X, y, alpha=0.85, s=12, edge=True):
    X = _np.asarray(X)
    y = _np.asarray(y)
    for k in (0, 1):
        m = y == k
        ax.scatter(X[m, 0], X[m, 1], s=s, c=_CMAP_PTS[k], alpha=alpha,
                   linewidths=(0.4 if edge else 0), edgecolors="white", zorder=3)


def boundaries(panels, X, y, title=None, ncols=None, figsize_per=3.1):
    """Decision boundaries side by side, on the same axes limits.

    panels : list of (label, predict_fn); predict_fn takes an (N, 2) float array
             and returns N predicted class labels.
    X, y   : the points to draw on top of every panel (the target task).
    """
    X = _np.asarray(X, dtype=float)
    pad = 0.6
    x0, x1 = X[:, 0].min() - pad, X[:, 0].max() + pad
    y0, y1 = X[:, 1].min() - pad, X[:, 1].max() + pad
    gx, gy = _np.meshgrid(_np.linspace(x0, x1, 220), _np.linspace(y0, y1, 220))
    grid = _np.c_[gx.ravel(), gy.ravel()].astype("float32")

    n = len(panels)
    ncols = ncols or min(n, 3)
    nrows = (n + ncols - 1) // ncols
    fig, axes = _plt.subplots(nrows, ncols,
                              figsize=(figsize_per * ncols, figsize_per * nrows))
    axes = _np.atleast_1d(axes).ravel()
    for ax, (label, fn) in zip(axes, panels):
        zz = _np.asarray(fn(grid)).reshape(gx.shape)
        ax.contourf(gx, gy, zz, levels=[-0.5, 0.5, 1.5],
                    colors=["#dfe4fb", "#fbe7d5"], alpha=0.95)
        ax.contour(gx, gy, zz, levels=[0.5], colors=[INK], linewidths=1.2)
        _scatter(ax, X, y)
        ax.set_title(label, fontsize=10.5, color=INK, fontweight="bold")
        ax.set_xticks([]), ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_edgecolor("#dfe3ee")
    for ax in axes[n:]:
        ax.axis("off")
    if title:
        fig.suptitle(title, fontsize=12, color=INK, fontweight="bold")
    fig.tight_layout()
    _plt.show()


def show_tasks(Xs, ys, Xt, yt, titles=("Source task", "Target task")):
    """The two datasets, same colours, same axes, so the shift is visible."""
    fig, axes = _plt.subplots(1, 2, figsize=(7.6, 3.5), sharex=True, sharey=True)
    for ax, X, y, t in zip(axes, (Xs, Xt), (ys, yt), titles):
        _scatter(ax, X, y, s=14)
        ax.set_title(t, fontsize=11, color=INK, fontweight="bold")
        ax.grid(alpha=0.15)
        for sp in ax.spines.values():
            sp.set_edgecolor("#dfe3ee")
    fig.tight_layout()
    _plt.show()


def sweep_plot(xs, panels, xlabel, title=None, logx=False):
    """One small axis per panel, shared x.

    panels: (ylabel, values, colour) or (ylabel, values, colour, ylim), where
    ylim is a (low, high) pair; either end may be None. Give a panel an
    explicit ylim whenever autoscaling would turn measurement noise into a
    trend.
    """
    fig, axes = _plt.subplots(1, len(panels), figsize=(3.4 * len(panels), 3.0))
    axes = _np.atleast_1d(axes).ravel()
    for ax, panel in zip(axes, panels):
        ylabel, vals, color = panel[0], panel[1], panel[2]
        ylim = panel[3] if len(panel) > 3 else None
        ax.plot(xs, vals, "o-", color=color, linewidth=2, markersize=6)
        ax.set_xlabel(xlabel, fontsize=9.5)
        ax.set_ylabel(ylabel, fontsize=9.5, color=color)
        if ylim:
            ax.set_ylim(ylim[0], ylim[1])
        if logx:
            ax.set_xscale("log", base=2)
        ax.set_xticks(list(xs))
        ax.get_xaxis().set_major_formatter(_plt.matplotlib.ticker.ScalarFormatter())
        ax.grid(alpha=0.2)
        ax.tick_params(labelsize=8.5)
        for sp in ax.spines.values():
            sp.set_edgecolor("#dfe3ee")
    if title:
        fig.suptitle(title, fontsize=11.5, color=INK, fontweight="bold")
    fig.tight_layout()
    _plt.show()


def pareto(records, annotate=True):
    """Target accuracy against trainable parameters, log x. One dot per method."""
    fig, ax = _plt.subplots(figsize=(6.6, 4.0))
    for r in records:
        c = METHOD_COLOR.get(r["method"], ACCENT)
        ax.scatter(r["trainable_params"], 100 * r["target_acc"], s=130, c=c,
                   edgecolors="white", linewidths=1.4, zorder=3)
        if annotate:
            ax.annotate(r["method"], (r["trainable_params"], 100 * r["target_acc"]),
                        textcoords="offset points", xytext=(9, 6),
                        fontsize=9.5, color=c, fontweight="bold")
    ax.set_xscale("log")
    ax.set_xlabel("trainable parameters (log scale)", fontsize=10)
    ax.set_ylabel("target accuracy (%)", fontsize=10)
    ax.grid(alpha=0.2)
    ax.margins(x=0.25)
    for sp in ax.spines.values():
        sp.set_edgecolor("#dfe3ee")
    ax.set_title("Cheap on the left, accurate at the top", fontsize=11.5,
                 color=INK, fontweight="bold")
    fig.tight_layout()
    _plt.show()


def normalised_bars(records, baseline=None):
    """Trainable parameters, estimated memory and checkpoint size, each as a
    percentage of full fine-tuning."""
    base = baseline or records[0]
    keys = [("trainable_params", "trainable parameters", TRAIN),
            ("train_state_bytes", "training-state memory", ACCENT2),
            ("ckpt_bytes", "task checkpoint", ACCENT)]
    names = [r["method"] for r in records]
    x = _np.arange(len(records))
    w = 0.26
    fig, ax = _plt.subplots(figsize=(1.7 * len(records) + 3.4, 4.0))
    for i, (key, label, color) in enumerate(keys):
        vals = [100.0 * r[key] / float(base[key]) for r in records]
        bars = ax.bar(x + (i - 1) * w, vals, w, label=label, color=color)
        for b_, v in zip(bars, vals):
            ax.text(b_.get_x() + b_.get_width() / 2, v * 1.08,
                    ("%.1f" % v) if v < 10 else ("%.0f" % v),
                    ha="center", fontsize=8, color=color, fontweight="bold")
    ax.set_yscale("log")
    ax.set_ylim(0.5, 400)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=9.5)
    ax.set_ylabel("% of full fine-tuning (log scale)", fontsize=10)
    ax.axhline(100, color=FULL, linewidth=1, linestyle="--", alpha=0.7)
    ax.legend(fontsize=9, frameon=False)
    ax.grid(alpha=0.2, axis="y")
    for sp in ax.spines.values():
        sp.set_edgecolor("#dfe3ee")
    fig.tight_layout()
    _plt.show()


def diff_grid(delta, threshold, width=64,
              title="The learned difference, one cell per parameter"):
    """The update before and after thresholding: grey is exactly zero.

    A 1-D update (the whole model's difference, parameter by parameter) is
    padded and folded into a grid; a 2-D one is drawn as it is.
    """
    from matplotlib.colors import LinearSegmentedColormap, PowerNorm
    d = _np.abs(_np.asarray(delta, dtype=float))
    if d.ndim == 1:
        pad = (-d.size) % width
        d = _np.concatenate([d, _np.full(pad, _np.nan)]).reshape(-1, width)
    kept = d >= threshold
    vmax = float(_np.nanmax(d)) or 1.0
    cmap = LinearSegmentedColormap.from_list("peft", ["#eef0f6", TRAIN])
    cmap.set_bad("#ffffff")
    norm = PowerNorm(gamma=0.4, vmin=0.0, vmax=vmax)

    fig, axes = _plt.subplots(1, 2, figsize=(8.4, 4.2))
    for ax, m, t in ((axes[0], d, "after training: dense, but mostly small"),
                     (axes[1], _np.where(kept, d, _np.where(_np.isnan(d), _np.nan, 0.0)),
                      "after thresholding at %.3g" % threshold)):
        ax.imshow(m, cmap=cmap, norm=norm, interpolation="nearest")
        ax.set_title(t, fontsize=10, color=INK, fontweight="bold")
        ax.set_xticks([]), ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_edgecolor("#dfe3ee")
    n_cells = int((~_np.isnan(d)).sum())
    nz_after = int(kept.sum())
    fig.suptitle("%s  ·  %s differences -> %s still nonzero (%.1f%% kept)"
                 % (title, _num(n_cells), _num(nz_after),
                    100.0 * nz_after / max(n_cells, 1)),
                 fontsize=11, color=INK, fontweight="bold")
    fig.tight_layout()
    _plt.show()


# ===========================================================================
#  §5  One interactive picture: the LoRA rank
# ===========================================================================
#  Plain HTML + JS with the data injected from Python, so they survive Colab,
#  Jupyter and a static export. Nothing is computed in the browser except the
#  arithmetic that is on screen anyway.
# ---------------------------------------------------------------------------

def lora_widget(d_in=64, d_out=64, ranks=(1, 2, 4, 8, 16), measured=None, default=4,
                alpha=8.0):
    """Move the rank and watch the two thin matrices, and the saving, change.

    measured: optional {rank: {"trainable": int, "target_acc": float}}
    """
    data = {"din": int(d_in), "dout": int(d_out), "ranks": [int(r) for r in ranks],
            "meas": measured or {}, "def": int(default), "alpha": float(alpha)}
    uid = _uid("lr", (d_in, d_out, tuple(ranks)))
    tmpl = r'''
<style>
#__UID__{font-family:__FONT__;border:1px solid #e6e8ee;border-radius:14px;padding:18px;max-width:840px;background:#fff;color:#24262b}
#__UID__ .hd{font-weight:800;font-size:15px;color:__INK__;margin-bottom:10px}
#__UID__ .btns{display:flex;gap:8px;margin-bottom:18px;flex-wrap:wrap}
#__UID__ .rb{cursor:pointer;border:1px solid #dfe3ee;border-radius:8px;padding:6px 14px;font-size:13px;font-weight:700;color:#555;background:#fff}
#__UID__ .rb.on{background:__TRAIN__;border-color:__TRAIN__;color:#fff}
#__UID__ .mx{border-radius:6px;display:flex;align-items:center;justify-content:center;color:#fff;font-size:11px;font-weight:700;transition:.25s}
#__UID__ .cap{font-size:11px;color:#7b8194;text-align:center;margin-top:5px;line-height:1.4}
#__UID__ .num{font-size:12.5px;color:#444;line-height:1.75}
#__UID__ .num b{color:__TRAIN__}
</style>
<div id="__UID__">
  <div class="hd">🧩 LoRA on one linear layer: a low-rank update beside a frozen W</div>
  <div class="btns"></div>
  <div style="display:flex;gap:30px;flex-wrap:wrap;align-items:flex-start">
    <div style="display:flex;gap:22px;align-items:center">
      <div><div class="mx mW" style="background:__FROZEN__"></div><div class="cap">W (frozen)<br><span class="wc"></span></div></div>
      <div style="font-size:20px;color:#9aa0b5">+</div>
      <div style="display:flex;gap:8px;align-items:center">
        <div><div class="mx mB" style="background:__TRAIN__"></div><div class="cap">B<br><span class="bc"></span></div></div>
        <div style="font-size:16px;color:#9aa0b5">x</div>
        <div><div class="mx mA" style="background:__TRAIN__"></div><div class="cap">A<br><span class="ac"></span></div></div>
      </div>
    </div>
    <div style="flex:1 1 250px" class="num">
      <div style="font-weight:800;color:__INK__;margin-bottom:6px">Cost of the update</div>
      <div class="calc"></div>
      <div class="meas" style="margin-top:10px"></div>
    </div>
  </div>
  <div style="margin-top:14px;font-size:12px;color:#555;background:#f6f7fb;border-radius:8px;padding:10px 12px;line-height:1.6">
    y = W x + <span style="color:__TRAIN__">(α/r) B A x</span> + b &nbsp;·&nbsp; B starts at zero, so the adapted layer starts out identical to the pretrained one.
  </div>
</div>
<script>
(function(){
  const D=__DATA__, root=document.getElementById("__UID__"), S=132;
  function draw(r){
    root.querySelectorAll(".rb").forEach(x=>x.classList.toggle("on", +x.dataset.r===r));
    const thin=Math.max(7, S*r/Math.max(D.din,D.dout));
    const W=root.querySelector(".mW"), A=root.querySelector(".mA"), B=root.querySelector(".mB");
    W.style.width=S+"px"; W.style.height=S+"px"; W.textContent=D.dout+"x"+D.din;
    A.style.width=S+"px"; A.style.height=thin+"px"; A.textContent=(thin>13?r+"x"+D.din:"");
    B.style.width=thin+"px"; B.style.height=S+"px"; B.textContent="";
    root.querySelector(".wc").textContent=(D.dout*D.din)+" numbers";
    root.querySelector(".ac").textContent=r+"x"+D.din+" = "+(r*D.din);
    root.querySelector(".bc").textContent=D.dout+"x"+r+" = "+(D.dout*r);
    const lora=r*(D.din+D.dout), full=D.din*D.dout;
    root.querySelector(".calc").innerHTML=
      "full ΔW: d<sub>out</sub>&middot;d<sub>in</sub> = <b style='color:__FULL__'>"+full+"</b><br>"+
      "LoRA: r&middot;(d<sub>in</sub>+d<sub>out</sub>) = <b>"+lora+"</b><br>"+
      "<div style='margin-top:6px;border-top:1px solid #eceef5;padding-top:6px'>"+
      (full/lora).toFixed(1)+"x fewer numbers to learn and to store</div>";
    const m=D.meas[r];
    root.querySelector(".meas").innerHTML = m ?
      ("<div style='background:#f6f7fb;border-radius:8px;padding:8px 10px;font-size:12px'>measured on the target task:<br><b style='color:__GREEN__'>"
       +(100*m.target_acc).toFixed(1)+"%</b> accuracy with <b>"+m.trainable+"</b> trainable parameters</div>") : "";
  }
  const btns=root.querySelector(".btns");
  D.ranks.forEach(r=>{
    const el=document.createElement("div"); el.className="rb"; el.dataset.r=r;
    el.textContent="r = "+r; el.addEventListener("click",()=>draw(r)); btns.appendChild(el);
  });
  draw(D.def);
})();
</script>'''
    display(HTML(tmpl.replace("__UID__", uid).replace("__FONT__", _FONT)
                 .replace("__INK__", INK).replace("__TRAIN__", TRAIN)
                 .replace("__FROZEN__", FROZEN).replace("__FULL__", FULL)
                 .replace("__GREEN__", GREEN)
                 .replace("__DATA__", _json.dumps(data))))


# ===========================================================================
#  §6  The model map, the final table, and the closing panels
# ===========================================================================

def model_map(entries, title="What is trainable inside the model",
              note="grey = frozen and shared · orange = trainable, saved per task"):
    """entries: list of (name, numel, trainable). Grouped by layer.

    A weight matrix is a wide block, a bias is a thin strip, and the block is
    orange exactly when that parameter carries a gradient.
    """
    groups = []
    for name, numel, tr in entries:
        layer = name.rsplit(".", 1)[0] if "." in name else name
        kind = name.rsplit(".", 1)[-1]
        if not groups or groups[-1][0] != layer:
            groups.append((layer, []))
        groups[-1][1].append((kind, int(numel), bool(tr)))

    biggest = max(n for _, items in groups for _, n, _ in items)
    cols = ""
    for layer, items in groups:
        blocks = ""
        for kind, numel, tr in items:
            color = TRAIN if tr else FROZEN
            if kind == "bias" or numel <= 64:
                blocks += ('<div style="height:11px;width:%dpx;background:%s;'
                           'border-radius:3px;margin-top:5px" title="%s"></div>'
                           '<div style="font-size:10px;color:#8a90a3">%s · %s</div>'
                           % (min(120, 26 + 94 * numel / float(biggest)), color, kind,
                              kind, _num(numel)))
            else:
                side = 34 + 62 * (numel / float(biggest)) ** 0.5
                blocks += ('<div style="height:%dpx;width:%dpx;background:%s;'
                           'border-radius:5px;display:flex;align-items:center;'
                           'justify-content:center;color:#fff;font-size:10.5px;'
                           'font-weight:700">%s</div>'
                           '<div style="font-size:10px;color:#8a90a3;margin-top:2px">'
                           '%s</div>' % (side, side * 1.35, color, _num(numel), kind))
        cols += ('<div style="text-align:center">'
                 '<div style="font-size:11.5px;font-weight:700;color:%s;margin-bottom:5px">'
                 '%s</div>%s</div>' % (INK, layer, blocks))
    _card('<div style="font-weight:800;font-size:14px;color:%s;margin-bottom:12px">%s</div>'
          '<div style="display:flex;gap:26px;align-items:flex-start;flex-wrap:wrap">%s</div>'
          '<div style="font-size:11.5px;color:#7b8194;margin-top:12px">%s</div>'
          % (INK, title, cols, note), maxw=800)


_CAVEAT = ("These measurements come from a tiny CPU model. The mechanisms scale to large "
           "models, but absolute timings and memory ratios depend on architecture, "
           "implementation, precision, hardware, kernels and optimiser.")


def results_table(records, extra=None):
    """One row per method, every column measured in the notebook.

    Numbers in the table; the two prose columns go underneath, where they can
    have the width they need instead of stretching every row.
    """
    cols = [
        ("Method", lambda r: '<b style="color:%s">%s</b>'
                             % (METHOD_COLOR.get(r["method"], ACCENT), r["method"])),
        ("Params<br><span style='font-weight:400;color:#8a90a3'>for inference</span>",
         lambda r: _num(r["total_params"])),
        ("Trainable<br><span style='font-weight:400;color:#8a90a3'>and % of total</span>",
         lambda r: "%s <span style='color:#8a90a3'>(%.2f%%)</span>"
                   % (_num(r["trainable_params"]),
                      100.0 * r["trainable_params"] / r["total_params"])),
        ("Training state<br><span style='font-weight:400;color:#8a90a3'>estimate</span>",
         lambda r: _human_bytes(r["train_state_bytes"])),
        ("Task<br>checkpoint", lambda r: _human_bytes(r["ckpt_bytes"])),
        ("Target<br>acc.", lambda r: "%.1f%%" % (100 * r["target_acc"])),
        ("Source acc.<br><span style='font-weight:400;color:#8a90a3'>after adapting</span>",
         lambda r: "%.1f%%" % (100 * r["source_acc"])),
        ("Train<br>time", lambda r: "%.2f s" % r["train_time_s"]),
        ("Inference", lambda r: "%.3f ms" % r["infer_ms"]),
        ("Path<br>changed", lambda r: "yes" if r.get("changes_path") else "no"),
        ("Mergeable", lambda r: "yes" if r.get("mergeable") else "no"),
    ]
    head = "".join('<th style="text-align:left;padding:7px 9px;font-size:11px;'
                   'color:#7b8194;text-transform:uppercase;letter-spacing:.3px;'
                   'border-bottom:1px solid #e6e8ee;vertical-align:bottom;'
                   'white-space:nowrap">%s</th>' % c for c, _ in cols)
    body = ""
    for r in records:
        body += "<tr>" + "".join(
            '<td style="padding:8px 9px;font-size:12px;border-bottom:1px solid #f2f3f8;'
            'white-space:nowrap">%s</td>' % f(r) for _, f in cols) + "</tr>"

    notes = ""
    for r in records:
        notes += ('<div style="display:flex;gap:12px;padding:7px 0;border-bottom:'
                  '1px solid #f2f3f8;font-size:12px;line-height:1.55">'
                  '<div style="flex:0 0 120px;font-weight:700;color:%s">%s</div>'
                  '<div style="flex:1 1 200px"><span style="color:%s">+</span> %s</div>'
                  '<div style="flex:1 1 200px"><span style="color:%s">-</span> %s</div></div>'
                  % (METHOD_COLOR.get(r["method"], ACCENT), r["method"],
                     GREEN, r.get("advantage", ""), FULL, r.get("limitation", "")))

    _card('<div style="font-size:12px;color:#7a4a45;background:#fdf1ef;border:1px solid '
          '#f3d8d4;border-radius:8px;padding:10px 12px;line-height:1.6;margin-bottom:12px">'
          '⚠️ %s</div>'
          '<div style="overflow-x:auto"><table style="border-collapse:collapse;color:#24262b">'
          '<tr>%s</tr>%s</table></div>'
          '<div style="margin-top:16px"><div style="font-size:11px;color:#7b8194;'
          'text-transform:uppercase;letter-spacing:.3px;margin-bottom:4px">'
          'key advantage &amp; key limitation</div>%s</div>%s'
          % (_CAVEAT, head, body, notes,
             ('<div style="font-size:11.5px;color:#7b8194;margin-top:10px">%s</div>' % extra)
             if extra else ""), maxw=1080)


def storage_projection(records, backbone_bytes, tasks=(1, 5, 20, 100)):
    """What the whole fleet costs to store, as the number of tasks grows."""
    fig, ax = _plt.subplots(figsize=(7.4, 4.0))
    x = _np.arange(len(tasks))
    w = 0.8 / len(records)
    for i, r in enumerate(records):
        shares = r.get("shares_backbone", r["method"] != "Full fine-tuning")
        vals = [(backbone_bytes if shares else 0) + n * r["ckpt_bytes"] for n in tasks]
        color = METHOD_COLOR.get(r["method"], ACCENT)
        bars = ax.bar(x + (i - (len(records) - 1) / 2.0) * w, vals, w,
                      label=r["method"], color=color)
        for b_, v in zip(bars, vals):
            ax.text(b_.get_x() + b_.get_width() / 2, v * 1.06, _human_bytes(v),
                    ha="center", fontsize=7.2, color=color, rotation=90)
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(["%d task%s" % (n, "" if n == 1 else "s") for n in tasks])
    ax.set_ylabel("total stored (log scale)", fontsize=10)
    ax.set_title("Storing the whole fleet: backbone once, module per task",
                 fontsize=11.5, color=INK, fontweight="bold")
    ax.legend(fontsize=8.5, frameon=False, ncol=3)
    ax.grid(alpha=0.2, axis="y")
    ax.set_ylim(top=max(n * records[0]["ckpt_bytes"] for n in tasks) * 6)
    for sp in ax.spines.values():
        sp.set_edgecolor("#dfe3ee")
    fig.tight_layout()
    _plt.show()


_SCENARIOS = [
    ("💾 100 client-specific models, stored and switched all day",
     "Every client gets their own adapted version. They live on disk, get loaded on demand, "
     "and are swapped in and out constantly.",
     "What matters here is the <b>per-task checkpoint</b> and how fast a module can be swapped "
     "onto a backbone that is already in memory. Look at the checkpoint column, then at the "
     "100-task bar. Full fine-tuning stores a hundred complete models; every PEFT row stores "
     "the backbone once. Whether the module changes the execution path matters less, because "
     "the same backbone stays loaded either way."),
    ("⚡ Inference latency must stay as close to the original as possible",
     "The adapted model sits behind a latency budget that the pretrained model already "
     "almost fills.",
     "Look at the inference column and at the <i>mergeable</i> row. Any method that leaves a "
     "branch in the graph keeps paying for it on every forward pass, for ever. A LoRA update "
     "can be folded into W once, after which the served model is bit-for-bit the same shape "
     "as the original. The "
     "notebook checked that numerically. BitFit and Diff Pruning also change no shapes: they "
     "only change values in place."),
    ("🎯 The target task is far from pretraining and accuracy is what counts",
     "Storage is cheap, there is one task, and every point of accuracy is worth paying for.",
     "This is the case where PEFT has the least to offer, and saying so is the right answer. "
     "Compare the target-accuracy column against the trainable-parameter column: whichever "
     "method came out on top here, the argument has to be about measured accuracy and the "
     "capacity that produced it, and a bias-only update has very little of it. Note also what "
     "happened to source accuracy: freezing the backbone did not keep the old behaviour."),
]


def choose_method_panel():
    """Three scenarios. No single right answer: the reveal names what to argue from."""
    uid = _uid("sc", "scenarios")
    items = "".join(
        '<div class="sc" data-i="%d">'
        '<div class="sc-t">%s</div><div class="sc-d">%s</div>'
        '<div class="sc-r">%s</div></div>' % (i, t, d, r)
        for i, (t, d, r) in enumerate(_SCENARIOS))
    tmpl = r'''
<style>
#__UID__{font-family:__FONT__;border:1px solid #e6e8ee;border-radius:14px;padding:18px;max-width:860px;background:#fff;color:#24262b}
#__UID__ .hd{font-weight:800;font-size:15px;color:__INK__;margin-bottom:4px}
#__UID__ .sub{font-size:12.5px;color:#666;margin-bottom:14px;line-height:1.6}
#__UID__ .sc{border:1px solid #e2e5ef;border-radius:10px;padding:12px 14px;margin-bottom:10px;cursor:pointer}
#__UID__ .sc:hover{border-color:__ACCENT__;background:#faf9ff}
#__UID__ .sc-t{font-weight:800;font-size:13.5px;color:__INK__}
#__UID__ .sc-d{font-size:12.5px;color:#555;margin-top:4px;line-height:1.6}
#__UID__ .sc-r{display:none;font-size:12.5px;color:#3b2d6b;margin-top:10px;padding-top:10px;border-top:1px dashed #dfe3ee;line-height:1.65}
#__UID__ .sc.open .sc-r{display:block}
</style>
<div id="__UID__">
  <div class="hd">🎯 Pick a method for each situation, then defend it</div>
  <div class="sub">Write two or three sentences for each before you click. There is no single
  correct method: what is being graded is whether you argue from the numbers you measured.</div>
  __ITEMS__
</div>
<script>
(function(){
  document.getElementById("__UID__").querySelectorAll(".sc").forEach(el=>{
    el.addEventListener("click",()=>el.classList.toggle("open"));
  });
})();
</script>'''
    display(HTML(tmpl.replace("__UID__", uid).replace("__FONT__", _FONT)
                 .replace("__INK__", INK).replace("__ACCENT__", ACCENT)
                 .replace("__ITEMS__", items)))


def takeaway():
    _card('<div style="font-size:15.5px;line-height:1.75;color:%s">'
          '<b>PEFT does not make the pretrained backbone disappear.</b> It changes '
          '<b style="color:%s">what has to be trained</b> and '
          '<b style="color:%s">what has to be stored for each new task</b>.'
          '<div style="font-size:13.5px;color:#555;margin-top:10px;line-height:1.7">'
          'Which method is right depends on which constraint is the real one: memory during '
          'training, storage across tasks, inference latency, adaptation capacity, or simply '
          'how much machinery you are willing to operate.</div></div>'
          % (INK, TRAIN, ACCENT), maxw=760)


# ===========================================================================
#  §7  Quiz banks  (options are shuffled at render time, so order means nothing)
# ===========================================================================
_MC_QUIZZES = {
    "why_peft": (
        "Why is one shared backbone plus small modules the interesting picture?",
        "A lab has one pretrained model and a hundred tasks to adapt it to.",
        ["Because the per-task cost collapses to the size of a module, while the backbone is "
         "trained, stored and loaded once",
         "Because a small module runs faster than the full model at inference",
         "Because the modules can be trained without ever running the pretrained model",
         "Because a hundred small modules together fit in the memory of one GPU, which a "
         "hundred full models would not"],
        0,
        "The saving is <b>per task</b>. The backbone is exactly as big as it always was and it "
        "still runs on every forward pass. A module does not replace it, it rides on it. What "
        "changes is that task number 101 costs a module rather than a model."),
    "frozen_forward": (
        "A parameter has <code>requires_grad = False</code>. What happens to it during training?",
        "The optimiser was built from the parameters that still require gradients.",
        ["It is still used in every forward pass, but no gradient is stored for it and the "
         "optimiser never updates it",
         "It is skipped in the forward pass, which is where the speed-up comes from",
         "It is removed from the model, so the model has fewer parameters than before",
         "It is used in the forward pass and gets a gradient, which is simply not applied"],
        0,
        "Freezing changes the <b>backward</b> side of the ledger only. The frozen weight is read "
        "on every forward pass, since the model would compute something else entirely without it, "
        "and it still occupies its memory. What disappears is its gradient buffer and its "
        "optimiser state."),
    "optimizer_state": (
        "In the simplified memory model, which parameters carry Adam's two moments?",
        "Adam keeps a running mean and a running variance per <i>optimised</i> parameter, 4 bytes "
        "each in FP32.",
        ["Only the trainable ones, because the optimiser was constructed from exactly those",
         "All of them, because the optimiser tracks the whole model's state",
         "All the weight matrices, but not the biases",
         "Only the parameters whose gradient happened to be non-zero at the last step"],
        0,
        "The optimiser only holds state for the parameters it was handed. That is why the "
        "estimate is 4&middot;N<sub>total</sub> + 4&middot;N<sub>trainable</sub> + "
        "8&middot;N<sub>trainable</sub>: the first term covers every weight in the model, the "
        "other two shrink with the trainable count."),
    "lora_count": (
        "A LoRA update is placed on a layer with d<sub>in</sub> = 512 and d<sub>out</sub> = 512, "
        "at rank r = 8. How many trainable numbers is that?",
        "A is r x d<sub>in</sub>, B is d<sub>out</sub> x r, and W itself stays frozen.",
        ["r(d<sub>in</sub> + d<sub>out</sub>) = 8 x 1024 = 8,192",
         "d<sub>in</sub> x d<sub>out</sub> = 262,144, since ΔW is the same shape as W",
         "r x d<sub>in</sub> x d<sub>out</sub> = 2,097,152",
         "r&sup2; = 64, because the update is squeezed through an r x r core"],
        0,
        "ΔW has the shape of W, but it is never <i>stored</i> in that shape: only the two thin "
        "factors are. 8,192 against 262,144 is a factor of 32, and the ratio improves as the "
        "layer gets wider at fixed rank."),
    "diff_memory": (
        "A Diff Pruning checkpoint holds a few hundred nonzero values. What does that tell you "
        "about the memory used while it was trained?",
        "Training kept a dense difference tensor for every adapted base parameter and an Adam "
        "state for each of them.",
        ["Nothing: the sparsity appears at the end, so training carried a dense difference for "
         "every adapted parameter",
         "That training used a few hundred parameters' worth of memory, since only those "
         "survived",
         "That training memory was about half of full fine-tuning, because half the updates were "
         "eventually thresholded away",
         "That the backbone was not needed during training, only the differences"],
        0,
        "What you store afterwards and what you had to allocate during optimisation are two "
        "different quantities. The sparse file is genuinely small; the training run was as "
        "expensive as full fine-tuning, plus the sparsity penalty. And the file needs indices or "
        "a mask on top of the values."),
    "forgetting": (
        "The backbone was frozen. What does that guarantee about the model's behaviour on the "
        "source task?",
        "In the notebook, source accuracy was measured again after every adaptation.",
        ["Nothing on its own: the task module changes what comes out, so predictions on the old "
         "task can change a lot",
         "That predictions on the source task are unchanged, since the original weights were "
         "never written to",
         "That source accuracy can drop only if the learning rate was too high",
         "That the source task is safe as long as the module has few parameters"],
        0,
        "Frozen weights are preserved <b>as values</b>, which is why you can always go back to "
        "the original model by dropping the module. With the module active, the computation is a "
        "different one, and the notebook measured the drop. Keeping the original behaviour is a "
        "consequence of being able to <i>detach</i> the module, not of freezing it."),
    "speedup": (
        "Training only 2.9% of the parameters. What should you expect for training time?",
        "The measured times are in the dashboard you have been updating.",
        ["A modest saving at best: the frozen backbone still runs forwards, and gradients still "
         "have to travel back through it to reach the trainable parts",
         "Roughly a 34x speed-up, matching the reduction in trainable parameters",
         "No change at all is possible, since the same number of operations always runs",
         "A slowdown, because freezing adds bookkeeping to every step"],
        0,
        "Backward work is skipped only for the parts of the graph that lie <i>after</i> the last "
        "trainable parameter on the way back. Everything before it is still traversed. The "
        "parameter count and the runtime are simply not the same axis, which is why the "
        "dashboard shows both."),
}

_TF_QUIZZES = {
    # 4 true, 4 false
    "counting": ("Total parameters, trainable parameters", [
        ("A frozen parameter still counts towards the model's total parameter count.", True),
        ("Freezing a parameter removes it from the forward pass.", False),
        ("A Linear(64, 64) layer holds 4096 weights and 64 biases.", True),
        ("Trainable parameters are the ones that get a gradient and an optimiser update.", True),
        ("Once most parameters are frozen, the model needs less memory to run a forward pass.",
         False),
        ("Building the optimiser from the parameters with requires_grad=True is what keeps "
         "optimiser state off the frozen ones.", True),
        ("The number of trainable parameters fixes how long one training step takes.", False),
        ("A model with 1% trainable parameters has 1% of the original storage footprint.",
         False),
    ]),
    # 4 true, 4 false
    "methods": ("BitFit and LoRA", [
        ("BitFit adds no new modules: it selects parameters the model already had.", True),
        ("B is initialised at zero so the adapted layer starts out identical to the pretrained "
         "one.", True),
        ("A rank of 1 gives a LoRA update more capacity than a rank of 16.", False),
        ("LoRA constrains the <i>update</i> to a low-rank form; the pretrained W is untouched "
         "and full-rank.", True),
        ("Raising the LoRA rank leaves the trainable parameter count unchanged.", False),
        ("Merging a LoRA update reduces what has to be stored for that task.", False),
        ("Both A and B being trainable is what lets the product BA change during training.",
         True),
        ("Bias-only tuning usually adapts as well as methods that learn structured weight "
         "changes.", False),
    ]),
    # 3 true, 3 false
    "systems": ("What actually gets cheaper", [
        ("Storing 100 adapted versions is where PEFT is most convincing.", True),
        ("A smaller task checkpoint implies a smaller training-state footprint.", False),
        ("The estimate 4N_total + 12N_trainable ignores activations and allocator overhead.",
         True),
        ("A PEFT run needs less memory for the pretrained weights than a full fine-tuning run.",
         False),
        ("Merging a LoRA update removes the extra branch from the served model.", True),
        ("PEFT reaches a higher accuracy than full fine-tuning whenever it is applied "
         "correctly.", False),
    ]),
}

_NUMBER_QUIZZES = {
    "storage_math": ("🔢 Twenty tasks, one billion parameters", [
        ("A pretrained model has <b>1 billion parameters in FP16</b> (2 bytes each). How many "
         "<b>GB</b> is one complete copy? (count 1 GB = 10<sup>9</sup> bytes)", 2.0, 0.01,
         "1e9 parameters x 2 bytes = 2e9 bytes = 2 GB."),
        ("Full fine-tuning for <b>20 tasks</b> means 20 complete copies. Total, in GB?",
         40.0, 0.1,
         "20 x 2 GB = 40 GB, and nothing is shared between them."),
        ("Now each task keeps only a module of <b>0.5% of the base parameters</b>, and the "
         "backbone is stored <b>once</b>. Total for the same 20 tasks, in GB?", 2.2, 0.05,
         "One module = 0.5% x 2 GB = 0.01 GB = 10 MB. So 2 GB + 20 x 0.01 GB = 2.2 GB, about "
         "18x less, and the gap grows with every task added."),
    ]),
    "layer_math": ("🔢 Count a layer before PyTorch tells you", [
        ("<code>nn.Linear(2, 64)</code>: how many <b>weights</b>?", 128.0, 0.01,
         "The weight matrix is (out_features x in_features) = 64 x 2 = 128."),
        ("<code>nn.Linear(2, 64)</code>: how many <b>biases</b>?", 64.0, 0.01,
         "One per output unit: 64."),
        ("The whole MLP is Linear(2,64) -> Linear(64,64) -> Linear(64,2). How many parameters "
         "in total?", 4482.0, 0.5,
         "192 + 4160 + 130 = 4482. The middle layer alone is 93% of the model."),
    ]),
    "lora_math": ("🔢 The size of a LoRA update", [
        ("Layer with d<sub>in</sub> = 64, d<sub>out</sub> = 64. How many numbers would a "
         "<b>full</b> ΔW contain?", 4096.0, 0.5,
         "64 x 64 = 4096, one for every entry of W."),
        ("With rank r = 4, how many trainable numbers do <b>A and B together</b> hold?",
         512.0, 0.5,
         "A is 4 x 64 = 256, B is 64 x 4 = 256, so 512 = r(d_in + d_out)."),
        ("How many times fewer is that than the full ΔW?", 8.0, 0.05,
         "4096 / 512 = 8. On a 512 x 512 layer at the same rank it would be 32x: the wider the "
         "layer, the better the ratio."),
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
#  §8  Final boss: timed true/false flash quiz with lives
# ===========================================================================
# Balanced pool (14 true / 14 false), phrased so neither answer is given away
# by the wording, and so that no statement answers another one.
_FLASH_POOL = [
    # --- parameters, frozen and trainable ----------------------------------
    ("A frozen parameter is still read on every forward pass.", True),
    ("Freezing a parameter frees the memory that held its value.", False),
    ("The optimiser holds state only for the parameters it was constructed from.", True),
    ("A model with 3% trainable parameters occupies 3% of the memory it did before.", False),
    ("Linear(64, 64) contributes 4160 parameters, biases included.", True),
    ("A PEFT run can skip loading the pretrained weights, since they are not being trained.",
     False),
    ("Freezing the backbone removes it from the backward pass entirely.", False),
    # --- the methods -------------------------------------------------------
    ("BitFit trains parameters the pretrained model already contained.", True),
    ("The scaling alpha/r in front of a LoRA update is fixed, not learned.", True),
    ("A larger LoRA rank means fewer trainable parameters.", False),
    ("LoRA keeps the pretrained weight matrix frozen and learns two thin factors.", True),
    ("LoRA replaces W with a low-rank matrix, so the served layer is low-rank.", False),
    ("Doubling the LoRA rank doubles the number of trainable LoRA parameters.", True),
    ("Which layers a LoRA update is placed on makes no difference to the outcome.", False),
    ("Diff Pruning stores one dense difference tensor per task after training.", False),
    ("Diff Pruning may allocate a difference variable for every adapted base parameter while "
     "it trains.", True),
    # --- the systems consequences -----------------------------------------
    ("Storing 100 adapted models is where sharing one backbone pays off most.", True),
    ("Cutting trainable parameters by 30x cuts training time by about 30x.", False),
    ("A merged LoRA layer costs the same at inference as the original layer.", True),
    ("Merging a LoRA update makes the per-task file smaller.", False),
    ("The estimate 4N_total + 12N_trainable includes activation memory.", False),
    ("Gradients and optimiser moments are what shrink when parameters are frozen.", True),
    ("A small task checkpoint proves the training run had a small memory footprint.", False),
    ("A sparse checkpoint has to carry indices or a mask on top of its nonzero values.", True),
    ("PEFT is a way of avoiding catastrophic forgetting with a guarantee.", False),
    ("Source-task accuracy can fall after adaptation even with every original weight frozen.",
     True),
    ("PEFT reaches higher accuracy than full fine-tuning when it is set up correctly.", False),
    ("The pretrained backbone still has to be loaded and executed under every PEFT method.",
     True),
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
    msg.innerHTML=won?"You can tell a trainable parameter from a stored one, and a checkpoint "
                      +"from a training state. That is the whole hour."
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
