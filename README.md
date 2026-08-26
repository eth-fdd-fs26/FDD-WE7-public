# FDD · WE7 — Parameter-efficient fine-tuning

Exercise notebook for week 7 of *Foundations of Data-Driven Engineering* (ETH Zürich).

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/eth-fdd-fs26/FDD-WE7-public/blob/main/2_peft_finetuning/exercise/02_peft_finetuning_student.ipynb)

## Notebook 02 — One backbone, many tasks

You own one pretrained model and a queue of people who all want it adapted to *their* task.
Fine-tuning once per task means a full checkpoint per task. **Parameter-efficient fine-tuning**
is the family of answers to that: freeze the pretrained model, train a small set of parameters,
ship only those.

By the end you will be able to

- tell **total** from **trainable** parameters, and say why the second drives gradient and
  optimiser memory;
- implement **BitFit**, **LoRA** and **Diff Pruning** from scratch, a few lines each;
- work out a LoRA update's parameter count *before* running anything;
- compare the methods on accuracy, memory, checkpoint size and runtime, and argue for one.

The pretrained model is a 4,482-parameter MLP, so every number in the notebook can be checked
by hand. About an hour, **CPU only** — no GPU, no downloads, no Transformers.

## How to run it

**On Colab** (recommended): click the badge above. The first cell clones this repo into your
session and the second installs the dependencies. No account or access token needed.

**Locally**: clone the repo and start Jupyter from the repo root.

```bash
git clone https://github.com/eth-fdd-fs26/FDD-WE7-public.git
cd FDD-WE7-public
pip install -r 2_peft_finetuning/exercise/requirements_peft.txt
jupyter notebook 2_peft_finetuning/exercise/02_peft_finetuning_student.ipynb
```

The setup cell detects that it is not on Colab, skips the clone, and locates the repo root on
disk instead.

## Files

| file | what it is |
|---|---|
| `2_peft_finetuning/exercise/02_peft_finetuning_student.ipynb` | the notebook — task cells contain `???` for you to fill in |
| `2_peft_finetuning/exercise/peft_viz.py` | display helpers: dashboards, diagrams, quizzes. You are not meant to read this one |
| `2_peft_finetuning/exercise/requirements_peft.txt` | torch (CPU), numpy, matplotlib, scikit-learn, ipython |

> 🧠 **Assumed background:** what a linear layer is, what a training loop does, and that
> gradients get computed and applied. Nothing about Transformers, attention or language models
> is assumed anywhere.
