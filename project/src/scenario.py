"""The banks scenario — four synthetic lenders, one shared credit-risk rule.

Four banks want to know whether a borrower will default. They cannot pool customer
records. Everything a scenario owes the rest of the project is here: the data, the words
the figures should use, and the colours. The engine in `fedcore.py` never learns
which of the two scenarios it is running.

    import scenario
    books = scenario.load()            # a fedcore.Federation of four Sites
    tr, te = scenario.split(books)     # books to learn from, cohorts to be scored on

TWO THINGS DIFFER FROM THE HOSPITALS SCENARIO, both on purpose.

NO POOLED STANDARDISATION. The features arrive already scaled against a published
reference population (see generator.REFERENCE), so the banks never have to agree a shared
mean and standard deviation before they can start. A scenario computes those
constants on the pooled records, which is a small leak it has to apologise for in one
sentence. Here there is nothing to apologise for.

TEST RECORDS ARE A FRESH COHORT, NOT A SLICE OF THE BOOK. Ashfield holds 150 customers.
Carving 30 of them off as a test set cannot resolve a per-bank comparison, and the whole
project is a per-bank comparison. Each bank is instead scored on 2,000 fresh customers
from the same population — next year's applicants. Synthetic customers cost nothing, and
this is the difference between a measurable result and a shrug. Measured in
CANDIDATE_REVIEW.md section 2.2.
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "core"))
import fedcore as fc                                            # noqa: E402
import generator as gen                                          # noqa: E402

FEATURES = gen.FEATURES
N_FEATURES = len(FEATURES)            # 13
N_PARAMS = fc.configure(N_FEATURES)   # 14

BUSINESS = gen.BUSINESS
BANKS = gen.BANKS
DICTIONARY = gen.DICTIONARY

#: The Dirichlet alpha the source package ships with. Nothing in this notebook uses it any
#: more: part 2 replaced it with the named divisions below, and part 5 now stresses FedAvg
#: with the worst of those instead. Kept only so a reader comparing against the source
#: package can find the number.
SKEW_ALPHA = 0.3
MIN_SITE_SIZE = 150

#: The six divisions part 2 offers, in the order a reader should meet them.
#:
#: Dirichlet(alpha) is the usual knob in the literature and it is the wrong one to hand a
#: student. It is a random draw, so the same alpha gives a different picture every seed;
#: it moves group sizes and default rates together, so nothing can be attributed; and
#: "alpha = 0.5" says nothing to anyone who has not read the papers.
#:
#: These six are deterministic and named after what they do. The last four hold group
#: sizes equal, so the only thing moving is how the outcome was handed out. Rows 3 and 5
#: are the pair the part turns on: both spread the default rate twelve times over, one by
#: sorting customers and one by dealing out defaults, and only the second costs anything.
#: Measured in checks/verify_splits.py.
SPLITS = [("even split", "iid"),
          ("the real banks", "natural"),
          ("sorted by debt-to-income", "sortby:%d" % FEATURES.index("debt_to_income_z")),
          ("one group gets 60% of all defaults", "concentrate:0.60"),
          ("one group gets 80% of all defaults", "concentrate:0.80"),
          ("one group gets 100% of all defaults", "concentrate:1.00")]

#: The two rows above that share a spread and differ in cost.
SPLIT_PAIR = ("sorted by debt-to-income", "one group gets 80% of all defaults")

READABLE = {
    "age_z": "age",
    "log_income_z": "annual income",
    "employment_years_z": "years in current job",
    "debt_to_income_z": "debt-to-income ratio",
    "credit_utilization_z": "credit utilisation",
    "late_payments_12m_z": "late payments (12m)",
    "credit_history_years_z": "credit history length",
    "loan_to_income_z": "loan size vs income",
    "interest_rate_z": "interest rate",
    "savings_to_income_z": "savings vs income",
    "num_credit_lines_z": "open credit lines",
    "home_owner": "owns a home",
    "self_employed": "self-employed",
}

READABLE_BUSINESS = {
    "age_years": "age (years)", "annual_income_k": "annual income (k)",
    "employment_years": "years in current job", "debt_to_income": "debt-to-income ratio",
    "credit_utilization": "credit utilisation", "late_payments_12m": "late payments (12m)",
    "credit_history_years": "credit history (years)", "loan_amount_k": "loan amount (k)",
    "interest_rate_pct": "interest rate (%)", "savings_balance_k": "savings (k)",
    "num_credit_lines": "open credit lines", "home_owner": "owns a home",
    "self_employed": "self-employed",
}

#: How one customer's filed values read on a credit file, in the order of FEATURES.
#: The model works in standardised units, which mean nothing to a reader; a figure that
#: shows a customer wants the number the branch would actually see.
_FILED = [
    ("{:.0f}", " years"), ("{:,.0f}", "k a year"), ("{:.0f}", " years in the job"),
    ("{:.2f}", ""), ("{:.0%}", ""), ("{:.0f}", ""),
    ("{:.0f}", " years"), ("{:,.0f}", "k"), ("{:.1f}", "%"),
    ("{:,.0f}", "k"), ("{:.0f}", ""), (None, ""), (None, ""),
]


def filed(x_row):
    """One customer's model features, written the way a credit file writes them."""
    vals = to_business(np.asarray(x_row, float).reshape(1, -1))[0]
    out = []
    for v, (fmt, unit) in zip(vals, _FILED):
        out.append("yes" if fmt is None and v > 0.5 else
                   "no" if fmt is None else fmt.format(v) + unit)
    return out


VOCAB = fc.Vocab(
    site="bank", sites="banks",
    member="customer", members="customers",
    positive="in default", positive_rate="default rate",
    setting="consortium",
    outcome_question="Will this customer default?",
    negative="paying",
    record="credit file", records="credit files",
)

#: Shortened for tight figure labels. The letter carries the ordering, so it stays.
SHORT_NAME = {"A Prime Metro": "A Prime", "B Digital Growth": "B Digital",
              "C Regional Mainstreet": "C Regional",
              "D Community High-Touch": "D Community"}

SITE_COLOUR = {"A Prime Metro": "#2f5d8a", "B Digital Growth": "#2e6b4e",
               "C Regional Mainstreet": "#b0761f", "D Community High-Touch": "#7b4b8a"}

#: Kept because `fedcore` accepts it from any scenario. This scenario does not carve
#: a test set out of the book — see the module docstring and `split`.
TRAIN_FRACTION = 1.0

#: Knobs the figure set needs that only this scenario can know. C = 1.35 is the
#: measured median per-customer gradient norm at w = 0; lot = 64 gives sampling rates
#: of 0.05 / 0.09 / 0.18 / 0.43, which is the spread the privacy figure is about.
#: 13 exchanges at E = 5 is where the federation reaches pooled loss (checks/).
#: AUC, not accuracy. At an 11.9% default rate a model that never predicts default
#: already scores 88%, so an accuracy grid is four columns of the same number.
#: Two figure titles are readings of THIS dataset and would be false on the other one.
#: The banks share a risk rule by construction, so no bank's model is badly wrong next
#: door — but in three of four columns someone else's model beats the local one, which
#: is a sharper argument for federating. And skew here is harmless until local work is
#: high, so the Dirichlet row is the one to highlight, not the real split.
FIGURES = dict(hero="A Prime Metro", C=1.35, lot=64, rounds=13, E=5, metric="auc",
               map_x="annual_income_k", map_y="debt_to_income",
               map_x_label="annual income (thousands)",
               map_y_label="debt-to-income ratio",
               crossgrid_title="Your own model is not the best model for your own customers.",
               stability_title="Every bank gains, and the smallest gains most.",
               stability_takeaway="D Community gains about two and a half times what the others "
                                  "average, on every one of the ten rebuilds. Three of the four gain "
                                  "every time; what separates D Community is the size of the gain, "
                                  "not its reliability. That difference is the argument for "
                                  "federating, and what follows prices it.",
               drift_title="Different customers travel together. Dealt-out defaults do not.",
               drift_highlight="the real banks",
               batching_takeaway="These weights are prediction errors, so they take either "
                                 "sign and can total nearly zero. One customer often carries "
                                 "several times an equal split, sometimes more than the "
                                 "whole update. Averaging changes what is disclosed without "
                                 "capping anybody. Part 6 adds the cap.",
               drift_takeaway="At one local step every division lands in the same place. "
                              "More local work only hurts the divisions where the customers "
                              "do not explain the defaults, and there it compounds fast.",
               noise_levels=[(16.0, "very strong"), (8.0, "strong"), (4.0, "moderate")])

SEED = gen.BOOK_SEED


def load(seed=SEED) -> fc.Federation:
    """The four banks' existing books. Deterministic for a given seed."""
    return fc.Federation([fc.Site(name, Xz, y)
                          for name, _, _, Xz, y, _ in gen.generate(seed=seed)])


def cohort(seed=gen.COHORT_SEED, multiple=gen.COHORT_MULTIPLE) -> fc.Federation:
    """A fresh scoring cohort per bank, drawn from the same four populations.

    Sized as a multiple of each bank's book, so the pooled cohort carries the
    consortium's own bank mix rather than a flattened one.
    """
    sizes = {b["name"]: b["n"] * multiple for b in gen.BANKS}
    return fc.Federation([fc.Site(name, Xz, y) for name, _, _, Xz, y, _
                          in gen.generate(sizes=sizes, seed=seed, first_id=500001)])


def split(federation=None, seed=0, train_fraction=None):
    """Books to learn from, cohorts to be scored on. Returns two Federations.

    `federation` is accepted so the call matches what `fedcore` expects; the
    books are regenerated if it is omitted. No customer appears in both.
    """
    return (load() if federation is None else federation), cohort(seed=gen.COHORT_SEED + seed)


def load_raw(seed=SEED) -> fc.Federation:
    """The same books in business units, for reading a customer record out loud."""
    return fc.Federation([fc.Site(name, raw, y)
                          for name, _, raw, _, y, _ in gen.generate(seed=seed)])


def oracle(seed=SEED):
    """The true default probability per customer. For auditing only — never a feature.

    Lets the notebook show students the ceiling: once the pooled model is within a
    whisker of this, more tuning is not the answer.
    """
    return {name: p for name, _, _, _, _, p in gen.generate(seed=seed)}


def customer_ids(seed=SEED):
    """One id per customer, unique across the federation. One row is one customer, which
    is what makes example-level DP customer-level DP here."""
    return {name: ids for name, ids, _, _, _, _ in gen.generate(seed=seed)}


def to_business(Xz):
    """Model features -> a readable customer file. Used by the reconstruction attack."""
    return gen.to_business(Xz)


if __name__ == "__main__":
    books, coh = split()
    print("banks scenario — four lenders, one shared rule\n")
    print(f"  {'bank':<20}{'book':>7}{'cohort':>8}{'default':>10}   profile")
    for s, c, spec in zip(books, coh, gen.BANKS):
        print(f"  {s.name:<20}{s.n:>7}{c.n:>8}{s.share_positive*100:>9.1f}%   {spec['blurb']}")
    Xb, yb = books.pooled()
    print(f"  {'TOTAL':<20}{len(yb):>7}{sum(c.n for c in coh):>8}{yb.mean()*100:>9.1f}%")
    print(f"\n  {N_FEATURES} features, {N_PARAMS} parameters")
