"""The data-generating process for the four-bank federation.

Four fictional banks lend to different populations under one shared credit-risk rule.
Every customer's default probability comes from the same logistic equation; only the
customers differ. That is what makes a single global model the right answer rather than
a compromise, and it is enforced here rather than asserted.

    P_k(x)     differs across banks      they serve different customers
    P_k(y)     differs across banks      so their default rates differ
    P_k(y|x)   IS THE SAME everywhere    creditworthiness means one thing

Two design choices are worth knowing about.

FEATURES ARE CORRELATED THROUGH TWO LATENT FACTORS. A customer's unobserved financial
`stress` pushes up their debt ratio, utilisation, late payments and the rate they were
offered, and pushes down their income and savings. `stability` moves employment length
and savings the other way. Real credit files behave like this, and it means the
reconstruction attack in the security part recovers a customer who reads as a person
rather than thirteen unrelated numbers.

THE INTEREST RATE IS PRICED OFF THE SAME LATENT STRESS, NOT OFF THE LABEL. A bank that
priced a loan using the outcome it is trying to predict would leak the answer into the
features. Here the rate is correlated with risk the way a real book is — through what the
bank could observe at origination — and carries no information about the realised default
beyond what the other features already carry. Because the rate is a feature, the target is
"will this already-priced loan default", not "should we approve this application".

Adapted from an external synthetic-data package (CC0-1.0), resized to teaching scale and
reduced to numpy. See CANDIDATE_REVIEW.md for what was measured and what was changed.
"""

from __future__ import annotations

import numpy as np

#: The thirteen model features, in the order the coefficient vector expects.
FEATURES = [
    "age_z", "log_income_z", "employment_years_z", "debt_to_income_z",
    "credit_utilization_z", "late_payments_12m_z", "credit_history_years_z",
    "loan_to_income_z", "interest_rate_z", "savings_to_income_z",
    "num_credit_lines_z", "home_owner", "self_employed",
]

#: The shared risk rule. These thirteen numbers decide who defaults, at every bank.
#: Units are "per reference standard deviation", so they are directly comparable:
#: +0.90 on late payments means one reference SD more moves the log-odds by 0.90.
TRUE_BETA = np.array([
    0.05,   # age_z                  older borrowers default very slightly more here
    -0.25,  # log_income_z           more income helps
    -0.15,  # employment_years_z     a longer job tenure helps
    0.65,   # debt_to_income_z       second strongest driver
    0.70,   # credit_utilization_z   running the cards hot
    0.90,   # late_payments_12m_z    strongest single signal
    -0.20,  # credit_history_years_z a long record helps
    0.45,   # loan_to_income_z       borrowing large relative to income
    0.25,   # interest_rate_z        the price the bank set at origination
    -0.30,  # savings_to_income_z    a buffer helps
    0.10,   # num_credit_lines_z
    -0.25,  # home_owner
    0.18,   # self_employed
], dtype=float)
TRUE_INTERCEPT = -2.30

#: The fixed reference population the z-scores are taken against. This is published,
#: not computed from anybody's book, so the banks never have to agree a shared statistic
#: before they can start. (Pooled standardisation would itself be a small leak.)
REFERENCE = {
    "age_years": (40.0, 10.0), "annual_income_k": (np.log(60.0), 0.50),
    "employment_years": (8.0, 6.0), "debt_to_income": (0.32, 0.15),
    "credit_utilization": (0.45, 0.22), "late_payments_12m": (1.2, 1.5),
    "credit_history_years": (10.0, 7.0), "loan_to_income": (0.35, 0.20),
    "interest_rate_pct": (12.0, 4.0), "savings_to_income": (0.20, 0.15),
    "num_credit_lines": (4.0, 2.0),
}

#: What each column means and, where it is derived, exactly how. The notebook shows this
#: table rather than asking students to read the generator.
DICTIONARY = {
    "age_years": ("age in years", "age_z = (age_years - 40) / 10"),
    "annual_income_k": ("annual income, thousands", "log_income_z = (ln income - ln 60) / 0.50"),
    "employment_years": ("years in the current job", "employment_years_z = (x - 8) / 6"),
    "debt_to_income": ("debt service over income", "debt_to_income_z = (x - 0.32) / 0.15"),
    "credit_utilization": ("share of revolving credit in use", "credit_utilization_z = (x - 0.45) / 0.22"),
    "late_payments_12m": ("late payments in the last year", "late_payments_12m_z = (x - 1.2) / 1.5"),
    "credit_history_years": ("length of credit history", "credit_history_years_z = (x - 10) / 7"),
    "loan_amount_k": ("loan size, thousands", "loan_to_income_z = (loan/income - 0.35) / 0.20"),
    "interest_rate_pct": ("rate set at origination", "interest_rate_z = (x - 12) / 4"),
    "savings_balance_k": ("savings balance, thousands", "savings_to_income_z = (savings/income - 0.20) / 0.15"),
    "num_credit_lines": ("open credit lines", "num_credit_lines_z = (x - 4) / 2"),
    "home_owner": ("1 if the customer owns a home", "used as-is"),
    "self_employed": ("1 if self-employed", "used as-is"),
}

#: The eleven business-unit columns a person can read, plus the two binaries.
BUSINESS = [
    "age_years", "annual_income_k", "employment_years", "debt_to_income",
    "credit_utilization", "late_payments_12m", "credit_history_years",
    "loan_amount_k", "interest_rate_pct", "savings_balance_k",
    "num_credit_lines", "home_owner", "self_employed",
]

#: Half the source package's scale. At full scale every bank can fit fourteen parameters
#: on its own and the federation gains almost nothing; halving restores a benefit that is
#: visible at every bank on every seed. Measured in CANDIDATE_REVIEW.md section 6.
BANKS = [
    dict(name="A Prime Metro", n=1250, blurb="large prime metropolitan lender",
         age=47, income=82, emp=11, dti=.29, util=.38, late=.60, history=14,
         lti=.32, rate=10.0, save=.25, lines=5.2, owner=.80, self_emp=.08),
    dict(name="B Digital Growth", n=700, blurb="online lender, younger thin-file customers",
         age=31, income=66, emp=5, dti=.33, util=.46, late=.70, history=6.5,
         lti=.38, rate=11.0, save=.19, lines=4.0, owner=.34, self_emp=.15),
    dict(name="C Regional Mainstreet", n=350, blurb="regional lender, lower income and self-employed mix",
         age=43, income=54, emp=9, dti=.37, util=.50, late=1.00, history=10,
         lti=.42, rate=13.0, save=.15, lines=3.5, owner=.61, self_emp=.30),
    dict(name="D Community High-Touch", n=150, blurb="small community lender, higher-risk book",
         age=38, income=48, emp=7, dti=.41, util=.56, late=1.20, history=8,
         lti=.46, rate=14.0, save=.12, lines=3.0, owner=.45, self_emp=.25),
]

#: Each bank is scored on a large fresh cohort rather than a slice of its own book.
#: D Community High-Touch holds 150 customers; a 30-record test set cannot resolve anything, and the
#: whole point of the project is a per-bank comparison. Synthetic customers are free.
#:
#: The cohort is a fixed MULTIPLE of each bank's book, not a fixed size, so the pooled
#: cohort has the same bank mix as the consortium itself. Equal-sized cohorts would
#: quietly reweight the global number towards the small, high-risk banks.
COHORT_MULTIPLE = 6

BOOK_SEED = 20260818
COHORT_SEED = 90260818


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -35, 35)))


def draw(spec, n, rng):
    """One bank's customers, in business units. Returns (raw, model) column blocks."""
    stress = rng.normal(size=n)       # unobserved financial pressure
    stability = rng.normal(size=n)    # unobserved steadiness
    e = rng.normal(size=(n, 11))

    age = np.clip(spec["age"] + 8.0 * e[:, 0], 21, 75)
    income = np.clip(np.exp(np.log(spec["income"])
                            + .34 * (-.28 * stress + .18 * stability + .88 * e[:, 1])), 16, 260)
    employment = np.clip(spec["emp"] + 3.8 * (.55 * stability + .84 * e[:, 2]), 0, age - 18)
    dti = np.clip(spec["dti"] + .085 * (.72 * stress + .69 * e[:, 3]), .02, .78)
    util = np.clip(spec["util"] + .13 * (.70 * stress + .71 * e[:, 4]), .01, .99)
    late = np.clip(rng.poisson(np.clip((spec["late"] + .12)
                                       * np.exp(.34 * stress + .16 * e[:, 5]), .03, 7.0)), 0, 12)
    history = np.clip(spec["history"] + 3.7 * (.30 * ((age - spec["age"]) / 8) + .95 * e[:, 6]),
                      .5, age - 18)
    lti = np.clip(spec["lti"] + .105 * (.62 * stress + .79 * e[:, 7]), .04, .90)
    loan = np.clip(income * lti, 2, 180)
    rate = np.clip(spec["rate"] + 1.8 * (.48 * stress + .88 * e[:, 8]), 3.0, 27.0)
    save_ratio = np.clip(spec["save"] + .095 * (-.42 * stress + .38 * stability + .82 * e[:, 9]),
                         0, .85)
    savings = income * save_ratio
    lines = np.clip(np.rint(spec["lines"] + 1.35 * e[:, 10]), 1, 10)
    home_owner = rng.binomial(1, np.clip(spec["owner"] + .06 * stability - .04 * stress, .05, .95))
    self_emp = rng.binomial(1, np.clip(spec["self_emp"] + .035 * stress, .02, .65))

    # Round to what a bank would actually store, THEN derive the model features from the
    # rounded values. Nothing the model sees carries precision a student cannot see.
    raw = np.column_stack([
        np.round(age, 1), np.round(income, 2), np.round(employment, 1), np.round(dti, 4),
        np.round(util, 4), late.astype(float), np.round(history, 1), np.round(loan, 2),
        np.round(rate, 3), np.round(savings, 2), lines.astype(float),
        home_owner.astype(float), self_emp.astype(float)])
    return raw, to_model(raw)


def to_model(raw):
    """Business units -> the thirteen model features. A fixed, published transform."""
    age, income, emp, dti, util, late, hist, loan, rate, sav, lines, owner, semp = raw.T
    r = REFERENCE
    return np.column_stack([
        (age - r["age_years"][0]) / r["age_years"][1],
        (np.log(income) - r["annual_income_k"][0]) / r["annual_income_k"][1],
        (emp - r["employment_years"][0]) / r["employment_years"][1],
        (dti - r["debt_to_income"][0]) / r["debt_to_income"][1],
        (util - r["credit_utilization"][0]) / r["credit_utilization"][1],
        (late - r["late_payments_12m"][0]) / r["late_payments_12m"][1],
        (hist - r["credit_history_years"][0]) / r["credit_history_years"][1],
        (loan / income - r["loan_to_income"][0]) / r["loan_to_income"][1],
        (rate - r["interest_rate_pct"][0]) / r["interest_rate_pct"][1],
        (sav / income - r["savings_to_income"][0]) / r["savings_to_income"][1],
        (lines - r["num_credit_lines"][0]) / r["num_credit_lines"][1],
        owner, semp])


def to_business(Xz):
    """The thirteen model features -> business units. Inverts `to_model` exactly.

    This is what turns a reconstructed gradient into a readable customer file.
    """
    Xz = np.atleast_2d(np.asarray(Xz, float))
    r = REFERENCE
    age = Xz[:, 0] * r["age_years"][1] + r["age_years"][0]
    income = np.exp(Xz[:, 1] * r["annual_income_k"][1] + r["annual_income_k"][0])
    emp = Xz[:, 2] * r["employment_years"][1] + r["employment_years"][0]
    dti = Xz[:, 3] * r["debt_to_income"][1] + r["debt_to_income"][0]
    util = Xz[:, 4] * r["credit_utilization"][1] + r["credit_utilization"][0]
    late = Xz[:, 5] * r["late_payments_12m"][1] + r["late_payments_12m"][0]
    hist = Xz[:, 6] * r["credit_history_years"][1] + r["credit_history_years"][0]
    loan = (Xz[:, 7] * r["loan_to_income"][1] + r["loan_to_income"][0]) * income
    rate = Xz[:, 8] * r["interest_rate_pct"][1] + r["interest_rate_pct"][0]
    sav = (Xz[:, 9] * r["savings_to_income"][1] + r["savings_to_income"][0]) * income
    lines = Xz[:, 10] * r["num_credit_lines"][1] + r["num_credit_lines"][0]
    return np.column_stack([age, income, emp, dti, util, late, hist, loan, rate, sav,
                            lines, Xz[:, 11], Xz[:, 12]])


def default_probability(Xz):
    """P(default | x) from the ONE shared rule. No bank gets its own definition."""
    return _sigmoid(TRUE_INTERCEPT + np.asarray(Xz, float) @ TRUE_BETA)


def generate(sizes=None, seed=BOOK_SEED, first_id=1):
    """Every bank's customers. Returns [(name, ids, raw, Xz, y, p_true), ...].

    One row is one customer and customer ids are unique across the federation, which is
    what lets the privacy part say "example-level DP is customer-level DP here" without
    a group-privacy argument.
    """
    rng = np.random.default_rng(seed)
    out, start = [], first_id
    for spec in BANKS:
        n = spec["n"] if sizes is None else sizes[spec["name"]]
        raw, Xz = draw(spec, n, rng)
        p = default_probability(Xz)
        # y ~ Bernoulli(p) IS the irreducible noise. Two customers with identical files
        # do not always get the same outcome, and no extra label flipping is needed.
        # Drawn exactly as the source package does, so at full scale this generator
        # reproduces its published records bit for bit. Checked by checks/verify_source.py.
        y = rng.binomial(1, p).astype(float)
        ids = np.array([f"C{v:06d}" for v in range(start, start + n)])
        start += n
        out.append((spec["name"], ids, raw, Xz, y, p))
    return out
