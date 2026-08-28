"""fedcore — the dataset-independent engine behind "One Model, Four ___".

This module knows about federated learning. It knows nothing about banks, customers or
credit. A *scenario* supplies the data and the vocabulary; everything
here works the same either way.

  src/scenario.py   synthetic credit-risk records, four banks

A scenario must call `configure(n_features=...)` once at import, then build a
`Federation` of `Site` objects. See `core/FLOW.md` for the teaching spine that both
scenarios share.

Design rules:
  * numpy only. No torch, no sklearn. The whole notebook runs on a Colab CPU in minutes.
  * one model everywhere: logistic regression. It carries prediction, the distributed
    gradient, weighted averaging, the leakage attack, both kinds of clipping and DP-SGD.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from math import exp, lgamma, log

import numpy as np

# ------------------------------------------------------------------ dimensions

#: Set by the active scenario via configure(). Logistic regression on N_FEATURES
#: inputs has N_FEATURES + 1 parameters, the last one being the bias.
N_FEATURES = 0
N_PARAMS = 1


def configure(n_features: int):
    """Tell the engine how wide the feature vector is. Called once by a scenario."""
    global N_FEATURES, N_PARAMS
    N_FEATURES = int(n_features)
    N_PARAMS = N_FEATURES + 1
    return N_PARAMS


# ------------------------------------------------------------------- vocabulary

@dataclass(frozen=True)
class Vocab:
    """The words one scenario uses, so figures and prose can be written once.

    banks: Vocab("bank", "banks", "customer", "customers", "defaulted", ...)
    banks:     Vocab("bank", "banks", "customer", "customers", "in default", ...)
    """
    site: str
    sites: str
    member: str            # one record: "customer"
    members: str
    positive: str          # what y == 1 means: "ill" / "in default"
    positive_rate: str     # "default rate"
    setting: str           # "consortium" / "consortium"
    outcome_question: str  # "will this customer default?"
    negative: str = "the rest"      # what y == 0 means: "healthy" / "paying"
    record: str = "record"          # what one row is called in the attack parts
    records: str = "records"


# ---------------------------------------------------------------- the federation

@dataclass(frozen=True)
class Site:
    """One participant's records. X is (n, N_FEATURES) standardised, y is (n,) in {0., 1.}."""
    name: str
    X: np.ndarray
    y: np.ndarray

    @property
    def n(self) -> int:
        return len(self.y)

    @property
    def share_positive(self) -> float:
        return float(self.y.mean())

    def __repr__(self) -> str:
        return f"<{self.name}: {self.n} records, {self.share_positive:.0%} positive>"


@dataclass(frozen=True)
class Federation:
    """The participants, plus the standardisation constants used to build them."""
    sites: list
    mean: np.ndarray = None
    std: np.ndarray = None

    def __iter__(self):
        return iter(self.sites)

    def __len__(self):
        return len(self.sites)

    def __getitem__(self, key):
        if isinstance(key, str):
            return next(s for s in self.sites if s.name.startswith(key))
        return self.sites[key]

    @property
    def names(self) -> list:
        return [s.name for s in self.sites]

    @property
    def sizes(self) -> np.ndarray:
        return np.array([s.n for s in self.sites], float)

    def pooled(self) -> tuple:
        """Every record in one place. The reference we are not allowed to deploy."""
        return (np.vstack([s.X for s in self.sites]),
                np.concatenate([s.y for s in self.sites]))


def standardise(sites):
    """Centre and scale on the pooled records. Returns (sites, mean, std).

    A simplification worth one sentence in the notebook: a real federation would have
    to agree these constants first, which it can do with exactly the weighted average
    of part 3.
    """
    allX = np.vstack([s.X for s in sites])
    mean, std = allX.mean(0), allX.std(0) + 1e-9
    return [replace(s, X=(s.X - mean) / std) for s in sites], mean, std


def split(federation, seed=0, train_fraction=0.657):
    """Split every participant into training and test records. Returns two Federations."""
    rng = np.random.default_rng(seed)
    tr, te = [], []
    for s in federation:
        idx = rng.permutation(s.n)
        k = int(round(s.n * train_fraction))
        tr.append(Site(s.name, s.X[idx[:k]], s.y[idx[:k]]))
        te.append(Site(s.name, s.X[idx[k:]], s.y[idx[k:]]))
    return (Federation(tr, federation.mean, federation.std),
            Federation(te, federation.mean, federation.std))

# ------------------------------------------------------------------- partitions

def _concentrate(y, share, n_sites, rng):
    """One group is handed `share` of every positive record; all groups end the same size.

    This is the dial part 2 turns. Unlike a Dirichlet draw it moves one thing only: how
    the outcome is spread. Group sizes stay equal, so nothing else can explain the result.
    """
    pos = rng.permutation(np.where(y == 1.0)[0])
    neg = rng.permutation(np.where(y == 0.0)[0])
    take = int(round(share * len(pos)))
    rest = np.array_split(pos[take:], n_sites - 1) if n_sites > 1 else []
    got = [pos[:take]] + [np.asarray(r, dtype=int) for r in rest]

    target = len(y) // n_sites                       # every group the same size
    parts, at = [], 0
    for k in range(n_sites):
        want = max(0, target - len(got[k])) if k < n_sites - 1 else len(neg) - at
        parts.append(np.concatenate([got[k], neg[at:at + want]]))
        at += want
    return [np.asarray(sorted(p.tolist()), dtype=int) for p in parts]


def partition_indices(y, regime, seed=0, n_sites=4, min_size=0, X=None):
    """Which record goes to which participant, as index arrays.

    Split out from `partition` so a figure can colour the original records by the group
    they landed in, which is impossible once the records have been copied into Sites.

    "iid"             random, equal sizes, same outcome mix everywhere
    "concentrate:P"   equal sizes, but one group is handed fraction P of every positive
                      record and the rest share what is left. Deterministic.
    "sortby:J"        sort every record by feature J and cut the order into equal blocks,
                      so each group holds one band of that feature. Deterministic.
    "dirichlet:A"     uneven by a random draw: smaller A is lumpier
    """
    rng = np.random.default_rng(1000 + seed)
    n = len(y)
    if regime.startswith("concentrate"):
        return _concentrate(y, float(regime.split(":")[1]), n_sites, rng)
    if regime.startswith("sortby"):
        if X is None:
            raise ValueError("sortby needs the records, not only the outcomes")
        order = np.argsort(X[:, int(regime.split(":")[1])], kind="stable")
        return [np.asarray(sorted(g.tolist()), dtype=int)
                for g in np.array_split(order, n_sites)]
    if regime == "iid":
        props = np.ones((2, n_sites)) / n_sites
        alpha = None
    elif regime.startswith("dirichlet"):
        alpha = float(regime.split(":")[1])
        props = np.array([rng.dirichlet([alpha] * n_sites) for _ in range(2)])
    else:
        raise ValueError(f"unknown regime {regime!r}")

    for _ in range(1000):
        parts = [[] for _ in range(n_sites)]
        for ci, cls in enumerate((0.0, 1.0)):
            idx = rng.permutation(np.where(y == cls)[0])
            ends = np.cumsum((props[ci] * len(idx)).astype(int))[:-1]
            for k, chunk in enumerate(np.split(idx, ends)):
                parts[k].extend(chunk.tolist())
        counts = np.array([len(p) for p in parts])
        if counts.min() >= min_size or alpha is None:
            break
        props = np.array([rng.dirichlet([alpha] * n_sites) for _ in range(2)])
    return [np.asarray(sorted(p), dtype=int) for p in parts]


def separability(X, groups, seed=0, steps=4000, lr=0.4, l2=1e-3):
    """Can a model work out which group a record belongs to, from the record alone?

    Softmax regression over the features, scored by BALANCED accuracy: the mean of the
    per-group recalls rather than the raw hit rate. Raw accuracy reads group size instead
    of group difference, because a probe that always names the biggest group scores that
    group's share and looks informative while having learnt nothing. Chance is 1/K.

    Returns (balanced accuracy on held-out records, chance).
    """
    rng = np.random.default_rng(seed)
    lab = np.concatenate([np.full(len(g), k) for k, g in enumerate(groups)])
    Z = X[np.concatenate(list(groups))]
    Z = (Z - Z.mean(0)) / np.where(Z.std(0) < 1e-9, 1.0, Z.std(0))
    Z = np.hstack([Z, np.ones((len(Z), 1))])

    perm = rng.permutation(len(Z))
    cut = int(0.7 * len(Z))
    tr, te = perm[:cut], perm[cut:]
    K = len(groups)
    Y = np.eye(K)[lab[tr]]
    W = np.zeros((Z.shape[1], K))
    prev = np.inf
    for i in range(steps):
        S = Z[tr] @ W
        P = np.exp(S - S.max(1, keepdims=True)); P /= P.sum(1, keepdims=True)
        W -= lr * (Z[tr].T @ (P - Y) / len(tr) + l2 * W)
        if i % 200 == 0:                       # stop once it has actually converged
            L = -np.log(np.clip(P[np.arange(len(tr)), lab[tr]], 1e-12, None)).mean()
            if abs(prev - L) < 1e-7:
                break
            prev = L
    pred = (Z[te] @ W).argmax(1)
    recalls = [float((pred[lab[te] == k] == k).mean()) if (lab[te] == k).any() else 0.0
               for k in range(K)]
    return float(np.mean(recalls)), 1.0 / K


def partition(federation, regime, seed=0, n_sites=4, min_size=0):
    """Re-divide the same records into a different federation.

    "natural"        the participants as they really are, untouched
    "iid"            random, equal sizes
    "concentrate:P"  equal sizes, one participant handed fraction P of the positives
    "sortby:J"       equal sizes, each participant holding one band of feature J
    "dirichlet:A"    uneven: smaller A spreads the two outcomes more unevenly

    `min_size` redraws until every participant holds at least that many records. A
    Dirichlet draw is allowed to be brutal, but a participant with nothing to train on
    is a broken experiment rather than a hard one.
    """
    if regime == "natural":
        return federation
    X, y = federation.pooled()
    groups = partition_indices(y, regime, seed=seed, n_sites=n_sites, min_size=min_size,
                               X=X)
    sites = [Site(f"Group {k + 1}", X[g], y[g]) for k, g in enumerate(groups)]
    return Federation(sites, federation.mean, federation.std)


# ------------------------------------------------------------------------ model

def init():
    """A model that predicts nothing: ten weights and a bias, all zero."""
    return np.zeros(N_PARAMS)


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -60, 60)))


def predict_proba(w, X):
    return _sigmoid(X @ w[:N_FEATURES] + w[N_FEATURES])


def predict(w, X):
    return (predict_proba(w, X) > 0.5).astype(float)


def loss(w, X, y):
    """Average cross-entropy. Lower is better; 0.693 is a model that knows nothing."""
    p = np.clip(predict_proba(w, X), 1e-9, 1 - 1e-9)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def gradient(w, X, y):
    """The average training direction over these records."""
    d = (predict_proba(w, X) - y) / len(y)
    return np.concatenate([X.T @ d, [d.sum()]])


def per_record_gradients(w, X, y):
    """One row per record: (p_i - y_i) * [x_i, 1]. Shape (n, N_PARAMS).

    This is the whole reason the project uses logistic regression. Record-level DP
    needs these, and here they are a single broadcast product rather than a backward pass.
    """
    r = (predict_proba(w, X) - y)[:, None]
    return np.hstack([r * X, r])


def train(w, X, y, steps, lr=0.5, batch=None, start=0):
    """Plain gradient descent inside one site.

    batch limits the round to that many records, the same ones for every step, beginning
    at `start` and wrapping; leave it None for the whole book. A site picks its sample
    once and works it rather than drawing a new one each step, and the window is fixed
    rather than random so the JavaScript port can be checked against this record for
    record.

    That is what the attack turns on. Every step contributes c_s * [x, 1] for the same x,
    so the update is (sum of c_s) * [x, 1]: the steps change the scalar and never the
    direction, and dividing the weight part by the bias part returns the customer however
    many steps were taken. Only a wider batch mixes people in.
    """
    w = np.asarray(w, float).copy()
    n = len(y)
    m = int(batch) if batch and batch < n else None
    idx = (np.arange(m) + start) % n if m is not None else None
    for _ in range(steps):
        if m is None:
            g = gradient(w, X, y)
        else:
            g = gradient(w, X[idx], y[idx])
        w = w - lr * g
    return w

# ------------------------------------------------------------------- federation

def aggregate_weighted(updates, sizes):
    """Combine site updates in proportion to how many records each one holds.

    This is the rule part 3 derives. It is not a convention: it is what reproducing
    the pooled result requires when sites hold different numbers of records.
    """
    sizes = np.asarray(sizes, float)
    return np.average(np.asarray(updates), axis=0, weights=sizes / sizes.sum())


def aggregate_plain(updates, sizes=None):
    """Treat every site equally. Measured 0.092 off the pooled gradient on the banks."""
    return np.mean(np.asarray(updates), axis=0)


@dataclass(frozen=True)
class Privacy:
    """Record-level DP-SGD settings. C is a clipping limit, z a noise multiplier.

    The defaults are the banks scenario's: C is the median contribution across all
    2,450 records, which is what part 7 asks the student to derive.
    """
    C: float = 1.35      # the measured median per-record contribution
    z: float = 4.0
    lot: int = 64        # expected records per noisy step, so q = lot / n


def dp_local(w, X, y, steps, lr, privacy, rng):
    """One site's local training, made private one record at a time.

    Poisson sampling, to match the accountant in `epsilon`. Every record is included
    independently with probability q = lot/n, so the batch size varies and the sum is
    divided by the expected lot rather than the actual count.
    """
    w = np.asarray(w, float).copy()
    n = len(y)
    q = min(1.0, privacy.lot / n)
    for _ in range(steps):
        chosen = rng.random(n) < q
        if not chosen.any():
            continue
        g = clip_and_noise(per_record_gradients(w, X[chosen], y[chosen]),
                           privacy.C, privacy.z, q * n, rng)
        w = w - lr * g
    return w


def clip_and_noise(record_grads, C, z, lot, rng):
    """Limit each record's contribution, add calibrated noise, then average.

    Clipping bounds how much one record can matter, which is what makes the noise
    size meaningful. The noise is what provides the guarantee. Reference version of
    the student's task 6.
    """
    norms = np.linalg.norm(record_grads, axis=1, keepdims=True)
    clipped = record_grads * np.minimum(1.0, C / (norms + 1e-12))
    return (clipped.sum(axis=0) + rng.normal(0.0, z * C, record_grads.shape[1])) / lot


def fed_round(w, sites, E, lr=0.5, aggregate=aggregate_weighted,
              privacy=None, rng=None):
    """One exchange: every site trains locally, sends its change, gets the average."""
    updates, sizes = [], []
    for h in sites:
        if h.n == 0:
            continue
        if privacy is None:
            local = train(w, h.X, h.y, E, lr)
        else:
            local = dp_local(w, h.X, h.y, E, lr, privacy, rng)
        updates.append(local - w)
        sizes.append(h.n)
    return w + aggregate(updates, sizes)


def run(sites, E=5, rounds=20, lr=0.5, aggregate=aggregate_weighted,
        privacy=None, seed=0, evaluate_on=None, w0=None, sample=None):
    """Run a federation and record what happened after every exchange.

    Returns a dict with the final model, and per-round test loss and accuracy when
    `evaluate_on` is given.

    `sample` draws that many participants per round instead of using all of them. With
    four participants you wait for everybody; with five hundred you cannot, and sampling
    is how every deployed system at that size actually runs.
    """
    rng = np.random.default_rng(seed)
    w = init() if w0 is None else np.asarray(w0, float).copy()
    sites = list(sites)
    hist = {"loss": [], "accuracy": []}
    Xe = ye = None
    if evaluate_on is not None:
        Xe, ye = evaluate_on.pooled()
    for _ in range(rounds):
        here = sites if sample is None else [
            sites[i] for i in rng.choice(len(sites), min(sample, len(sites)),
                                         replace=False)]
        w = fed_round(w, here, E, lr, aggregate, privacy, rng)
        if Xe is not None:
            hist["loss"].append(loss(w, Xe, ye))
            hist["accuracy"].append(accuracy(w, Xe, ye))
    hist["w"] = w
    hist["exchanges"] = rounds
    hist["local_steps"] = rounds * E
    return hist


def exchanges_to_target(sites, evaluate_on, E, lr=0.5, target=None,
                        tolerance=1.01, cap=200, aggregate=aggregate_weighted):
    """How many exchanges before the federation matches pooled training?

    Returns None if it never does within `cap`. Part 4 reports this per split,
    never averaged: at ten local steps and above the outcome is bimodal.
    """
    Xe, ye = evaluate_on.pooled()
    if target is None:
        Xp, yp = sites.pooled() if isinstance(sites, Federation) else (
            np.vstack([h.X for h in sites]), np.concatenate([h.y for h in sites]))
        target = loss(train(init(), Xp, yp, 20000, lr), Xe, ye)
    w = init()
    for r in range(1, cap + 1):
        w = fed_round(w, sites, E, lr, aggregate)
        if loss(w, Xe, ye) <= target * tolerance:
            return r
    return None

# -------------------------------------------------------------------- privacy

def epsilon(q, z, steps, delta=1e-5, orders=range(2, 65)):
    """Privacy loss for `steps` noisy steps of the Poisson-subsampled Gaussian mechanism.

    Standard Renyi accounting over integer orders, then converted to (epsilon, delta).
    The sampling model here must match `dp_local`, which is why both are Poisson: an
    accountant that assumes one sampling scheme while the training loop uses another
    reports a number that is not about the thing that ran.
    """
    if z <= 0:
        return float("inf")
    best = float("inf")
    for a in orders:
        terms = []
        for k in range(a + 1):
            log_binom = lgamma(a + 1) - lgamma(k + 1) - lgamma(a - k + 1)
            terms.append(log_binom
                         + (a - k) * log(max(1 - q, 1e-300))
                         + k * log(max(q, 1e-300))
                         + k * (k - 1) / (2 * z * z))
        m = max(terms)
        rdp = (m + log(sum(exp(t - m) for t in terms))) / (a - 1)
        best = min(best, rdp * steps + log(1 / delta) / (a - 1))
    return best


def rdp_order(q, z, a):
    """Renyi divergence of one Poisson-subsampled Gaussian step at integer order a."""
    terms = []
    for k in range(a + 1):
        log_binom = lgamma(a + 1) - lgamma(k + 1) - lgamma(a - k + 1)
        terms.append(log_binom
                     + (a - k) * log(max(1 - q, 1e-300))
                     + k * log(max(q, 1e-300))
                     + k * (k - 1) / (2 * z * z))
    m = max(terms)
    return (m + log(sum(exp(t - m) for t in terms))) / (a - 1)


def epsilon_schedule(phases, z, delta=1e-5, orders=range(2, 65)):
    """Privacy loss over phases that sampled at different rates.

    `phases` is a list of (q, steps). A site whose book is still filling samples a smaller
    fraction of it at every later meeting, so its q falls over time; Renyi accounting
    composes by adding the per-order divergences, phase by phase, before converting.
    With one phase this is exactly `epsilon`.
    """
    if z <= 0:
        return float("inf")
    best = float("inf")
    for a in orders:
        total = sum(rdp_order(q, z, a) * steps for q, steps in phases)
        best = min(best, total + log(1 / delta) / (a - 1))
    return best


def epsilon_per_site(sites, privacy, rounds, E, delta=1e-5):
    """What each site's records actually get. The gap is the point.

    A small site cannot subsample: once the lot is as large as the site, the sampling
    rate is 1, so it gets no amplification and a much weaker guarantee for the same noise.
    """
    steps = rounds * E
    return {h.name: epsilon(min(1.0, privacy.lot / h.n), privacy.z, steps, delta)
            for h in sites}


def contribution_sizes(w, sites):
    """Every record's gradient norm, pooled. Part 7 picks the clipping limit from these."""
    return np.concatenate([np.linalg.norm(per_record_gradients(w, h.X, h.y), axis=1)
                           for h in sites])

# --------------------------------------------------------------------- attacks

def reconstruct(update, lr=None):
    """Recover the record an update was computed from.

    For one record and one step, grad_w = (p - y)x and grad_b = (p - y), so dividing
    one by the other returns x exactly.

    Assumptions, and they are the whole story: one record, one step, the model is
    known, no noise has been added, and the bias gradient is not zero. Change any of
    the first two and this stops identifying an individual, which is not the same as
    saying nothing leaks.
    """
    g = np.asarray(update, float)
    if lr is not None:            # an update is -lr * gradient; the ratio cancels lr anyway
        g = -g / lr
    if abs(g[N_FEATURES]) < 1e-15:
        raise ValueError("the bias gradient is zero, so the ratio is undefined")
    return g[:N_FEATURES] / g[N_FEATURES]


def isolate(aggregate_update, my_updates, n_participants=None, my_sizes=None,
            victim_size=None):
    """Pull one participant's update back out of an average you are part of.

    Two versions, because the arithmetic depends on how the coordinator combines:

        plain average   g_v = K * G - sum(g_mine)
        size-weighted   g_v = (N * G - sum_k n_k g_k) / n_v

    The second is the one that matters here: this project's coordinator weights by
    size (part 3 derives why), so the plain formula recovers the wrong vector. Pass
    `my_sizes` and `victim_size` to use it.

    Assumptions, and they are the whole story: the victim is the only contributor you
    do not already control, you know the weights, and aggregation is synchronous. In
    general a group of colluders learns the weighted sum of the honest updates, which
    is revealing when few honest participants remain.

    Isolating a participant's update is NOT the same as isolating one of its records.
    A normal update is computed over many records and often several local steps.
    Reconstructing an individual on top of this needs the isolated contribution to be
    a single-record gradient — which is what `reconstruct` requires and what part 6
    has to say out loud.
    """
    G = np.asarray(aggregate_update, float)
    ups = np.asarray(my_updates, float).reshape(-1, N_PARAMS)
    if my_sizes is None:
        if n_participants is None:
            raise ValueError("give n_participants for a plain average, or my_sizes and "
                             "victim_size for the size-weighted aggregation")
        return n_participants * G - ups.sum(axis=0)
    my_sizes = np.asarray(my_sizes, float)
    if victim_size is None:
        raise ValueError("victim_size is needed to undo the size weighting")
    N = my_sizes.sum() + victim_size
    return (N * G - (my_sizes[:, None] * ups).sum(axis=0)) / victim_size


# --------------------------------------------------------------------- metrics

def accuracy(w, X, y):
    return float((predict(w, X) == y).mean())


def auc(w, X, y):
    """Undefined when a site's records all share one outcome."""
    s = predict_proba(w, X)
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    return float((pos[:, None] > neg[None, :]).mean()
                 + 0.5 * (pos[:, None] == neg[None, :]).mean())


def wilson(correct, n, z=1.96):
    """95% interval for an accuracy. Wilson, not the textbook one: at n=16 the normal
    approximation produces intervals that run past 0 and 1."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = correct / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def evaluate(w, sites, min_n=20):
    """Per-site accuracy with its interval, plus the global figure.

    `worst` skips sites too small or too one-sided to carry a meaningful number.
    """
    rows = {}
    for h in sites:
        if h.n == 0:
            continue
        a = accuracy(w, h.X, h.y)
        lo, hi = wilson(round(a * h.n), h.n)
        rows[h.name] = {"accuracy": a, "low": lo, "high": hi, "n": h.n,
                        "auc": auc(w, h.X, h.y), "share_positive": h.share_positive,
                        "evaluable": bool(h.n >= min_n and 0 < h.y.mean() < 1)}
    X, y = sites.pooled()
    usable = [r["accuracy"] for r in rows.values() if r["evaluable"]]
    return {"per_site": rows,
            "global": accuracy(w, X, y),
            "global_auc": auc(w, X, y),
            "worst": min(usable) if usable else float("nan")}
