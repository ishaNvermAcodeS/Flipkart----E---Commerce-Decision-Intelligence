"""Shared helpers for part1.py ... part4.py.

Author: Ishan Verma (iitmd_dsai_2508836)

Importing this module never runs a Part: it only defines functions.
Everything here is plain Python + NumPy array arithmetic; no ready-made
model or metric routine is used.
"""
import csv
import os

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(ROOT, "data", "records.csv")
OUT_DIR = os.path.join(ROOT, "outputs")

FEATURES = [
    "customer_tenure_months",        # X1
    "complaint_count_90d",           # X2
    "discount_order_share",          # X3
    "avg_order_value_inr_hundreds",  # X4
    "loyalty_score",                 # X5
    "cod_flag",                      # X6
    "plus_member_flag",              # X7
]
OUTCOME = "churned"
VALUE = "order_value_inr_hundreds"


# --------------------------------------------------------------------------- data
def load_data():
    """Return (X, y, value) in file order. X has the seven predictors X1..X7."""
    with open(DATA_PATH, newline="") as f:
        rows = list(csv.DictReader(f))
    X = np.array([[float(r[c]) for c in FEATURES] for r in rows], dtype=float)
    y = np.array([int(r[OUTCOME]) for r in rows], dtype=int)
    v = np.array([float(r[VALUE]) for r in rows], dtype=float)
    return X, y, v


def sample_mean(a, axis=None):
    a = np.asarray(a, dtype=float)
    n = a.shape[0] if axis == 0 else a.size
    return a.sum(axis=axis) / n


def sample_sd(a, axis=None):
    """Sample standard deviation (divisor n - 1), written out by hand."""
    a = np.asarray(a, dtype=float)
    if axis is None:
        m = a.sum() / a.size
        return float(np.sqrt(((a - m) ** 2).sum() / (a.size - 1)))
    n = a.shape[axis]
    m = a.sum(axis=axis, keepdims=True) / n
    return np.sqrt(((a - m) ** 2).sum(axis=axis) / (n - 1))


# --------------------------------------------------------------------------- split
def stratified_split(y):
    """Task 2.1 split: for each class (0 first, then 1) in file order, the first
    floor(7 * count / 10) rows go to train, the rest to test. Both sets keep file order."""
    train_idx, test_idx = [], []
    for cls in (0, 1):
        idx = [i for i in range(len(y)) if y[i] == cls]
        k = (7 * len(idx)) // 10
        train_idx += idx[:k]
        test_idx += idx[k:]
    return np.array(sorted(train_idx)), np.array(sorted(test_idx))


def get_split():
    """Load data, split it, and standardise X1..X7 with training mean and sample SD."""
    X, y, v = load_data()
    tr, te = stratified_split(y)
    mu = sample_mean(X[tr], axis=0)
    sd = sample_sd(X[tr], axis=0)
    Xtr = (X[tr] - mu) / sd
    Xte = (X[te] - mu) / sd
    return {
        "X": X, "y": y, "v": v, "tr": tr, "te": te,
        "Xtr": Xtr, "Xte": Xte, "ytr": y[tr], "yte": y[te],
        "mu": mu, "sd": sd,
    }


# --------------------------------------------------------------------------- logistic
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def bce(y, p):
    """Binary cross-entropy averaged over rows."""
    return float(-np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)) / len(y))


def train_logistic(X, y, eta, n_updates, record=()):
    """Full-batch gradient descent from w = 0, b = 0.
    Returns (w, b, losses) where losses maps update-count -> training loss."""
    n, d = X.shape
    w = np.zeros(d)
    b = 0.0
    losses = {}
    if 0 in record:
        losses[0] = bce(y, sigmoid(X @ w + b))
    for t in range(1, n_updates + 1):
        p = sigmoid(X @ w + b)
        err = p - y
        gw = (X.T @ err) / n
        gb = err.sum() / n
        w = w - eta * gw
        b = b - eta * gb
        if t in record:
            losses[t] = bce(y, sigmoid(X @ w + b))
    return w, b, losses


def fit_part2_logistic(split):
    """The Task 2.2 model: eta = 0.1, 1000 updates on the 560 training rows."""
    return train_logistic(split["Xtr"], split["ytr"], 0.1, 1000, record=(0, 1, 10, 100, 1000))


# --------------------------------------------------------------------------- metrics
def confusion(y, pred):
    y = np.asarray(y)
    pred = np.asarray(pred)
    tp = int(np.sum((pred == 1) & (y == 1)))
    fp = int(np.sum((pred == 1) & (y == 0)))
    tn = int(np.sum((pred == 0) & (y == 0)))
    fn = int(np.sum((pred == 0) & (y == 1)))
    return tp, fp, tn, fn


def _div(a, b):
    return a / b if b != 0 else 0.0


def metrics(tp, fp, tn, fn):
    acc = _div(tp + tn, tp + fp + tn + fn)
    prec = _div(tp, tp + fp)
    rec = _div(tp, tp + fn)
    spec = _div(tn, tn + fp)
    fpr = _div(fp, fp + tn)
    f1 = _div(2 * prec * rec, prec + rec)
    return {"accuracy": acc, "precision": prec, "recall": rec,
            "specificity": spec, "fpr": fpr, "f1": f1}


def predict_at(p, tau):
    """Positive when probability >= tau."""
    return (np.asarray(p) >= tau).astype(int)


# --------------------------------------------------------------------------- KNN
def knn_votes(Xref, yref, Xq, k):
    """Number of positive neighbours among the K nearest reference rows for each query.
    Neighbours ranked by Euclidean distance, ties by reference position (stable sort)."""
    votes = np.empty(len(Xq), dtype=int)
    for i, q in enumerate(Xq):
        d = np.sqrt(np.sum((Xref - q) ** 2, axis=1))
        order = np.argsort(d, kind="stable")[:k]
        votes[i] = int(yref[order].sum())
    return votes


def knn_predict(Xref, yref, Xq, k):
    """Positive when at least (K + 1) / 2 of the K neighbours are positive (K odd)."""
    return (2 * knn_votes(Xref, yref, Xq, k) >= k + 1).astype(int)


def knn_cv(Xtr, ytr, ks=range(1, 22, 2), folds=5):
    """5-fold CV on training rows; fold = position mod 5. Returns list of (K, mean F1)."""
    pos = np.arange(len(ytr))
    fold = pos % folds
    out = []
    for k in ks:
        f1s = []
        for f in range(folds):
            ref = fold != f
            hold = fold == f
            pred = knn_predict(Xtr[ref], ytr[ref], Xtr[hold], k)
            f1s.append(metrics(*confusion(ytr[hold], pred))["f1"])
        out.append((k, sum(f1s) / folds))
    return out


def best_k(cv_rows):
    """Highest mean F1; ties go to the smaller K (rows are in ascending K)."""
    best = cv_rows[0]
    for row in cv_rows[1:]:
        if row[1] > best[1]:
            best = row
    return best[0]


# --------------------------------------------------------------------------- cost
COST_FP = 50   # INR per false alarm (coupon to a customer who would have stayed)
COST_FN = 400  # INR per missed churner


def cost(fp, fn):
    return COST_FP * fp + COST_FN * fn


def choose_cost_tau(p_train, y_train):
    """tau = k/20, k = 1..19; lowest training cost, ties to the smaller tau."""
    best_tau, best_cost = None, None
    for k in range(1, 20):
        tau = k / 20
        _, fp, _, fn = confusion(y_train, predict_at(p_train, tau))
        c = cost(fp, fn)
        if best_cost is None or c < best_cost:
            best_tau, best_cost = tau, c
    return best_tau, best_cost


# --------------------------------------------------------------------------- output
def fmt(v):
    """Integers as integers, bools as yes/no, strings as-is, floats with 4 dp."""
    if isinstance(v, (bool, np.bool_)):
        return "yes" if v else "no"
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, str):
        return v
    return f"{float(v):.4f}"


def write_results(name, items):
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, name)
    with open(path, "w") as f:
        for k, v in items:
            f.write(f"{k} = {fmt(v)}\n")
    print(f"wrote {os.path.relpath(path, ROOT)} ({len(items)} keys)")


def write_csv(name, header, rows):
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, name)
    with open(path, "w", newline="") as f:
        f.write(",".join(header) + "\n")
        for r in rows:
            f.write(",".join(fmt(x) for x in r) + "\n")
    print(f"wrote {os.path.relpath(path, ROOT)} ({len(rows)} rows)")
