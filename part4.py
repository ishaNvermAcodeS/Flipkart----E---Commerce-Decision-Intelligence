"""Part 4 - Model choice, a hand-run boosting calculation and network sizing.

Author: Ishan Verma (iitmd_dsai_2508836)
Writes outputs/part4_results.txt, part4_splits.csv, part4_voting.csv,
       part4_adaboost.csv, part4_architecture.csv
Reuses the Task 2.1 split and the Part 2 models through common.py.
"""
import math

import numpy as np

from common import (best_k, confusion, fit_part2_logistic, get_split, knn_cv, knn_votes,
                    metrics, predict_at, sigmoid, write_csv, write_results)


# --------------------------------------------------------------------------- impurity
def entropy(pos, n):
    if n == 0:
        return 0.0
    h = 0.0
    for c in (pos, n - pos):
        p = c / n
        if p > 0:
            h -= p * math.log2(p)
    return h


def gini(pos, n):
    if n == 0:
        return 0.0
    p1 = pos / n
    p0 = 1 - p1
    return 1 - (p1 ** 2 + p0 ** 2)


def split_gains(mask, y):
    n = len(y)
    P = int(y.sum())
    ny, py = int(mask.sum()), int(y[mask].sum())
    nn, pn = n - ny, P - py
    ig = entropy(P, n) - (ny / n * entropy(py, ny) + nn / n * entropy(pn, nn))
    gg = gini(P, n) - (ny / n * gini(py, ny) + nn / n * gini(pn, nn))
    return ny, py, nn, pn, ig, gg


def split_masks(X):
    """Candidate splits in original units; 'yes' side is where the condition holds."""
    return {
        "S1": X[:, 5] == 1,      # X6 = 1 (cash on delivery)
        "S2": X[:, 6] == 1,      # X7 = 1 (plus member)
        "S3": X[:, 2] > 0.5,     # X3 > 0.5 (discount share)
        "S4": X[:, 0] <= 24,     # X1 <= 24 (tenure months)
    }


def argmax_first(d):
    """Key with the largest value; ties go to the first key in order."""
    best = None
    for k, val in d.items():
        if best is None or val > d[best]:
            best = k
    return best


# --------------------------------------------------------------------------- conv sizing
def conv_out(size, f, pad, stride):
    return (size - f + 2 * pad) // stride + 1


def main():
    s = get_split()
    X, tr, te = s["X"], s["tr"], s["te"]
    ytr, yte = s["ytr"], s["yte"]
    res = []

    # ---- 4.1 purest split on the 560 training rows (original units)
    Xtr_raw, Xte_raw = X[tr], X[te]
    n, P = len(ytr), int(ytr.sum())
    res += [("root_entropy", entropy(P, n)), ("root_gini", gini(P, n))]
    masks_tr = split_masks(Xtr_raw)
    split_rows, ig, gg = [], {}, {}
    for name, m in masks_tr.items():
        ny, py, nn, pn, i_g, g_g = split_gains(m, ytr)
        ig[name], gg[name] = i_g, g_g
        split_rows.append([name, ny, py, nn, pn, i_g, g_g])
    write_csv("part4_splits.csv",
              ["split", "n_yes", "positives_yes", "n_no", "positives_no", "info_gain", "gini_gain"],
              split_rows)
    best_ig = argmax_first(ig)
    res += [("best_split_by_info_gain", best_ig), ("best_split_by_gini_gain", argmax_first(gg))]

    # ---- 4.2 stump, then hard and soft voting on the test rows
    m_tr = masks_tr[best_ig]
    rate_yes = ytr[m_tr].sum() / m_tr.sum()
    rate_no = ytr[~m_tr].sum() / (~m_tr).sum()
    res += [("stump_rate_yes", rate_yes), ("stump_rate_no", rate_no)]
    m_te = split_masks(Xte_raw)[best_ig]
    p_stump = np.where(m_te, rate_yes, rate_no)

    w, b, _ = fit_part2_logistic(s)
    p_log = sigmoid(s["Xte"] @ w + b)
    k = best_k(knn_cv(s["Xtr"], ytr))
    votes = knn_votes(s["Xtr"], ytr, s["Xte"], k)
    p_knn = votes / k                      # share of positive neighbours

    pred = {
        "logistic": predict_at(p_log, 0.3),
        "knn": (2 * votes >= k + 1).astype(int),
        "stump": predict_at(p_stump, 0.3),
    }
    pred["hard_vote"] = ((pred["logistic"] + pred["knn"] + pred["stump"]) >= 2).astype(int)
    pred["soft_vote"] = predict_at((p_log + p_knn + p_stump) / 3, 0.3)
    vote_rows = []
    for name in ("logistic", "knn", "stump", "hard_vote", "soft_vote"):
        tp, fp, tn, fn = confusion(yte, pred[name])
        mt = metrics(tp, fp, tn, fn)
        vote_rows.append([name, tp, fp, tn, fn, mt["precision"], mt["recall"], mt["f1"]])
    write_csv("part4_voting.csv", ["model", "tp", "fp", "tn", "fn", "precision", "recall", "f1"], vote_rows)

    # ---- 4.3 two rounds of AdaBoost by hand
    yb = np.array([1, 1, 1, -1, -1, -1, 1, -1])
    h = [np.array([1, 1, -1, -1, -1, -1, -1, -1]),
         np.array([-1, 1, 1, -1, 1, -1, 1, -1])]
    wts = np.full(8, 1 / 8)
    alphas, ada_rows = [], []
    for t in range(2):
        eps = float(wts[h[t] != yb].sum())
        alpha = 0.5 * math.log((1 - eps) / eps)
        unnorm = wts * np.exp(-alpha * yb * h[t])
        after = unnorm / unnorm.sum()
        for p in range(8):
            ada_rows.append([t + 1, p + 1, wts[p], unnorm[p], after[p]])
        res += [(f"round{t + 1}_error", eps), (f"round{t + 1}_alpha", alpha)]
        alphas.append(alpha)
        wts = after
    write_csv("part4_adaboost.csv",
              ["round", "point", "weight_before", "weight_unnormalised", "weight_after"], ada_rows)
    ens = np.sign(alphas[0] * h[0] + alphas[1] * h[1])
    res.append(("ensemble_accuracy", float((ens == yb).sum()) / 8))

    # ---- 4.4 out-of-bag share
    res.append(("oob_fraction", (1 - 1 / 560) ** 560))

    # ---- 4.5 network sizing (counting only; nothing is trained)
    mlp = [("mlp_hidden1", 784 * 16 + 16), ("mlp_hidden2", 16 * 16 + 16), ("mlp_output", 16 * 10 + 10)]
    s1 = conv_out(28, 5, 2, 1)          # 28 -> conv1 -> 28
    p1 = conv_out(s1, 2, 0, 2)          # pool -> 14
    s2 = conv_out(p1, 5, 0, 1)          # conv2 -> 10
    p2 = conv_out(s2, 2, 0, 2)          # pool -> 5
    flat = p2 * p2 * 16                 # 400
    cnn = [("cnn_conv1", 5 * 5 * 1 * 6 + 6),
           ("cnn_conv2", 5 * 5 * 6 * 16 + 16),
           ("cnn_dense1", flat * 32 + 32),
           ("cnn_dense2", 32 * 10 + 10)]
    print(f"  CNN shapes: conv1 {s1}x{s1}x6, pool1 {p1}x{p1}x6, conv2 {s2}x{s2}x16, "
          f"pool2 {p2}x{p2}x16, flatten {flat}")
    write_csv("part4_architecture.csv", ["layer", "parameters"], [list(r) for r in mlp + cnn])
    res += [("mlp_total", sum(p for _, p in mlp)), ("cnn_total", sum(p for _, p in cnn))]
    res += [("fc_layer_params", 128 * 128 * 3 * 1024 + 1024),
            ("conv_layer_params", 3 * 3 * 3 * 64 + 64)]
    d_in, d_h, d_out = 64, 128, 16
    res.append(("rnn_total", d_h * d_in + d_h * d_h + d_out * d_h + d_h + d_out))

    write_results("part4_results.txt", res)


if __name__ == "__main__":
    main()
