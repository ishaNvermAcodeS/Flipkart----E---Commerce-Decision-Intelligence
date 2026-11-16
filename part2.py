"""Part 2 - Classification and evaluation (churn).

Author: Ishan Verma (iitmd_dsai_2508836)
Writes outputs/part2_results.txt, part2_thresholds.csv, part2_knn_cv.csv
"""
import math

import numpy as np

from common import (best_k, bce, choose_cost_tau, confusion, cost, fit_part2_logistic,
                    get_split, knn_cv, knn_predict, metrics, predict_at, sigmoid,
                    train_logistic, write_csv, write_results)


def main():
    s = get_split()
    Xtr, Xte, ytr, yte = s["Xtr"], s["Xte"], s["ytr"], s["yte"]
    res = []

    # ---- 2.1 split
    res += [("train_rows", len(ytr)), ("train_positives", int(ytr.sum())),
            ("test_rows", len(yte)), ("test_positives", int(yte.sum()))]

    # ---- 2.2 logistic regression from scratch
    w, b, losses = fit_part2_logistic(s)
    for t in (0, 1, 10, 100, 1000):
        res.append((f"loss_epoch_{t}", losses[t]))
    res += [(f"weight_X{j + 1}", w[j]) for j in range(7)]
    res.append(("bias", b))
    p_te = sigmoid(Xte @ w + b)
    p_tr = sigmoid(Xtr @ w + b)
    res.append(("test_bce", bce(yte, p_te)))

    # ---- 2.3 learning-rate effect
    for eta in (0.01, 0.1, 1.0):
        _, _, l = train_logistic(Xtr, ytr, eta, 200, record=(200,))
        res.append((f"loss_epoch_200_eta_{eta}", l[200]))

    # ---- 2.4 odds ratios
    res += [(f"odds_ratio_X{j + 1}", math.exp(w[j])) for j in range(7)]

    # ---- 2.5 two thresholds on the test rows
    for tau in (0.5, 0.3):
        tp, fp, tn, fn = confusion(yte, predict_at(p_te, tau))
        m = metrics(tp, fp, tn, fn)
        res += [(f"tau_{tau}_tp", tp), (f"tau_{tau}_fp", fp),
                (f"tau_{tau}_tn", tn), (f"tau_{tau}_fn", fn),
                (f"tau_{tau}_accuracy", m["accuracy"]), (f"tau_{tau}_precision", m["precision"]),
                (f"tau_{tau}_recall", m["recall"]), (f"tau_{tau}_f1", m["f1"]),
                (f"tau_{tau}_specificity", m["specificity"])]
    blind = confusion(yte, np.zeros(len(yte), dtype=int))
    res.append(("blind_accuracy", metrics(*blind)["accuracy"]))

    # ---- 2.6 threshold sweep (ROC points)
    rows = []
    for k in range(1, 10):
        tau = k / 10
        tp, fp, tn, fn = confusion(yte, predict_at(p_te, tau))
        m = metrics(tp, fp, tn, fn)
        rows.append([tau, tp, fp, tn, fn, m["recall"], m["fpr"]])
    write_csv("part2_thresholds.csv", ["tau", "tp", "fp", "tn", "fn", "tpr", "fpr"], rows)

    # ---- 2.7 KNN with 5-fold CV on training rows
    cv = knn_cv(Xtr, ytr)
    write_csv("part2_knn_cv.csv", ["K", "mean_cv_f1"], [[k, f] for k, f in cv])
    k_best = best_k(cv)
    tp, fp, tn, fn = confusion(yte, knn_predict(Xtr, ytr, Xte, k_best))
    m = metrics(tp, fp, tn, fn)
    res += [("knn_best_k", k_best), ("knn_tp", tp), ("knn_fp", fp), ("knn_tn", tn), ("knn_fn", fn),
            ("knn_accuracy", m["accuracy"]), ("knn_precision", m["precision"]),
            ("knn_recall", m["recall"]), ("knn_f1", m["f1"])]

    # ---- 2.8 alert threshold chosen with money (on training rows)
    tau_c, cost_tr = choose_cost_tau(p_tr, ytr)
    _, fp, _, fn = confusion(yte, predict_at(p_te, tau_c))
    _, fp5, _, fn5 = confusion(yte, predict_at(p_te, 0.5))
    res += [("cost_tau", tau_c), ("cost_train_at_tau", cost_tr),
            ("cost_test_at_tau", cost(fp, fn)),
            ("cost_test_at_0.5", cost(fp5, fn5)),
            ("cost_test_flag_nobody", cost(0, int(yte.sum())))]

    write_results("part2_results.txt", res)


if __name__ == "__main__":
    main()
