"""Part 1 - Estimation, inference and a regression baseline (all 800 rows).

Author: Ishan Verma (iitmd_dsai_2508836)
Writes outputs/part1_results.txt
"""
import math

import numpy as np

from common import load_data, sample_mean, sample_sd, write_results


def simple_ols(x, y):
    """Closed-form one-predictor regression: slope = cov / var (both n - 1)."""
    n = len(x)
    mx = x.sum() / n
    my = y.sum() / n
    cov = np.sum((x - mx) * (y - my)) / (n - 1)
    var = np.sum((x - mx) ** 2) / (n - 1)
    slope = cov / var
    intercept = my - slope * mx
    fitted = intercept + slope * x
    tss = float(np.sum((y - my) ** 2))
    ess = float(np.sum((fitted - my) ** 2))
    rss = float(np.sum((y - fitted) ** 2))
    return {
        "slope": slope, "intercept": intercept, "fitted": fitted,
        "tss": tss, "ess": ess, "rss": rss,
        "r2": 1 - rss / tss,
        "rmse": math.sqrt(rss / n),
        "mae": float(np.sum(np.abs(y - fitted)) / n),
    }


def residual_halves(fitted, resid):
    """Sort by fitted value (stable, so ties keep row order); SD of residuals in each half."""
    order = np.argsort(fitted, kind="stable")
    half = len(order) // 2
    lower = sample_sd(resid[order[:half]])
    upper = sample_sd(resid[order[half:]])
    return lower, upper, upper / lower


def adj_r2(r2, n, p):
    return 1 - (1 - r2) * (n - 1) / (n - p - 1)


def main():
    X, y, value = load_data()
    n = len(value)
    x4 = X[:, 3]
    res = []

    # ---- 1.1 estimate with an interval
    mean_value = sample_mean(value)
    sd_value = sample_sd(value)
    se_value = sd_value / math.sqrt(n)
    res += [("n", n), ("mean_value", mean_value), ("sd_value", sd_value),
            ("se_value", se_value),
            ("ci95_low", mean_value - 1.96 * se_value),
            ("ci95_high", mean_value + 1.96 * se_value)]

    # ---- 1.2 two-tailed z-test of H0: mu = 8.0
    z_stat = (mean_value - 8.0) / se_value
    res += [("z_stat", z_stat), ("reject_h0", bool(abs(z_stat) > 1.96))]

    # ---- 1.3 sample size for margin of error E = 0.5
    E = 0.5
    n_required = math.ceil((1.96 * sd_value / E) ** 2)
    res += [("n_required", n_required)]

    # ---- 1.4 closed-form simple regression of value on X4
    m1 = simple_ols(x4, value)
    res += [("reg_slope", m1["slope"]), ("reg_intercept", m1["intercept"]),
            ("tss", m1["tss"]), ("ess", m1["ess"]), ("rss", m1["rss"]),
            ("tss_equals_ess_plus_rss", bool(abs(m1["tss"] - m1["ess"] - m1["rss"]) < 1e-6 * m1["tss"])),
            ("r2", m1["r2"]), ("rmse", m1["rmse"]), ("mae", m1["mae"])]

    # ---- 1.5 same line by gradient descent on standardised X4
    mx4 = sample_mean(x4)
    sdx4 = sample_sd(x4)
    z = (x4 - mx4) / sdx4
    m, c, eta = 0.0, 0.0, 0.1

    def cost_j(m, c):
        e = value - (m * z + c)
        return float(np.sum(e ** 2) / n)

    gd = {0: cost_j(m, c)}
    for t in range(1, 501):
        e = value - (m * z + c)
        dm = -(2.0 / n) * np.sum(z * e)
        dc = -(2.0 / n) * np.sum(e)
        m -= eta * dm
        c -= eta * dc
        if t in (1, 10, 100, 500):
            gd[t] = cost_j(m, c)
    gd_slope = m / sdx4
    gd_intercept = c - m * mx4 / sdx4
    res += [("gd_J0", gd[0]), ("gd_J1", gd[1]), ("gd_J10", gd[10]),
            ("gd_J100", gd[100]), ("gd_J500", gd[500]),
            ("gd_slope", gd_slope), ("gd_intercept", gd_intercept),
            ("gd_slope_gap", abs(gd_slope - m1["slope"]))]

    # ---- 1.6 funnel check, then log-log remedy
    lo, up, ratio = residual_halves(m1["fitted"], value - m1["fitted"])
    res += [("resid_sd_lower_raw", lo), ("resid_sd_upper_raw", up), ("resid_sd_ratio_raw", ratio)]
    lx4 = np.log(x4)
    ly = np.log(value)
    m2 = simple_ols(lx4, ly)
    lo2, up2, ratio2 = residual_halves(m2["fitted"], ly - m2["fitted"])
    res += [("loglog_intercept", m2["intercept"]), ("loglog_slope", m2["slope"]),
            ("loglog_r2", m2["r2"]), ("loglog_rmse", m2["rmse"]),
            ("resid_sd_lower_log", lo2), ("resid_sd_upper_log", up2), ("resid_sd_ratio_log", ratio2)]

    # ---- 1.7 multiple regression by the normal equation
    D = np.column_stack([np.ones(n), lx4, X[:, 0], X[:, 1], X[:, 2], X[:, 4], X[:, 5], X[:, 6]])
    beta = np.linalg.solve(D.T @ D, D.T @ ly)
    fitted = D @ beta
    rss = float(np.sum((ly - fitted) ** 2))
    tss = float(np.sum((ly - sample_mean(ly)) ** 2))
    mr_r2 = 1 - rss / tss
    res += [(f"mr_beta_{j}", beta[j]) for j in range(8)]
    res += [("mr_r2", mr_r2), ("mr_adj_r2", adj_r2(mr_r2, n, 7)),
            ("loglog_adj_r2", adj_r2(m2["r2"], n, 1))]

    write_results("part1_results.txt", res)


if __name__ == "__main__":
    main()
