"""Part 3 - Segmentation and dimensionality reduction (all 800 rows, X1..X5).

Author: Ishan Verma (iitmd_dsai_2508836)
Writes outputs/part3_results.txt, part3_segments.csv, part3_hierarchical.csv
"""
import numpy as np

from common import load_data, sample_mean, sample_sd, write_csv, write_results


# --------------------------------------------------------------------------- K-means
def kmeans(Z, k):
    """Pinned K-means: start rows 0, floor(n/k), 2*floor(n/k), ...; nearest centroid by
    squared distance (ties to lower index, which argmin gives); empty centroids stay put;
    stop at the first pass with no change. Returns (labels, centroids, wcss, passes)."""
    n = len(Z)
    step = n // k
    C = Z[[j * step for j in range(k)]].copy()
    labels = None
    passes = 0
    while True:
        d2 = ((Z[:, None, :] - C[None, :, :]) ** 2).sum(axis=2)
        new = np.argmin(d2, axis=1)
        passes += 1
        if labels is not None and np.array_equal(new, labels):
            break
        labels = new
        for j in range(k):
            members = Z[labels == j]
            if len(members) > 0:
                C[j] = members.sum(axis=0) / len(members)
    wcss = float(((Z - C[labels]) ** 2).sum())
    return labels, C, wcss, passes


# --------------------------------------------------------------------------- hierarchical
NAMES = ["E1", "E2", "E3", "E4", "E5", "E6"]
DIST = np.array([
    [0.0, 7.9, 7.4, 2.9, 2.9, 6.1],
    [7.9, 0.0, 1.4, 5.9, 5.3, 2.0],
    [7.4, 1.4, 0.0, 5.8, 4.6, 1.3],
    [2.9, 5.9, 5.8, 0.0, 2.8, 4.5],
    [2.9, 5.3, 4.6, 2.8, 0.0, 3.4],
    [6.1, 2.0, 1.3, 4.5, 3.4, 0.0],
])


def linkage_distance(a, b, rule):
    pair = [DIST[i, j] for i in a for j in b]
    if rule == "single":
        return min(pair)
    if rule == "complete":
        return max(pair)
    return sum(pair) / len(pair)  # average


def cluster_name(members):
    return "+".join(NAMES[i] for i in sorted(members))


def agglomerate(rule):
    """Return the list of merges (cluster_a, cluster_b, height) and the final clusters at each step."""
    clusters = [[i] for i in range(len(NAMES))]
    merges = []
    history = [list(map(list, clusters))]
    while len(clusters) > 1:
        best = None
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                d = linkage_distance(clusters[i], clusters[j], rule)
                if best is None or d < best[0]:
                    best = (d, i, j)
        d, i, j = best
        a, b = clusters[i], clusters[j]
        if min(b) < min(a):
            a, b = b, a
        merges.append((cluster_name(a), cluster_name(b), d))
        merged = sorted(a + b)
        clusters = [c for idx, c in enumerate(clusters) if idx not in (i, j)] + [merged]
        clusters.sort(key=min)
        history.append(list(map(list, clusters)))
    return merges, history


def clusters_at_cut(merges):
    """Cut in the largest gap between consecutive merge heights; 6 minus merges below the cut."""
    h = [m[2] for m in merges]
    gaps = [h[i + 1] - h[i] for i in range(len(h) - 1)]
    g = int(np.argmax(gaps))          # first largest gap
    merges_below = g + 1
    return len(NAMES) - merges_below, merges_below


# --------------------------------------------------------------------------- main
def main():
    X, y, v = load_data()
    X5 = X[:, :5]
    n = len(X5)
    mu = sample_mean(X5, axis=0)
    sd = sample_sd(X5, axis=0)
    Z = (X5 - mu) / sd
    res = []

    # ---- 3.1 K-means for K = 2..6
    wcss = {}
    fits = {}
    for k in range(2, 7):
        labels, C, w, passes = kmeans(Z, k)
        wcss[k] = w
        fits[k] = labels
        print(f"  K={k}: WCSS={w:.4f}, passes={passes}")
        res.append((f"wcss_k{k}", w))

    # ---- 3.2 elbow
    elbow = None
    for k in (3, 4, 5):
        if wcss[k] - wcss[k + 1] < 0.5 * (wcss[k - 1] - wcss[k]):
            elbow = k
            break
    res.append(("elbow_k", elbow if elbow is not None else "none"))

    # ---- 3.3 profile K = 3 segments (ordered by ascending mean X3)
    labels = fits[3]
    x3_means = [X5[labels == j, 2].mean() for j in range(3)]
    order = np.argsort(x3_means, kind="stable")
    seg_of_row = np.empty(n, dtype=object)
    seg_rows = []
    for letter, j in zip("ABC", order):
        mask = labels == j
        seg_of_row[mask] = letter
        size = int(mask.sum())
        means = [X5[mask, c].sum() / size for c in range(5)]
        seg_rows.append([letter, size] + means + [y[mask].sum() / size, v[mask].sum() / size])
        res.append((f"size_segment_{letter}", size))
    write_csv("part3_segments.csv",
              ["segment", "size", "mean_X1", "mean_X2", "mean_X3", "mean_X4", "mean_X5",
               "outcome_rate", "mean_value"], seg_rows)

    # ---- 3.4 hierarchical clustering on the supplied table
    h_rows = []
    for rule in ("single", "complete", "average"):
        merges, history = agglomerate(rule)
        for no, (a, b, h) in enumerate(merges, start=1):
            h_rows.append([rule, no, a, b, h])
        k_cut, below = clusters_at_cut(merges)
        res.append((f"{rule}_clusters_at_cut", k_cut))
        print(f"  {rule}: clusters at cut = {[cluster_name(c) for c in history[below]]}")
    write_csv("part3_hierarchical.csv", ["linkage", "merge_no", "cluster_a", "cluster_b", "height"], h_rows)

    # ---- 3.5 PCA by the seven steps
    mean_vec = Z.sum(axis=0) / n                     # (1) mean vector: zeros after standardising
    Xc = Z - mean_vec                                # (2) centre
    S = (Xc.T @ Xc) / (n - 1)                        # (3) covariance matrix
    evals, evecs = np.linalg.eigh(S)                 # (4) eigen-decomposition
    idx = np.argsort(evals, kind="stable")[::-1]     # (5) sort largest first
    evals = evals[idx]
    evecs = evecs[:, idx]
    for j in range(evecs.shape[1]):                  # sign convention
        if evecs[np.argmax(np.abs(evecs[:, j])), j] < 0:
            evecs[:, j] = -evecs[:, j]
    W = evecs[:, :2]                                 # (6) keep top two
    P = Xc @ W                                       # (7) project
    total = evals.sum()
    ratios = evals / total
    cum = np.cumsum(ratios)
    print(f"  PCA mean vector max |entry| = {np.abs(mean_vec).max():.2e}")
    res.append(("eigenvalue_sum", total))
    res += [(f"eigenvalue_{j + 1}", evals[j]) for j in range(5)]
    res += [(f"explained_ratio_{j + 1}", ratios[j]) for j in range(5)]
    res += [(f"cumulative_ratio_{j + 1}", cum[j]) for j in range(5)]
    res.append(("components_for_80pct", int(np.argmax(cum >= 0.8)) + 1))
    res += [(f"pc1_loading_X{j + 1}", W[j, 0]) for j in range(5)]
    res += [(f"pc2_loading_X{j + 1}", W[j, 1]) for j in range(5)]
    for r in range(3):
        res += [(f"row{r + 1}_pc1", P[r, 0]), (f"row{r + 1}_pc2", P[r, 1])]
    for letter in "ABC":
        mask = seg_of_row == letter
        res.append((f"mean_pc1_segment_{letter}", P[mask, 0].sum() / mask.sum()))

    write_results("part3_results.txt", res)


if __name__ == "__main__":
    main()
