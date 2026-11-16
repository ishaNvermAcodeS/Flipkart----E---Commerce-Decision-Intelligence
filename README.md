Domain: E-commerce

# Flipkart E-commerce Decision Intelligence — Capstone

**Student:** Ishan Verma · **Roll number:** iitmd_dsai_2508836
**Program:** Minor in Artificial Intelligence & Data Science, IIT Mandi

> All data is synthetic, produced by `generate_data.py`. Nothing here reflects Flipkart's real
> records, products, policies or performance; the brand is only the setting for the decisions.

Every number below is computed by the scripts in this repository (plain Python + NumPy; no
ready-made model, metric, clustering or PCA routine) and is quoted from the committed files in
`outputs/`. Money is in Indian Rupees (INR); the value column is in hundreds of INR.

### Repository layout

```
README.md            this file (all written answers)
requirements.txt     numpy==2.2.6
generate_data.py     provided generator, committed verbatim
data/records.csv     generator output (800 rows, 10 columns)
common.py            shared helpers: data loading, Task 2.1 split, logistic regression,
                     KNN, metrics, cost rule, output formatting (importing it runs nothing)
part1.py             Part 1 - estimation, z-test, regression (closed form, GD, normal equation)
part2.py             Part 2 - logistic regression, thresholds, KNN with CV, cost threshold
part3.py             Part 3 - K-means, segment profiles, hierarchical clustering, PCA
part4.py             Part 4 - splits, stump and voting, AdaBoost, OOB, network sizing
outputs/             every results file and CSV named in the brief
```

## How to run

From a clean clone, at the repository root, with Python 3.10 or newer:

```bash
pip install -r requirements.txt
python generate_data.py
python part1.py
python part2.py
python part3.py
python part4.py
```

`python generate_data.py` prints `800 10 170 c6f249ba693f1ebeed318d6d5e18de23`. The four Part
scripts then regenerate exactly the committed files in `outputs/`. Part 4 imports the Part 2
models from `common.py`, so the scripts can be run in the order above or individually after
the data exists.

## Part 1 findings

**R1.1 — Planning claim.** H0 (μ = 8.0, i.e. an average order value of INR 800) is **rejected**:
`z_stat` = 2.8930, which is above the two-tailed critical value 1.96 at α = 0.05
(`mean_value` = 8.9382, `se_value` = 0.3243, 95% interval 8.3026 to 9.5738). A Type I error here
would be telling the category team that the average order value is no longer INR 800 when in
truth it still is, and re-planning budgets for nothing. The gap is `mean_value` − 8.0 = 0.9382
column units, about INR 94 per order or roughly 12% above plan. I think that matters in practice:
a 12% under-statement of basket size changes revenue and inventory plans for a category, though
the interval says the true gap could be anywhere from about INR 30 to INR 157, so the plan
should be updated with that range, not a single point.

**R1.2 — Sample size.** To estimate the mean within ±0.5 units (INR 50) at 95% confidence,
`n_required` = 1293 rows are needed, against the 800 available. The current sample therefore
only gives a margin of about ±0.64 units (1.96 × 0.3243); the planning team should either
accept a margin of roughly ±INR 64 or collect about 493 more records before relying on an
INR 50 precision. The large `sd_value` (9.1723) — order values are very spread out — is what
drives the requirement up.

**R1.3 — Gradient descent vs closed form.** `gd_slope_gap` = 0.0000 (below 0.0001); the
gradient-descent line (`gd_slope` = 0.6701, `gd_intercept` = −3.6598) is the closed-form line.
The mean-squared-error cost of a straight line is a convex bowl with a single minimum, and with
X4 standardised a step of η = 0.1 shrinks the error each update, so 500 updates walk all the
way to that unique minimum (`gd_J100` = `gd_J500` = 40.3698 = `rss` / n).

**R1.4 — Funnel check.** On the raw scale `resid_sd_ratio_raw` = 3.7525 (residual SD 2.3155 in
the lower half of fitted values vs 8.6891 in the upper half): the raw model shows a clear
**funnel** — errors grow with the predicted order value. After the log remedy
(ln(value) on ln(X4)) `resid_sd_ratio_log` = 1.0396 (0.5165 vs 0.5369), so the log-log model
**removes the funnel**: residual spread is essentially constant across the range.

**R1.5 — Multiple regression.** Coefficients with |β| > 0.05 (excluding the constant):
**ln(X4)** (`mr_beta_1` = 1.1576), **X2 complaints** (`mr_beta_3` = 0.0940), **X3 discount share**
(`mr_beta_4` = 0.1067) and **X6 cash-on-delivery flag** (`mr_beta_6` = 0.1518). X1, X5 and X7
fall below the mechanical 0.05 cut (−0.0074, 0.0028, 0.0172); as the brief notes, the cut is
not an importance ranking because the units differ. `mr_adj_r2` = 0.7403 against
`loglog_adj_r2` = 0.6561: adjusted R², which already penalises the six extra predictors, rises
by about 0.084, so the extra predictors earn their place.

## Part 2 findings

Split (Task 2.1): 560 training rows with 119 churners, 240 test rows with 51 churners.

**R2.1 — Learning rate.** After 200 updates the training loss is lowest for η = 1.0 (0.3703),
then η = 0.1 (0.3776), then η = 0.01 (0.5091). A learning rate that is too small moves so
slowly that after 200 steps it is still far from the minimum, while the large rate is
fine here — the cross-entropy loss is convex and the features are standardised, so η = 1.0
converges fastest without overshooting (a much larger rate could diverge).

**R2.2 — Strongest predictors.** The three largest absolute weights are **X3 discount order
share** (0.9013), **X1 tenure** (−0.5554) and **X2 complaints in 90 days** (0.5505). The largest
odds ratio is `odds_ratio_X3` = 2.4629: a customer whose share of discounted orders is one
training SD higher has about 2.5 times the odds of not buying in the next 90 days, other
things equal — deal-driven shoppers are the most likely to drift away.

**R2.3 — Accuracy is weak here.** `blind_accuracy` = 0.7875 (predict "no churn" for everyone)
against `tau_0.5_accuracy` = 0.8042: the model is only 1.7 points better than a rule that
catches no churner at all, because about 79% of test customers stay anyway. Accuracy rewards
the easy majority and hides the misses that cost money. Lowering τ from 0.5 to 0.3 raised
recall from 0.4706 to 0.7451 (24 → 38 of 51 churners caught) and lowered precision from
0.5455 to 0.4634 (false alarms 20 → 44); F1 rose from 0.5053 to 0.5714.

**R2.4 — Logistic vs KNN.** Logistic regression at τ = 0.3: `f1` = 0.5714, `recall` = 0.7451.
KNN with the cross-validated `knn_best_k` = 5: `knn_f1` = 0.3696, `knn_recall` = 0.3333. The
logistic model catches more than twice as many churners and is better on both scores.

**R2.5 — Deployment choice.** I would deploy the **logistic regression at τ = 0.15**
(`cost_tau`), chosen by the INR 50 / INR 400 cost rule. On the test rows it costs
`cost_test_at_tau` = INR 6,050, against `cost_test_at_0.5` = INR 11,800 at the default
threshold and `cost_test_flag_nobody` = INR 20,400 for doing nothing. The threshold was
chosen on the training rows so that the test rows stay an untouched, honest estimate of the
cost the team will see in production; choosing τ on the test rows would make that number
optimistic.

## Part 3 findings

**R3.1 — Elbow.** WCSS for K = 2…6: 2262.6753, 1602.8471, 1494.7732, 1323.7157, 1278.1691.
The four successive drops are:

| Drop | Value |
|---|---|
| WCSS(2) − WCSS(3) | 659.8282 |
| WCSS(3) − WCSS(4) | 108.0739 |
| WCSS(4) − WCSS(5) | 171.0575 |
| WCSS(5) − WCSS(6) | 45.5466 |

The elbow supports **K = 3** (`elbow_k` = 3): going from 3 to 4 clusters gains only 108.07,
far less than half of the 659.83 gained going from 2 to 3. The drops do not shrink steadily
(the 4 → 5 drop is larger than the 3 → 4 drop); the pinned starting rows give one reproducible
solution, not necessarily the global optimum for each K, so this zig-zag is a property of the
starting points rather than evidence for K = 5.

**R3.2 — Segments** (from `part3_segments.csv`, means in original units):

| Segment | Size | Tenure (mo) | Complaints | Discount share | Avg order (INR 100s) | Loyalty | Churn rate | Order value | Name | Action |
|---|---|---|---|---|---|---|---|---|---|---|
| A | 322 | 47.5 | 1.95 | 0.225 | 12.53 | 62.2 | 0.0186 | 3.60 | **Loyal long-tenure regulars** | Keep them with membership perks and early access, not discounts; cross-sell to grow their small baskets. |
| B | 255 | 28.6 | 6.07 | 0.453 | 28.72 | 55.0 | 0.2353 | 17.12 | **High-spend, complaint-prone shoppers** | Route their complaints to priority service-recovery; they bring the largest orders, so fixing friction protects the most revenue. |
| C | 223 | 14.4 | 4.23 | 0.712 | 16.51 | 38.5 | 0.4664 | 7.29 | **Bargain-focused newcomers** | Send the retention coupon here first and move them into the loyalty programme before the discount habit lapses. |

The segment with the highest `outcome_rate` is **C, the bargain-focused newcomers, at 0.4664**
(almost half churn), against 0.2353 for B and 0.0186 for A. Segment A is loyal but not
"premium" — its order values are the smallest — so I named it from the numbers.

**R3.3 — Hierarchical clustering.** At the cut in the largest gap, **single linkage** gives
**4 clusters: {E2, E3, E6}, {E1}, {E4}, {E5}** (heights 1.3, 1.4, 2.8, 2.9, 3.4; largest gap
1.4 → 2.8 after merge 2). **Complete linkage** gives **2 clusters: {E1, E4, E5} and
{E2, E3, E6}** (heights 1.3, 2.0, 2.8, 2.9, 7.9; largest gap 2.9 → 7.9). The two rules disagree
because single linkage measures the *closest* pair between clusters, so its final merge happens
at only 3.4 (E5–E6 are close) and the biggest jump in its dendrogram falls early, between the
tight E2/E3/E6 group and the looser E1/E4/E5 merges. Complete linkage measures the *farthest*
pair, so joining the two groups costs 7.9 (E1–E2) and that one dramatic jump dominates the cut,
leaving the two natural groups. (Average linkage agrees with complete: 2 clusters.)

**R3.4 — PCA.** After standardising, the mean vector is all zeros (largest absolute entry about
2e-15, i.e. rounding error), so centring changes nothing, and the eigenvalues sum to 5.0000,
the number of standardised variables. PC1 explains 0.5227 of the variance. Its loadings are
**+0.5317 tenure (X1)** and **+0.4418 loyalty (X5)** against **−0.5187 discount share (X3)**,
**−0.4100 complaints (X2)** and −0.2916 average order value (X4): PC1 is a
"settled loyalty vs bargain-and-friction" axis. **3 components** are needed to keep at least
80% of the variance (cumulative 0.7593 after two, 0.8616 after three). The segment means on
PC1 are A = 1.7773, B = −0.7831, C = −1.6709: the three K-means segments line up along this
single axis, from loyal regulars at the positive end to bargain-focused newcomers at the
negative end, which is also the order of their churn rates.

## Part 4 findings

**R4.1 — Best split.** S4 (tenure ≤ 24 months) is best on both criteria
(info gain 0.1252, Gini gain 0.0606, vs 0.1046 / 0.0494 for S3). It separates 188 customers
of whom 86 churn (45.7%) from 372 of whom only 33 churn (8.9%), so its children are far purer
than the root (entropy 0.7462). A tree grown until every leaf is pure ends up carving out
leaves of one or two customers that fit the noise of this particular training set and do not
repeat on new customers; a maximum depth (or a minimum number of records per leaf) stops it.

**R4.2 — What to ship.** I would ship the **logistic regression** (τ = 0.3 here: `f1` = 0.5714,
`recall` = 0.7451). Combining models did **not** help: `hard_vote` reached f1 = 0.5344 and
recall = 0.6863, and `soft_vote` matched the logistic recall (0.7451) but with five more false
alarms, giving f1 = 0.5507. The stump has the highest recall (0.7647) but a lower f1 (0.5532),
and KNN (f1 = 0.3696, recall = 0.3333) drags the hard vote down.

**R4.3 — AdaBoost.** After round 1 the two points h1 misclassified (points 3 and 7) each carry
weight 0.2500, a **total of 0.5000** — boosting moves half the weight onto the mistakes. Round 2
trusts h2 more (`round2_alpha` = 0.8047 vs `round1_alpha` = 0.5493) because, measured on those
re-weighted points, h2's error is only 0.1667 compared with h1's 0.2500, and α grows as the
weighted error falls. The two-rule ensemble classifies 0.7500 of the eight points correctly.
(The out-of-bag share for a bootstrap of 560 rows is `oob_fraction` = 0.3676.)

**R4.4 — Network sizing.**

Model A, option 2 (CNN on a 28 × 28 × 1 input):

| Layer | Output shape (H × W × D) | Parameters |
|---|---|---|
| Input | 28 × 28 × 1 | 0 |
| Conv 5×5, 6 filters, stride 1, pad 2 (+ ReLU) | 28 × 28 × 6 | 156 |
| Max-pool 2×2, stride 2 | 14 × 14 × 6 | 0 |
| Conv 5×5, 16 filters, stride 1, pad 0 (+ ReLU) | 10 × 10 × 16 | 2,416 |
| Max-pool 2×2, stride 2 | 5 × 5 × 16 | 0 |
| Flatten | 1 × 1 × 400 | 0 |
| Dense 32 (+ ReLU) | 1 × 1 × 32 | 12,832 |
| Dense 10 (output) | 1 × 1 × 10 | 330 |
| **Total** | | **15,734** |

For comparison, option 1 (MLP 784 → 16 → 16 → 10) has `mlp_total` = 13,002 parameters. The
scaling example shows why convolution matters for bigger pictures: a dense layer on a
128 × 128 × 3 input with 1,024 neurons needs `fc_layer_params` = 50,332,672 parameters, while 64
filters of 3 × 3 need only `conv_layer_params` = 1,792, because a filter's weights are shared
across every position.

Model B (RNN): W_xh is 128 × 64, W_hh is 128 × 128 and W_hy is 16 × 128 (with b_h of length 128
and b_y of length 16), giving `rnn_total` = 26,768. That count does not grow with longer
reviews because the same W_xh, W_hh and b_h are reused at every time step; a longer sequence
only means more steps through the same weights.

For the 10-class model I would use **ReLU** in the hidden layers and a **softmax** output with
**categorical cross-entropy** loss. In a very deep plain network the gradient reaching the
early layers is a product of many per-layer factors, so it can shrink towards zero (vanish) and
those layers stop learning. A skip connection adds the block's input to its output
(y = F(x) + x), which gives the gradient a direct identity path back to the early layers, so
they keep receiving a useful signal even when F's own gradient is small.

## Model-choice memo

**To:** Head of Retention · **From:** Ishan Verma, Customer Analytics · **Re:** Churn-alert model for production

**Recommendation.** Put the **logistic-regression model** into production. Of the five
structured-data models I tested on 240 held-out customers, it had the best balance of catching
churners and avoiding wasted coupons: at a 0.3 threshold it caught 38 of 51 churners
(recall 0.7451) with an F1 of 0.5714. KNN caught only 17 (recall 0.3333), and neither hard
voting (F1 0.5344) nor soft voting (F1 0.5507) beat logistic regression on its own, so the
added complexity of an ensemble is not justified.

**Threshold and cost.** Using our cost rule (INR 50 per wasted coupon, INR 400 per customer lost
without warning), the lowest-cost threshold on the training customers is **0.15**. At that
threshold the logistic model's cost on the held-out customers is **INR 6,050**, against
INR 11,800 at the default 0.5 threshold and INR 20,400 if we flag nobody. A low threshold is
right because a missed churner costs eight times a wasted coupon.

**Risk.** The evidence rests on one split with only 51 churners in the test set, on synthetic
data, and on cost figures that are approximate; a few customers either way would move these
numbers. I recommend a monitored pilot comparing coupon and no-coupon groups before full
roll-out, with the threshold re-checked quarterly.

**Image and text models.** Neither Model A (product-picture sorter) nor Model B (review-topic
model) is worth building before this churn model is in production, because they do not address
the retention losses that the structured model already reduces.

## Limitations

1. **Synthetic data.** All 800 records come from a generator with three hidden groups and a
   built-in logistic churn rule; real Flipkart data would be messier (missing values, seasonality,
   drift), and these results say nothing about real customers.
2. **One split and few positives.** Every Part 2 and Part 4 metric comes from one pinned 560/240
   split with only 51 churners in the test set; one or two customers change recall by about
   0.02–0.04, and the chosen K (5) and threshold (0.15) are valid for this split only.
3. **Approximate, fixed costs.** The INR 50 / INR 400 costs are rough averages; the real cost of a
   coupon and the margin lost per churner vary by customer and segment, which would move the
   optimal threshold. The threshold was also searched on a coarse grid of steps of 0.05.
4. **Simple models.** The logistic model is linear in standardised features with no
   interactions, and its probabilities were not checked for calibration; the order-value
   regressions show heteroscedasticity on the raw scale.
5. **K-means depends on its pinned starts.** The segments come from one pinned initialisation that
   is not guaranteed to be the global optimum (the WCSS drops are not monotone), and K = 3 was
   fixed by the brief.
6. **No causal claim.** The models predict who is likely to churn; they do not show that a coupon
   will change that behaviour. Only a controlled test can measure the coupon's effect.
