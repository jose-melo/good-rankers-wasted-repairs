# Testbeds, verdict rule and localiser settings

This note details each testbed and localiser of the paper. Every number below is reproduced by the code of this
repository; the files named are the ones that define or compute it.

## S1. Testbeds

| testbed | d (tested) | units | rows per unit | w | windows | calibration blocks of w rows (minimum ⌈d/0.10⌉) | faults |
|---|---|---|---|---|---|---|---|
| Causal graphs (4 linear, 4 nonlinear) | 8 | 160 (8 graphs × 20 data seeds) | 500 fitting, 500 calibration, 3,000 pool | 1, 10, 50, 250 | 10 per setting and unit | rows drawn independently; no blocks | offset 1 and 3 sd, stuck, onset, broken dependence, two, half, value-dependent (w = 1 only), selection, propagating 1 and 3 sd |
| CATS (simulated control process) | 10, plus 7 command channels as inputs | 20 | shared reference: 30,000 consecutive calibration rows | 50, 250 | consecutive (Figure 2) | B = 600 at w = 50, 120 at w = 250 (minimum 100) | the sensor faults on consecutive rows; no change |
| L-Town (simulated water network) | 33 | 20 re-simulated years | 4 weeks fitting, 330 days calibration, 7-day test week | 24 hourly rows | 7 per setting and unit | B = 330 (minimum 330) | offset 0.25 sd and 1 sd, stuck, onset, broken dependence; no change |
| Light tunnel (Causal Chambers, physical; inputs hidden) | 9 | 20, reshuffles of one recording | 2,000 fitting, 1,000 calibration | 10, 50, 250 | 10 consecutive no-change windows per w, plus 10 drawn no-change windows | 0 (calibration rows are a shuffled sample), so no-change windows are drawn | offsets 1 and 3 sd, stuck, onset, broken dependence, two, many, selection (injected in software) |
| Wind tunnel (Causal Chambers, physical; four barometers removed) | 7 | 8 per injected fault | one recording | 10 (also 50, 250) | consecutive and drawn | B = 247 / 49 / 9 at w = 10 / 50 / 250 (minimum 70) | the injected sensor faults |

Definitions: `wasted_repairs/testbeds/*.py` (one file per testbed) and `wasted_repairs/harness.py` (injected faults).

**Fault definitions** (`harness.inject_units`). An offset adds ±mag × sd to one feature; stuck holds the window's first
value; onset (`half`) adds ±3 sd from the middle of the window; broken dependence (`decoupled`) replaces the column with
values from other pool rows, keeping its marginal; `two` and `many` add ±3 sd to 2 and to max(2, ⌊d/2⌋) features;
value-dependent (`valdep`, w = 1) offsets by 3 sd the strongest correlate of the row's most extreme feature; selection
draws rows with a probability that depends on one feature and acts on nothing. S holds the corrupted features; under
selection S is empty; under a propagating change S holds the shifted feature. On L-Town, onset adds 1 sd and broken
dependence replays the sensor's own values from 168 h earlier.

**Data sources.** Causal graphs are simulated (`testbeds/graphs.py`). CATS v2 (Fleith, Zenodo 10.5281/zenodo.8338435),
thinned to one row every 10 s. L-Town: leak-free EPANET simulations of the BattLeDIM network (Vrachimis et al. 2022,
Zenodo 10.5281/zenodo.4017659); each evaluation year perturbs the published demand patterns (`scripts/simulate_ltown.py`).
Light tunnel and wind tunnel: Causal Chambers datasets `lt_walks_v1` and `wt_walks_v1` (Gamella et al. 2025).

The wind tunnel is analysed without its four barometers, which follow atmospheric pressure: every localiser named a
barometer in most no-change windows of a separate recording of the same tunnel. This view was chosen after seeing those
windows; the reported units come from a different recording.

## S2. Scoring and verdict rule

All of it is in `wasted_repairs/score.py`.

- **Moved features.** Under selection and under a propagating change, a feature whose mean moves by at least 0.2 sd lies
  outside S and counts as a wasted repair when flagged; one that moves by 0.1 to 0.2 sd is left out of the count.
- **Averaging and bounds.** Rates are averaged within units, then over units; a one-sided 97.5 % t bound over units is
  taken on each side.
- **Verdict, read in this order**: undetermined if fewer than two units; above the level if the lower bound exceeds 0.10;
  within margin if the upper bound is at most 0.12; undetermined otherwise.
- **Abstention.** A localiser that flags nothing in a setting, with rate and power both 0, is reported as abstaining,
  never as within margin.
- **Resolution.** Near the level, graph intervals at w = 250 have a median half-width of 0.0132 and at most 0.0228 (19
  cells with rate in [0.05, 0.15]); on consecutive CATS windows at w = 250 the leave-one-out test's half-widths run from
  0.0208 to 0.0768.
- **Dependence between units.** Bounds treat units as independent. On CATS and the two tunnels the units share one
  recording, so their bounds hold given that recording; no correction is applied.
- **Count of split settings** (`paper/count_split.py`). A good ranker has a point estimate of hit-at-1 above 0.9 (a cut we
  chose, without an interval). A setting with at least two good rankers splits when one is within margin and another above
  the level. Graph settings at w = 1, 10, 50, 250 and consecutive CATS settings at w = 50, 250 enter. Result: 23 of 32
  settings split; 24 of 32 if the leave-one-out test is kept under propagating changes; 18 of 28 without it. Of the 23,
  the conditional test is the only good ranker above the level in 13; in 8 of the other 10 every good ranker above the
  level is also above it with no change at the same testbed and window.

## S3. Localiser settings

Code: `wasted_repairs/localisers/`.

- **Per-feature KS** (Rabanser et al. 2019) and **ARM's rank test with a known split** (Peng et al. 2026): one test per
  feature, Holm's procedure at 0.10 (Holm 1979). ARM uses the authors' `arm_core.py` with 9,999 null draws.
- **Conditional test**, our residual form of the null of Kulinski et al. (2020): a one-sided Mann–Whitney test of the
  absolute residuals of a predictor of xⱼ from all other features, window against calibration rows (Bates et al. 2023);
  Holm at 0.10. Predictors are gradient-boosted trees (graphs, CATS, wind tunnel) or ridge regression (light tunnel,
  L-Town).
- **Classifier** (Hinder et al. 2023): a feature's score is the AUC a gradient-boosted reference-versus-window classifier
  loses when that feature is permuted; it is thresholded at the 0.90 quantile of its largest score on no-change windows of
  separate units. **FSL-Net** (Barrabés et al. 2025) runs pretrained, at its authors' threshold (0.5). Neither threshold
  states a level for the units scored here. Both need windows of at least 20 rows, so neither runs on the wind tunnel
  (w = 10).
- **Block-calibrated KS and conditional test** (on time-ordered rows): each window statistic is ranked among the same
  statistic on disjoint blocks of w consecutive calibration rows.

### The leave-one-out conformal localiser (`wasted_repairs/localisers/loo.py`)

1. For every pair j ≠ k, fit a predictor of xⱼ from all features except xⱼ and xₖ, and take its residuals on the
   window and on held-out calibration rows.
2. **Drawn rows.** A two-sided rank test of the signed residuals and a one-sided rank test of their absolute values are
   combined as twice the smaller p-value. Calibration residuals are weighted by a density ratio that a logistic classifier
   estimates on the window from the inputs of that predictor only, so the weights of the predictor that leaves out xₖ never
   read xₖ (weighted conformal p-values under covariate shift, Tibshirani et al. 2019).
3. The first-stage p-value of xⱼ is the largest over k. On drawn rows a screen for row selection tests what remains of the
   full prediction of xⱼ after its regression on xⱼ itself.
4. **Re-test.** Holm's procedure at 0.10 on the first-stage p-values and on marginal rank tests gives an implicated set I.
   Each feature rejected at the first stage is re-tested with predictors that leave out every other feature of I and, in
   turn, one more, keeping the largest p-value.
5. The final p-value of xⱼ is the largest of its first-stage, re-test and (drawn rows) screen p-values; Holm's procedure at
   0.10 on them gives the flagged set R.

**Consecutive rows (block form).** Weights and screen are dropped. Each p-value ranks the window among B disjoint blocks of
w consecutive calibration rows on two statistics, the mean absolute residual and the absolute mean residual: for each of the
B + 1 items, a and b are its exceedance counts on the two statistics, its key is (min(a, b), max(a, b)), and p is the share of
items whose key is lexicographically at most the window's. The smallest p-value is 1/(B + 1), so the test requires
B ≥ ⌈d/0.10⌉ and otherwise sets every p-value to 1, which is an abstention. The block form was settled on separate CATS
units; every reported number comes from seeds, graphs and recordings disjoint from them.

**What is guaranteed and what is measured.** If one feature s is acted on and every other feature keeps its joint law with
the inputs of its leave-out-s predictor, that predictor's p-value is super-uniform up to two approximations, the normal
approximation of the rank tests and the window-estimated weights. The final p-value is a maximum that includes it, even
though I is chosen on the same window, so Holm's procedure keeps the wasted-repair rate at or below 0.10 up to these
approximations. With several acted-on features, under selection and on time-ordered rows the rates are measured; block
p-values are approximately valid when the process is stationary and blocks are weakly dependent (Chernozhukov et al. 2018).
The test does not cover propagating process faults, where the acted-on and the moved features differ.
