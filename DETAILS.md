# Testbeds, verdict rule and localiser settings

## Testbeds

| testbed | features | units | window (rows) | faults |
|---|---|---|---|---|
| Causal graphs (4 linear, 4 nonlinear) | 8 | 160 (8 graphs × 20 seeds) | 1, 10, 50, 250 | offset 1 and 3 sd, stuck, onset, broken dependence, two, half, value-dependent, selection, propagating 1 and 3 sd |
| CATS (simulated control process) | 10 | 20 | 50, 250, consecutive | sensor faults, no change |
| L-Town (simulated water network) | 33 | 20 simulated years | 24 hourly | offset 0.25 and 1 sd, stuck, onset, broken dependence, no change |
| Light tunnel (Causal Chambers) | 9 | 20 | 250 | offsets 1 and 3 sd, stuck, onset, broken dependence, two, many, selection |
| Wind tunnel (Causal Chambers, barometers removed) | 7 | 8 per fault | 10 | sensor faults |

All faults are injected in software (`wasted_repairs/harness.py`); the testbeds are in `wasted_repairs/testbeds/`.

- **Offset** adds ±mag × sd to one feature; **stuck** holds the window's first value; **onset** adds the offset from the
  middle of the window; **broken dependence** replaces the column with its own values from other rows.
- **Two / half** act on 2 features or on half of them. **Selection** keeps rows with a probability that depends on one
  feature and acts on nothing. A **propagating** change is applied to one feature and passed to its descendants.
- The acted-on set S holds the corrupted features; it is empty under selection.

Data: CATS v2 (Zenodo 10.5281/zenodo.8338435), L-Town from the BattLeDIM network (Zenodo 10.5281/zenodo.4017659,
`scripts/simulate_ltown.py`), Causal Chambers `lt_walks_v1` and `wt_walks_v1`.

## Scoring (`wasted_repairs/score.py`)

- A window wastes a repair when it flags a feature outside S. Rates are averaged within units, then over units.
- Verdict from one-sided 97.5 % t bounds over units: above the level if the lower bound exceeds 0.10, within margin if
  the upper bound is at most 0.12, undetermined otherwise.
- A localiser that flags nothing in a setting abstains; it is not counted as within margin.
- Split settings (`paper/count_split.py`): among settings where at least two localisers rank an acted-on feature first in
  more than 0.9 of windows, those where one is within margin and another above the level (23 of 32).

## Localisers (`wasted_repairs/localisers/`)

- **Per-feature KS** and **ARM's rank test with a known split** (the authors' `arm_core.py`): one test per feature, Holm
  at 0.10.
- **Conditional test**: rank test of the residuals of each feature predicted from all the others, Holm at 0.10.
- **Classifier**: permutation importance of a reference-versus-window classifier, thresholded on no-change windows of
  separate units. **FSL-Net**: pretrained, at its authors' threshold of 0.5. Both need windows of 20 rows or more.
- **Block calibration** (time-ordered rows): the window statistic is ranked among disjoint blocks of consecutive
  calibration rows.

**Leave-one-out conformal localiser** (`loo.py`)

1. For every pair j ≠ k, predict feature j from all features except j and k.
2. Each predictor gives a conformal p-value: the rank of the window's residuals among those of held-out calibration rows.
3. Feature j takes the largest p-value over k; Holm at 0.10 gives the flagged set.
4. With several implicated features, each flagged feature is re-tested with predictors that leave out the others.
5. On time-ordered rows the window is ranked among B blocks of calibration rows; with B below d/0.10 the test abstains.
