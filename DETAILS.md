# Testbeds, scoring and localisers

## Testbeds

| testbed | features | units | window (rows) |
|---|---|---|---|
| Causal graphs (4 linear, 4 nonlinear) | 8 | 160 (8 graphs × 20 seeds) | 1, 10, 50, 250 |
| CATS (simulated control process) | 10 | 20 | 50, 250, consecutive |
| L-Town (simulated water network) | 33 | 20 simulated years | 24 hourly |
| Light tunnel (Causal Chambers) | 9 | 20 | 250 |
| Wind tunnel (Causal Chambers, barometers removed) | 7 | 8 per fault | 10 |

Faults are injected in software (`wasted_repairs/harness.py`): an offset of 1 or 3 sd, a stuck value, an onset in the
middle of the window, a broken dependence (the column replaced by its own values from other rows), two or half of the
features offset, row selection (acts on nothing) and, on the graphs, a change that propagates to descendants.

## Scoring (`wasted_repairs/score.py`)

- A window wastes a repair when it flags a feature outside the acted-on set. Rates are averaged over units.
- Verdict from one-sided 97.5 % t bounds over units: above the level if the lower bound exceeds 0.10, within margin if
  the upper bound is at most 0.12, undetermined otherwise.
- `paper/count_split.py` counts the settings where two localisers both rank an acted-on feature first in more than 0.9
  of windows and one is within margin while the other is above the level (23 of 32).

## Localisers (`wasted_repairs/localisers/`)

- **KS** and **ARM's rank test** (known split, the authors' `arm_core.py`): one test per feature, Holm at 0.10.
- **Conditional test**: rank test of the residuals of each feature predicted from all the others, Holm at 0.10.
- **Classifier**: permutation importance, thresholded on no-change windows of separate units. **FSL-Net**: pretrained,
  at its authors' threshold of 0.5. Both need windows of 20 rows or more.
- **Block calibration** (time-ordered rows): the window is ranked among blocks of consecutive calibration rows.
- **Leave-one-out conformal localiser** (`loo.py`): feature j is predicted from all features except j and one other, k;
  each predictor gives a conformal p-value against held-out calibration rows, feature j takes the largest over k, and
  Holm at 0.10 gives the flagged set. Flagged features are re-tested with the other implicated features left out. With
  fewer than d/0.10 calibration blocks the block form abstains.
