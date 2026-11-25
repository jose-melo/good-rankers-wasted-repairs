# Good Rankers, Wasted Repairs: Scoring Feature-Shift Localisers by the Repairs They Waste

Code and results of the paper. A feature-shift localiser compares a reference sample with a recent window and names
the features in which a shift originates, so that they can be inspected or repaired. Each intact feature it names is a
wasted repair. The repository scores localisers by their wasted-repair rate, the chance that a window names at least one
feature nobody acted on, per type of feature shift, with a verdict at level 0.10 and power beside it, on five testbeds
(random causal graphs, CATS, the L-Town water network, and the light tunnel and wind tunnel of the Causal Chambers). It
also contains the leave-one-out conformal localiser used as a reference.

![Figure 1](assets/figure1.png)

*Figure 1. A fault replaces the recorded column of x3 while the true chain stays intact. (1) Its histogram is unchanged;
its correlation with x2 falls from 0.79 to 0.00. (2) Each localiser computes a p-value per feature, and Holm's procedure
at level 0.10 flags features. KS (per-feature Kolmogorov–Smirnov) flags nothing, the conditional test also flags the
intact x2 and x4 (red: wasted repairs), and ours flags x3 alone. (3) Over all windows on causal graphs (160 units,
w = 250), the conditional test and ours rank x3 first about equally often (0.97, 0.99) and waste a repair in 0.98 and
under 0.01 of windows; arrows: better direction.*

The testbeds, the verdict rule and the settings of every localiser are detailed in the
[supplementary note](SUPPLEMENT.md).

## Installation

Python 3.12.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Two parts run in their own environments, only needed to rerun the experiments:

- FSL-Net (pretrained model of Barrabés et al., CC BY-NC 4.0): `scripts/setup_fslnet.sh` installs it at the commit used
  into `.venv-fslnet` and prints the `FSLNET_PYTHON` variable to export.
- The L-Town simulations (Python 3.14, `pip install -r requirements-ltown.txt`); point `LTOWN_PYTHON` at that
  interpreter.

## Data

```bash
scripts/download_data.sh
```

downloads every dataset from its official source into `data/` (about 2 GB; set `GRWR_DATA` to put it elsewhere) and
checks its checksum:

| testbed | source |
|---|---|
| causal graphs | simulated by the code (`wasted_repairs/testbeds/graphs.py`), nothing to download |
| CATS | Controlled Anomalies Time Series v2, Zenodo [10.5281/zenodo.8338435](https://doi.org/10.5281/zenodo.8338435) (`data.csv`, `metadata.csv`) |
| L-Town | BattLeDIM, Zenodo [10.5281/zenodo.4017659](https://doi.org/10.5281/zenodo.4017659) (`L-TOWN_Real.inp`, `dataset_configuration.yaml`); the years are then simulated by `scripts/simulate_ltown.py` |
| light tunnel, wind tunnel | Causal Chambers datasets `lt_walks_v1` and `wt_walks_v1` ([causal-chamber](https://github.com/juangamella/causal-chamber)), plus the wind-tunnel regime targets and the published graph |

The script also fetches the authors' `arm_core.py` of ARM (arXiv:2608.01691, ancillary file, CC BY 4.0), which the ARM
localiser loads unmodified.

## Reproducing the results

The scored results of every setting the paper reports are in `results/` (one CSV per testbed; `results/toy.json` is the
toy window of Figure 1), so the figures and numbers regenerate in seconds without rerunning the experiments. A row is
one setting (`type`, `mag`, window length `w`) of one `localiser`: `rate` is the wasted-repair rate with its one-sided
97.5 % bounds `rate_lo` and `rate_hi` over `n_units` units, `verdict` is within / above (the level) / undetermined, and
`power` and `hit1` (hit-at-1) stand beside it.

```bash
cd paper
python figure1.py        # Figure 1   -> paper/out/figure1.pdf
python figure2.py        # Figure 2   -> paper/out/figure2.pdf
python count_split.py    # 23 of 32 settings split; 13 by the conditional null; 8 of the other 10 with no change
python text_numbers.py   # every number quoted in the text
python table_settings.py # per-setting values: wasted-repair rate and power, no change / offset / broken dependence
```

`paper/make_all.sh` runs them all (`text_numbers.py` includes the count). `text_numbers.py` prints, among others, .97 and .99 (x3 ranked first by the
conditional test and ours), .98 and <.01 (their wasted-repair rates), 38 of 42 graph settings within margin for ours,
and .25 under a 3 sd propagating fault (numbers print as in the figures: two decimals without the leading zero).

To recompute `results/` from the data:

```bash
export FSLNET_PYTHON=$PWD/.venv-fslnet/bin/python LTOWN_PYTHON=/path/to/ltown-env/bin/python
scripts/run_experiments.sh 8      # 8 parallel jobs
```

This prepares CATS, simulates the 21 L-Town years, runs every localiser on every unit (`jobs/eval_*.txt`) and on the
units that set the classifier's threshold (`jobs/pool_*.txt`), scores them (`wasted_repairs/score.py`) and rewrites
`results/`. It takes about 30 CPU-hours on single-threaded jobs (causal graphs about 13, CATS about 10, the other testbeds
1 to 2 each). One unit can be run alone, for example:

```bash
python -m wasted_repairs.run --testbed graphs --dataset lin5 --seed 4000 --localisers KS,COND,LOO
```

Localisers: `KS` (per-feature Kolmogorov–Smirnov), `ARM` (ARM's rank test at the known split), `COND` (conditional
test), `CLF` (classifier), `FSL` (FSL-Net), `KS_B` and `COND_B` (calibrated on blocks of consecutive rows), `LOO` (ours).

## Citation

```bibtex
@inproceedings{goodrankers2027,
  title     = {Good Rankers, Wasted Repairs: Scoring Feature-Shift Localisers by the Repairs They Waste},
  author    = {De Melo Costa, J. L. and others},
  booktitle = {European Symposium on Artificial Neural Networks, Computational Intelligence and Machine Learning (ESANN)},
  year      = {2027},
  note      = {Under review}
}
```

## Licence

MIT (see `LICENSE`). The downloaded data and third-party code keep their own licences: CATS, L-Town and the Causal
Chambers data CC BY 4.0, the Causal Chambers scripts MIT, ARM's `arm_core.py` CC BY 4.0, FSL-Net CC BY-NC 4.0.
