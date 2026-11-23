# Does the Source of Randomness Change Who Is Arbitrary?

Code and results for the ECE750T final report by Xinyu Li (University of Waterloo, Fall 2026).
The experiment trains 2,121 neural-network models on ACSIncome (California, 2018), varying one
source of randomness at a time (resampling, initial weights, data order), and compares who is
flagged as arbitrary, how often women and men are flagged, and the error-rate gaps after abstention.

## Setup

Tested with Python 3.12.7 on Windows 11, CPU only (versions from `step1_report.json`):

```
pip install -r requirements.txt
```

On the first run, `folktables` downloads the ACS data into `data/acs`.
Run every command from this folder; the scripts read and write the folder `results`.
Code comments are in Chinese.

## Reproducing the results

**0. Preliminary checks** (run before the main experiment):

```
python step1_check.py
python step2_pilot.py
```

`step1_check.py` records the library versions, checks on synthetic data that each source of
randomness is controlled by its own seed, and downloads the data (`step1_report.json`).
`step2_pilot.py` is the pilot on the training part of split 0 that chose 20 epochs
(`step2_report.json`, `step2_curve.csv`, `step2_curve.png`); it never uses the test people.

**1. Train the models** (about 17 minutes per split with 16 worker processes):

```
python run_main.py
```

This trains 707 models in each of the three splits and writes `results/split{0,1,2}.npz`
(all predicted probabilities) and `results/split{0,1,2}_report.json`.
For a quick test, `python run_main.py --splits 0 --n-models 3` writes to `results_smoke` instead.
The `.npz` files (about 112 MB each) are not in this repository because of GitHub's file-size
limit; training is deterministic, so this step regenerates them exactly.

**2. Run the analyses** (no retraining):

```
python analysis_rq1.py
python analysis_rq1_threshold.py
python analysis_rq1_margin.py
python analysis_rq2.py
python analysis_rq3.py
python make_figures.py results/rq2_summary.csv results/rq3_summary.csv figures
```

**3. Optional checks.** `python check_rerun_order.py` retrains a few models and compares them
with the saved predictions; `python rerun_all.py` retrains all models (about 50 minutes) and
confirms that every prediction is reproduced exactly.

## Where each result comes from

| Report | File |
|---|---|
| Appendix: number of epochs (pilot) | `step2_report.json`, `step2_curve.csv` |
| Table 1 (conditions and seeds) | `run_main.py` |
| Table 3 (accuracy, share flagged) | `results/split*_report.json` (`first_look`) |
| Table 4 (RQ1) | `results/rq1_summary.csv` |
| Table 5 (SC thresholds 0.6 and 0.9) | `results/rq1_threshold.csv` |
| Table 6 (margin control) | `results/rq1_margin.csv` |
| Figure 1a, Figure 2a (RQ2) | `results/rq2_summary.csv` |
| Figure 1b-c, Figure 2b-c, Table 8 (RQ3) | `results/rq3_summary.csv` |
| Table 7 (noise checks) | `arb noise` rows in `rq2_summary.csv`; `fpr noise`, `fnr noise` rows in `rq3_summary.csv` |
| Reproducibility checks | `results/split*_report.json` (`checks`), `results/rerun_all_report.json` |

The `log_*.txt` files in `results` are the console output of the runs used in the report.
All gaps are women minus men; intervals are 95% intervals from 1,000 bootstrap draws of the test people.
