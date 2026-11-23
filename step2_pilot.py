"""Step 2: choose the number of training epochs and measure run time.

Run from the project folder (needs common.py next to it):
    python step2_pilot.py

Uses only the TRAINING part of split 0. A validation set is cut out of the
training rows to choose the number of epochs; the test rows are never used.
Self-consistency is NOT computed here: hypotheses must be written down first.

Outputs:
    step2_report.json   summary numbers
    step2_curve.csv     per-epoch training loss, validation accuracy, seconds
    step2_curve.png     the same curve as a figure
"""
import json
import os
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from common import load_acs_income, predict_proba, preprocess, stratified_split, train_mlp

MAX_EPOCHS = 30
SPLIT_SEED = 0
REPORT = {}


def main():
    t0 = time.perf_counter()
    df, y, group = load_acs_income()
    train_idx, test_idx = stratified_split(y, group, seed=SPLIT_SEED, test_frac=0.2)
    # Validation set from the training rows only (10% of them).
    fit_pos, val_pos = stratified_split(y[train_idx], group[train_idx], seed=100, test_frac=0.1)
    fit_idx, val_idx = train_idx[fit_pos], train_idx[val_pos]
    X, names = preprocess(df, fit_idx)
    X_fit, y_fit, X_val, y_val = X[fit_idx], y[fit_idx], X[val_idx], y[val_idx]

    REPORT["data"] = dict(
        rows=int(len(y)), n_features=int(X.shape[1]),
        fit_rows=int(len(fit_idx)), val_rows=int(len(val_idx)), test_rows_untouched=int(len(test_idx)),
        load_and_preprocess_seconds=round(time.perf_counter() - t0, 1),
        cpu_count=os.cpu_count())
    print("Data:", REPORT["data"])

    # Reference model: logistic regression with a deterministic solver.
    t = time.perf_counter()
    lr_model = LogisticRegression(max_iter=2000).fit(X_fit, y_fit)
    REPORT["logistic_regression"] = dict(
        val_accuracy=float((lr_model.predict(X_val) == y_val).mean()),
        seconds=round(time.perf_counter() - t, 1))
    print("Logistic regression:", REPORT["logistic_regression"])

    # Determinism on the real data: same seeds twice for 2 epochs.
    p1 = predict_proba(train_mlp(X_fit, y_fit, 1, 1, epochs=2), X_val)
    p2 = predict_proba(train_mlp(X_fit, y_fit, 1, 1, epochs=2), X_val)
    REPORT["determinism_max_prob_diff"] = float(np.abs(p1 - p2).max())
    print("Same seeds twice, max prob diff:", REPORT["determinism_max_prob_diff"],
          "(must be 0.0)")

    # Learning curve for one MLP.
    rows = []
    clock = {"last": time.perf_counter()}

    def on_epoch(epoch, model, mean_loss):
        now = time.perf_counter()
        acc = float(((predict_proba(model, X_val) >= 0.5) == y_val).mean())
        rows.append(dict(epoch=epoch, train_loss=mean_loss, val_accuracy=acc,
                         epoch_seconds=now - clock["last"]))
        clock["last"] = time.perf_counter()
        print(f"   epoch {epoch:2d}: train loss {mean_loss:.4f}, val accuracy {acc:.4f}, "
              f"{rows[-1]['epoch_seconds']:.1f} s")

    print(f"Training one MLP for {MAX_EPOCHS} epochs...")
    train_mlp(X_fit, y_fit, init_seed=0, order_seed=0, epochs=MAX_EPOCHS, on_epoch=on_epoch)
    curve = pd.DataFrame(rows)
    curve.to_csv("step2_curve.csv", index=False)

    sec_per_epoch = float(curve["epoch_seconds"].median())
    best = curve.loc[curve["val_accuracy"].idxmax()]
    # Total runs: about 700 models per dataset and split, 3 splits, ACSIncome only.
    n_models = 700 * 3
    estimate = {}
    for e in (10, 20, 30):
        hours_one_process = n_models * e * sec_per_epoch / 3600
        estimate[f"{e}_epochs"] = dict(
            hours_one_process=round(hours_one_process, 1),
            hours_if_4_parallel=round(hours_one_process / 4, 1))
    REPORT["learning_curve"] = dict(
        median_seconds_per_epoch=round(sec_per_epoch, 2),
        best_val_accuracy=float(best["val_accuracy"]), best_epoch=int(best["epoch"]),
        val_accuracy_at_10_20_30=[float(curve.loc[curve.epoch == e, "val_accuracy"].iloc[0])
                                  for e in (10, 20, 30) if e <= MAX_EPOCHS])
    REPORT["runtime_estimate_acsincome_2100_models"] = estimate
    print("Learning curve:", REPORT["learning_curve"])
    print("Run time estimate:", estimate)

    fig, ax1 = plt.subplots(figsize=(6, 3.5))
    ax1.plot(curve["epoch"], curve["val_accuracy"], marker="o", ms=3, label="validation accuracy")
    ax1.axhline(REPORT["logistic_regression"]["val_accuracy"], ls="--", c="gray",
                label="logistic regression")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("validation accuracy")
    ax2 = ax1.twinx()
    ax2.plot(curve["epoch"], curve["train_loss"], c="tab:red", alpha=0.6, label="training loss")
    ax2.set_ylabel("training loss")
    fig.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig("step2_curve.png", dpi=150)

    Path("step2_report.json").write_text(json.dumps(REPORT, indent=2))
    print("\nSaved step2_report.json, step2_curve.csv, step2_curve.png")


if __name__ == "__main__":
    main()
