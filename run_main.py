"""Main experiment: train 101 models per condition and save their test predictions.

Run from the project folder (needs common.py next to it):
    python run_main.py --splits 0 --n-models 3     # smoke test, a few minutes
    python run_main.py --splits 0                  # split 0, full size
    python run_main.py --splits 1 2                # the other two splits

For each split (step numbers refer to the explanation doc):
  Step 2  split the people 80/20 into training and test, one-hot encode
          using the training rows only
  Step 3  train a reference model: no resampling, init seed 0, order seed 0
  Step 5  train n models (default 101) for each condition and set:
              resampling_A, resampling_B   only the bootstrap seed changes
              init_A, init_B               only the initialization seed changes
              order_A, order_B             only the minibatch-order seed changes
              all                          all three change (reference condition)
          A and B are two independent sets, used for the repeatability baseline.
  Step 6  every model predicts every test person; the probabilities are saved
  Checks  rerun two models in the main process (must be identical), and
          logistic regression with shuffled rows (labels must not change)

Output (folder "results", or "results_smoke" when --n-models is not 101):
    split{s}.npz          predictions and test-set information
    split{s}_report.json  sizes, timing, sanity checks, a first look
A split whose .npz already exists is skipped (use --force to redo it).
"""
import argparse
import json
import multiprocessing as mp
import os
import shutil
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

from common import load_acs_income, predict_proba, preprocess, stratified_split, train_mlp

FIXED = 0                          # seed for a source that is held fixed
SET_START = {"A": 1, "B": 10001}   # set A uses seeds 1..101, set B 10001..10101
ALL_START = (20001, 30001, 40001)  # "all" condition: bootstrap, init, order seeds


# ---------------------------------------------------------------- worker side
# Each worker process loads the training and test inputs once, then trains
# one model per job. Windows starts workers fresh ("spawn"), so they read the
# arrays from the cache folder instead of receiving them from the main process.
_W = {}


def _init_worker(cache_dir):
    d = Path(cache_dir)
    _W["X_train"] = np.load(d / "X_train.npy")
    _W["y_train"] = np.load(d / "y_train.npy")
    _W["X_test"] = np.load(d / "X_test.npy")


def _train_one(job):
    name, k, boot, init, order, epochs = job
    model = train_mlp(_W["X_train"], _W["y_train"], init_seed=init, order_seed=order,
                      boot_seed=boot, epochs=epochs)
    return name, k, predict_proba(model, _W["X_test"]).astype(np.float32)


# ---------------------------------------------------------------- job list
def make_jobs(n_models, epochs):
    """One job = one model: (condition, index, boot_seed, init_seed, order_seed, epochs).

    boot_seed None means: train on the training set as it is (no resampling).
    """
    jobs = []
    for s in ("A", "B"):
        for k in range(n_models):
            seed = SET_START[s] + k
            jobs.append((f"resampling_{s}", k, seed, FIXED, FIXED, epochs))  # data changes
            jobs.append((f"init_{s}", k, None, seed, FIXED, epochs))         # weights change
            jobs.append((f"order_{s}", k, None, FIXED, seed, epochs))        # order changes
    b0, i0, o0 = ALL_START
    for k in range(n_models):
        jobs.append(("all", k, b0 + k, i0 + k, o0 + k, epochs))              # all change
    return jobs


def self_consistency(probs):
    """SC for every test person from a (models x people) probability table."""
    B = probs.shape[0]
    b1 = (probs >= 0.5).sum(axis=0)  # models predicting income above $50k
    b0 = B - b1
    return 1.0 - 2.0 * b0 * b1 / (B * (B - 1))


# ---------------------------------------------------------------- one split
def run_split(split, n_models, epochs, workers, out_dir):
    t_start = time.perf_counter()
    print(f"\n=== Split {split} ===", flush=True)

    # Step 2: split and encode (preprocessing fitted on training rows only)
    df, y, group = load_acs_income()
    train_idx, test_idx = stratified_split(y, group, seed=split, test_frac=0.2)
    X, _ = preprocess(df, train_idx)
    X_train, y_train, X_test = X[train_idx], y[train_idx], X[test_idx]
    y_test, g_test = y[test_idx], group[test_idx]
    report = dict(split=split, N=int(len(y)), m=int(len(train_idx)), n=int(len(test_idx)),
                  d=int(X.shape[1]), n_models=n_models, epochs=epochs, workers=workers)
    print(f"N = {report['N']}, training m = {report['m']}, test n = {report['n']}, "
          f"d = {report['d']} columns", flush=True)

    cache = out_dir / f"cache_split{split}"
    cache.mkdir(parents=True, exist_ok=True)
    np.save(cache / "X_train.npy", X_train)
    np.save(cache / "y_train.npy", y_train)
    np.save(cache / "X_test.npy", X_test)

    # Step 3: reference model (all seeds 0, no resampling); used later as the
    # single model for RQ3 and for the margin |p - 0.5| in the RQ1 control
    t = time.perf_counter()
    ref_prob = predict_proba(train_mlp(X_train, y_train, init_seed=FIXED, order_seed=FIXED,
                                       boot_seed=None, epochs=epochs), X_test)
    report["seconds_one_model"] = round(time.perf_counter() - t, 1)
    print(f"Reference model: {report['seconds_one_model']} s, test accuracy "
          f"{((ref_prob >= 0.5) == y_test).mean():.4f}", flush=True)

    # Step 5: all other models, in parallel
    jobs = make_jobs(n_models, epochs)
    conditions = sorted({j[0] for j in jobs})
    probs = {c: np.full((n_models, len(test_idx)), np.nan, dtype=np.float32) for c in conditions}
    print(f"Training {len(jobs)} models with {workers} worker processes...", flush=True)
    t = time.perf_counter()
    ctx = mp.get_context("spawn")
    with ctx.Pool(workers, initializer=_init_worker, initargs=(str(cache),)) as pool:
        for i, (name, k, p) in enumerate(pool.imap_unordered(_train_one, jobs), 1):
            probs[name][k] = p
            if i % 25 == 0 or i == len(jobs):
                el = time.perf_counter() - t
                print(f"   {i:4d}/{len(jobs)} models, {el / 60:5.1f} min elapsed, "
                      f"about {el / i * (len(jobs) - i) / 60:5.1f} min left", flush=True)
    report["minutes_all_models"] = round((time.perf_counter() - t) / 60, 1)
    assert all(np.isfinite(v).all() for v in probs.values()), "some models are missing"

    # Checks 1: rerun two models here; must match the workers' results exactly
    checks = {}
    for name, k, boot, init, order, ep in (jobs[0], jobs[1]):  # resampling_A 0, init_A 0
        again = predict_proba(train_mlp(X_train, y_train, init_seed=init, order_seed=order,
                                        boot_seed=boot, epochs=ep), X_test).astype(np.float32)
        checks[f"rerun {name} model {k}: max prob diff (must be 0.0)"] = \
            float(np.abs(again - probs[name][k]).max())

    # Checks 2: logistic regression has no random init and should not depend on
    # row order; its labels must not change when the rows are shuffled
    lr_p = LogisticRegression(max_iter=2000).fit(X_train, y_train).predict_proba(X_test)[:, 1]
    for seed in (1, 2):
        perm = np.random.default_rng(seed).permutation(len(y_train))
        p2 = LogisticRegression(max_iter=2000).fit(X_train[perm], y_train[perm]) \
            .predict_proba(X_test)[:, 1]
        checks[f"logistic regression, rows shuffled (seed {seed}): label flips (must be 0)"] = \
            int(((p2 >= 0.5) != (lr_p >= 0.5)).sum())
    checks["logistic regression test accuracy"] = float(((lr_p >= 0.5) == y_test).mean())
    report["checks"] = checks
    print("Checks:", flush=True)
    for key, val in checks.items():
        print(f"   {key}: {val}", flush=True)

    # First look (point values only; intervals come in the analysis script)
    first = {}
    for c in conditions:
        acc = ((probs[c] >= 0.5) == y_test[None, :]).mean(axis=1)
        sc = self_consistency(probs[c])
        first[c] = dict(model_acc_mean=round(float(acc.mean()), 4),
                        model_acc_min=round(float(acc.min()), 4),
                        model_acc_max=round(float(acc.max()), 4),
                        share_any_disagreement=round(float((sc < 1).mean()), 4),
                        share_arbitrary_sc_below_075=round(float((sc < 0.75).mean()), 4))
    report["first_look"] = first
    print("First look (one split, no intervals yet):", flush=True)
    print(f"   {'condition':14s} {'accuracy mean [min, max]':28s} "
          f"{'any disagreement':>17s} {'arbitrary':>10s}", flush=True)
    for c, r in first.items():
        print(f"   {c:14s} {r['model_acc_mean']:.4f} [{r['model_acc_min']:.4f}, "
              f"{r['model_acc_max']:.4f}]   {r['share_any_disagreement']:16.1%} "
              f"{r['share_arbitrary_sc_below_075']:10.1%}", flush=True)

    # Save
    np.savez(out_dir / f"split{split}.npz", test_idx=test_idx, y_test=y_test, group_test=g_test,
             ref_prob=ref_prob, lr_prob=lr_p, **{f"prob_{c}": probs[c] for c in conditions})
    report["minutes_total"] = round((time.perf_counter() - t_start) / 60, 1)
    (out_dir / f"split{split}_report.json").write_text(json.dumps(report, indent=2))
    shutil.rmtree(cache, ignore_errors=True)
    print(f"Saved {out_dir / f'split{split}.npz'} ({report['minutes_total']} min total)",
          flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--n-models", type=int, default=101)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--workers", type=int, default=max(1, min(16, (os.cpu_count() or 2) - 2)))
    ap.add_argument("--out", default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out or ("results" if args.n_models == 101 else "results_smoke"))
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output folder: {out_dir.resolve()}")
    for split in args.splits:
        if (out_dir / f"split{split}.npz").exists() and not args.force:
            print(f"Split {split} already done, skipped (use --force to redo it).")
            continue
        run_split(split, args.n_models, args.epochs, args.workers, out_dir)


if __name__ == "__main__":  # required on Windows for the worker processes
    main()
