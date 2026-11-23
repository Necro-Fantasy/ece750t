"""全部重跑检查：用和原来完全相同的种子，重新训练所有模型，逐个和保存的结果比较。

为什么要做：run_main.py 和 check_rerun_order.py 只抽查了 4 个代表性模型。
这里把三次划分的 2,121 个模型（加 3 个参照模型）全部重跑，确认每一个都逐位复现。
不会覆盖 results 里原来的结果，只写一个报告文件。

在项目文件夹里运行（需要 run_main.py、common.py，以及 results/split0.npz 等）：
    python rerun_all.py | Tee-Object -FilePath results\\log_rerun_all.txt
三次划分大约 50 分钟（和当初训练时间一样）。只跑某几次划分：--splits 0

输出：屏幕表格 + results/rerun_all_report.json
"""
import argparse
import json
import multiprocessing as mp
import os
import shutil
import time
from pathlib import Path

import numpy as np

from common import load_acs_income, predict_proba, preprocess, stratified_split, train_mlp
# 直接用 run_main.py 里的任务清单和工作进程函数，保证种子规则和当初完全一样
from run_main import FIXED, _init_worker, _train_one, make_jobs


def rerun_split(split, n_models, epochs, workers, results):
    saved = np.load(results / f"split{split}.npz")
    df, y, group = load_acs_income()
    train_idx, test_idx = stratified_split(y, group, seed=split, test_frac=0.2)
    assert np.array_equal(test_idx, saved["test_idx"]), "test people differ from the saved run"
    X, _ = preprocess(df, train_idx)
    X_train, y_train, X_test = X[train_idx], y[train_idx], X[test_idx]

    cache = results / f"cache_rerun_split{split}"
    cache.mkdir(parents=True, exist_ok=True)
    np.save(cache / "X_train.npy", X_train)
    np.save(cache / "y_train.npy", y_train)
    np.save(cache / "X_test.npy", X_test)

    # 参照模型（种子全为 0，不重抽）
    ref = predict_proba(train_mlp(X_train, y_train, init_seed=FIXED, order_seed=FIXED, boot_seed=None,
                                  epochs=epochs), X_test).astype(np.float32)
    ref_diff = float(np.abs(ref - saved["ref_prob"].astype(np.float32)).max())

    # 其余所有模型：并行重跑，每跑完一个就和保存的那一行比较
    jobs = make_jobs(n_models, epochs)
    diffs = {}  # (条件, 编号) -> 最大概率差
    t = time.perf_counter()
    ctx = mp.get_context("spawn")
    with ctx.Pool(workers, initializer=_init_worker, initargs=(str(cache),)) as pool:
        for i, (name, k, p) in enumerate(pool.imap_unordered(_train_one, jobs), 1):
            diffs[(name, k)] = float(np.abs(p - saved[f"prob_{name}"][k]).max())
            if i % 50 == 0 or i == len(jobs):
                el = time.perf_counter() - t
                print(f"   {i:4d}/{len(jobs)} models rechecked, {el / 60:5.1f} min elapsed, "
                      f"about {el / i * (len(jobs) - i) / 60:5.1f} min left", flush=True)
    shutil.rmtree(cache, ignore_errors=True)

    summary = {"reference model": dict(checked=1, not_identical=int(ref_diff != 0.0), max_diff=ref_diff)}
    for c in sorted({name for name, _ in diffs}):
        ds = [d for (name, _), d in diffs.items() if name == c]
        summary[c] = dict(checked=len(ds), not_identical=sum(d != 0.0 for d in ds), max_diff=max(ds))
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--n-models", type=int, default=101)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--workers", type=int, default=max(1, min(16, (os.cpu_count() or 2) - 2)))
    ap.add_argument("--results", default="results")
    args = ap.parse_args()
    results = Path(args.results)

    report, total, bad = {}, 0, 0
    for split in args.splits:
        print(f"\n=== Split {split}: retraining every model with the same seeds ===", flush=True)
        s = rerun_split(split, args.n_models, args.epochs, args.workers, results)
        report[f"split {split}"] = s
        print(f"   {'condition':16s} {'checked':>8s} {'not identical':>14s} {'max prob diff':>14s}")
        for c, r in s.items():
            print(f"   {c:16s} {r['checked']:8d} {r['not_identical']:14d} {r['max_diff']:14.3g}")
            total += r["checked"]
            bad += r["not_identical"]
    print(f"\nModels rechecked: {total}; not identical: {bad}")
    print("ALL MODELS REPRODUCED EXACTLY" if bad == 0 else "SOME MODELS DIFFER - send me this log")
    (results / "rerun_all_report.json").write_text(json.dumps(report, indent=2))
    print(f"Saved {results / 'rerun_all_report.json'}")


if __name__ == "__main__":  # Windows 上多进程必须有这一行
    main()
