"""补充检查：order 条件和 all 条件的模型，用同样的种子重跑，结果必须完全一样。

run_main.py 里只重跑了 resampling_A 0 号和 init_A 0 号，两者的顺序种子都是 0，
没有直接检查"顺序种子在变"的模型。这里补上：
    order_A 0 号：不重抽，初始权重种子 0，顺序种子 1（只有顺序在变）
    all     0 号：重抽种子 20001，初始权重种子 30001，顺序种子 40001（三个都在变）

在项目文件夹里运行（需要 results/split0.npz、split1.npz、split2.npz）：
    python check_rerun_order.py | Tee-Object -FilePath results\\log_check_rerun_order.txt
每次划分大约半分钟。
"""
import argparse
from pathlib import Path

import numpy as np

from common import load_acs_income, predict_proba, preprocess, stratified_split, train_mlp

# 和 run_main.py 里 make_jobs 的种子规则完全一致
CASES = [
    ("order_A", 0, dict(boot_seed=None, init_seed=0, order_seed=1)),
    ("all", 0, dict(boot_seed=20001, init_seed=30001, order_seed=40001)),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--epochs", type=int, default=20)
    args = ap.parse_args()
    out = Path(args.results)

    df, y, group = load_acs_income()
    all_ok = True
    for split in (0, 1, 2):
        path = out / f"split{split}.npz"
        if not path.exists():
            print(f"split {split}: {path} not found, skipped")
            continue
        saved = np.load(path)
        # 和 run_main.py 完全相同的划分和编码
        train_idx, test_idx = stratified_split(y, group, seed=split, test_frac=0.2)
        assert np.array_equal(test_idx, saved["test_idx"]), "test people differ from the saved run"
        X, _ = preprocess(df, train_idx)
        X_train, y_train, X_test = X[train_idx], y[train_idx], X[test_idx]
        for name, k, seeds in CASES:
            again = predict_proba(train_mlp(X_train, y_train, epochs=args.epochs, **seeds),
                                  X_test).astype(np.float32)
            diff = float(np.abs(again - saved[f"prob_{name}"][k]).max())
            ok = diff == 0.0
            all_ok &= ok
            print(f"split {split}: rerun {name} model {k} {seeds}: max prob diff = {diff} "
                  f"(must be 0.0) -> {'PASS' if ok else 'FAIL'}")
    print("\nAll reruns identical:" , "YES" if all_ok else "NO")


if __name__ == "__main__":
    main()
