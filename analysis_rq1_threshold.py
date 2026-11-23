"""RQ1 阈值敏感性：把 arbitrary 的阈值从 0.75 换成 0.6 和 0.9，结论还一样吗？

实验计划里写了 "I also report 0.6 and 0.9"。这里用和 analysis_rq1.py 完全相同的计算
（直接调用它的函数），只把阈值换掉：
    SC < 0.6  严：101 票里少数派 >= 28 票才算 arbitrary
    SC < 0.75 原来的：少数派 >= 15 票
    SC < 0.9  松：少数派 >= 6 票
判断规则和原来一样：Delta 的 95% 区间 > 0，且三次划分都成立。
另外照样报告包含比例（探索性）。秩相关不依赖阈值，不用重算。

在项目文件夹里运行（需要 analysis_rq1.py 和 results/split0.npz 等）：
    python analysis_rq1_threshold.py | Tee-Object -FilePath results\\log_rq1_threshold.txt
输出：屏幕表格 + results/rq1_threshold.csv（0.75 也一起重算，方便对照）
"""
import csv
from pathlib import Path

import numpy as np

import analysis_rq1 as A  # 用同一套计算，只改 A.THRESHOLD

THRESHOLDS = [0.6, 0.75, 0.9]


def minority_votes(t, B=101):
    """SC < t 等价于少数派至少多少票。"""
    for b in range(B // 2 + 1):
        if 1 - 2 * b * (B - b) / (B * (B - 1)) < t:
            return b
    return None


def main():
    splits = [s for s in (0, 1, 2) if (A.RESULTS / f"split{s}.npz").exists()]
    print(f"Splits found: {splits}; bootstrap draws: {A.N_BOOT}")
    all_rows = []
    for t in THRESHOLDS:
        A.THRESHOLD = t
        rng = np.random.default_rng(A.BOOT_SEED)  # 每个阈值用同一个种子，结果可复现
        print(f"\n######## Threshold SC < {t} (minority >= {minority_votes(t)} of 101 votes) ########")
        for s in splits:
            rows, sizes, n = A.analyse_split(s, rng)
            get = {r["measure"]: r for r in rows}
            f = lambda r: f"{r['point']:.3f} [{r['lo']:.3f}, {r['hi']:.3f}]"
            print(f"split {s}: arbitrary share (set A): " + ", ".join(
                f"{c} {sizes[f'{c}_A'] / n:.2%}" for c in A.CONDS))
            print("   baseline: " + ", ".join(f"{c} {get[f'baseline {c}']['point']:.3f}" for c in A.CONDS))
            for p, q in A.PAIRS:
                d = get[f"delta {p}-{q}"]
                print(f"   {p + '-' + q:18s} cross {get[f'cross {p}-{q}']['point']:.3f}   "
                      f"Delta {f(d)}   rule met: {'yes' if d['lo'] > 0 else 'no'}")
            for small, big in A.CONTAIN:
                print(f"   containment {small:5s} in {big:11s} {f(get[f'containment {small} in {big}'])}   "
                      f"(baseline {get[f'containment baseline {small}']['point']:.3f})")
            for r in rows:
                if not r["measure"].startswith("spearman"):
                    all_rows.append(dict(threshold=t, **r))

    # 汇总：每个阈值下，三对条件的结论
    print("\n=== Summary: does the RQ1 decision hold at every threshold? ===")
    print(f"   {'threshold':>9s} {'pair':18s} {'splits with rule met':>21s} {'Delta range':>14s} "
          f"{'containment (small in big)':>27s}")
    contain_of = {("resampling", "init"): "containment init in resampling",
                  ("resampling", "order"): "containment order in resampling",
                  ("init", "order"): "containment init in order"}
    for t in THRESHOLDS:
        for p, q in A.PAIRS:
            ds = [r for r in all_rows if r["threshold"] == t and r["measure"] == f"delta {p}-{q}"]
            cs = [r["point"] for r in all_rows if r["threshold"] == t and r["measure"] == contain_of[(p, q)]]
            met = sum(r["lo"] > 0 for r in ds)
            pts = [r["point"] for r in ds]
            print(f"   {t:9.2f} {p + '-' + q:18s} {f'{met} / {len(ds)}':>21s} "
                  f"{min(pts):6.3f}-{max(pts):.3f} {min(cs):17.3f}-{max(cs):.3f}")

    out = A.RESULTS / "rq1_threshold.csv"
    with open(out, "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=["threshold", "split", "measure", "point", "lo", "hi"])
        wr.writeheader()
        for r in all_rows:
            wr.writerow({k: (f"{v:.6f}" if isinstance(v, float) and k != "threshold" else v) for k, v in r.items()})
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
