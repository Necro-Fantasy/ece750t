"""RQ1 分析：三个条件找到的 arbitrary 人，是不是同一批人？

在项目文件夹里运行（需要 results/split0.npz、split1.npz、split2.npz）：
    python analysis_rq1.py

对每次划分、每一对条件（resampling-init、resampling-order、init-order）算：
  1. arbitrary 集合：101 个模型里 SC < 0.75 的测试者
  2. baseline（重复性基线）：同一个条件 A、B 两组集合的 Jaccard 重叠度
  3. 交叉重叠度：两个条件之间的 Jaccard（A/B 两两配对共 4 种，取平均）
  4. Delta = min(两个条件各自的 baseline) - 交叉重叠度
  5. 95% 区间：把测试者有放回地重抽 1,000 次，每次重算，取第 2.5 和第 97.5 百分位
     判断规则（实验计划里事先定好）：Delta 的区间整体 > 0，且三次划分都成立
     → 两个条件找到的人确实不同
另外两个量只报告，不参与判断：
  6. SC 的秩相关（Spearman）：同一条件 A 对 B，和跨条件对比
  7. 包含比例（探索性，11/2 看到结果后决定加的）：
     小集合里的人，有多大比例也在大集合里

输出：屏幕上的表格 + results/rq1_summary.csv
屏幕输出用英文，避免 Windows 管道（Tee-Object）遇到中文时出编码错误。
"""
import csv
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

RESULTS = Path("results")
CONDS = ["resampling", "init", "order"]
PAIRS = list(combinations(CONDS, 2))  # (resampling, init), (resampling, order), (init, order)
# 包含比例的方向：小集合 → 大集合（三次划分里 init < order < resampling）
CONTAIN = [("init", "resampling"), ("order", "resampling"), ("init", "order")]
THRESHOLD = 0.75
N_BOOT = 1000
BOOT_SEED = 2026  # 重抽测试者用的种子，固定下来保证结果可复现


def self_consistency(probs):
    """从 (模型数 × 测试人数) 的概率表，算出每个测试者的 SC。"""
    B = probs.shape[0]
    b1 = (probs >= 0.5).sum(axis=0)
    b0 = B - b1
    return 1.0 - 2.0 * b0 * b1 / (B * (B - 1))


def jaccard(x, y, w):
    """Jaccard = 两边都有的人数 / 至少一边有的人数。
    x、y 是 0/1 向量（1 = 这个人在集合里）；w 是每个人被算几次：
    不重抽时全是 1，重抽时就是这个人在这次重抽里被抽到的次数。"""
    union = w @ np.maximum(x, y)
    return np.nan if union == 0 else (w @ (x * y)) / union


def containment(small, big, w):
    """small 集合里的人，有多大比例也在 big 集合里。"""
    size = w @ small
    return np.nan if size == 0 else (w @ (small * big)) / size


def measures(arb, w):
    """在给定权重 w 下，算出这次划分要报告的所有重叠类的量。"""
    out = {}
    base = {c: jaccard(arb[f"{c}_A"], arb[f"{c}_B"], w) for c in CONDS}
    for c in CONDS:
        out[f"baseline {c}"] = base[c]
    for s, t in PAIRS:
        cross = np.mean([jaccard(arb[f"{s}_{a}"], arb[f"{t}_{b}"], w) for a in "AB" for b in "AB"])
        out[f"cross {s}-{t}"] = cross
        out[f"delta {s}-{t}"] = min(base[s], base[t]) - cross
    for small, big in CONTAIN:
        # 同一条件 A、B 互相包含的比例 = 这个量的"重复性基线"
        out[f"containment baseline {small}"] = np.mean(
            [containment(arb[f"{small}_A"], arb[f"{small}_B"], w),
             containment(arb[f"{small}_B"], arb[f"{small}_A"], w)])
        out[f"containment {small} in {big}"] = np.mean(
            [containment(arb[f"{small}_{a}"], arb[f"{big}_{b}"], w) for a in "AB" for b in "AB"])
    return out


def analyse_split(split, rng):
    z = np.load(RESULTS / f"split{split}.npz")
    sc = {f"{c}_{g}": self_consistency(z[f"prob_{c}_{g}"]) for c in CONDS for g in "AB"}
    arb = {k: (v < THRESHOLD).astype(np.float64) for k, v in sc.items()}
    n = len(z["y_test"])

    # 点估计：所有人各算一次
    point = measures(arb, np.ones(n))
    # 95% 区间：把测试者有放回地重抽 N_BOOT 次（不重新训练任何模型）
    boots = {k: [] for k in point}
    for _ in range(N_BOOT):
        w = np.bincount(rng.integers(0, n, n), minlength=n).astype(np.float64)
        for k, v in measures(arb, w).items():
            boots[k].append(v)
    rows = []
    for k, v in point.items():
        lo, hi = np.nanpercentile(boots[k], [2.5, 97.5])
        rows.append(dict(split=split, measure=k, point=v, lo=lo, hi=hi))

    # 秩相关：只报告点估计
    rho = {}
    for c in CONDS:
        rho[f"spearman baseline {c}"] = spearmanr(sc[f"{c}_A"], sc[f"{c}_B"])[0]
    for s, t in PAIRS:
        rho[f"spearman {s}-{t}"] = np.mean(
            [spearmanr(sc[f"{s}_{a}"], sc[f"{t}_{b}"])[0] for a in "AB" for b in "AB"])
    for k, v in rho.items():
        rows.append(dict(split=split, measure=k, point=v, lo=np.nan, hi=np.nan))

    sizes = {k: int(v.sum()) for k, v in arb.items()}
    return rows, sizes, n


def show(rows, split, sizes, n):
    get = {r["measure"]: r for r in rows if r["split"] == split}
    f = lambda r: f"{r['point']:.3f} [{r['lo']:.3f}, {r['hi']:.3f}]"
    print(f"\n=== Split {split} (test people n = {n}) ===")
    print("Arbitrary set sizes (SC < 0.75):")
    for c in CONDS:
        a, b = sizes[f"{c}_A"], sizes[f"{c}_B"]
        print(f"   {c:11s} A {a:6,d} ({a / n:.2%})   B {b:6,d} ({b / n:.2%})")
    print("Repeatability baseline (same condition, set A vs set B), Jaccard [95% interval]:")
    for c in CONDS:
        print(f"   {c:11s} {f(get[f'baseline {c}'])}")
    print("Between conditions:")
    print(f"   {'pair':18s} {'cross Jaccard':24s} {'Delta [95% interval]':24s} rule met?")
    for s, t in PAIRS:
        d = get[f"delta {s}-{t}"]
        print(f"   {s + '-' + t:18s} {f(get[f'cross {s}-{t}']):24s} {f(d):24s} "
              f"{'yes' if d['lo'] > 0 else 'no'}")
    print("Spearman rank correlation of SC (reported only):")
    for c in CONDS:
        print(f"   baseline {c:11s} {get[f'spearman baseline {c}']['point']:.3f}")
    for s, t in PAIRS:
        print(f"   {s + '-' + t:20s} {get[f'spearman {s}-{t}']['point']:.3f}")
    print("Containment (exploratory): share of the smaller set that is also in the larger set")
    for small, big in CONTAIN:
        print(f"   {small:5s} in {big:11s} {f(get[f'containment {small} in {big}'])}   "
              f"(baseline {small} A<->B: {f(get[f'containment baseline {small}'])})")


def main():
    splits = [s for s in (0, 1, 2) if (RESULTS / f"split{s}.npz").exists()]
    print(f"Splits found: {splits}; bootstrap draws: {N_BOOT}")
    rng = np.random.default_rng(BOOT_SEED)
    all_rows = []
    for s in splits:
        rows, sizes, n = analyse_split(s, rng)
        all_rows += rows
        show(rows, s, sizes, n)

    # 汇总：三次划分都满足规则，才算"找到的人不同"
    print("\n=== RQ1 decision (pre-registered rule: Delta interval > 0 in every split) ===")
    for s, t in PAIRS:
        ds = [r for r in all_rows if r["measure"] == f"delta {s}-{t}"]
        met = [r["lo"] > 0 for r in ds]
        verdict = "DIFFERENT people" if len(ds) == 3 and all(met) else "not shown to differ"
        detail = ", ".join(f"split {r['split']}: {r['point']:.3f} [{r['lo']:.3f}, {r['hi']:.3f}]" for r in ds)
        print(f"   {s + '-' + t:18s} {verdict:20s} ({detail})")

    out = RESULTS / "rq1_summary.csv"
    with open(out, "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=["split", "measure", "point", "lo", "hi"])
        wr.writeheader()
        for r in all_rows:
            wr.writerow({k: (f"{v:.6f}" if isinstance(v, float) else v) for k, v in r.items()})
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
