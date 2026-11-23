"""RQ1 的 margin 控制：差别是不是只来自"离 0.5 很近、怎么都会翻"的人？

在项目文件夹里运行（需要 results/split0.npz、split1.npz、split2.npz）：
    python analysis_rq1_margin.py

做法（实验计划里写的）：
  1. 用参照模型（种子全为 0）的预测概率 p，算每个测试者的 margin = |p - 0.5|，
     也就是"离判断边界有多远"
  2. 按 margin 从小到大把测试者分成 10 组，每组人数一样（第 1 组离 0.5 最近）
  3. 在每一组里面，重新做一遍 RQ1 的比较：baseline、交叉 Jaccard、Delta
具体规则（11/2 运行这个脚本之前定下来的）：
  - 一组里，参与比较的 4 个集合（两个条件的 A、B 组）都至少有 20 个 arbitrary 的人，
    这一组才算"可比"；人太少的组不参与判断（Jaccard 没有意义）
  - 合并 Delta = 各个可比组的 Delta 按该组 arbitrary 人数（4 个集合的并集）加权平均
  - 判断：合并 Delta 的 95% 区间 > 0，且三次划分都成立
    → 差别不能只用"离边界近"来解释
  - 95% 区间：在每一组里面把测试者有放回地重抽 1,000 次（分层重抽）
另外（探索性）：每组里的包含比例，看 init 和 order 的差别在组内是否还在。

输出：屏幕表格 + results/rq1_margin.csv
"""
import csv
import warnings
from itertools import combinations
from pathlib import Path

import numpy as np

# 离 0.5 很远的组里几乎没有 arbitrary 的人，算不出 Jaccard（结果是 NaN），numpy 会警告。
# 这些组本来就不参与判断，所以把这条警告关掉，免得误以为出错。
warnings.filterwarnings("ignore", message="All-NaN slice encountered")

RESULTS = Path("results")
CONDS = ["resampling", "init", "order"]
PAIRS = list(combinations(CONDS, 2))
CONTAIN = [("init", "resampling"), ("order", "resampling"), ("init", "order")]
THRESHOLD = 0.75
N_BINS = 10
MIN_ARB = 20      # 一组里每个集合至少要有这么多 arbitrary 的人才参与比较
N_BOOT = 1000
BOOT_SEED = 2027


def self_consistency(probs):
    B = probs.shape[0]
    b1 = (probs >= 0.5).sum(axis=0)
    b0 = B - b1
    return 1.0 - 2.0 * b0 * b1 / (B * (B - 1))


def jaccard(x, y, w):
    union = w @ np.maximum(x, y)
    return np.nan if union == 0 else (w @ (x * y)) / union


def containment(small, big, w):
    size = w @ small
    return np.nan if size == 0 else (w @ (small * big)) / size


def bin_measures(a, w):
    """一组人里的所有量。a：这一组人的 arbitrary 0/1 向量（字典）；w：每人被算几次。"""
    out = {}
    base = {c: jaccard(a[f"{c}_A"], a[f"{c}_B"], w) for c in CONDS}
    for s, t in PAIRS:
        cross = np.mean([jaccard(a[f"{s}_{x}"], a[f"{t}_{y}"], w) for x in "AB" for y in "AB"])
        out[f"delta {s}-{t}"] = min(base[s], base[t]) - cross
    for small, big in CONTAIN:
        out[f"containment baseline {small}"] = np.mean(
            [containment(a[f"{small}_A"], a[f"{small}_B"], w), containment(a[f"{small}_B"], a[f"{small}_A"], w)])
        out[f"containment {small} in {big}"] = np.mean(
            [containment(a[f"{small}_{x}"], a[f"{big}_{y}"], w) for x in "AB" for y in "AB"])
    return out


def analyse_split(split, rng):
    z = np.load(RESULTS / f"split{split}.npz")
    arb = {f"{c}_{g}": (self_consistency(z[f"prob_{c}_{g}"]) < THRESHOLD).astype(np.float64)
           for c in CONDS for g in "AB"}
    margin = np.abs(z["ref_prob"] - 0.5)
    bins = np.array_split(np.argsort(margin, kind="stable"), N_BINS)  # 第 1 组离 0.5 最近

    info, point, weight, ok = [], [], {}, {}
    for b, idx in enumerate(bins):
        a = {k: v[idx] for k, v in arb.items()}
        counts = {k: int(v.sum()) for k, v in a.items()}
        info.append(dict(bin=b + 1, n=len(idx), lo=float(margin[idx].min()), hi=float(margin[idx].max()),
                         counts=counts))
        point.append(bin_measures(a, np.ones(len(idx))))
        for s, t in PAIRS:  # 可比性 + 合并时的权重（4 个集合的并集人数）
            sets = [a[f"{s}_A"], a[f"{s}_B"], a[f"{t}_A"], a[f"{t}_B"]]
            ok[(b, f"delta {s}-{t}")] = min(int(x.sum()) for x in sets) >= MIN_ARB
            weight[(b, f"delta {s}-{t}")] = float(np.max(sets, axis=0).sum())
        for small, big in CONTAIN:
            sets = [a[f"{small}_A"], a[f"{small}_B"]]
            for key in (f"containment {small} in {big}", f"containment baseline {small}"):
                ok[(b, key)] = min(int(x.sum()) for x in sets) >= MIN_ARB
                weight[(b, key)] = float(sum(x.sum() for x in sets))

    keys = list(point[0])

    def pooled(per_bin):
        """把可比组的值按权重合并成一个数。"""
        out = {}
        for k in keys:
            ws = [(weight[(b, k)], per_bin[b][k]) for b in range(N_BINS) if ok[(b, k)]]
            out[k] = np.nan if not ws else sum(w * v for w, v in ws) / sum(w for w, _ in ws)
        return out

    pooled_point = pooled(point)
    # 分层重抽：每一组里面各自有放回地抽，每次重算所有组和合并值
    boot_bins = {(b, k): [] for b in range(N_BINS) for k in keys}
    boot_pool = {k: [] for k in keys}
    for _ in range(N_BOOT):
        per_bin = []
        for b, idx in enumerate(bins):
            w = np.bincount(rng.integers(0, len(idx), len(idx)), minlength=len(idx)).astype(np.float64)
            m = bin_measures({k: v[idx] for k, v in arb.items()}, w)
            per_bin.append(m)
            for k in keys:
                boot_bins[(b, k)].append(m[k])
        for k, v in pooled(per_bin).items():
            boot_pool[k].append(v)

    rows = []
    for b in range(N_BINS):
        for k in keys:
            lo, hi = np.nanpercentile(boot_bins[(b, k)], [2.5, 97.5])
            rows.append(dict(split=split, bin=b + 1, measure=k, point=point[b][k], lo=lo, hi=hi,
                             comparable=ok[(b, k)]))
    for k in keys:
        lo, hi = np.nanpercentile(boot_pool[k], [2.5, 97.5])
        rows.append(dict(split=split, bin="pooled", measure=k, point=pooled_point[k], lo=lo, hi=hi,
                         comparable=True))
    return rows, info


def show(rows, info, split):
    get = {(r["bin"], r["measure"]): r for r in rows}
    print(f"\n=== Split {split}: 10 groups by distance to 0.5 (group 1 = closest) ===")
    print(f"   {'grp':>3s} {'margin range':>15s} {'people':>7s}   arbitrary people in set A: "
          f"{'resampling':>10s} {'init':>6s} {'order':>6s}")
    for d in info:
        c = d["counts"]
        print(f"   {d['bin']:3d} {d['lo']:6.3f} - {d['hi']:6.3f} {d['n']:7d}   {'':27s}"
              f"{c['resampling_A']:10d} {c['init_A']:6d} {c['order_A']:6d}")

    def cell(b, k):
        r = get[(b, k)]
        return f"{r['point']:6.3f}" if r["comparable"] else "     -"

    print("Delta inside each group ('-' = fewer than 20 arbitrary people in a set):")
    print(f"   {'grp':>3s} " + " ".join(f"{s + '-' + t:>17s}" for s, t in PAIRS))
    for d in info:
        print(f"   {d['bin']:3d} " + " ".join(f"{cell(d['bin'], f'delta {s}-{t}'):>17s}" for s, t in PAIRS))
    print("   pooled Delta [95% interval]:")
    for s, t in PAIRS:
        r = get[("pooled", f"delta {s}-{t}")]
        n_ok = sum(get[(b, f"delta {s}-{t}")]["comparable"] for b in range(1, N_BINS + 1))
        n_pos = sum(get[(b, f"delta {s}-{t}")]["comparable"] and get[(b, f"delta {s}-{t}")]["lo"] > 0
                    for b in range(1, N_BINS + 1))
        print(f"      {s + '-' + t:18s} {r['point']:.3f} [{r['lo']:.3f}, {r['hi']:.3f}]   "
              f"comparable groups: {n_ok}, groups with interval > 0: {n_pos}")
    print("Containment inside each group (exploratory), pooled [95% interval]:")
    for small, big in CONTAIN:
        r = get[("pooled", f"containment {small} in {big}")]
        rb = get[("pooled", f"containment baseline {small}")]
        print(f"      {small:5s} in {big:11s} {r['point']:.3f} [{r['lo']:.3f}, {r['hi']:.3f}]   "
              f"(baseline {small} A<->B: {rb['point']:.3f} [{rb['lo']:.3f}, {rb['hi']:.3f}])")


def main():
    splits = [s for s in (0, 1, 2) if (RESULTS / f"split{s}.npz").exists()]
    print(f"Splits found: {splits}; groups: {N_BINS}; minimum arbitrary people per set: {MIN_ARB}; "
          f"bootstrap draws: {N_BOOT}")
    rng = np.random.default_rng(BOOT_SEED)
    all_rows = []
    for s in splits:
        rows, info = analyse_split(s, rng)
        all_rows += rows
        show(rows, info, s)

    print("\n=== Margin control decision (rule: pooled Delta interval > 0 in every split) ===")
    for s, t in PAIRS:
        ps = [r for r in all_rows if r["bin"] == "pooled" and r["measure"] == f"delta {s}-{t}"]
        holds = len(ps) == 3 and all(r["lo"] > 0 for r in ps)
        verdict = "difference REMAINS within groups" if holds else "explained by distance to 0.5"
        detail = ", ".join(f"split {r['split']}: {r['point']:.3f} [{r['lo']:.3f}, {r['hi']:.3f}]" for r in ps)
        print(f"   {s + '-' + t:18s} {verdict:34s} ({detail})")

    out = RESULTS / "rq1_margin.csv"
    with open(out, "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=["split", "bin", "measure", "point", "lo", "hi", "comparable"])
        wr.writeheader()
        for r in all_rows:
            wr.writerow({k: (f"{v:.6f}" if isinstance(v, float) else v) for k, v in r.items()})
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
