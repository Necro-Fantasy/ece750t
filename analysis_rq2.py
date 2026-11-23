"""RQ2 分析：男女之间的差距，在三种随机性下是不是一样？

在项目文件夹里运行（需要 analysis_rq1.py 和 results/split0.npz 等）：
    python analysis_rq2.py | Tee-Object -FilePath results\\log_rq2.txt

每次划分、每个条件，分别对男、女算：
  1. arbitrary 比例（SC < 0.75 的人占多少）。在 Cooper 的做法里，arbitrary 的人
     就是被弃权、转给人工的人，所以它同时也是"弃权比例"。
  2. 平均 SC（不依赖阈值）。补充分析：不在 10/31 的实验计划里，11/3 运行前加入，
     依据是 proposal 里 RQ2 的原话 "gap ... in self-consistency and abstention"。
  差距 = 女 - 男（和 RQ3、论文统一）。resampling、init、order 各有 A、B 两组，差距取两组的平均。
  11/12 修改：原来是"男 - 女"，现在改成"女 - 男"。只改符号：数值大小、区间宽度、
  判断结果都不变（重抽测试者的种子 BOOT_SEED 没变，抽到的人完全一样）。

判断（双侧，因为事先不预测方向）：
  差距的差 = 条件 1 的差距 - 条件 2 的差距；
  它的 95% 区间不包含 0，且三次划分都成立 → 两个条件的男女差距不同。
  主结论只看 arbitrary 比例（实验计划里事先定好）；平均 SC 用同一规则判断，
  但只作为补充，不改变主结论。
另外报告（不参与判断）：
  - 噪声对照：同一条件 A 组的差距 - B 组的差距（应该接近 0）
  - 女 / 男 的比值（探索性，11/3 加的；11/12 随差距方向一起改成 女 / 男）：各条件 arbitrary 比例本身大小不同，
    看相对差，避免把"规模不同"误当成"公平性不同"
95% 区间：把测试者有放回地重抽 1,000 次（不重新训练任何模型）。
输出：屏幕表格 + results/rq2_summary.csv
"""
import csv

import numpy as np

import analysis_rq1 as A  # 用同一个 SC 函数、同样的条件名和文件位置

BOOT_SEED = 2028
GROUPS = {"men": 0, "women": 1}  # group_test：0 = 男，1 = 女


def group_stats(arb, sc, male, female, w):
    """在权重 w 下，算一组模型的男、女 arbitrary 比例和平均 SC。"""
    wm, wf = w * male, w * female
    return dict(arb_men=(wm @ arb) / wm.sum(), arb_women=(wf @ arb) / wf.sum(),
                sc_men=(wm @ sc) / wm.sum(), sc_women=(wf @ sc) / wf.sum())


def measures(data, male, female, w):
    """所有要报告的量（一个字典）。data[组名] = (arb 0/1 向量, SC 向量)。"""
    st = {k: group_stats(a, s, male, female, w) for k, (a, s) in data.items()}
    out = {}
    for c in A.CONDS + ["all"]:
        sets = [f"{c}_A", f"{c}_B"] if c != "all" else ["all"]
        for m in ("arb", "sc"):
            men = np.mean([st[k][f"{m}_men"] for k in sets])
            women = np.mean([st[k][f"{m}_women"] for k in sets])
            out[f"{m} men {c}"], out[f"{m} women {c}"] = men, women
            out[f"{m} gap {c}"] = women - men  # 差距 = 女 - 男
            if m == "arb":
                out[f"arb ratio {c}"] = women / men if men > 0 else np.nan  # 女 / 男
            if c != "all":  # 噪声对照：同一条件 A 组的差距 - B 组的差距
                gA = st[f"{c}_A"][f"{m}_women"] - st[f"{c}_A"][f"{m}_men"]
                gB = st[f"{c}_B"][f"{m}_women"] - st[f"{c}_B"][f"{m}_men"]
                out[f"{m} noise {c}"] = gA - gB
    for s, t in A.PAIRS:
        for m in ("arb", "sc"):
            out[f"{m} gapdiff {s}-{t}"] = out[f"{m} gap {s}"] - out[f"{m} gap {t}"]
    return out


def analyse_split(split, rng):
    z = np.load(A.RESULTS / f"split{split}.npz")
    data = {}
    for key in [f"{c}_{g}" for c in A.CONDS for g in "AB"] + ["all"]:
        sc = A.self_consistency(z[f"prob_{key}"])
        data[key] = ((sc < A.THRESHOLD).astype(np.float64), sc)
    g = z["group_test"]
    male, female = (g == GROUPS["men"]).astype(np.float64), (g == GROUPS["women"]).astype(np.float64)
    n = len(g)
    point = measures(data, male, female, np.ones(n))
    boots = {k: [] for k in point}
    for _ in range(A.N_BOOT):
        w = np.bincount(rng.integers(0, n, n), minlength=n).astype(np.float64)
        for k, v in measures(data, male, female, w).items():
            boots[k].append(v)
    rows = []
    for k, v in point.items():
        lo, hi = np.nanpercentile(boots[k], [2.5, 97.5])
        rows.append(dict(split=split, measure=k, point=v, lo=lo, hi=hi))
    y = z["y_test"]
    info = dict(men=int(male.sum()), women=int(female.sum()),
                base_men=float(y[g == 0].mean()), base_women=float(y[g == 1].mean()))
    return rows, info


def show(rows, info, split):
    get = {r["measure"]: r for r in rows}
    pp = lambda r: f"{100 * r['point']:+.2f} [{100 * r['lo']:+.2f}, {100 * r['hi']:+.2f}]"
    sc4 = lambda r: f"{r['point']:+.4f} [{r['lo']:+.4f}, {r['hi']:+.4f}]"
    print(f"\n=== Split {split}: test people men {info['men']}, women {info['women']}; "
          f"share above $50k: men {info['base_men']:.1%}, women {info['base_women']:.1%} ===")
    print("Arbitrary (= abstained) share by sex, average of sets A and B:")
    print(f"   {'condition':11s} {'men':>7s} {'women':>7s}   {'gap women-men, points [95%]':30s} {'ratio women/men [95%]':22s}")
    for c in A.CONDS + ["all"]:
        r = get[f"arb ratio {c}"]
        print(f"   {c:11s} {get[f'arb men {c}']['point']:7.2%} {get[f'arb women {c}']['point']:7.2%}   "
              f"{pp(get[f'arb gap {c}']):30s} {r['point']:.3f} [{r['lo']:.3f}, {r['hi']:.3f}]")
    print("Mean self-consistency by sex:")
    for c in A.CONDS + ["all"]:
        print(f"   {c:11s} men {get[f'sc men {c}']['point']:.4f}  women {get[f'sc women {c}']['point']:.4f}   "
              f"gap {sc4(get[f'sc gap {c}'])}")
    print("Noise check, same condition: gap in set A minus gap in set B (should include 0):")
    for c in A.CONDS:
        print(f"   {c:11s} arbitrary share {pp(get[f'arb noise {c}'])} points   mean SC {sc4(get[f'sc noise {c}'])}")
    print("Difference of gaps between conditions (two-sided rule: interval excludes 0):")
    for s, t in A.PAIRS:
        ra, rs = get[f"arb gapdiff {s}-{t}"], get[f"sc gapdiff {s}-{t}"]
        ex = lambda r: "yes" if r["lo"] > 0 or r["hi"] < 0 else "no"
        print(f"   {s + '-' + t:18s} arbitrary share {pp(ra)} points, excludes 0: {ex(ra):3s}   "
              f"mean SC {sc4(rs)}, excludes 0: {ex(rs)}")


def main():
    splits = [s for s in (0, 1, 2) if (A.RESULTS / f"split{s}.npz").exists()]
    print(f"Splits found: {splits}; threshold SC < {A.THRESHOLD}; bootstrap draws: {A.N_BOOT}")
    rng = np.random.default_rng(BOOT_SEED)
    all_rows = []
    for s in splits:
        rows, info = analyse_split(s, rng)
        all_rows += rows
        show(rows, info, s)

    print("\n=== RQ2 decision (two-sided: interval excludes 0 in every split) ===")
    print("    arbitrary share = pre-registered main result; mean SC = supplementary, not in the plan")
    for m, label in (("arb", "arbitrary share"), ("sc", "mean SC (suppl.)")):
        for s, t in A.PAIRS:
            ds = [r for r in all_rows if r["measure"] == f"{m} gapdiff {s}-{t}"]
            ex = [r["lo"] > 0 or r["hi"] < 0 for r in ds]
            verdict = "gap DIFFERS" if len(ds) == 3 and all(ex) else "not shown to differ"
            print(f"   {label:15s} {s + '-' + t:18s} {verdict:20s} (excludes 0 in {sum(ex)} of {len(ds)} splits)")

    out = A.RESULTS / "rq2_summary.csv"
    with open(out, "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=["split", "measure", "point", "lo", "hi"])
        wr.writeheader()
        for r in all_rows:
            wr.writerow({k: (f"{v:.6f}" if isinstance(v, float) else v) for k, v in r.items()})
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
