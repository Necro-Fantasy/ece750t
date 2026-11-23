"""RQ3 分析：弃权以后，男女的错误率是不是接近公平？在三种随机性下都是这样吗？

在项目文件夹里运行（需要 analysis_rq1.py 和 results/split0.npz 等；不重新训练任何模型）：
    python analysis_rq3.py | Tee-Object -FilePath results\\log_rq3.txt

两个错误率（每个性别分别算）：
  FPR（误报率）= 收入其实 <= 5 万的人里，被预测成 > 5 万的比例
  FNR（漏报率）= 收入其实 >  5 万的人里，被预测成 <= 5 万的比例
  差距 = 女 - 男（实验计划第 3 节的定义）

实验计划里事先写好的（10/31）：
  - 弃权：一组 101 个模型投票；SC < 0.75 的人不给预测（转给人工），其余的人用多数票
  - 比较：弃权后、在没被弃权的人身上的 FPR、FNR 差距  vs  单个模型在所有人身上的差距
  - 判断：两个差距都比单个模型的小（95% 区间，三次划分都成立）→ Cooper 的发现在这个条件下成立
计划没写清、运行前补全的细节（11/4，看结果之前定下）：
  - "单个模型"= 同一组 101 个模型各自单独预测时的平均表现（期望值）。这和 Cooper et al.
    的 baseline 一样（他们用的是 "expected error of individual models"）。因为 FPR 对每个
    模型是线性的，101 个模型的平均 FPR = 每个人"投 1 的模型比例 q"在负样本里的平均。
  - "小"= 离 0 更近，所以比绝对值：减小量 = |单个模型的差距| - |弃权后的差距|
  - A、B 两组各算一次减小量，取平均（和 RQ2 一样）
  - 单侧：减小量的 95% 区间下限 > 0；FPR 和 FNR 都满足，且三次划分都成立 → 成立
补充（不在实验计划里，只报告，不参与判断）：
  - 单个模型、只在没被弃权的人身上：分开"去掉了难判的人"和"投票"两个作用
  - 多数票、在所有人身上（不弃权）
  - 参照模型（三个种子都是 0 的那一个模型）在所有人身上：字面意义上的"一个模型"
  - 噪声对照：同一条件 A 组的减小量 - B 组的减小量
  - 按性别 x 真实标签的弃权比例、准确率（帮助理解，不判断）
95% 区间：把测试者有放回地重抽 1,000 次（不重新训练任何模型）。
输出：屏幕表格 + results/rq3_summary.csv
"""
import csv

import numpy as np

import analysis_rq1 as A  # 同一个 SC 函数、阈值、条件名和文件位置

BOOT_SEED = 2029
SETS = [f"{c}_{g}" for c in A.CONDS for g in "AB"] + ["all"]


def rate(num, mask, w):
    """在 mask 选中的人里，num 的加权平均。
    num：每个人的 0/1（或 0~1 之间的数）；mask：0/1，1 = 这个人算在分母里；
    w：每个人被抽到几次（不重抽时全是 1）。"""
    wm = w * mask
    den = wm.sum()
    return np.nan if den == 0 else (wm @ num) / den


def set_stats(q, kept, y, men, women, w):
    """一组 101 个模型的所有量。
    q：每个人"投 1 的模型比例"（0~1）；kept：1 = 没被弃权；y：真实标签。"""
    vote = (q >= 0.5).astype(np.float64)  # 多数票（101 个模型，不会平票）
    neg, pos = (y == 0).astype(np.float64), (y == 1).astype(np.float64)
    out = {}
    for gname, g in (("men", men), ("women", women)):
        # 主比较：单个模型（101 个模型的平均）在所有人身上 vs 多数票在没被弃权的人身上
        out[f"fpr base {gname}"] = rate(q, g * neg, w)
        out[f"fnr base {gname}"] = rate(1 - q, g * pos, w)
        out[f"fpr ens {gname}"] = rate(vote, g * neg * kept, w)
        out[f"fnr ens {gname}"] = rate(1 - vote, g * pos * kept, w)
        # 补充 1：单个模型，只看没被弃权的人
        out[f"fpr indiv_kept {gname}"] = rate(q, g * neg * kept, w)
        out[f"fnr indiv_kept {gname}"] = rate(1 - q, g * pos * kept, w)
        # 补充 2：多数票，所有人（不弃权）
        out[f"fpr vote_all {gname}"] = rate(vote, g * neg, w)
        out[f"fnr vote_all {gname}"] = rate(1 - vote, g * pos, w)
        # 帮助理解：弃权比例（按真实标签分开）
        out[f"abstain neg {gname}"] = rate(1 - kept, g * neg, w)
        out[f"abstain pos {gname}"] = rate(1 - kept, g * pos, w)
    everyone = np.ones_like(y, dtype=np.float64)
    out["acc base"] = rate(q * y + (1 - q) * (1 - y), everyone, w)  # 单个模型的平均准确率
    out["acc ens"] = rate((vote == y).astype(np.float64), kept, w)   # 多数票在没被弃权的人上
    out["coverage"] = rate(kept, everyone, w)                         # 没被弃权的人占多少
    for m in ("fpr", "fnr"):
        for model in ("base", "ens", "indiv_kept", "vote_all"):
            out[f"{m} gap {model}"] = out[f"{m} {model} women"] - out[f"{m} {model} men"]
        # 判断用的量：差距（绝对值）减小了多少
        out[f"{m} reduction"] = abs(out[f"{m} gap base"]) - abs(out[f"{m} gap ens"])
    return out


def measures(data, ref_pred, y, men, women, w):
    """所有要报告的量。data[组名] = (q, kept)。"""
    st = {k: set_stats(q, kept, y, men, women, w) for k, (q, kept) in data.items()}
    out = {}
    for c in A.CONDS + ["all"]:
        sets = [f"{c}_A", f"{c}_B"] if c != "all" else ["all"]
        for key in st[sets[0]]:
            out[f"{key} {c}"] = np.mean([st[s][key] for s in sets])  # A、B 两组取平均
        if c != "all":
            for m in ("fpr", "fnr"):  # 噪声对照
                out[f"{m} noise {c}"] = st[f"{c}_A"][f"{m} reduction"] - st[f"{c}_B"][f"{m} reduction"]
    # 补充 3：参照模型（一个模型）在所有人身上
    neg, pos = (y == 0).astype(np.float64), (y == 1).astype(np.float64)
    for m, num, lab in (("fpr", ref_pred, neg), ("fnr", 1 - ref_pred, pos)):
        out[f"{m} ref men"] = rate(num, men * lab, w)
        out[f"{m} ref women"] = rate(num, women * lab, w)
        out[f"{m} gap ref"] = out[f"{m} ref women"] - out[f"{m} ref men"]
    return out


def analyse_split(split, rng):
    z = np.load(A.RESULTS / f"split{split}.npz")
    y = z["y_test"].astype(np.float64)
    g = z["group_test"]
    men, women = (g == 0).astype(np.float64), (g == 1).astype(np.float64)
    data = {}
    for key in SETS:
        probs = z[f"prob_{key}"]
        q = (probs >= 0.5).mean(axis=0)                      # 投 1 的模型比例 = B1 / 101
        kept = (A.self_consistency(probs) >= A.THRESHOLD).astype(np.float64)  # 1 = 不弃权
        data[key] = (q, kept)
    ref_pred = (z["ref_prob"] >= 0.5).astype(np.float64)
    n = len(y)
    point = measures(data, ref_pred, y, men, women, np.ones(n))
    boots = {k: [] for k in point}
    for _ in range(A.N_BOOT):
        w = np.bincount(rng.integers(0, n, n), minlength=n).astype(np.float64)
        for k, v in measures(data, ref_pred, y, men, women, w).items():
            boots[k].append(v)
    rows = []
    for k, v in point.items():
        lo, hi = np.nanpercentile(boots[k], [2.5, 97.5])
        rows.append(dict(split=split, measure=k, point=v, lo=lo, hi=hi))
    info = dict(men=int(men.sum()), women=int(women.sum()),
                men_pos=int((men * y).sum()), women_pos=int((women * y).sum()))
    return rows, info


def show(rows, info, split):
    get = {r["measure"]: r for r in rows}
    p = lambda k: get[k]["point"]
    ci = lambda k: f"{100 * p(k):+.2f} [{100 * get[k]['lo']:+.2f}, {100 * get[k]['hi']:+.2f}]"
    print(f"\n=== Split {split}: men {info['men']} (above $50k: {info['men_pos']}), "
          f"women {info['women']} (above $50k: {info['women_pos']}) ===")
    print("Coverage and accuracy (average of sets A and B):")
    print(f"   {'condition':11s} {'kept':>7s} {'abstain: men neg/pos':>22s} {'women neg/pos':>15s} "
          f"{'acc single (all)':>17s} {'acc vote (kept)':>16s}")
    for c in A.CONDS + ["all"]:
        print(f"   {c:11s} {p(f'coverage {c}'):7.2%} {p(f'abstain neg men {c}'):10.2%} / {p(f'abstain pos men {c}'):6.2%}  "
              f"{p(f'abstain neg women {c}'):6.2%} / {p(f'abstain pos women {c}'):6.2%} "
              f"{p(f'acc base {c}'):17.4f} {p(f'acc ens {c}'):16.4f}")
    for m, label in (("fpr", "FPR (true <= $50k, predicted > $50k)"), ("fnr", "FNR (true > $50k, predicted <= $50k)")):
        print(f"{label}, in percent; gap = women - men:")
        print(f"   {'condition':11s} {'single model on everyone':>30s}   {'vote after abstention':>30s}   "
              f"{'reduction |single|-|vote| [95%]':>32s}")
        for c in A.CONDS + ["all"]:
            print(f"   {c:11s} men {100 * p(f'{m} base men {c}'):5.2f} women {100 * p(f'{m} base women {c}'):5.2f} "
                  f"gap {100 * p(f'{m} gap base {c}'):+6.2f}   "
                  f"men {100 * p(f'{m} ens men {c}'):5.2f} women {100 * p(f'{m} ens women {c}'):5.2f} "
                  f"gap {100 * p(f'{m} gap ens {c}'):+6.2f}   {ci(f'{m} reduction {c}')}")
    print("Supplementary gaps (women - men, percent; not part of the decision):")
    print(f"   {'condition':11s} {'FPR: single on kept':>20s} {'vote on all':>12s}   {'FNR: single on kept':>20s} {'vote on all':>12s}")
    for c in A.CONDS + ["all"]:
        print(f"   {c:11s} {100 * p(f'fpr gap indiv_kept {c}'):20.2f} {100 * p(f'fpr gap vote_all {c}'):12.2f}   "
              f"{100 * p(f'fnr gap indiv_kept {c}'):20.2f} {100 * p(f'fnr gap vote_all {c}'):12.2f}")
    print(f"   reference model (one model, all seeds 0) on everyone: FPR gap {ci('fpr gap ref')}, FNR gap {ci('fnr gap ref')}")
    print("Noise check: reduction in set A minus reduction in set B (should include 0):")
    for c in A.CONDS:
        print(f"   {c:11s} FPR {ci(f'fpr noise {c}')}   FNR {ci(f'fnr noise {c}')}")


def main():
    splits = [s for s in (0, 1, 2) if (A.RESULTS / f"split{s}.npz").exists()]
    print(f"Splits found: {splits}; threshold SC < {A.THRESHOLD}; bootstrap draws: {A.N_BOOT}")
    rng = np.random.default_rng(BOOT_SEED)
    all_rows = []
    for s in splits:
        rows, info = analyse_split(s, rng)
        all_rows += rows
        show(rows, info, s)

    print("\n=== RQ3 decision (pre-registered, one-sided: lower bound of the reduction > 0, "
          "for both FPR and FNR, in every split) ===")
    for c in A.CONDS:
        ok = {}
        for m in ("fpr", "fnr"):
            ds = [r for r in all_rows if r["measure"] == f"{m} reduction {c}"]
            ok[m] = [r["lo"] > 0 for r in ds]
        holds = len(ok["fpr"]) == 3 and all(ok["fpr"]) and all(ok["fnr"])
        print(f"   {c:11s} {'HOLDS' if holds else 'not shown':10s} "
              f"(FPR lower bound > 0 in {sum(ok['fpr'])} of {len(ok['fpr'])} splits, "
              f"FNR in {sum(ok['fnr'])} of {len(ok['fnr'])})")

    out = A.RESULTS / "rq3_summary.csv"
    with open(out, "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=["split", "measure", "point", "lo", "hi"])
        wr.writeheader()
        for r in all_rows:
            wr.writerow({k: (f"{v:.6f}" if isinstance(v, float) else v) for k, v in r.items()})
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
