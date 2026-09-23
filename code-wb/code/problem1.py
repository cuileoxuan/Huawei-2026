# -*- coding: utf-8 -*-
"""
problem1.py —— 问题一：单点往返运输能力与货箱组批方案

内容：
(1) 3 机型 × 15 服务区最大安全载荷矩阵（二分搜索 + 返航安全余量约束）
(2) 各服务区货箱组批：枚举可行批次 + 集合划分 0-1 整数规划（COPT）
    字典序优化：最少架次 -> 最小总能耗 -> 最小累计作业时间
(3) 返航安全余量 rho 的敏感性分析
结果输出：results/Q1_最大安全载荷矩阵.csv、results/Q1_单点组批.csv、
          results/Q1_敏感性分析.csv，并打印汇总。
"""
import os
import itertools
import numpy as np
import pandas as pd

from common import ProblemData, get_copt_model

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(OUT, exist_ok=True)

D = ProblemData()
MODELS = ["A", "B", "C"]


# ------------------------------------------------------------ 1. 最大安全载荷
def roundtrip_energy(g, aid, q):
    """机型 g 向服务区 aid 单点往返（去程载荷 q、返程载荷 0）的总能耗。"""
    seg_out = D.segment("O01", aid)
    seg_back = D.segment(aid, "O01")
    _, e1 = D.seg_time_energy(g, seg_out, q)
    _, e2 = D.seg_time_energy(g, seg_back, 0.0)
    return e1 + e2


def max_safe_payload(g, aid, rho=None):
    """二分搜索机型 g 在服务区 aid 的最大安全载荷（kg）。"""
    m = D.models[g]
    rho = m["rho"] if rho is None else rho
    limit = (1 - rho) * m["Euse"]
    if roundtrip_energy(g, aid, 0.0) > limit:      # 空载往返都不可行
        return 0.0
    lo, hi = 0.0, m["Q"]
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if roundtrip_energy(g, aid, mid) <= limit:
            lo = mid
        else:
            hi = mid
    return lo


def payload_matrix(rho=None):
    mat = pd.DataFrame(index=D.area_ids, columns=MODELS, dtype=float)
    for g in MODELS:
        for aid in D.area_ids:
            mat.loc[aid, g] = round(max_safe_payload(g, aid, rho), 3)
    return mat


# ------------------------------------------------------------ 2. 货箱组批
def feasible_batches(aid, boxes, Q_safe):
    """
    枚举服务区 aid 的全部可行批次：(机型, 货箱子集)。
    约束：总质量 <= min(机型最大载货, 最大安全载荷)、总体积 <= 机型容积、
          往返能耗 <= (1-rho)*Euse（按实际批次重量验算）。
    """
    batches = []
    ids = [b["id"] for b in boxes]
    n = len(ids)
    for g in MODELS:
        m = D.models[g]
        cap = min(m["Q"], Q_safe[g])
        for r in range(1, n + 1):
            for comb in itertools.combinations(range(n), r):
                w = sum(boxes[k]["w"] for k in comb)
                if w > cap + 1e-9:
                    continue
                v = sum(boxes[k]["v"] for k in comb)
                if v > m["V"] + 1e-9:
                    continue
                if roundtrip_energy(g, aid, w) > (1 - m["rho"]) * m["Euse"] + 1e-9:
                    continue
                batches.append((g, tuple(ids[k] for k in comb), w, v))
    return batches


def batch_cost(g, aid, bid_tuple):
    """批次的架次时间与能耗。"""
    w = sum(D.box_w(b) for b in bid_tuple)
    m = D.models[g]
    so = D.eval_sortie(g, [aid], [list(bid_tuple)])
    return so["T"], so["E"]


def solve_batching(aid, boxes, Q_safe, verbose=False):
    """对单个服务区做字典序集合划分优化。返回选中的批次列表。"""
    from common import milp_binary_solve
    batches = feasible_batches(aid, boxes, Q_safe)
    ids = [b["id"] for b in boxes]
    NB = len(batches)
    costs = [batch_cost(g, aid, bt) for (g, bt, _, _) in batches]
    # 覆盖矩阵：每箱恰好被覆盖一次
    A = np.zeros((len(ids), NB))
    for r, (_, bt, _, _) in enumerate(batches):
        for b in bt:
            A[ids.index(b), r] = 1.0
    b_eq = np.ones(len(ids))
    ones = np.ones((1, NB))
    e_row = np.array([[c[1] for c in costs]])

    # 第一层：最少架次
    x = milp_binary_solve(np.ones(NB), A_eq=A, b_eq=b_eq)
    if x is None:
        raise RuntimeError(f"{aid} 组批模型不可行")
    n_star = int(round(x.sum()))
    # 第二层：架次固定下最小能耗
    x2 = milp_binary_solve(e_row[0], A_eq=np.vstack([A, ones]),
                           b_eq=np.r_[b_eq, n_star])
    if x2 is None:
        x2 = x
    e_star = float(e_row[0] @ x2)
    # 第三层：架次+能耗固定下最小累计作业时间
    x = milp_binary_solve(np.array([c[0] for c in costs]),
                          A_eq=np.vstack([A, ones]), b_eq=np.r_[b_eq, n_star],
                          A_ub=e_row, b_ub=np.array([e_star + 1e-6]))
    if x is None:
        x = x2
    t_star = float(np.array([c[0] for c in costs]) @ x)
    sel = [r for r in range(NB) if x[r] > 0.5]
    chosen = []
    for r in sel:
        g, bt, w, v = batches[r]
        T, E = costs[r]
        chosen.append(dict(area=aid, model=g, boxes=list(bt), w=w, v=v,
                           T=T, E=E, soc=1 - E / D.models[g]["Euse"]))
    return chosen, dict(n=n_star, E=e_star, T=t_star, n_candidates=NB)


# ------------------------------------------------------------ 主流程
def main(rho=None, tag=""):
    print("=" * 70)
    print("问题一：单点往返运输能力与货箱组批")
    print("=" * 70)

    # (1) 最大安全载荷矩阵
    mat = payload_matrix(rho)
    print("\n[1] 3 机型 × 15 服务区最大安全载荷矩阵 (kg)")
    print(mat.to_string())
    mat.to_csv(os.path.join(OUT, f"Q1_最大安全载荷矩阵{tag}.csv"),
               encoding="utf-8-sig")

    Q_safe = {g: {aid: mat.loc[aid, g] for aid in D.area_ids} for g in MODELS}

    # (2) 货箱组批（字典序优化）
    area_boxes = {}
    for b in D.boxes:
        area_boxes.setdefault(b["area"], []).append(b)

    all_rows, tot = [], dict(N=0, E=0.0, T=0.0)
    print("\n[2] 各服务区货箱组批方案（字典序：最少架次→最小能耗→最短时间）")
    for aid in D.area_ids:
        boxes = area_boxes[aid]
        Qs = {g: Q_safe[g][aid] for g in MODELS}
        chosen, info = solve_batching(aid, boxes, Qs)
        tot["N"] += info["n"]; tot["E"] += info["E"]; tot["T"] += info["T"]
        print(f"  {aid}: {len(boxes)} 箱 -> {info['n']} 架次 "
              f"(候选批次 {info['n_candidates']}), "
              f"能耗 {info['E']:.3f} kWh, 累计作业时间 {info['T']:.0f} s")
        for k, c in enumerate(chosen, 1):
            all_rows.append(dict(
                架次编号=f"P1-{aid}-{k:02d}", 服务区编号=aid, 机型编号=c["model"],
                货箱编号列表=";".join(c["boxes"]), 总质量kg=round(c["w"], 3),
                总体积m3=round(c["v"], 4), 往返时间s=round(c["T"], 1),
                架次能耗kWh=round(c["E"], 4), 返航SOCpct=round(100 * c["soc"], 2)))
    df = pd.DataFrame(all_rows)
    df.to_csv(os.path.join(OUT, f"Q1_单点组批{tag}.csv"),
              index=False, encoding="utf-8-sig")
    print(f"\n  合计：{tot['N']} 架次，总能耗 {tot['E']:.3f} kWh，"
          f"累计作业时间 {tot['T']:.0f} s")
    return mat, df


def sensitivity():
    """(3) 返航安全余量敏感性分析。"""
    print("\n[3] 返航安全余量敏感性分析")
    rows = []
    for rho in [0.10, 0.15, 0.20, 0.25, 0.30]:
        mat = payload_matrix(rho)
        n_sorties = 0
        for aid in D.area_ids:
            boxes = [b for b in D.boxes if b["area"] == aid]
            Qs = {g: mat.loc[aid, g] for g in MODELS}
            _, info = solve_batching(aid, boxes, Qs)
            n_sorties += info["n"]
        rows.append(dict(rho=rho, 总架次=n_sorties,
                         A型平均载荷kg=round(mat["A"].mean(), 2),
                         B型平均载荷kg=round(mat["B"].mean(), 2),
                         C型平均载荷kg=round(mat["C"].mean(), 2),
                         最小安全载荷kg=round(mat.values.min(), 2)))
        print(f"  rho={rho:.2f}: 总架次={n_sorties}, "
              f"平均最大安全载荷 A/B/C = {mat['A'].mean():.2f}/"
              f"{mat['B'].mean():.2f}/{mat['C'].mean():.2f} kg")
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "Q1_敏感性分析.csv"),
              index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
    sensitivity()
