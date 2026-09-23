# -*- coding: utf-8 -*-
"""
problem4.py —— 问题四：救援任务分区与资源配置优化

流程：
1. 继承问题三的联合调度方案（results/q3_solution.json）。
2. 并查集（Union-Find）：同一运输架次涉及多个服务区 -> 必须同组；
   同一中继架次服务多个架次 -> 涉及服务区亦并入同一不可拆任务块。
3. 对 K=2 与 K=3，枚举任务块的全部分区方案（精确枚举，等价于求解
   约束图分区问题），目标按字典序：总资源配置规模最小 -> 组间工作量
   最均衡（最大工作量差最小）。
4. 每个任务组独立核算资源需求：按时间轴计算各类资源的
   “最大并发占用量”（运输无人机按机型、共享电池按机型含充电占用、
   中继无人机、中继能源组件），资源不跨组调配。
5. 对比两种分区：资源规模、冗余（相对现有库存）、工作量均衡、资源缺口。

输出：results/Q4_分区配置.csv、results/q4_report.json
"""
import os
import json
import itertools
import numpy as np
import pandas as pd

from common import ProblemData

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
D = ProblemData()
AREA_IDS = D.area_ids


# ------------------------------------------------------------ 并查集
class UnionFind:
    def __init__(self, items):
        self.p = {x: x for x in items}

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra


def task_blocks(routes, relays):
    """不可拆任务块：仅按题目规定的“同一运输架次涉及多个服务区”合并。"""
    uf = UnionFind(AREA_IDS)
    for rt in routes:                       # 同架次多服务区 -> 同组
        for s in rt["stops"][1:]:
            uf.union(rt["stops"][0], s)
    blocks = {}
    for a in AREA_IDS:
        blocks.setdefault(uf.find(a), []).append(a)
    return list(blocks.values())


# ------------------------------------------------------------ 资源峰值核算
def peak_overlap(intervals):
    """区间列表的最大并发占用量（区间图色数 = 最大重叠数）。"""
    if not intervals:
        return 0
    ev = []
    for s, e in intervals:
        ev.append((s, 1)); ev.append((e - 1e-6, -1))
    ev.sort()
    cur = peak = 0
    for _, d in ev:
        cur += d
        peak = max(peak, cur)
    return peak


def group_resources(routes_g, relays_g):
    """核算一个任务组独立执行所需的各类资源数量与工作量。"""
    res = {}
    workload = 0.0
    for g in ["A", "B", "C"]:
        uav_iv, bat_iv = [], []
        for rt in routes_g:
            if rt["model"] != g:
                continue
            uav_iv.append((rt["start"], rt["end"]))
            chg = D.charge_time(
                1 - rt["E"] / D.models[g]["Euse"], D.batteries[g]["Tfull"])
            bat_iv.append((rt["start"], rt["end"] + chg))
            workload += rt["T"]
        res[f"UAV_{g}"] = peak_overlap(uav_iv)
        res[f"BAT_{g}"] = peak_overlap(bat_iv)
    rly_iv = [(m["start_act"], m["ret_act"] + D.relay["t_turn"])
              for m in relays_g]
    ecp_iv = [(m["start_act"],
               m["ret_act"] + D.charge_time(m["soc_end"],
                                            D.relay_batt["Tfull"]))
              for m in relays_g]
    res["RELAY"] = peak_overlap(rly_iv)
    res["ECOMP"] = peak_overlap(ecp_iv)
    workload += sum(m["ret_act"] - m["start_act"] for m in relays_g)
    return res, workload


def evaluate_partition(blocks, assign, routes, relays):
    """assign[i] = 任务块 i 的组号。返回 (总资源, 工作量差, 明细)。
    注：跨组服务的中继架次在每个被服务的组分别计入（资源不可跨组调配，
    各组须独立具备该时段的中继保障能力，属保守核算）。"""
    K = max(assign) + 1
    groups = [[] for _ in range(K)]
    for bi, gi in enumerate(assign):
        groups[gi] += blocks[bi]
    total, wl, detail = 0, [], []
    for gi, areas in enumerate(groups):
        rg = [rt for rt in routes if any(s in areas for s in rt["stops"])]
        mg = [m for m in relays
              if any(s in areas for r in routes if r["sortie"] in m["cover"]
                     for s in r["stops"])]
        res, w = group_resources(rg, mg)
        total += sum(res.values())
        wl.append(w)
        detail.append(dict(areas=areas, res=res, workload=w))
    imbalance = (max(wl) - min(wl)) if wl else 0.0
    return total, imbalance, detail


def optimize_partition(blocks, K, routes, relays):
    """
    枚举任务块到 K 个组的全部分配（精确枚举，等价于约束图分区最优解）。
    返回 Pareto 两端最优：
      res_best  —— 字典序 (总资源, 工作量差)：资源配置规模最小方案
      bal_best  —— 字典序 (工作量差, 总资源)：组间工作量最均衡方案
    """
    m = len(blocks)
    if m < K:
        return None
    res_best = bal_best = None
    for assign in itertools.product(range(K), repeat=m):
        if len(set(assign)) < K:        # 每组非空
            continue
        # 破除组标号对称性：第一个出现的标号必须递增
        first = {}
        ok = True
        for i, a in enumerate(assign):
            if a not in first:
                first[a] = len(first)
            if first[a] != a or a > i:
                ok = False
                break
        if not ok:
            continue
        total, imb, detail = evaluate_partition(blocks, assign, routes, relays)
        if res_best is None or (total, imb) < res_best[0]:
            res_best = ((total, imb), assign, detail)
        if bal_best is None or (imb, total) < bal_best[0]:
            bal_best = ((imb, total), assign, detail)
    return res_best, bal_best


# ------------------------------------------------------------ 主流程
def main():
    print("=" * 70)
    print("问题四：救援任务分区与资源配置优化")
    print("=" * 70)
    fp = os.path.join(OUT, "q3_solution.json")
    if not os.path.exists(fp):
        import problem3
        problem3.main()
    q3 = json.load(open(fp, encoding="utf-8"))
    routes, relays = q3["routes"], q3.get("relays", [])

    blocks = task_blocks(routes, relays)
    print(f"\n[1] 不可拆任务块（并查集）: {len(blocks)} 个")
    for i, b in enumerate(blocks, 1):
        print(f"  C{i}: {sorted(b)}")

    inventory = {
        "UAV_A": sum(1 for u, g in D.uavs.items() if g == "A"),
        "UAV_B": sum(1 for u, g in D.uavs.items() if g == "B"),
        "UAV_C": sum(1 for u, g in D.uavs.items() if g == "C"),
        "BAT_A": D.batteries["A"]["n"], "BAT_B": D.batteries["B"]["n"],
        "BAT_C": D.batteries["C"]["n"],
        "RELAY": len(D.relay_uavs), "ECOMP": D.relay_batt["n"],
    }

    rows, report = [], {}
    for K in (2, 3):
        result = optimize_partition(blocks, K, routes, relays)
        if result is None:
            print(f"\n[2] K={K}: 不可拆任务块仅 {len(blocks)} 个，"
                  f"无法划分为 {K} 个非空任务组")
            continue
        res_best, bal_best = result
        for scheme, (key, assign, detail) in (("资源最小", res_best),
                                              ("均衡最优", bal_best)):
            v1, v2 = key
            print(f"\n[2] K={K} [{scheme}]: 总资源={sum(d['res'][r] for d in detail for r in d['res'])}, "
                  f"组间最大工作量差={max(d['workload'] for d in detail) - min(d['workload'] for d in detail):.0f}s")
            tot_need = {r: 0 for r in inventory}
            for gi, d in enumerate(detail, 1):
                print(f"  任务组 {gi}: 服务区={sorted(d['areas'])}, "
                      f"工作量={d['workload']:.0f}s, 资源={d['res']}")
                for r in inventory:
                    tot_need[r] += d["res"][r]
                rows.append(dict(
                    K=K, 方案=scheme, 任务组编号=f"G{gi}",
                    服务区列表=";".join(sorted(d["areas"])),
                    A型运输无人机数=d["res"]["UAV_A"],
                    B型运输无人机数=d["res"]["UAV_B"],
                    C型运输无人机数=d["res"]["UAV_C"],
                    A型电池组数=d["res"]["BAT_A"],
                    B型电池组数=d["res"]["BAT_B"],
                    C型电池组数=d["res"]["BAT_C"],
                    中继无人机数=d["res"]["RELAY"],
                    中继能源组件数=d["res"]["ECOMP"]))
            print(f"  合计需求: {tot_need}")
            gaps, redund = {}, {}
            for r in inventory:
                gaps[r] = max(0, tot_need[r] - inventory[r])
                redund[r] = max(0, inventory[r] - tot_need[r])
            print(f"  相对库存冗余: {redund}")
            if any(gaps.values()):
                print(f"  ⚠ 资源缺口: {gaps}")
                print("  缺口原因: 分区后资源不可跨组调配，各组须独立覆盖"
                      "本组时间轴上的并发峰值，导致重复配置。")
            else:
                print("  无资源缺口。")
            report.setdefault(K, {})[scheme] = dict(
                assign=[sorted(d["areas"]) for d in detail],
                total_need=tot_need, gap=gaps, redundancy=redund,
                detail=detail)

    pd.DataFrame(rows).to_csv(os.path.join(OUT, "Q4_分区配置.csv"),
                              index=False, encoding="utf-8-sig")
    with open(os.path.join(OUT, "q4_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1, default=str)
    print(f"\n结果已保存到 {OUT}")


if __name__ == "__main__":
    main()
