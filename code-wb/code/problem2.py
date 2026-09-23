# -*- coding: utf-8 -*-
"""
problem2.py —— 问题二：异构无人机多点多架次运输调度

求解框架：
1. 路线层：贪心构造 + ALNS 大邻域搜索（destroy/repair + 2-opt + 机型替换），
   每个架次可访问多个服务区（默认不超过 MAX_STOPS 个），货箱不可拆分，
   载荷/体积/返航安全余量逐架次精确验算（能耗按剩余载荷逐航段动态计算）。
2. 资源层：事件驱动贪心排程器，把架次分配到 8 架实体无人机（按机型匹配）
   与共享电池池（SOC + 两阶段充电周转），确定各架次开始时刻。
3. 目标（标量化的字典序）：
   硬约束违约（医疗/首批超时） >> 加权配送 tardiness >> 全部任务完成时间
   >> 运输能耗 >> 架次数。

输出：results/Q2_运输架次.csv、results/Q2_逐箱交付.csv，
      results/q2_solution.json（供问题三继承）。
"""
import os
import json
import math
import random
import numpy as np
import pandas as pd

from common import ProblemData

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(OUT, exist_ok=True)

D = ProblemData()
MODELS = ["A", "B", "C"]
MAX_STOPS = 3                 # 单个架次最多访问服务区数
random.seed(2026)

# 目标权重（标量化字典序）
W_HARD = 1e9        # 医疗/首批箱超时（按箱）
W_LATE = 1e4        # 医疗/首批箱超时秒数
W_TARD = 1.0        # 普通箱 加权tardiness = prio * 迟到秒数
W_CMAX = 5.0        # 全部任务完成时间
W_ENER = 500.0      # 总能耗 kWh
W_SORT = 2000.0     # 架次数

BOX = {b["id"]: b for b in D.boxes}
BOXIDS = list(BOX.keys())
AREA_BOXES = {}
for b in D.boxes:
    AREA_BOXES.setdefault(b["area"], []).append(b["id"])

BATT_IDS = {g: [f"B{g}{i+1:02d}" for i in range(D.batteries[g]["n"])] for g in MODELS}
UAV_BY_MODEL = {g: [u for u, gg in D.uavs.items() if gg == g] for g in MODELS}

_eval_cache = {}


def route_eval(g, stops, stop_boxes):
    """带缓存的架次评估（物理可行性 + 时间/能耗）。"""
    key = (g, tuple(stops),
           tuple(tuple(sorted(sb)) for sb in stop_boxes))
    if key not in _eval_cache:
        _eval_cache[key] = D.eval_sortie(g, stops, stop_boxes)
    return _eval_cache[key]


def route_feasible(g, stops, stop_boxes):
    m = D.models[g]
    w = sum(BOX[b]["w"] for sb in stop_boxes for b in sb)
    v = sum(BOX[b]["v"] for sb in stop_boxes for b in sb)
    if w > m["Q"] + 1e-9 or v > m["V"] + 1e-9:
        return None
    ev = route_eval(g, stops, stop_boxes)
    return ev if ev["feasible"] else None


def best_model(stops, stop_boxes):
    """为给定 stops/装载选择能耗最小的可行机型。"""
    best = None
    for g in MODELS:
        ev = route_feasible(g, stops, stop_boxes)
        if ev and (best is None or ev["E"] < best["E"]):
            best = ev
    return best


# ------------------------------------------------------------ 资源排程器
def schedule(routes):
    """
    routes: list of dict(model, stops, stop_boxes, eval)
    返回 (可行, 目标分量字典, 排程明细)
    贪心规则：按架次内最早货箱时限排序，依次分配“最早可用的同型无人机 +
    最早可用的同型电池”。
    """
    def hard_ddl(rt):
        """架次内最紧急的硬时限（首批截止/医疗期望），无硬时限箱则为 inf。"""
        ds = []
        for sb in rt["stop_boxes"]:
            for b in sb:
                bi = BOX[b]
                if bi["first"]:
                    ds.append(bi["t_first"])
                elif bi["mtype"] == "医疗物资":
                    ds.append(bi["t_due"])
        return min(ds) if ds else float("inf")

    order = sorted(range(len(routes)),
                   key=lambda r: (hard_ddl(routes[r]),
                                  min(BOX[b]["t_due"]
                                      for sb in routes[r]["stop_boxes"]
                                      for b in sb)))
    uav_avail = {u: 0.0 for u in D.uavs}
    batt_avail = {bid: 0.0 for g in MODELS for bid in BATT_IDS[g]}
    assign = [None] * len(routes)

    for r in order:
        rt = routes[r]
        g = rt["model"]
        ev = rt["eval"]
        best = None
        for u in UAV_BY_MODEL[g]:
            for bid in BATT_IDS[g]:
                s = max(uav_avail[u], batt_avail[bid])
                if best is None or s < best[0]:
                    best = (s, u, bid)
        s, u, bid = best
        end = s + ev["T"]
        soc = ev["soc_end"]
        chg = D.charge_time(soc, D.batteries[g]["Tfull"])
        uav_avail[u] = end
        batt_avail[bid] = end + chg
        deliver = {st: s + ev["depart"][st] for st in rt["stops"]}
        assign[r] = dict(uav=u, batt=bid, start=s, end=end, deliver=deliver)

    # ---- 目标分量 ----
    hard_cnt, hard_sec, tard, cmax, etot = 0, 0.0, 0.0, 0.0, 0.0
    for r, rt in enumerate(routes):
        a = assign[r]
        cmax = max(cmax, a["end"])
        etot += rt["eval"]["E"]
        for st, sb in zip(rt["stops"], rt["stop_boxes"]):
            for b in sb:
                bi = BOX[b]
                t = a["deliver"][st]
                late = t - bi["t_due"]
                if bi["mtype"] == "医疗物资" or bi["first"]:
                    ddl = bi["t_first"] if bi["first"] else bi["t_due"]
                    if t > ddl + 1e-6:
                        hard_cnt += 1
                        hard_sec += t - ddl
                elif late > 0:
                    tard += bi["prio"] * late
    obj = (W_HARD * hard_cnt + W_LATE * hard_sec + W_TARD * tard
           + W_CMAX * cmax + W_ENER * etot + W_SORT * len(routes))
    comp = dict(obj=obj, hard=hard_cnt, hard_sec=hard_sec, tard=tard,
                cmax=cmax, E=etot, n=len(routes))
    return comp, assign


# ------------------------------------------------------------ 初始解构造
def greedy_construct():
    """按 (期望送达时间, 优先系数) 排序逐箱插入。"""
    routes = []          # dict(model, stops, stop_boxes)
    order = sorted(BOXIDS, key=lambda b: (BOX[b]["t_due"], -BOX[b]["prio"]))
    for b in order:
        area = BOX[b]["area"]
        best = None     # (deltaE, route_idx, new_stops, new_sb, ev)
        for ri, rt in enumerate(routes):
            for pos in range(len(rt["stops"]) + 1):
                if pos < len(rt["stops"]) and rt["stops"][pos] == area:
                    ns, nsb = list(rt["stops"]), [list(x) for x in rt["stop_boxes"]]
                    nsb[pos].append(b)
                else:
                    if len(rt["stops"]) >= MAX_STOPS or area in rt["stops"]:
                        continue
                    ns = list(rt["stops"]); ns.insert(pos, area)
                    nsb = [list(x) for x in rt["stop_boxes"]]; nsb.insert(pos, [b])
                ev = route_feasible(rt["model"], ns, nsb)
                if ev is None:
                    continue
                dE = ev["E"] - rt["eval"]["E"]
                if best is None or dE < best[0]:
                    best = (dE, ri, ns, nsb, ev)
        if best:
            _, ri, ns, nsb, ev = best
            routes[ri].update(stops=ns, stop_boxes=nsb, eval=ev)
        else:
            ev = best_model([area], [[b]])
            if ev is None:
                raise RuntimeError(f"货箱 {b} 单箱也无法运输，请检查数据")
            routes.append(dict(model=ev["model"], stops=[area],
                               stop_boxes=[[b]], eval=ev))
    # 机型再优化
    for rt in routes:
        ev = best_model(rt["stops"], rt["stop_boxes"])
        if ev and ev["E"] < rt["eval"]["E"] - 1e-9:
            rt.update(model=ev["model"], eval=ev)
    return routes


# ------------------------------------------------------------ ALNS
def copy_routes(routes):
    return [dict(model=rt["model"], stops=list(rt["stops"]),
                 stop_boxes=[list(x) for x in rt["stop_boxes"]], eval=rt["eval"])
            for rt in routes]


def two_opt(rt):
    """单架次内服务区访问顺序优化（全排列，站点数 <= MAX_STOPS）。"""
    import itertools
    best = rt["eval"]
    best_perm = rt["stops"]
    for perm in itertools.permutations(range(len(rt["stops"]))):
        ns = [rt["stops"][k] for k in perm]
        nsb = [rt["stop_boxes"][k] for k in perm]
        ev = route_feasible(rt["model"], ns, nsb)
        if ev and ev["E"] < best["E"] - 1e-9:
            best, best_perm = ev, list(ns)
    if best_perm != rt["stops"]:
        idx = [rt["stops"].index(s) for s in best_perm]
        rt["stops"] = best_perm
        rt["stop_boxes"] = [rt["stop_boxes"][k] for k in idx]
        rt["eval"] = best


def destroy(routes, q):
    removed = []
    rs = copy_routes(routes)
    all_boxes = [(ri, b) for ri, rt in enumerate(rs)
                 for sb in rt["stop_boxes"] for b in sb]
    random.shuffle(all_boxes)
    for ri, b in all_boxes[:q]:
        removed.append(b)
        for si, sb in enumerate(rs[ri]["stop_boxes"]):
            if b in sb:
                sb.remove(b)
                break
    # 清理空站点并保证 stops 与 stop_boxes 对齐
    new_rs = []
    for rt in rs:
        pairs = [(s, sb) for s, sb in zip(rt["stops"], rt["stop_boxes"]) if sb]
        if pairs:
            rt["stops"] = [p[0] for p in pairs]
            rt["stop_boxes"] = [p[1] for p in pairs]
            ev = best_model(rt["stops"], rt["stop_boxes"])
            rt.update(model=ev["model"], eval=ev)
            new_rs.append(rt)
    return new_rs, removed


def repair(rs, removed):
    """regret 插入修复。"""
    random.shuffle(removed)
    removed.sort(key=lambda b: (BOX[b]["t_due"], -BOX[b]["prio"]))
    for b in removed:
        area = BOX[b]["area"]
        cand = []
        for ri, rt in enumerate(rs):
            for pos in range(len(rt["stops"]) + 1):
                if pos < len(rt["stops"]) and rt["stops"][pos] == area:
                    ns, nsb = list(rt["stops"]), [list(x) for x in rt["stop_boxes"]]
                    nsb[pos].append(b)
                else:
                    if len(rt["stops"]) >= MAX_STOPS or area in rt["stops"]:
                        continue
                    ns = list(rt["stops"]); ns.insert(pos, area)
                    nsb = [list(x) for x in rt["stop_boxes"]]; nsb.insert(pos, [b])
                for g in MODELS:
                    ev = route_feasible(g, ns, nsb)
                    if ev:
                        cand.append((ev["E"] - rt["eval"]["E"], ri, ns, nsb, ev))
        if cand:
            cand.sort(key=lambda x: x[0])
            _, ri, ns, nsb, ev = cand[0]
            rs[ri].update(model=ev["model"], stops=ns, stop_boxes=nsb, eval=ev)
        else:
            ev = best_model([area], [[b]])
            rs.append(dict(model=ev["model"], stops=[area],
                           stop_boxes=[[b]], eval=ev))
    return rs


def destroy_worst(routes, assign, q):
    """针对当前解中造成硬超时/高 tardiness 的货箱进行移除。"""
    pen = {}
    for rt, a in zip(routes, assign):
        for st, sb in zip(rt["stops"], rt["stop_boxes"]):
            for b in sb:
                bi = BOX[b]
                t = a["deliver"][st]
                p = 0.0
                if bi["first"] and t > bi["t_first"]:
                    p += 1e6 + (t - bi["t_first"])
                elif bi["mtype"] == "医疗物资" and t > bi["t_due"]:
                    p += 1e6 + (t - bi["t_due"])
                elif t > bi["t_due"]:
                    p += bi["prio"] * (t - bi["t_due"])
                pen[b] = p + random.random() * 100
    worst = sorted(pen, key=pen.get, reverse=True)[:q]
    rs = copy_routes(routes)
    for b in worst:
        for rt in rs:
            hit = False
            for sb in rt["stop_boxes"]:
                if b in sb:
                    sb.remove(b); hit = True; break
            if hit:
                break
    new_rs = []
    for rt in rs:
        pairs = [(s, sb) for s, sb in zip(rt["stops"], rt["stop_boxes"]) if sb]
        if pairs:
            rt["stops"] = [p[0] for p in pairs]
            rt["stop_boxes"] = [p[1] for p in pairs]
            ev = best_model(rt["stops"], rt["stop_boxes"])
            rt.update(model=ev["model"], eval=ev)
            new_rs.append(rt)
    return new_rs, worst


def alns(iters=1500, q_max=8, seed=2026, verbose=True):
    random.seed(seed)
    cur = greedy_construct()
    cur_comp, cur_assign = schedule(cur)
    best, best_comp = copy_routes(cur), cur_comp
    if verbose:
        print(f"  初始解: obj={cur_comp['obj']:.3e}, 架次={cur_comp['n']}, "
              f"硬违约={cur_comp['hard']}, Cmax={cur_comp['cmax']:.0f}s, "
              f"E={cur_comp['E']:.2f}kWh")
    T0, alpha = 1e6, 0.995
    for it in range(iters):
        q = random.randint(2, q_max)
        if random.random() < 0.5:
            rs, removed = destroy_worst(cur, cur_assign, q)
        else:
            rs, removed = destroy(cur, q)
        rs = repair(rs, removed)
        for rt in rs:
            if len(rt["stops"]) > 1:
                two_opt(rt)
        comp, assign = schedule(rs)
        T = T0 * alpha ** it
        if comp["obj"] <= cur_comp["obj"] or \
                random.random() < math.exp(-(comp["obj"] - cur_comp["obj"]) / max(T, 1)):
            cur, cur_comp, cur_assign = rs, comp, assign
            if comp["obj"] < best_comp["obj"]:
                best, best_comp = copy_routes(rs), comp
        if verbose and it % 500 == 499:
            print(f"    iter {it+1}: best_obj={best_comp['obj']:.3e}, "
                  f"架次={best_comp['n']}, 硬违约={best_comp['hard']}, "
                  f"Cmax={best_comp['cmax']:.0f}s, E={best_comp['E']:.2f}kWh")
    return best, best_comp


# ------------------------------------------------------------ 主流程
def main(iters=1500, seeds=(2026, 7, 42)):
    print("=" * 70)
    print("问题二：异构无人机多点多架次运输调度（ALNS + 资源排程）")
    print("=" * 70)
    best, best_comp = None, None
    for sd in seeds:
        print(f"--- ALNS seed={sd} ---")
        b, c = alns(iters=iters, seed=sd, verbose=(sd == seeds[0]))
        if best is None or c["obj"] < best_comp["obj"]:
            best, best_comp = b, c
        print(f"  seed={sd} 最优: obj={c['obj']:.3e}, 架次={c['n']}, "
              f"硬违约={c['hard']}, Cmax={c['cmax']:.0f}s, E={c['E']:.2f}kWh")
    comp, assign = schedule(best)

    print("\n[最终结果]")
    print(f"  架次数: {comp['n']}")
    print(f"  医疗/首批超时箱数: {comp['hard']} (超时 {comp['hard_sec']:.0f}s)")
    print(f"  普通箱加权 tardiness: {comp['tard']:.0f}")
    print(f"  全部任务完成时间 Cmax: {comp['cmax']:.0f} s")
    print(f"  总运输能耗: {comp['E']:.3f} kWh")

    rows_s, rows_b, sol = [], [], []
    for i, (rt, a) in enumerate(zip(best, assign), 1):
        sid = f"P2-{i:03d}"
        rows_s.append(dict(
            架次编号=sid, 无人机编号=a["uav"], 机型编号=rt["model"],
            电池编号=a["batt"], 开始时刻s=round(a["start"], 1),
            访问服务区顺序="->".join(["O01"] + rt["stops"] + ["O01"]),
            返回O01时刻s=round(a["end"], 1),
            架次能耗kWh=round(rt["eval"]["E"], 4)))
        for st, sb in zip(rt["stops"], rt["stop_boxes"]):
            for b in sb:
                rows_b.append(dict(货箱编号=b, 架次编号=sid, 服务区编号=st,
                                   交付完成时刻s=round(a["deliver"][st], 1)))
        sol.append(dict(sortie=sid, model=rt["model"], uav=a["uav"],
                        batt=a["batt"], start=a["start"], end=a["end"],
                        stops=rt["stops"], stop_boxes=rt["stop_boxes"],
                        E=rt["eval"]["E"], T=rt["eval"]["T"],
                        deliver={k: v for k, v in a["deliver"].items()}))
    df_s = pd.DataFrame(rows_s)
    df_b = pd.DataFrame(rows_b).sort_values("交付完成时刻s")
    df_s.to_csv(os.path.join(OUT, "Q2_运输架次.csv"),
                index=False, encoding="utf-8-sig")
    df_b.to_csv(os.path.join(OUT, "Q2_逐箱交付.csv"),
                index=False, encoding="utf-8-sig")
    with open(os.path.join(OUT, "q2_solution.json"), "w", encoding="utf-8") as f:
        json.dump(dict(routes=sol, comp=comp), f, ensure_ascii=False, indent=1)

    print("\n[资源可行性检验]")
    # 无人机时间轴不重叠
    ok = True
    for u in D.uavs:
        iv = sorted([(a["start"], a["end"]) for rt, a in zip(best, assign)
                     if a["uav"] == u])
        for k in range(1, len(iv)):
            if iv[k][0] < iv[k - 1][1] - 1e-6:
                ok = False
                print(f"  ✗ 无人机 {u} 时间轴重叠")
    # 电池占用+充电不重叠，SOC 不低于返航下限
    for g in MODELS:
        for bid in BATT_IDS[g]:
            iv = []
            for rt, a in zip(best, assign):
                if a["batt"] == bid:
                    chg = D.charge_time(rt["eval"]["soc_end"],
                                        D.batteries[g]["Tfull"])
                    iv.append((a["start"], a["end"] + chg))
            iv.sort()
            for k in range(1, len(iv)):
                if iv[k][0] < iv[k - 1][1] - 1e-6:
                    ok = False
                    print(f"  ✗ 电池 {bid} 占用/充电重叠")
            for rt, a in zip(best, assign):
                if a["batt"] == bid and rt["eval"]["soc_end"] < D.models[g]["rho"] - 1e-9:
                    ok = False
                    print(f"  ✗ 电池 {bid} 返航 SOC 低于下限")
    print("  ✓ 全部资源可行性检验通过" if ok else "  存在违约，请检查")
    print(f"\n结果已保存到 {OUT}")
    return best, comp, assign


if __name__ == "__main__":
    main()
