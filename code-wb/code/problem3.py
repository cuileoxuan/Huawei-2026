# -*- coding: utf-8 -*-
"""
problem3.py —— 问题三：通信约束下的运输与中继联合调度

流程：
1. 继承问题二的运输方案（results/q2_solution.json，缺失则自动重跑问题二）。
2. 对每个运输架次重建三维轨迹（爬升/巡航/下降/投送悬停），按 dt 采样
   逐时刻判定与固定网关 G01 的直连链路（LOS + FSPL + 双向链路预算），
   提取“直连不可用”的通信缺口区间（时空通信需求区间）。
3. 在 DEM 范围内离散生成候选中继悬停点（水平网格 × 离地高度），
   对每个缺口区间预计算覆盖关系（接入链路 U↔R 与回传链路 R↔G01 同时可用）。
4. 候选中继任务生成（同点可覆盖的不重叠区间合并）+ 集合覆盖 MILP（COPT），
   目标：中继架次数最少，次级目标能耗最低；逐任务验算中继能量约束。
5. 中继资源排程（2 架中继无人机 + 6 组能源组件 + 两阶段充电周转）。
6. 输出：results/Q3_中继架次.csv、results/Q3_通信保障.csv、
         results/q3_solution.json（供问题四继承）。
"""
import os
import json
import math
import itertools
import numpy as np
import pandas as pd

from common import ProblemData, milp_binary_solve

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(OUT, exist_ok=True)

D = ProblemData()
DT = 5.0                    # 通信状态采样步长 s
GRID_STEP = 400.0           # 候选点水平网格步长 m
HOVER_HEIGHTS = [60, 120, 180, 240, 300]   # 候选离地高度 m（<= h_max）
GAP_MAX = 900.0             # 同一中继任务允许跨越的最大区间间隙 s

BOX = {b["id"]: b for b in D.boxes}
G_POS = D.gateway_pos()
LMAX_UG = D.lmax_ug()
LMAX_UR = D.lmax_ur()
LMAX_RG = D.lmax_rg()


# ------------------------------------------------------------ 轨迹重建
def sortie_trajectory(route):
    """
    将运输架次 route(q2_solution 格式) 重建为 (t, x, y, z) 轨迹点列。
    t 从架次开始（准备）计时；监测从起飞到返回降落。
    返回 (times, pos[N,3], 起飞机时刻)
    """
    m = D.models[route["model"]]
    stops = route["stops"]
    n_boxes = [len(sb) for sb in route["stop_boxes"]]
    nodes = ["O01"] + stops + ["O01"]
    t_now = m["t_prep"] + m["t_load"] * sum(n_boxes)
    t_takeoff = t_now
    pts = []          # (t, x, y, z)

    def add(t, x, y, z):
        pts.append((t, x, y, z))

    q_weights = [sum(BOX[b]["w"] for b in sb) for sb in route["stop_boxes"]]
    for k in range(len(nodes) - 1):
        i, j = nodes[k], nodes[k + 1]
        seg = D.segment(i, j)
        xi, yi = D.nodes[i]["xy"]; xj, yj = D.nodes[j]["xy"]
        ai, aj = D.work_alt[i], D.work_alt[j]
        hc = seg["h_cruise"]
        # 爬升（在 i 点垂直上升）
        t_up = seg["h_up"] / m["v_up"]
        for tt in np.arange(0, t_up, DT):
            z = ai + (hc - ai) * tt / t_up if t_up > 0 else hc
            add(t_now + tt, xi, yi, z)
        t_now += t_up
        # 巡航（水平移动，高度 hc）
        t_cr = seg["d"] / m["vc"]
        for tt in np.arange(0, t_cr, DT):
            x = xi + (xj - xi) * tt / t_cr if t_cr > 0 else xj
            y = yi + (yj - yi) * tt / t_cr if t_cr > 0 else yj
            add(t_now + tt, x, y, hc)
        t_now += t_cr
        # 下降（在 j 点垂直下降）
        t_dn = seg["h_dn"] / m["v_dn"]
        for tt in np.arange(0, t_dn, DT):
            z = hc - (hc - aj) * tt / t_dn if t_dn > 0 else aj
            add(t_now + tt, xj, yj, z)
        t_now += t_dn
        # 投送悬停
        if j != "O01":
            idx = stops.index(j)
            t_srv = m["t_hand"] + m["t_perbox"] * n_boxes[idx]
            for tt in np.arange(0, t_srv, DT):
                add(t_now + tt, xj, yj, aj)
            t_now += t_srv
    add(t_now, D.depot["xy"][0], D.depot["xy"][1], D.work_alt["O01"])
    arr = np.array([(p[1], p[2], p[3]) for p in pts])
    times = np.array([p[0] for p in pts])
    return times, arr, t_takeoff


def find_outages(route):
    """返回该架次直连不可用区间 [(t_s_rel, t_e_rel, positions)]（相对架次开始）。"""
    times, pos, t0 = sortie_trajectory(route)
    mask = times >= t0 - 1e-9
    times, pos = times[mask], pos[mask]
    # 先向量化 FSPL 带通判定，仅对不确定带做 LOS 检测
    d3 = np.linalg.norm(pos - np.array(G_POS)[None, :], axis=1) / 1000.0
    d3 = np.maximum(d3, 1e-6)
    lfspl = 32.45 + 20 * math.log10(D.comm["f"]) + 20 * np.log10(d3)
    ok = np.empty(len(pos), dtype=bool)
    fail = lfspl > LMAX_UG                       # 无遮挡也超门限
    pass_anyway = lfspl + D.comm["Lobs"] <= LMAX_UG   # 遮挡也可用
    ok[fail] = False
    ok[pass_anyway] = True
    uncertain = ~(fail | pass_anyway)
    for i in np.where(uncertain)[0]:
        ok[i] = not D.los_blocked(tuple(pos[i]), G_POS)
    intervals = []
    i = 0
    n = len(ok)
    while i < n:
        if not ok[i]:
            j = i
            while j + 1 < n and not ok[j + 1]:
                j += 1
            intervals.append((times[i], times[j], pos[i:j + 1]))
            i = j + 1
        else:
            i += 1
    return intervals


# ------------------------------------------------------------ 候选中继点
def candidate_points(out_pos_all):
    """以缺口位置包围盒为中心生成候选悬停点。"""
    lo = out_pos_all.min(axis=0)[:2] - 1500.0
    hi = out_pos_all.max(axis=0)[:2] + 1500.0
    pts = []
    xs = np.arange(lo[0], hi[0] + 1, GRID_STEP)
    ys = np.arange(lo[1], hi[1] + 1, GRID_STEP)
    for x in xs:
        for y in ys:
            z_ter = D.dem_elev(x, y)
            for h in HOVER_HEIGHTS:
                if h > D.relay["h_max"]:
                    continue
                pts.append((x, y, z_ter + h, h))
    return pts


def backhaul_filter(cands):
    """回传链路 R↔G01 可行性（每个候选点只算一次）。"""
    keep = []
    for (x, y, z, h) in cands:
        if D.link_ok((x, y, z), G_POS, LMAX_RG):
            keep.append((x, y, z, h))
    return keep


def point_covers_interval(p3, positions, max_check=15):
    """候选点是否覆盖某缺口区间：先 FSPL 粗筛，再 LOS 抽查。"""
    # 距离粗筛：与区间内所有采样点的最大三维距离
    d = np.linalg.norm(positions - np.array(p3)[None, :], axis=1) / 1000.0
    dmax = max(d.max(), 1e-6)
    lfspl = 32.45 + 20 * math.log10(D.comm["f"]) + 20 * math.log10(dmax)
    if lfspl > LMAX_UR:          # 无遮挡都超门限，必不可行
        return False
    if len(positions) > max_check:
        idx = np.linspace(0, len(positions) - 1, max_check).astype(int)
        samples = positions[idx]
    else:
        samples = positions
    for q in samples:
        if not D.link_ok(tuple(q), p3, LMAX_UR):
            return False
    return True


# ------------------------------------------------------------ 中继任务
def mission_from_point(pt, t_s, t_e):
    """由候选点与覆盖时段构造中继任务并验算能量。返回 None=不可行。"""
    x, y, z, h = pt
    r = D.relay
    t_out, E_out, _ = D.relay_fly(np.array([x, y]), z)
    t_back, E_back, _ = D.relay_fly(np.array([x, y]), z)   # 对称
    # 时间轴：开始(准备) -> 起飞 -> 飞抵 -> 建链 -> 服务(t_s..t_e) -> 返航
    # 可行性 1：t=0 立即出发也要能在 t_s 前完成建链
    if r["t_prep"] + t_out + r["t_link"] > t_s + 1e-9:
        return None
    T_hover = max(0.0, t_e - (t_s - r["t_link"]))
    E_hover = (r["P_hover"] + r["P_comm"]) * T_hover / 3600.0
    E_tot = E_out + E_back + E_hover
    # 可行性 2：返航安全余量
    if E_tot > (1 - r["rho"]) * r["Euse"] + 1e-9:
        return None
    start = max(0.0, t_s - r["t_link"] - t_out - r["t_prep"])
    ret = t_e + t_back
    return dict(point=(x, y, z), h_agl=h, t_s=t_s, t_e=t_e,
                start=start, ret=ret, T_hover=T_hover, E=E_tot,
                t_out=t_out, t_back=t_back,
                soc_end=1 - E_tot / r["Euse"])


from functools import lru_cache


@lru_cache(maxsize=None)
def _relay_fly_cached(xr, yr, zr):
    """按取整坐标缓存中继往返飞行时间/能耗。"""
    t, e, d = D.relay_fly(np.array([float(xr), float(yr)]), float(zr))
    return t, e


def mission_from_point_fast(pt, t_s, t_e):
    """带缓存的 mission_from_point（用取整坐标近似往返飞行）。"""
    x, y, z, h = pt
    r = D.relay
    t_out, E_out = _relay_fly_cached(round(x), round(y), round(z))
    t_back, E_back = t_out, E_out
    if r["t_prep"] + t_out + r["t_link"] > t_s + 1e-9:
        return None
    T_hover = max(0.0, t_e - (t_s - r["t_link"]))
    E_hover = (r["P_hover"] + r["P_comm"]) * T_hover / 3600.0
    E_tot = E_out + E_back + E_hover
    if E_tot > (1 - r["rho"]) * r["Euse"] + 1e-9:
        return None
    start = max(0.0, t_s - r["t_link"] - t_out - r["t_prep"])
    ret = t_e + t_back
    return dict(point=(x, y, z), h_agl=h, t_s=t_s, t_e=t_e,
                start=start, ret=ret, T_hover=T_hover, E=E_tot,
                t_out=t_out, t_back=t_back,
                soc_end=1 - E_tot / r["Euse"])


def plan_relays(intervals, feas_lists):
    """
    时空集合覆盖中继规划：
    1) 对同一悬停点，允许一架中继无人机同时服务多架运输无人机
       （仅要求每架运输无人机任一时刻至多一个通信提供方），
       将可连续覆盖的缺口区间合并为候选中继任务；
    2) 集合覆盖 MILP（COPT）：最少中继架次 + 低能耗；
    3) 上机排程：中继任务不可延迟（延迟会使缺口失去覆盖），
       放不下的任务拆成单区间任务重试，仍放不下记为未覆盖，
       交由外层联合迭代延迟相应运输架次。
    返回 (任务列表, 未覆盖区间编号列表, occ_u, occ_c)
    """
    r = D.relay
    uavs = D.relay_uavs
    comps = [f"E{i+1:02d}" for i in range(D.relay_batt["n"])]
    NI = len(intervals)

    # ---- 1. 候选中继任务生成 ----
    point_ivs = {}
    for i, fl in enumerate(feas_lists):
        for pt in fl:
            point_ivs.setdefault(pt, []).append(i)
    cands = []
    for pt, ivs in point_ivs.items():
        ivs = sorted(set(ivs), key=lambda i: intervals[i]["t_s"])

        def emit(group, pt=pt):
            if not group:
                return
            t_s = min(intervals[k]["t_s"] for k in group)
            t_e = max(intervals[k]["t_e"] for k in group)
            m = mission_from_point_fast(pt, t_s, t_e)
            if m is not None:
                m["cover"] = list(group)
                cands.append(m)

        group = []
        for i in ivs:
            trial = group + [i]
            t_s = min(intervals[k]["t_s"] for k in trial)
            t_e = max(intervals[k]["t_e"] for k in trial)
            if mission_from_point_fast(pt, t_s, t_e) is not None:
                group = trial
            else:
                emit(group)
                group = [i]
        emit(group)

    # ---- 2. 集合覆盖 MILP ----
    NM = len(cands)
    A = np.zeros((NI, NM))
    for k, m in enumerate(cands):
        for i in m["cover"]:
            A[i, k] = 1.0
    cost = np.array([1.0 + 0.002 * m["E"] + 0.0002 * (m["t_e"] - m["t_s"])
                     for m in cands])
    x = milp_binary_solve(cost, A_ub=-A, b_ub=-np.ones(NI), timelimit=60)
    if x is None:                       # 回退贪心集合覆盖
        chosen_idx, covered = [], set()
        order = sorted(range(NM),
                       key=lambda k: (-len(cands[k]["cover"]), cands[k]["E"]))
        for k in order:
            new = [i for i in cands[k]["cover"] if i not in covered]
            if new:
                chosen_idx.append(k)
                covered.update(new)
            if len(covered) == NI:
                break
        x = np.zeros(NM)
        x[chosen_idx] = 1
    chosen = [cands[k] for k in range(NM) if x[k] > 0.5]

    # ---- 3. 上机排程 ----
    occ_u = {u: [] for u in uavs}
    occ_c = {c: [] for c in comps}
    final, uncovered = [], set()

    def try_place(m):
        chg = D.charge_time(m["soc_end"], D.relay_batt["Tfull"])
        for u in uavs:
            if not all(m["ret"] + r["t_turn"] <= s or m["start"] >= e
                       for s, e in occ_u[u]):
                continue
            for cid in comps:
                if not all(m["ret"] + chg <= s or m["start"] >= e
                           for s, e in occ_c[cid]):
                    continue
                m.update(uav=u, comp=cid, start_act=m["start"],
                         t_e_act=m["t_e"], ret_act=m["ret"])
                occ_u[u].append((m["start"], m["ret"] + r["t_turn"]))
                occ_c[cid].append((m["start"], m["ret"] + chg))
                final.append(m)
                return True
        return False

    covered = set()
    for m in sorted(chosen, key=lambda mm: mm["start"]):
        new_cover = [i for i in m["cover"] if i not in covered]
        if not new_cover:
            continue                        # 完全冗余，直接弃用以释放资源
        m["cover"] = new_cover
        if try_place(m):
            covered.update(new_cover)
            continue
        # 拆成单区间任务逐个尝试
        for i in m["cover"]:
            itv = intervals[i]
            pt = (m["point"][0], m["point"][1], m["point"][2], m["h_agl"])
            mm = mission_from_point_fast(pt, itv["t_s"], itv["t_e"])
            if mm is not None and try_place(mm):
                mm["cover"] = [i]
                covered.add(i)
            else:
                uncovered.add(i)
    final.sort(key=lambda m: m["start"])
    return final, sorted(uncovered), occ_u, occ_c


# ------------------------------------------------------------ 联合调整
def _fit_delta(occ, s, e, d0=0.0):
    """求最小 d>=d0 使窗口 [s+d, e+d] 与占用区间 occ 不冲突。"""
    d = d0
    for _ in range(100):
        moved = False
        for s0, e0 in sorted(occ):
            if s + d < e0 - 1e-9 and e + d > s0 + 1e-9:
                d = e0 - s
                moved = True
        if not moved:
            return d
    return d


def _joint_delta(occ_u, occ_c, m, chg):
    """中继无人机与能源组件同时可用的最小延迟。"""
    r = D.relay
    d = 0.0
    for _ in range(50):
        du = _fit_delta(occ_u, m["start"], m["ret"] + r["t_turn"], d)
        dc = _fit_delta(occ_c, m["start"], m["ret"] + chg, d)
        dn = max(du, dc)
        if dn <= d + 1e-9:
            return d
        d = dn
    return d


def shift_route(rt, d):
    rt["start"] += d
    rt["end"] += d
    for k in rt["deliver"]:
        rt["deliver"][k] += d


def delay_sortie_cascade(routes, sortie_id, delta):
    """推迟指定架次，并沿同无人机/同电池链级联推迟后续架次。"""
    by_id = {r["sortie"]: r for r in routes}
    shift_route(by_id[sortie_id], delta)
    for _ in range(100):
        moved = False
        rs = sorted(routes, key=lambda x: x["start"])
        for i in range(len(rs)):
            for j in range(i + 1, len(rs)):
                a, b = rs[i], rs[j]
                if a["uav"] == b["uav"] and b["start"] < a["end"] - 1e-9:
                    shift_route(b, a["end"] - b["start"])
                    moved = True
                if a["batt"] == b["batt"]:
                    chg = D.charge_time(
                        1 - a["E"] / D.models[a["model"]]["Euse"],
                        D.batteries[a["model"]]["Tfull"])
                    need = a["end"] + chg
                    if b["start"] < need - 1e-9:
                        shift_route(b, need - b["start"])
                        moved = True
        if not moved:
            break


# ------------------------------------------------------------ 主流程
def load_q2():
    fp = os.path.join(OUT, "q2_solution.json")
    if not os.path.exists(fp):
        import problem2
        problem2.main()
    return json.load(open(fp, encoding="utf-8"))


def main():
    print("=" * 70)
    print("问题三：通信约束下的运输与中继联合调度")
    print("=" * 70)
    q2 = load_q2()
    routes = q2["routes"]

    # ---- 1. 直连链路扫描，提取通信缺口区间（相对时刻，延迟时整体平移）----
    print("\n[1] 运输轨迹通信扫描 (dt=5s)")
    intervals = []
    for rt in routes:
        ivs = find_outages(rt)
        for (t_s, t_e, pos) in ivs:
            if t_e - t_s < DT:      # 忽略瞬时抖动
                continue
            intervals.append(dict(sortie=rt["sortie"], rel_s=t_s, rel_e=t_e,
                                  t_s=rt["start"] + t_s, t_e=rt["start"] + t_e,
                                  positions=pos))
        if ivs:
            tot = sum(t_e - t_s for t_s, t_e, _ in ivs)
            print(f"  {rt['sortie']} ({'->'.join(rt['stops'])}): "
                  f"{len(ivs)} 个缺口区间, 累计 {tot:.0f}s")
    print(f"  共 {len(intervals)} 个中继通信需求区间")

    rows_comm = []
    delay_log = []
    relay_rows = []
    if not intervals:
        print("  全部架次全程直连 G01，无需中继。")
    else:
        # ---- 2. 候选点与覆盖关系（位置不随延迟变化，只算一次）----
        print("\n[2] 生成候选中继悬停点并预计算覆盖关系")
        all_pos = np.vstack([itv["positions"] for itv in intervals])
        cands = candidate_points(all_pos)
        cands = backhaul_filter(cands)
        print(f"  回传可行候选点: {len(cands)} 个")
        feas_lists = []
        for itv in intervals:
            feas_lists.append([
                pt for pt in cands
                if point_covers_interval((pt[0], pt[1], pt[2]),
                                         itv["positions"])])
        print(f"  各区间接入可行点数: "
              f"min={min(len(f) for f in feas_lists)}, "
              f"max={max(len(f) for f in feas_lists)}")

        # ---- 3. 中继规划 + 运输—通信联合迭代（延迟不可覆盖架次）----
        print("\n[3] 资源感知贪心中继规划 + 运输—通信联合迭代")
        chosen, uncovered = [], list(range(len(intervals)))
        uavs = D.relay_uavs
        comps = [f"E{i+1:02d}" for i in range(D.relay_batt["n"])]
        for it in range(12):
            # 同步区间绝对时刻
            st = {r["sortie"]: r["start"] for r in routes}
            for itv in intervals:
                itv["t_s"] = st[itv["sortie"]] + itv["rel_s"]
                itv["t_e"] = st[itv["sortie"]] + itv["rel_e"]
            chosen, uncovered, occ_u, occ_c = plan_relays(intervals,
                                                          feas_lists)
            print(f"  第 {it+1} 轮: 中继任务 {len(chosen)} 个, "
                  f"未覆盖区间 {len(uncovered)} 个")
            if not uncovered:
                break
            # 为每个未覆盖区间求“最小延迟使其可排”的架次延迟量
            # （若延迟会导致医疗/首批箱超时，加大惩罚以保护硬时限）
            BOX = {b["id"]: b for b in D.boxes}

            def hard_penalty(sortie_id, d):
                rt = next(r for r in routes if r["sortie"] == sortie_id)
                pen = 0
                for st, sb in zip(rt["stops"], rt["stop_boxes"]):
                    for b in sb:
                        bi = BOX[b]
                        ddl = bi["t_first"] if bi["first"] else (
                            bi["t_due"] if bi["mtype"] == "医疗物资" else None)
                        if ddl is not None and rt["deliver"][st] + d > ddl:
                            pen += 1e6
                return pen

            best = None
            for idx in uncovered:
                itv = intervals[idx]
                for pt in feas_lists[idx]:
                    m = mission_from_point_fast(pt, itv["t_s"], itv["t_e"])
                    if m is None:
                        continue
                    chg = D.charge_time(m["soc_end"], D.relay_batt["Tfull"])
                    for u in uavs:
                        for cid in comps:
                            d = _joint_delta(occ_u[u], occ_c[cid], m, chg)
                            key = d + hard_penalty(itv["sortie"], d)
                            if best is None or key < best[0]:
                                best = (key, d, idx)
            if best is None or best[1] > 1e9:
                print("  ⚠ 无法通过延迟解决，请增加中继资源")
                break
            _, d, idx = best
            sid = intervals[idx]["sortie"]
            print(f"  联合调整: 架次 {sid} 推迟 {d:.0f}s "
                  f"(区间 [{intervals[idx]['t_s']:.0f},"
                  f"{intervals[idx]['t_e']:.0f}])")
            delay_sortie_cascade(routes, sid, d)
            delay_log.append((sid, d))
        if uncovered:
            print(f"  ⚠ 仍有 {len(uncovered)} 个区间未能安排中继，"
                  f"已保留为未覆盖状态")
        E_relay = sum(m["E"] for m in chosen)
        if delay_log:
            tot_d = sum(d for _, d in delay_log)
            print(f"  运输架次联合调整 {len(delay_log)} 次, 累计推迟 {tot_d:.0f}s")
        cmax_joint = max(max(r["end"] for r in routes),
                         max(m["ret_act"] for m in chosen))
        print(f"  中继总能耗: {E_relay:.3f} kWh")
        print(f"  联合任务完成时间: {cmax_joint:.0f} s "
              f"(运输 {max(r['end'] for r in routes):.0f}s)")
        for i, m in enumerate(chosen, 1):
            rid = f"R-{i:03d}"
            m["rid"] = rid
            lon = D.depot["lon"] + math.degrees(
                m["point"][0] / (6371000.0 * D._cos0))
            lat = D.depot["lat"] + math.degrees(m["point"][1] / 6371000.0)
            t_linked = m["start_act"] + D.relay["t_prep"] + m["t_out"] \
                + D.relay["t_link"]
            relay_rows.append(dict(
                中继架次编号=rid, 中继无人机编号=m["uav"], 能源组件编号=m["comp"],
                开始时刻s=round(m["start_act"], 1), 悬停经度=round(lon, 7),
                悬停纬度=round(lat, 7), 悬停海拔m=round(m["point"][2], 1),
                建链完成时刻s=round(t_linked, 1),
                服务结束时刻s=round(m["t_e_act"], 1),
                返回O01时刻s=round(m["ret_act"], 1),
                架次能耗kWh=round(m["E"], 4)))
            print(f"  {rid}: {m['uav']}/{m['comp']} 悬停({lon:.5f},{lat:.5f},"
                  f"{m['point'][2]:.0f}m) 服务 {m['t_s']:.0f}-{m['t_e_act']:.0f}s, "
                  f"E={m['E']:.3f}kWh")

        # 区间 -> 中继架次映射
        itv_relay = {}
        for m in chosen:
            for i in m["cover"]:
                itv_relay[i] = m["rid"]

        # ---- 5. 通信保障明细（按架次汇总直连/中继阶段）----
        iv_by_sortie = {}
        for i, itv in enumerate(intervals):
            iv_by_sortie.setdefault(itv["sortie"], []).append((i, itv))
        for rt in routes:
            sid = rt["sortie"]
            ivs = sorted(iv_by_sortie.get(sid, []), key=lambda x: x[1]["t_s"])
            t0 = rt["start"]
            t1 = rt["end"]
            cur = t0
            if not ivs:
                rows_comm.append(dict(运输架次编号=sid, 通信阶段="全程",
                                      开始时刻s=round(t0, 1), 结束时刻s=round(t1, 1),
                                      保障方式="G01直连", 中继架次编号=""))
            for i, itv in ivs:
                if itv["t_s"] > cur + 1e-6:
                    rows_comm.append(dict(运输架次编号=sid, 通信阶段="飞行/投送",
                                          开始时刻s=round(cur, 1),
                                          结束时刻s=round(itv["t_s"], 1),
                                          保障方式="G01直连", 中继架次编号=""))
                rows_comm.append(dict(运输架次编号=sid, 通信阶段="飞行/投送",
                                      开始时刻s=round(itv["t_s"], 1),
                                      结束时刻s=round(itv["t_e"], 1),
                                      保障方式="中继",
                                      中继架次编号=itv_relay.get(i, "")))
                cur = itv["t_e"]
            if ivs and cur < t1 - 1e-6:
                rows_comm.append(dict(运输架次编号=sid, 通信阶段="飞行/投送",
                                      开始时刻s=round(cur, 1), 结束时刻s=round(t1, 1),
                                      保障方式="G01直连", 中继架次编号=""))

        with open(os.path.join(OUT, "q3_solution.json"), "w", encoding="utf-8") as f:
            json.dump(dict(routes=routes, relays=[{
                **{k: v for k, v in m.items() if k != "cover"},
                "cover": [intervals[i]["sortie"] for i in m["cover"]]}
                for m in chosen],
                intervals=[dict(sortie=itv["sortie"], t_s=itv["t_s"],
                                t_e=itv["t_e"]) for itv in intervals]),
                f, ensure_ascii=False, indent=1)

    df_r = pd.DataFrame(relay_rows)
    df_c = pd.DataFrame(rows_comm)
    df_r.to_csv(os.path.join(OUT, "Q3_中继架次.csv"),
                index=False, encoding="utf-8-sig")
    df_c.to_csv(os.path.join(OUT, "Q3_通信保障.csv"),
                index=False, encoding="utf-8-sig")
    print(f"\n结果已保存到 {OUT}")


if __name__ == "__main__":
    main()
