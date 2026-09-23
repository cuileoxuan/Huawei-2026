# -*- coding: utf-8 -*-
"""问题三方案独立核验（全分辨率 DT=1 s，仅依赖本目录脚本）。

核验内容：
1. 每个缺口区间是否被某个中继架次**全程**覆盖：
   - 接入链路 运输无人机 <-> 中继（门限 LMAX_UR）
   - 回传链路 中继 <-> G01（门限 LMAX_RG）
   并输出各链路的最小裕量（Lmax - Lpath）。
2. 中继任务的建链时限与能量约束（含返航安全余量）。

用法：python _verify_q3_link.py [results/q3_solution.json]
"""
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import problem3 as P            # noqa: E402
from common import G_GRAV, KWH  # noqa: E402

DT_VERIFY = 1.0                 # 核验采样步长 s（独立于 problem3 的 DT）
NO_TERRAIN_CLEARANCE = 0.0      # 与 los_blocked 口径一致


def path_loss(p1, p2, f_mhz):
    """总传播损耗 = FSPL + 遮挡附加损耗（LOS 被地形遮挡时 +Lobs）。"""
    d_km = max(math.dist(tuple(p1), tuple(p2)) / 1000.0, 1e-6)
    lfspl = 32.45 + 20 * math.log10(f_mhz) + 20 * math.log10(d_km)
    blocked = P.D.los_blocked(tuple(p1), tuple(p2), clearance=NO_TERRAIN_CLEARANCE)
    return lfspl + (P.D.comm["Lobs"] if blocked else 0.0), blocked


def main():
    fp = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "results",
                                                            "q3_solution.json")
    sol = json.load(open(fp, encoding="utf-8"))
    D, f = P.D, P.D.comm["f"]
    routes = {r["sortie"]: r for r in sol["routes"]}
    relays = sol["relays"]
    intervals = sol["intervals"]

    # 以 1 s 重建全部架次轨迹（核验分辨率独立于求解分辨率）
    old_dt = P.DT
    P.DT = DT_VERIFY
    try:
        traj = {}
        for sid, rt in routes.items():
            t, pos, t0 = P.sortie_trajectory(rt)
            m = t >= t0 - 1e-9
            traj[sid] = (rt["start"] + t[m], pos[m])
    finally:
        P.DT = old_dt

    print("=" * 78)
    print("问题三链路核验：DT=%.0f s 重建轨迹，接入/回传双链路全分辨率判定" % DT_VERIFY)
    print("门限: U<->G01 %.0f dB, U<->R %.0f dB, R<->G01 %.0f dB"
          % (P.LMAX_UG, P.LMAX_UR, P.LMAX_RG))
    print("=" * 78)

    bad = 0
    worst_ur = worst_rg = None
    for m in relays:
        p = tuple(m["point"])
        # ---- 回传链路（与时间无关）----
        lp, blk = path_loss(p, P.G_POS, f)
        margin_rg = P.LMAX_RG - lp
        if worst_rg is None or margin_rg < worst_rg[0]:
            worst_rg = (margin_rg, m["rid"])
        # ---- 建链时限 / 能耗（独立重算：不信任 JSON 中经取整缓存算出的飞行值）----
        xy = np.array(m["point"][:2])
        t_fly, E_fly, _ = D.relay_fly(xy, float(m["point"][2]))
        t_deadline = m["start_act"] + D.relay["t_prep"] + t_fly + D.relay["t_link"]
        ok_deadline = t_deadline <= m["t_s"] + 1e-6
        T_hover = max(0.0, m["t_e_act"] - (m["t_s"] - D.relay["t_link"]))
        E_recalc = 2 * E_fly + (D.relay["P_hover"] + D.relay["P_comm"]) * T_hover / 3600.0
        cap = (1 - D.relay["rho"]) * D.relay["Euse"]
        ok_energy = E_recalc <= cap + 1e-9
        print("\n%s %s/%s 悬停海拔 %.0f m (离地 %.0f m), 服务窗口 [%.0f, %.0f]"
              % (m["rid"], m["uav"], m["comp"], p[2], m["h_agl"],
                 m["t_s"], m["t_e_act"]))
        print("   回传 R<->G01: 损耗 %.2f dB (遮挡=%s), 裕量 %.2f dB"
              % (lp, blk, margin_rg))
        print("   建链时限: 出发 %.0f + 准备 %.0f + 飞抵 %.0f + 建链 %.0f"
              " = %.0f %s 服务开始 %.0f  [%s]"
              % (m["start_act"], D.relay["t_prep"], t_fly, D.relay["t_link"],
                 t_deadline, "<=" if ok_deadline else ">", m["t_s"],
                 "OK" if ok_deadline else "违约"))
        print("   能耗 重算 %.3f / %.3f kWh (JSON 存储 %.3f)  [%s]"
              % (E_recalc, cap, m["E"], "OK" if ok_energy else "违约"))
        if not (ok_deadline and ok_energy and margin_rg >= 0):
            bad += 1
        # ---- 接入链路（逐区间、全分辨率）----
        idxs = m.get("cover_idx")
        if idxs is None:
            idxs = [i for i, itv in enumerate(intervals)
                    if itv["sortie"] in m.get("cover", [])
                    and m["t_s"] - 1e-6 <= itv["t_s"]
                    and itv["t_e"] <= m["t_e_act"] + 1e-6]
        for i in idxs:
            itv = intervals[i]
            t_abs, pos = traj[itv["sortie"]]
            sel = (t_abs >= itv["t_s"] - 1e-9) & (t_abs <= itv["t_e"] + 1e-9)
            qs = pos[sel]
            m_ur = None
            n_gap = n_bad = 0
            for q in qs:
                lp_u, _ = path_loss(q, p, f)
                mg = P.LMAX_UR - lp_u
                # 该采样点上直连是否真的不可用（1 s 分辨率）
                lp_g, _ = path_loss(q, P.G_POS, f)
                need = lp_g > P.LMAX_UG
                if need:
                    n_gap += 1
                    if m_ur is None or mg < m_ur:
                        m_ur = mg
                    if mg < 0:
                        n_bad += 1
            if m_ur is None:
                m_ur = float("nan")
            if worst_ur is None or (m_ur == m_ur and m_ur < worst_ur[0]):
                worst_ur = (m_ur, "%s/%s" % (m["rid"], itv["sortie"]))
            flag = "OK" if n_bad == 0 else "违约 %d 点" % n_bad
            if n_bad:
                bad += 1
            print("   接入 区间%d %-7s [%7.0f,%7.0f] 采样 %4d 点, "
                  "其中需中继 %4d 点, 最小裕量 %6.2f dB  [%s]"
                  % (i, itv["sortie"], itv["t_s"], itv["t_e"], len(qs),
                     n_gap, m_ur, flag))

    # ---- 需求侧全局扫描：1 s 分辨率下，任一时刻直连不可用则必须处于某中继服务窗口 ----
    # （比“逐区间核对”更强：可发现 5 s 求解采样漏掉的直连缺口）
    print("\n" + "-" * 78)
    print("需求侧全局扫描（1 s 分辨率）：直连不可用时刻必须被中继窗口覆盖")
    wins = [(m["t_s"], m["t_e_act"]) for m in relays]
    lfspl_ug = P.LMAX_UG
    n_gap_pts = n_uncovered = 0
    for sid, (t_abs, pos) in traj.items():
        d3 = np.linalg.norm(pos - np.array(P.G_POS)[None, :], axis=1) / 1000.0
        lfspl = 32.45 + 20 * math.log10(f) + 20 * np.log10(np.maximum(d3, 1e-6))
        for tq, q, ls in zip(t_abs, pos, lfspl):
            if ls + D.comm["Lobs"] <= lfspl_ug:      # 即使遮挡也可用 -> 直连 OK
                continue
            if ls > lfspl_ug or P.D.los_blocked(tuple(q), P.G_POS):
                n_gap_pts += 1                       # 直连不可用
                if not any(ws - 1e-6 <= tq <= we + 1e-6 for ws, we in wins):
                    n_uncovered += 1
                    if n_uncovered <= 10:
                        print("  ✗ %s t=%.0f 直连不可用且无中继窗口覆盖" % (sid, tq))
    print("  直连不可用采样点: %d, 其中无中继窗口覆盖: %d  [%s]"
          % (n_gap_pts, n_uncovered, "OK" if n_uncovered == 0 else "违约"))
    if n_uncovered:
        bad += 1

    print("\n" + "=" * 78)
    print("最小接入裕量 %.2f dB (%s)" % worst_ur)
    print("最小回传裕量 %.2f dB (%s)" % worst_rg)
    print("结论: %s" % ("全部满足" if bad == 0 else "存在 %d 处违约" % bad))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
