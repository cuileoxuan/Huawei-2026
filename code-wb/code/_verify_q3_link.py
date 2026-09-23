# -*- coding: utf-8 -*-
"""最终判定：只对"服务窗口包含该区间"的中继计入，检查是否存在全程达标的中继点。"""
import sys, json, math
import numpy as np
sys.path.insert(0, r"D:\desktop\D题\code")
import problem3 as P
D = P.D
sol = json.load(open(r"D:\desktop\D题\code\results\q3_solution.json", encoding="utf-8"))
routes = {r["sortie"]: r for r in sol["routes"]}
traj = {}
for sid, rt in routes.items():
    t, pos, t0 = P.sortie_trajectory(rt)
    m = t >= t0 - 1e-9
    traj[sid] = (rt["start"] + t[m], pos[m])
relays = {r["rid"]: r for r in sol["relays"]}
bad_total = 0
for itv in sol["intervals"]:
    t_abs, pos = traj[itv["sortie"]]
    sel = (t_abs >= itv["t_s"] - 1e-9) & (t_abs <= itv["t_e"] + 1e-9)
    qs = pos[sel]
    cand = {}
    for rid, r in relays.items():
        if r["t_s"] - 1e-6 <= itv["t_s"] and itv["t_e"] <= r["t_e"] + 1e-6:
            bad = [q for q in qs if not D.link_ok(tuple(q), tuple(r["point"]), P.LMAX_UR)]
            cand[rid] = bad
    best = min((len(v) for v in cand.values()), default=None)
    mark = "OK " if best == 0 else "!! "
    detail = ", ".join("%s:%d" % (k, len(v)) for k, v in cand.items())
    if best != 0:
        bad_total += best
        print("%s%-7s [%7.0f,%7.0f] n=%3d 可用中继(窗口包含)=%s" % (mark, itv["sortie"], itv["t_s"], itv["t_e"], len(qs), detail))
        for rid, v in cand.items():
            if len(v) == len(qs):
                continue
    else:
        print("%s%-7s [%7.0f,%7.0f] n=%3d  %s" % (mark, itv["sortie"], itv["t_s"], itv["t_e"], len(qs), detail))
print()
print("存在越限的区间数(按窗口内最优中继计):", bad_total)
