# -*- coding: utf-8 -*-
"""
visualize.py —— 论文用关键图表生成

输入：code/results/ 下四个问题的结果文件（CSV/JSON）
输出：code/results/figures/*.png

图表清单：
  fig_q1_payload_heatmap.png   问题一 3×15 最大安全载荷热力图
  fig_q1_sensitivity.png       问题一 返航安全余量敏感性
  fig_q2_routes.png            问题二 运输路线图（DEM 底图）
  fig_q2_gantt.png             问题二 运输无人机甘特图
  fig_q2_battery.png           问题二 共享电池占用/充电甘特图
  fig_q3_map.png               问题三 中继悬停点与通信缺口分布图
  fig_q3_status.png            问题三 各运输架次通信状态时间轴
  fig_q3_relay_gantt.png       问题三 中继无人机与能源组件甘特图
  fig_q4_partition.png         问题四 任务分区图（K=2 / K=3 均衡方案）
  fig_q4_resources.png         问题四 各分区资源需求与库存对比
"""
import os
import json
import math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from common import ProblemData

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

CODE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(CODE, "results")
FIG = os.path.join(OUT, "figures")
os.makedirs(FIG, exist_ok=True)

D = ProblemData()
MODEL_COLOR = {"A": "#4C9BE0", "B": "#E0884C", "C": "#5CB85C"}


def dem_extent():
    """DEM 在局部平面坐标下的显示范围与高程矩阵。"""
    lon = D.dem_lon
    lat = D.dem_lat
    x = (lon - D.depot["lon"]) * math.pi / 180 * 6371000.0 * D._cos0
    y = (lat - D.depot["lat"]) * math.pi / 180 * 6371000.0
    Z = D.dem.copy()
    Z[Z < D.dem_nodata + 1] = np.nan
    return x, y, Z


def draw_map(ax):
    x, y, Z = dem_extent()
    ax.pcolormesh(x / 1000, y / 1000, Z, cmap="terrain", shading="auto",
                  alpha=0.85)
    ax.set_xlabel("相对 O01 东向距离 (km)")
    ax.set_ylabel("相对 O01 北向距离 (km)")
    # 节点
    dp = D.depot["xy"] / 1000
    ax.plot(*dp, marker="s", color="red", ms=10, zorder=5)
    ax.annotate("O01", dp, textcoords="offset points", xytext=(6, 6),
                color="red", weight="bold")
    for aid, a in D.areas.items():
        p = a["xy"] / 1000
        ax.plot(*p, marker="o", color="black", ms=5, zorder=5)
        ax.annotate(aid, p, textcoords="offset points", xytext=(4, 4),
                    fontsize=8)


def load_json(name):
    return json.load(open(os.path.join(OUT, name), encoding="utf-8"))


# ------------------------------------------------------------ 问题一
def fig_q1():
    mat = pd.read_csv(os.path.join(OUT, "Q1_最大安全载荷矩阵.csv"),
                      index_col=0)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(mat.values.astype(float), cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(3), mat.columns)
    ax.set_yticks(range(len(mat)), mat.index)
    for i in range(len(mat)):
        for j in range(3):
            ax.text(j, i, f"{mat.values[i, j]:.1f}", ha="center",
                    va="center", fontsize=8)
    ax.set_title("问题一：3 机型 × 15 服务区最大安全载荷 (kg)")
    fig.colorbar(im, label="最大安全载荷 (kg)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_q1_payload_heatmap.png"), dpi=150)
    plt.close(fig)

    sen = pd.read_csv(os.path.join(OUT, "Q1_敏感性分析.csv"))
    fig, ax1 = plt.subplots(figsize=(7, 4.5))
    ax1.plot(sen["rho"], sen["总架次"], "o-", color="#C0392B", label="总架次数")
    ax1.set_xlabel("返航安全余量 ρ")
    ax1.set_ylabel("总架次数", color="#C0392B")
    ax1.tick_params(axis="y", labelcolor="#C0392B")
    ax2 = ax1.twinx()
    ax2.plot(sen["rho"], sen["A型平均载荷kg"], "s--", label="A 型")
    ax2.plot(sen["rho"], sen["B型平均载荷kg"], "^--", label="B 型")
    ax2.plot(sen["rho"], sen["C型平均载荷kg"], "d--", label="C 型")
    ax2.set_ylabel("平均最大安全载荷 (kg)")
    ax2.legend(loc="lower left")
    ax1.set_title("问题一：返航安全余量敏感性")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_q1_sensitivity.png"), dpi=150)
    plt.close(fig)


# ------------------------------------------------------------ 问题二
def fig_q2():
    q2 = load_json("q2_solution.json")
    routes = q2["routes"]

    # 路线图
    fig, ax = plt.subplots(figsize=(9, 7))
    draw_map(ax)
    cmap = plt.cm.tab20
    for k, rt in enumerate(routes):
        nodes = ["O01"] + rt["stops"] + ["O01"]
        xs = [D.nodes[n]["xy"][0] / 1000 for n in nodes]
        ys = [D.nodes[n]["xy"][1] / 1000 for n in nodes]
        ax.plot(xs, ys, color=cmap(k % 20), lw=1.4, alpha=0.9)
    ax.set_title(f"问题二：运输路线图（{len(routes)} 架次）")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_q2_routes.png"), dpi=150)
    plt.close(fig)

    # 无人机甘特图
    fig, ax = plt.subplots(figsize=(10, 4.5))
    uavs = sorted({rt["uav"] for rt in routes})
    for rt in routes:
        y = uavs.index(rt["uav"])
        ax.barh(y, (rt["end"] - rt["start"]) / 60, left=rt["start"] / 60,
                color=MODEL_COLOR[rt["model"]], edgecolor="black", lw=0.4)
        ax.text((rt["start"] + rt["end"]) / 120, y,
                "->".join(rt["stops"]), ha="center", va="center", fontsize=7)
    ax.set_yticks(range(len(uavs)), uavs)
    ax.set_xlabel("时间 (min)")
    ax.set_title("问题二：运输无人机甘特图")
    ax.legend(handles=[Patch(color=c, label=f"{g} 型")
                       for g, c in MODEL_COLOR.items()])
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_q2_gantt.png"), dpi=150)
    plt.close(fig)

    # 电池甘特图（占用 + 充电）
    fig, ax = plt.subplots(figsize=(10, 5.5))
    batts = sorted({rt["batt"] for rt in routes})
    for rt in routes:
        y = batts.index(rt["batt"])
        g = rt["model"]
        chg = D.charge_time(1 - rt["E"] / D.models[g]["Euse"],
                            D.batteries[g]["Tfull"])
        ax.barh(y, (rt["end"] - rt["start"]) / 60, left=rt["start"] / 60,
                color=MODEL_COLOR[g], edgecolor="black", lw=0.4)
        ax.barh(y, chg / 60, left=rt["end"] / 60,
                color=MODEL_COLOR[g], alpha=0.3, hatch="//",
                edgecolor="gray", lw=0.3)
    ax.set_yticks(range(len(batts)), batts)
    ax.set_xlabel("时间 (min)")
    ax.set_title("问题二：共享电池占用（实色=飞行，斜纹=充电）甘特图")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_q2_battery.png"), dpi=150)
    plt.close(fig)


# ------------------------------------------------------------ 问题三
def fig_q3():
    q3 = load_json("q3_solution.json")
    routes, relays, intervals = q3["routes"], q3["relays"], q3["intervals"]

    # 地图：路线 + 中继点 + 缺口区
    fig, ax = plt.subplots(figsize=(9, 7))
    draw_map(ax)
    for rt in routes:
        nodes = ["O01"] + rt["stops"] + ["O01"]
        xs = [D.nodes[n]["xy"][0] / 1000 for n in nodes]
        ys = [D.nodes[n]["xy"][1] / 1000 for n in nodes]
        ax.plot(xs, ys, color="gray", lw=0.8, alpha=0.6)
    for itv in intervals:
        rt = next(r for r in routes if r["sortie"] == itv["sortie"])
        # 缺口大致位置：目的服务区
        p = D.nodes[rt["stops"][0]]["xy"] / 1000
        ax.plot(*p, marker="x", color="purple", ms=9, mew=2, zorder=6)
    for m in relays:
        lon = D.depot["lon"] + math.degrees(m["point"][0] / (6371000 * D._cos0))
        lat = D.depot["lat"] + math.degrees(m["point"][1] / 6371000)
        x = m["point"][0] / 1000
        y = m["point"][1] / 1000
        ax.plot(x, y, marker="*", color="#C0392B", ms=16, zorder=7)
        ax.annotate(m["rid"], (x, y), textcoords="offset points",
                    xytext=(6, 6), color="#C0392B", weight="bold")
    ax.set_title("问题三：中继悬停点（红星）与通信缺口架次（紫叉）")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_q3_map.png"), dpi=150)
    plt.close(fig)

    # 各架次通信状态时间轴
    comm = pd.read_csv(os.path.join(OUT, "Q3_通信保障.csv"))
    sids = list(dict.fromkeys(comm["运输架次编号"]))
    fig, ax = plt.subplots(figsize=(10, 6))
    for _, row in comm.iterrows():
        y = sids.index(row["运输架次编号"])
        color = "#5CB85C" if row["保障方式"] == "G01直连" else "#C0392B"
        ax.barh(y, (row["结束时刻s"] - row["开始时刻s"]) / 60,
                left=row["开始时刻s"] / 60, color=color, lw=0)
    ax.set_yticks(range(len(sids)), sids, fontsize=7)
    ax.set_xlabel("时间 (min)")
    ax.set_title("问题三：各运输架次通信状态（绿=G01直连，红=中继保障）")
    ax.legend(handles=[Patch(color="#5CB85C", label="G01 直连"),
                       Patch(color="#C0392B", label="中继保障")])
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_q3_status.png"), dpi=150)
    plt.close(fig)

    # 中继甘特图
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    ru = sorted({m["uav"] for m in relays})
    rc = sorted({m["comp"] for m in relays})
    for m in relays:
        y = ru.index(m["uav"])
        axes[0].barh(y, (m["ret_act"] - m["start_act"]) / 60,
                     left=m["start_act"] / 60, color="#C0392B",
                     edgecolor="black", lw=0.4)
        axes[0].text((m["start_act"] + m["ret_act"]) / 120, y, m["rid"],
                     ha="center", va="center", fontsize=7, color="white")
        y2 = rc.index(m["comp"])
        chg = D.charge_time(m["soc_end"], D.relay_batt["Tfull"])
        axes[1].barh(y2, (m["ret_act"] - m["start_act"]) / 60,
                     left=m["start_act"] / 60, color="#E0884C",
                     edgecolor="black", lw=0.4)
        axes[1].barh(y2, chg / 60, left=m["ret_act"] / 60,
                     color="#E0884C", alpha=0.3, hatch="//",
                     edgecolor="gray", lw=0.3)
    axes[0].set_yticks(range(len(ru)), ru)
    axes[0].set_title("问题三：中继无人机甘特图")
    axes[1].set_yticks(range(len(rc)), rc)
    axes[1].set_title("中继能源组件占用（实色=任务，斜纹=充电）")
    axes[1].set_xlabel("时间 (min)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_q3_relay_gantt.png"), dpi=150)
    plt.close(fig)


# ------------------------------------------------------------ 问题四
def fig_q4():
    rep = load_json("q4_report.json")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    colors = ["#FF6D01", "#2979FF", "#00C853"]
    for ax, K in zip(axes, ("2", "3")):
        if K not in rep:
            continue
        scheme = rep[K].get("均衡最优") or list(rep[K].values())[0]
        draw_map(ax)
        for gi, areas in enumerate(scheme["assign"]):
            for aid in areas:
                p = D.areas[aid]["xy"] / 1000
                ax.plot(*p, marker="o", color=colors[gi], ms=12, zorder=6,
                        mec="white", mew=1.5)
                ax.annotate(aid, p, textcoords="offset points",
                            xytext=(5, 5), fontsize=9, fontweight="bold",
                            color=colors[gi], zorder=7,
                            path_effects=[
                                __import__("matplotlib.patheffects", fromlist=["withStroke"]).withStroke(
                                    linewidth=2.5, foreground="white")])
        ax.set_title(f"问题四：K={K} 任务分区（均衡最优方案）")
        ax.legend(handles=[Patch(color=colors[i], label=f"任务组 {i+1}")
                           for i in range(len(scheme["assign"]))])
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_q4_partition.png"), dpi=150)
    plt.close(fig)

    # 资源需求 vs 库存
    res_names = [("UAV_A", "A型无人机"), ("UAV_B", "B型无人机"),
                 ("UAV_C", "C型无人机"), ("BAT_A", "A型电池"),
                 ("BAT_B", "B型电池"), ("BAT_C", "C型电池"),
                 ("RELAY", "中继无人机"), ("ECOMP", "能源组件")]
    inventory = {"UAV_A": 4, "UAV_B": 2, "UAV_C": 2, "BAT_A": 6,
                 "BAT_B": 4, "BAT_C": 4, "RELAY": 2, "ECOMP": 6}
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)
    for ax, K in zip(axes, ("2", "3")):
        if K not in rep:
            continue
        schemes = rep[K]
        xpos = np.arange(len(res_names))
        width = 0.35
        for si, (sname, sdata) in enumerate(schemes.items()):
            vals = [sdata["total_need"][k] for k, _ in res_names]
            ax.bar(xpos + si * width, vals, width, label=sname)
        inv = [inventory[k] for k, _ in res_names]
        ax.plot(xpos + width / 2, inv, "kD--", label="现有库存", ms=5)
        ax.set_xticks(xpos + width / 2, [n for _, n in res_names],
                      rotation=30, fontsize=8)
        ax.set_title(f"K={K} 分区资源需求 vs 库存")
        ax.legend(fontsize=8)
    axes[0].set_ylabel("数量")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_q4_resources.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    fig_q1()
    print("fig_q1 done")
    fig_q2()
    print("fig_q2 done")
    fig_q3()
    print("fig_q3 done")
    fig_q4()
    print("fig_q4 done")
    print(f"全部图表已输出到 {FIG}")
