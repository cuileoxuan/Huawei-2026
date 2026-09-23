# -*- coding: utf-8 -*-
"""
common.py —— 公共数据与物理计算模块
2026 年研究生数学建模竞赛 D 题：山区洪涝灾害下无人机运输与通信协同优化

统一实现：
1. 数据读取（调度中心/服务区、货箱、运输无人机、中继无人机、通信链路参数、30m DEM）
2. 坐标转换（经纬度 -> 局部平面直角坐标，单位 m）
3. 航段计算：水平距离 / 巡航海拔 / 爬升高度 / 下降高度
4. 运输无人机航段时间与能耗（按附录 2 公式，含剩余载荷动态更新）
5. 电池/能源组件两阶段充电模型
6. 通信链路：地形遮挡（LOS）、自由空间传播损耗、双向链路预算、可用性判定
"""
import os
import math
import numpy as np
import pandas as pd
import scipy.io as sio

# ---------------------------------------------------------------- 路径
def _find_root():
    """从本文件所在目录向上查找同时含“数据/”与提交模板的项目根目录。

    支持两种布局：
    1. 官方提交布局：code/ 直接位于根目录（如 D题/）之下；
    2. 仓库布局：D题/ 是 code/ 目录某个祖先的兄弟目录（如 code-wb/code 与 D题/ 平级）。
    """
    d = os.path.dirname(os.path.abspath(__file__))
    for _ in range(6):
        # 布局 1：本目录即根目录
        if (os.path.isdir(os.path.join(d, "数据"))
                and os.path.isfile(os.path.join(d, "结果提交模板.xlsx"))):
            return d
        # 布局 2：D题/ 为本目录的兄弟目录
        cand = os.path.join(d, "D题")
        if (os.path.isdir(os.path.join(cand, "数据"))
                and os.path.isfile(os.path.join(cand, "结果提交模板.xlsx"))):
            return cand
        parent = os.path.dirname(d)
        if parent == d:
            break
        d = parent
    # 退回“代码位于 <ROOT>/code/”的默认约定
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


ROOT = _find_root()                                     # D:\desktop\D题
DATA = os.path.join(ROOT, "数据")
BASE = os.path.join(DATA, "无人机应急物资运输基础数据")
GEO = os.path.join(DATA, "镇龙乡地理空间数据", "镇龙乡及周边地理数据")
DEM_MAT = os.path.join(GEO, "数字高程模型数据（DEM）", "镇龙乡及周边30米DEM.mat")

G_GRAV = 9.80665          # 重力加速度 m/s^2
KWH = 3.6e6               # 1 kWh = 3.6e6 J
R_EARTH = 6371000.0       # 地球半径 m


# ---------------------------------------------------------------- 数据读取
class ProblemData:
    """集中管理全部输入数据。"""

    def __init__(self):
        # ---- 调度中心与服务区（按内容定位行，避免空行错位）----
        df = pd.read_excel(os.path.join(BASE, "调度中心与服务区.xlsx"), header=None)
        c0 = df[0].astype(str)
        r_depot = df.index[c0 == "O01"][0]
        self.depot = dict(id="O01", name=df.iat[r_depot, 1],
                          lon=float(df.iat[r_depot, 2]),
                          lat=float(df.iat[r_depot, 3]),
                          alt=float(df.iat[r_depot, 4]))
        self.areas = {}          # S001 -> dict
        for r in df.index[c0.str.match(r"^S\d{3}$", na=False)]:
            aid = str(df.iat[r, 0])
            self.areas[aid] = dict(id=aid, name=df.iat[r, 1],
                                   lon=float(df.iat[r, 2]),
                                   lat=float(df.iat[r, 3]),
                                   alt=float(df.iat[r, 4]),
                                   pop=float(df.iat[r, 5]))
        self.area_ids = list(self.areas.keys())          # S001..S015

        # ---- 运输无人机机型 / 实体机 / 共享电池 ----
        df = pd.read_excel(os.path.join(BASE, "运输无人机数据.xlsx"), header=None)
        c0 = df[0].astype(str)
        is_model = c0.isin(["A", "B", "C"]) & df[1].astype(str).str.contains("多旋翼", na=False)
        self.models = {}         # A/B/C -> dict
        for r in df.index[is_model]:
            g = str(df.iat[r, 0])
            self.models[g] = dict(
                id=g, name=df.iat[r, 1],
                m0=float(df.iat[r, 2]),        # 含电池空载总质量 kg
                Q=float(df.iat[r, 3]),         # 最大载货质量 kg
                V=float(df.iat[r, 4]),         # 可用装载体积 m^3
                vc=float(df.iat[r, 5]),        # 巡航速度 m/s
                L0=float(df.iat[r, 6]),        # 空载标准航程 m
                LF=float(df.iat[r, 7]),        # 满载标准航程 m
                Euse=float(df.iat[r, 8]),      # 电池可用能量 kWh
                rho=float(df.iat[r, 9]) / 100.0,   # 返航电量下限(安全余量)比例
                t_prep=float(df.iat[r, 10]),   # 工位固定准备时间 s
                t_load=float(df.iat[r, 11]),   # 每箱装载时间 s
                t_hand=float(df.iat[r, 12]),   # 接收点基础交接时间 s
                t_perbox=float(df.iat[r, 13]), # 每箱增加交接时间 s
                v_up=float(df.iat[r, 14]),     # 最大爬升速度 m/s
                v_dn=float(df.iat[r, 15]),     # 最大下降速度 m/s
                eta_up=float(df.iat[r, 16]),   # 爬升能耗效率
            )
        # ---- 实体无人机 ----
        self.uavs = {}           # U01 -> model
        for r in df.index[c0.str.match(r"^U\d{2}$", na=False)]:
            self.uavs[str(df.iat[r, 0])] = str(df.iat[r, 1])
        # ---- 共享电池（编号为 A/B/C 且第 2 列为数值的行）----
        self.batteries = {}      # model -> dict(n, Tfull)
        is_batt = c0.isin(["A", "B", "C"]) & pd.to_numeric(df[1], errors="coerce").notna()
        for r in df.index[is_batt]:
            g = str(df.iat[r, 0])
            self.batteries[g] = dict(n=int(df.iat[r, 1]),
                                     Tfull=float(df.iat[r, 2]))

        # ---- 中继无人机 ----
        df = pd.read_excel(os.path.join(BASE, "中继无人机数据.xlsx"), header=None)
        c0 = df[0].astype(str)
        r_r = df.index[(c0 == "R") & df[1].astype(str).str.contains("多旋翼", na=False)][0]
        self.relay = dict(
            id="R", name=df.iat[r_r, 1],
            m=float(df.iat[r_r, 4]),         # 计划起飞总质量 kg
            vc=float(df.iat[r_r, 5]),        # 巡航速度 m/s
            Pc=float(df.iat[r_r, 6]),        # 巡航功率 kW
            Euse=float(df.iat[r_r, 7]),      # 能源组件可用能量 kWh
            rho=float(df.iat[r_r, 8]) / 100.0,
            t_prep=float(df.iat[r_r, 9]),    # 固定准备时间 s
            t_link=float(df.iat[r_r, 10]),   # 建链时间 s
            t_turn=float(df.iat[r_r, 11]),   # 架次周转时间 s
            v_up=float(df.iat[r_r, 12]),
            v_dn=float(df.iat[r_r, 13]),
            eta_up=float(df.iat[r_r, 14]),
            P_hover=float(df.iat[r_r, 16]),  # 悬停功率 kW
            P_comm=float(df.iat[r_r, 17]),   # 通信附加功率 kW
            h_max=float(df.iat[r_r, 18]),    # 最大悬停离地高度 m
        )
        self.relay_uavs = [str(df.iat[r, 0])
                           for r in df.index[c0.str.match(r"^R\d{2}$", na=False)]]
        r_b = df.index[(c0 == "R") & pd.to_numeric(df[1], errors="coerce").notna()][0]
        self.relay_batt = dict(n=int(df.iat[r_b, 1]),
                               Tfull=float(df.iat[r_b, 2]))

        # ---- 通信链路参数 ----
        df = pd.read_excel(os.path.join(BASE, "通信链路参数.xlsx"), header=None)
        valid = pd.to_numeric(df[4], errors="coerce").notna() & df[3].notna()
        p = {str(df.iat[r, 3]): float(df.iat[r, 4]) for r in df.index[valid]}
        self.comm = dict(
            f=p["f"], Lsys=p["Lsys"], Lobs=p["Lobs"],
            Psens=p["Psens"], M=p["M"],
            Pt_U=20.0, G_U=3.0,            # 运输无人机
            Pt_Ra=20.0, G_Ra=6.0,          # 中继接入端
            Pt_Rb=19.0, G_Rb=8.0,          # 中继回传端
            Pt_G=27.0, G_G=12.0,           # 固定网关
            hG=20.0,                       # 网关天线离地高度 m
        )
        # 逐行读取以防御列顺序变化
        for r in df.index[valid]:
            cat, name, val = str(df.iat[r, 0]), str(df.iat[r, 1]), float(df.iat[r, 4])
            if cat == "运输无人机":
                if "发射功率" in name: self.comm["Pt_U"] = val
                if "天线增益" in name: self.comm["G_U"] = val
            elif cat == "中继接入端":
                if "发射功率" in name: self.comm["Pt_Ra"] = val
                if "天线增益" in name: self.comm["G_Ra"] = val
            elif cat == "中继回传端":
                if "发射功率" in name: self.comm["Pt_Rb"] = val
                if "天线增益" in name: self.comm["G_Rb"] = val
            elif cat.startswith("固定网关"):
                if "发射功率" in name: self.comm["Pt_G"] = val
                if "天线增益" in name: self.comm["G_G"] = val
                if "离地高度" in name: self.comm["hG"] = val

        # ---- 货箱（逐箱清单，共 80 箱）----
        df = pd.read_excel(os.path.join(BASE, "物资需求与配送时限.xlsx"),
                           sheet_name="逐箱货箱清单")
        self.boxes = []
        for _, row in df.iterrows():
            self.boxes.append(dict(
                id=str(row["货箱编号"]), area=str(row["服务区编号"]),
                mtype=str(row["物资类型"]), w=float(row["单箱质量（kg）"]),
                v=float(row["单箱体积（m³）"]),
                first=(str(row["是否首批保障"]) == "是"),
                t_first=(float(row["首批截止时间（s）"])
                         if pd.notna(row["首批截止时间（s）"]) else None),
                t_due=float(row["期望送达时间（s）"]),
                prio=float(row["应急优先系数"]),
            ))

        # ---- DEM ----
        m = sio.loadmat(DEM_MAT)
        self.dem = m["dem"].astype(np.float64)           # (1309,1486) 行=纬度
        self.dem_lat = m["latitude"].ravel()             # 1309
        self.dem_lon = m["longitude"].ravel()            # 1486
        self.dem_nodata = float(m["nodata"].ravel()[0])

        # ---- 平面坐标（以 O01 为原点的局部切平面，单位 m）----
        lat0 = math.radians(self.depot["lat"])
        self._cos0 = math.cos(lat0)

        def to_xy(lon, lat):
            x = math.radians(lon - self.depot["lon"]) * R_EARTH * self._cos0
            y = math.radians(lat - self.depot["lat"]) * R_EARTH
            return np.array([x, y])

        self.depot["xy"] = to_xy(self.depot["lon"], self.depot["lat"])
        for a in self.areas.values():
            a["xy"] = to_xy(a["lon"], a["lat"])
        # 节点表：O01 + S001..S015
        self.nodes = {"O01": self.depot}
        self.nodes.update(self.areas)
        # 作业高度：O01 取地面海拔；服务区取地面海拔以上 30 m
        self.work_alt = {"O01": self.depot["alt"]}
        for aid, a in self.areas.items():
            self.work_alt[aid] = a["alt"] + 30.0

        self._seg_cache = {}     # (i,j) -> 航段地形参数

    # ------------------------------------------------------------ DEM 工具
    def dem_elev_vec(self, xs, ys):
        """向量化 DEM 高程双线性插值。xs/ys 为等长数组，返回高程数组。"""
        xs = np.asarray(xs, dtype=float)
        ys = np.asarray(ys, dtype=float)
        lon = self.depot["lon"] + np.degrees(xs / (R_EARTH * self._cos0))
        lat = self.depot["lat"] + np.degrees(ys / R_EARTH)
        jx = np.clip(np.searchsorted(self.dem_lon, lon) - 1,
                     0, len(self.dem_lon) - 2)
        iy = np.clip(np.searchsorted(-self.dem_lat, -lat) - 1,
                     0, len(self.dem_lat) - 2)
        x0 = self.dem_lon[jx]; x1 = self.dem_lon[jx + 1]
        y0 = self.dem_lat[iy]; y1 = self.dem_lat[iy + 1]
        z00 = self.dem[iy, jx]; z01 = self.dem[iy, jx + 1]
        z10 = self.dem[iy + 1, jx]; z11 = self.dem[iy + 1, jx + 1]
        nd = self.dem_nodata + 1
        for zz in (z00, z01, z10, z11):
            bad = zz < nd
            if bad.any():
                zz[bad] = np.nan
        z00 = np.where(np.isnan(z00), np.nanmean(np.stack([z01, z10, z11]), axis=0), z00)
        z01 = np.where(np.isnan(z01), np.nanmean(np.stack([z00, z10, z11]), axis=0), z01)
        z10 = np.where(np.isnan(z10), np.nanmean(np.stack([z00, z01, z11]), axis=0), z10)
        z11 = np.where(np.isnan(z11), np.nanmean(np.stack([z00, z01, z10]), axis=0), z11)
        tx = (lon - x0) / (x1 - x0)
        ty = (y0 - lat) / (y0 - y1)
        return ((1 - tx) * (1 - ty) * z00 + tx * (1 - ty) * z01
                + (1 - tx) * ty * z10 + tx * ty * z11)

    def dem_elev(self, x, y):
        """局部平面坐标 (x,y) 处的 DEM 高程（双线性插值），越界取最近边界。"""
        return float(self.dem_elev_vec(np.array([x]), np.array([y]))[0])

    def in_dem(self, x, y):
        """局部平面坐标 (x,y) 是否落在 DEM 覆盖范围内。

        dem_elev 对越界坐标会静默取最近边界像元，故生成候选点前须先过滤，
        以保证悬停位置位于 DEM 覆盖范围内（题目要求）。
        """
        lon = self.depot["lon"] + math.degrees(x / (R_EARTH * self._cos0))
        lat = self.depot["lat"] + math.degrees(y / R_EARTH)
        return (self.dem_lon[0] - 1e-12 <= lon <= self.dem_lon[-1] + 1e-12
                and self.dem_lat[-1] - 1e-12 <= lat <= self.dem_lat[0] + 1e-12)

    def terrain_profile(self, xy1, xy2, step=30.0):
        """沿线采样点坐标与地形高程（向量化）。"""
        d = float(np.linalg.norm(xy2 - xy1))
        n = max(2, int(d / step) + 1)
        ts = np.linspace(0.0, 1.0, n)
        pts = xy1[None, :] + (xy2 - xy1)[None, :] * ts[:, None]
        ele = self.dem_elev_vec(pts[:, 0], pts[:, 1])
        return pts, ele, d

    def max_terrain(self, xy1, xy2):
        _, ele, _ = self.terrain_profile(xy1, xy2)
        return float(ele.max())

    def los_blocked(self, p1, p2, step=30.0, clearance=0.0):
        """三维点 p1->p2 视线是否被地形遮挡。p=(x,y,alt)。返回 True=遮挡。"""
        xy1, xy2 = np.array(p1[:2]), np.array(p2[:2])
        pts, ele, d = self.terrain_profile(xy1, xy2, step)
        ts = np.linspace(0.0, 1.0, len(ele))
        line_alt = p1[2] + (p2[2] - p1[2]) * ts
        # 端点处不判（端点本身可能贴地）
        mid = (ts > 1e-6) & (ts < 1 - 1e-6)
        return bool(np.any(ele[mid] > line_alt[mid] - clearance + 1e-9))

    # ------------------------------------------------------------ 航段计算
    def segment(self, ni, nj):
        """节点间航段参数：水平距离 d、巡航海拔、爬升/下降高度。"""
        key = (ni, nj)
        if key in self._seg_cache:
            return self._seg_cache[key]
        a, b = self.nodes[ni], self.nodes[nj]
        d = float(np.linalg.norm(a["xy"] - b["xy"]))
        h_max = self.max_terrain(a["xy"], b["xy"])
        h_cruise = h_max + 50.0
        h_up = max(0.0, h_cruise - self.work_alt[ni])
        h_dn = max(0.0, h_cruise - self.work_alt[nj])
        seg = dict(d=d, h_cruise=h_cruise, h_up=h_up, h_dn=h_dn)
        self._seg_cache[key] = seg
        return seg

    def equiv_range(self, g, q):
        """机型 g 携带载荷 q 时的等效航程 L_g(q)（附录 2 公式）。"""
        m = self.models[g]
        ratio = min(max(q / m["Q"], 0.0), 1.0)
        return m["L0"] - (m["L0"] - m["LF"]) * ratio ** 1.5

    def seg_time_energy(self, g, seg, q):
        """机型 g 以剩余载荷 q 飞过航段 seg 的时间(s)与能耗(kWh)。"""
        m = self.models[g]
        t = seg["h_up"] / m["v_up"] + seg["d"] / m["vc"] + seg["h_dn"] / m["v_dn"]
        # 水平巡航能耗：按等效航程占比折算电池可用能量
        E_hor = m["Euse"] * seg["d"] / self.equiv_range(g, q)
        # 爬升附加能耗：(空机+载荷)·g·h / eta_up
        E_up = (m["m0"] + q) * G_GRAV * seg["h_up"] / m["eta_up"] / KWH
        return t, E_hor + E_up

    def eval_sortie(self, g, stops, stop_boxes):
        """
        评估一个运输架次：O01 -> stops[0] -> ... -> O01。
        stops: 服务区 id 列表；stop_boxes: 每个 stop 投下的货箱 id 列表。
        返回 dict(可行性与逐航段时间/能耗/各站到达与服务完成时刻)。
        时间轴约定：t=0 为架次开始（固定准备），之后装载、起飞、飞行、交接、返航。
        """
        m = self.models[g]
        route = ["O01"] + list(stops) + ["O01"]
        t_now = m["t_prep"] + m["t_load"] * sum(len(b) for b in stop_boxes)
        q = sum(self.box_w(b) for bl in stop_boxes for b in bl)
        E_tot, T_fly = 0.0, 0.0
        legs, arrive, depart = [], {}, {}
        for k in range(len(route) - 1):
            i, j = route[k], route[k + 1]
            seg = self.segment(i, j)
            t, e = self.seg_time_energy(g, seg, q)
            E_tot += e; T_fly += t
            t_now += t
            legs.append(dict(i=i, j=j, q=q, t=t, e=e))
            if j != "O01":
                arrive[j] = t_now
                n_b = len(stop_boxes[stops.index(j)]) if j in stops else 0
                t_serv = m["t_hand"] + m["t_perbox"] * n_b
                depart[j] = t_now + t_serv
                t_now += t_serv
                q -= sum(self.box_w(b) for b in stop_boxes[stops.index(j)])
        feasible = E_tot <= (1 - m["rho"]) * m["Euse"] + 1e-9
        return dict(model=g, stops=list(stops), legs=legs, E=E_tot,
                    T=t_now, T_fly=T_fly, arrive=arrive, depart=depart,
                    ret=t_now, soc_end=1 - E_tot / m["Euse"], feasible=feasible)

    def box_w(self, bid):
        return self._boxmap[bid]["w"]

    def box_v(self, bid):
        return self._boxmap[bid]["v"]

    @property
    def _boxmap(self):
        if not hasattr(self, "__boxmap"):
            self.__boxmap = {b["id"]: b for b in self.boxes}
        return self.__boxmap

    # ------------------------------------------------------------ 充电模型
    @staticmethod
    def charge_time(soc_end, Tfull):
        """两阶段等效充电模型：从 SOC=soc_end 充至 100% 所需时间(s)。"""
        s = min(max(soc_end, 0.0), 1.0)
        if s < 0.90:
            return Tfull * (0.65 * (0.90 - s) / 0.90 + 0.35)
        return Tfull * 0.35 * (1.0 - s) / 0.10

    # ------------------------------------------------------------ 通信模型
    def gateway_pos(self):
        """固定网关 G01 通信端点三维位置。"""
        return (self.depot["xy"][0], self.depot["xy"][1],
                self.depot["alt"] + self.comm["hG"])

    def _pth(self):
        c = self.comm
        return c["Psens"] + c["M"]        # 有效接收门限 dBm

    def lmax_ug(self):
        """运输无人机 <-> G01 双向链路门限 dB。"""
        c = self.comm; pth = self._pth()
        a = c["Pt_U"] + c["G_U"] + c["G_G"] - c["Lsys"] - pth
        b = c["Pt_G"] + c["G_G"] + c["G_U"] - c["Lsys"] - pth
        return min(a, b)

    def lmax_ur(self):
        """运输无人机 <-> 中继（接入链路）双向门限 dB。"""
        c = self.comm; pth = self._pth()
        a = c["Pt_U"] + c["G_U"] + c["G_Ra"] - c["Lsys"] - pth
        b = c["Pt_Ra"] + c["G_Ra"] + c["G_U"] - c["Lsys"] - pth
        return min(a, b)

    def lmax_rg(self):
        """中继（回传端）<-> G01 双向门限 dB。"""
        c = self.comm; pth = self._pth()
        a = c["Pt_Rb"] + c["G_Rb"] + c["G_G"] - c["Lsys"] - pth
        b = c["Pt_G"] + c["G_G"] + c["G_Rb"] - c["Lsys"] - pth
        return min(a, b)

    def link_ok(self, p1, p2, lmax):
        """三维链路可用性：FSPL + 遮挡附加损耗 <= lmax。含快速带通判定。"""
        d3 = math.dist(p1, p2) / 1000.0          # km
        d3 = max(d3, 1e-6)
        lfspl = 32.45 + 20 * math.log10(self.comm["f"]) + 20 * math.log10(d3)
        if lfspl > lmax:                          # 无遮挡也超门限 -> 不可用
            return False
        if lfspl + self.comm["Lobs"] <= lmax:     # 即使遮挡也可用 -> 免 LOS
            return True
        loss = lfspl + (self.comm["Lobs"] if self.los_blocked(p1, p2) else 0.0)
        return loss <= lmax

    # ------------------------------------------------------------ 中继飞行
    def relay_fly(self, xy_to, h_hover):
        """
        中继无人机 O01 -> 悬停点(xy_to, 海拔 h_hover) 单程时间与能耗。

        飞行剖面：O01 地面 -> 垂直爬升至剖面最高点 h_top -> 水平巡航至悬停点
        正上方 -> 垂直升降到悬停海拔 h_hover。其中
            h_top = max(巡航海拔, 悬停海拔), 巡航海拔 = 航线最高地形 + 50。
        悬停海拔高于巡航海拔时须先爬到悬停高度（原实现直接按巡航海拔算下降，
        会漏掉这段爬升，低估往返时间与能耗）。
        """
        r = self.relay
        a = self.depot["xy"]
        d = float(np.linalg.norm(xy_to - a))
        h_cruise = self.max_terrain(a, xy_to) + 50.0
        h_top = max(h_cruise, h_hover)
        h_up = max(0.0, h_top - self.work_alt["O01"])
        h_dn = max(0.0, h_top - h_hover)
        t_cruise = d / r["vc"]
        t = h_up / r["v_up"] + t_cruise + h_dn / r["v_dn"]
        E = (r["Pc"] * t_cruise / 3600.0
             + r["m"] * G_GRAV * h_up / r["eta_up"] / KWH)
        return t, E, d


# ---------------------------------------------------------------- 求解器
def get_copt_model(name="mip", timelimit=None):
    """创建 COPT 模型（用户环境已安装 coptpy）。"""
    from coptpy import Envr, COPT
    env = Envr()
    m = env.createModel(name)
    if timelimit:
        m.setParam(COPT.Param.TimeLimit, timelimit)
    m.setParam(COPT.Param.Logging, 0)
    return m


_COPT_ENV = None


def _copt_env():
    """模块级单例 COPT 环境，避免频繁创建/销毁导致的原生崩溃。"""
    global _COPT_ENV
    if _COPT_ENV is None:
        from coptpy import Envr
        _COPT_ENV = Envr()
    return _COPT_ENV


def milp_binary_solve(c, A_eq=None, b_eq=None, A_ub=None, b_ub=None,
                      timelimit=300):
    """
    求解 0-1 整数线性规划：min c'x  s.t. A_eq x = b_eq, A_ub x <= b_ub, x∈{0,1}^n。
    优先使用 COPT（coptpy）；若不可用或未求得最优解则回退 scipy.optimize.milp
    (HiGHS)。返回最优解向量 np.ndarray；不可行时返回 None。
    """
    c = np.asarray(c, dtype=float)
    n = len(c)

    # ---- 首选 COPT ----
    try:
        from coptpy import COPT
        m = _copt_env().createModel("milp")
        try:
            m.setParam(COPT.Param.TimeLimit, timelimit)
            m.setParam(COPT.Param.Logging, 0)
            x = m.addVars(n, vtype=COPT.BINARY)
            if A_eq is not None:
                for row, rhs in zip(np.atleast_2d(A_eq), np.atleast_1d(b_eq)):
                    m.addConstr(sum(float(row[j]) * x[j] for j in range(n)
                                    if row[j] != 0) == float(rhs))
            if A_ub is not None:
                for row, rhs in zip(np.atleast_2d(A_ub), np.atleast_1d(b_ub)):
                    m.addConstr(sum(float(row[j]) * x[j] for j in range(n)
                                    if row[j] != 0) <= float(rhs))
            m.setObjective(sum(float(c[j]) * x[j] for j in range(n)),
                           COPT.MINIMIZE)
            m.solve()
            if m.status == COPT.OPTIMAL:
                return np.array([x[j].x for j in range(n)])
        finally:
            try:
                m.close()
            except Exception:
                pass
    except Exception:
        pass
    # ---- 回退 HiGHS ----
    try:
        from scipy.optimize import milp, LinearConstraint, Bounds
        cons = []
        if A_eq is not None:
            A_eq = np.atleast_2d(A_eq); b_eq = np.atleast_1d(b_eq)
            cons.append(LinearConstraint(A_eq, b_eq, b_eq))
        if A_ub is not None:
            A_ub = np.atleast_2d(A_ub); b_ub = np.atleast_1d(b_ub)
            cons.append(LinearConstraint(A_ub, -np.inf, b_ub))
        res = milp(c, integrality=np.ones(n), bounds=Bounds(0, 1),
                   constraints=cons,
                   options=dict(time_limit=timelimit, disp=False))
        if res.x is not None and (res.success or res.status == 1):
            return np.round(res.x)
    except Exception:
        pass
    return None
