# -*- coding: utf-8 -*-
"""
export_results.py —— 将四个问题的结果 CSV 按“结果提交模板.xlsx”汇总为
results/结果提交.xlsx（模板各 sheet 列名保持不变）。
运行顺序：problem1 -> problem2 -> problem3 -> problem4 -> export_results
"""
import os
import pandas as pd

CODE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(CODE)
OUT = os.path.join(CODE, "results")
TEMPLATE = os.path.join(ROOT, "结果提交模板.xlsx")
DEST = os.path.join(OUT, "结果提交.xlsx")


def load(name):
    fp = os.path.join(OUT, name)
    return pd.read_csv(fp) if os.path.exists(fp) else None


def main():
    sheets = {}

    df = load("Q1_单点组批.csv")
    if df is not None:
        df.columns = ["架次编号", "服务区编号", "机型编号", "货箱编号列表",
                      "总质量（kg）", "总体积（m³）", "往返时间（s）",
                      "架次能耗（kWh）", "返航SOC（%）"]
        sheets["Q1_单点组批"] = df

    df = load("Q2_运输架次.csv")
    if df is not None:
        df.columns = ["架次编号", "无人机编号", "机型编号", "电池编号",
                      "开始时刻（s）", "访问服务区顺序", "返回O01时刻（s）",
                      "架次能耗（kWh）"]
        sheets["Q2_运输架次"] = df

    df = load("Q2_逐箱交付.csv")
    if df is not None:
        df.columns = ["货箱编号", "架次编号", "服务区编号", "交付完成时刻（s）"]
        sheets["Q2_逐箱交付"] = df

    df = load("Q3_中继架次.csv")
    if df is not None:
        df.columns = ["中继架次编号", "中继无人机编号", "能源组件编号",
                      "开始时刻（s）", "悬停经度（°）", "悬停纬度（°）",
                      "悬停海拔（m）", "建链完成时刻（s）", "服务结束时刻（s）",
                      "返回O01时刻（s）", "架次能耗（kWh）"]
        sheets["Q3_中继架次"] = df

    df = load("Q3_通信保障.csv")
    if df is not None:
        df.columns = ["运输架次编号", "通信阶段", "开始时刻（s）",
                      "结束时刻（s）", "保障方式", "中继架次编号"]
        sheets["Q3_通信保障"] = df

    df = load("Q4_分区配置.csv")
    if df is not None:
        # 模板无“方案”列，将其并入任务组编号前缀
        if "方案" in df.columns:
            df["任务组编号"] = df["方案"] + "-" + df["任务组编号"].astype(str)
            df = df.drop(columns=["方案"])
        df.columns = ["K（2或3）", "任务组编号", "服务区列表",
                      "A型运输无人机数", "B型运输无人机数", "C型运输无人机数",
                      "A型电池组数", "B型电池组数", "C型电池组数",
                      "中继无人机数", "中继能源组件数"]
        sheets["Q4_分区配置"] = df

    with pd.ExcelWriter(DEST, engine="openpyxl") as w:
        for name in ["Q1_单点组批", "Q2_运输架次", "Q2_逐箱交付",
                     "Q3_中继架次", "Q3_通信保障", "Q4_分区配置"]:
            if name in sheets:
                sheets[name].to_excel(w, sheet_name=name, index=False)
    print(f"已生成 {DEST}")
    for k, v in sheets.items():
        print(f"  {k}: {len(v)} 行")


if __name__ == "__main__":
    main()
