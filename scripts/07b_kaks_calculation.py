import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.transforms as mtransforms
from matplotlib.colors import LinearSegmentedColormap, Normalize

# ================= 1. 全局科研排版设置 =================
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]

# ================= 2. 路径设置 =================
current_dir = os.path.dirname(os.path.abspath(__file__)) if "__file__" in locals() else os.getcwd()

kaks_file = os.path.join(current_dir, "FINAL_KaKs_Results.csv")
rename_file = os.path.join(current_dir, "rename.csv")

output_csv = os.path.join(current_dir, "FINAL_KaKs_Annotated_ForPlot.csv")
output_png = os.path.join(current_dir, "FINAL_KaKs_Annotated_Barplot.png")
output_pdf = os.path.join(current_dir, "FINAL_KaKs_Annotated_Barplot.pdf")

# ================= 3. 读取文件 =================
if not os.path.exists(kaks_file):
    raise FileNotFoundError(f"找不到 Ka/Ks 结果文件：{kaks_file}")

if not os.path.exists(rename_file):
    raise FileNotFoundError(f"找不到 rename.csv：{rename_file}")

df = pd.read_csv(kaks_file)
rename_df = pd.read_csv(rename_file)

# ================= 4. 检查必要列 =================
required_kaks_cols = ["Barley_ID", "Wheat_ID", "Ka (dN)", "Ks (dS)", "Ka/Ks"]
for col in required_kaks_cols:
    if col not in df.columns:
        raise ValueError(f"FINAL_KaKs_Results.csv 缺少必要列：{col}")

required_rename_cols = ["GeneID", "Annoname"]
for col in required_rename_cols:
    if col not in rename_df.columns:
        raise ValueError(f"rename.csv 缺少必要列：{col}")

# ================= 5. 清理重复行 =================
# 你的文件里有重复同源对，这里去重
df = df.drop_duplicates(
    subset=["Barley_ID", "Wheat_ID", "Ka (dN)", "Ks (dS)", "Ka/Ks"]
).copy()

# 数值化，确保画的是 Ka/Ks，不是 Ka 或 Ks
df["Ka/Ks"] = pd.to_numeric(df["Ka/Ks"], errors="coerce")
df["Ka (dN)"] = pd.to_numeric(df["Ka (dN)"], errors="coerce")
df["Ks (dS)"] = pd.to_numeric(df["Ks (dS)"], errors="coerce")

df = df.dropna(subset=["Ka/Ks"]).copy()

# ================= 6. 构建 ID -> 注释名 映射 =================
rename_df["GeneID"] = rename_df["GeneID"].astype(str).str.strip()
rename_df["Annoname"] = rename_df["Annoname"].astype(str).str.strip()

id_to_name = dict(zip(rename_df["GeneID"], rename_df["Annoname"]))

def get_label(gene_id):
    """
    如果 gene_id 在 rename.csv 中，显示注释名 + (*)
    如果不在 rename.csv 中，显示原始 ID
    """
    gene_id = str(gene_id).strip()
    if gene_id in id_to_name:
        return f"{id_to_name[gene_id]} (*)"
    else:
        return gene_id

df["Barley_Label"] = df["Barley_ID"].apply(get_label)
df["Wheat_Label"] = df["Wheat_ID"].apply(get_label)

# ================= 7. 按 Ka/Ks 排序 =================
df = df.sort_values("Ka/Ks", ascending=False).reset_index(drop=True)

# 保存一份带注释名的表，方便你检查
df.to_csv(output_csv, index=False, encoding="utf-8-sig")

print("用于绘图的数据：")
print(df[["Barley_ID", "Barley_Label", "Wheat_ID", "Wheat_Label", "Ka (dN)", "Ks (dS)", "Ka/Ks"]])

print("\nKa/Ks 范围：")
print(f"min = {df['Ka/Ks'].min():.4f}")
print(f"max = {df['Ka/Ks'].max():.4f}")

if (df["Ka/Ks"] > 1).sum() == 0:
    print("✅ 当前所有同源对 Ka/Ks 均小于 1，主要表现为纯化选择。")
else:
    print(f"⚠️ 有 {(df['Ka/Ks'] > 1).sum()} 个同源对 Ka/Ks > 1。")

# ================= 8. 颜色映射 =================
cmap = LinearSegmentedColormap.from_list(
    "kaks_custom",
    ["#5B6FB8", "#68BDB5", "#CDE8A9", "#B01764"]
)

vmin = df["Ka/Ks"].min()
vmax = df["Ka/Ks"].max()

if vmax == vmin:
    vmax = vmin + 1e-6

norm = Normalize(vmin=vmin, vmax=vmax)
bar_colors = [cmap(norm(v)) for v in df["Ka/Ks"]]

# ================= 9. 绘图 =================
n = len(df)
fig_height = max(6.5, 0.55 * n + 1.8)

fig, ax = plt.subplots(figsize=(12, fig_height), facecolor="white")
ax.set_facecolor("white")

y = np.arange(n)

bars = ax.barh(
    y,
    df["Ka/Ks"],
    height=0.80,
    color=bar_colors,
    edgecolor="#8A8A8A",
    linewidth=1.0,
    zorder=3
)

# x 轴范围固定到 1.2，这样能显示 Ka/Ks = 1 的中性选择线
xmax = 1.2
ax.set_xlim(0, xmax)
ax.set_ylim(-0.8, n - 0.2)
ax.invert_yaxis()

# 去掉默认 y 轴标签，手动写双行标签
ax.set_yticks(y)
ax.set_yticklabels([""] * n)

# x 轴网格
ax.grid(axis="x", linestyle="--", linewidth=0.6, alpha=0.25, zorder=0)
ax.set_axisbelow(True)

# Ka/Ks = 1 红色虚线
ax.axvline(
    x=1.0,
    color="#FF6F61",
    linestyle="--",
    linewidth=1.8,
    alpha=0.95,
    label="Ka/Ks = 1.0 (Neutral Selection)",
    zorder=2
)

# 坐标轴标题
ax.set_xlabel(
    "Ka/Ks Ratio (Evolutionary Selection Pressure)",
    fontsize=13,
    fontweight="bold",
    labelpad=10
)

ax.set_title(
    "Annotated Gene Pairs Selection Pressure (Horizontal View)",
    fontsize=17,
    fontweight="bold",
    pad=18
)

# 边框
for spine in ax.spines.values():
    spine.set_linewidth(1.0)
    spine.set_color("#999999")

# ================= 10. 左侧双行标签 + 右侧数值 =================
text_transform = mtransforms.blended_transform_factory(ax.transAxes, ax.transData)

for i, row in df.iterrows():
    yy = y[i]

    # 第一行：大麦，蓝色
    ax.text(
        -0.015,
        yy - 0.16,
        str(row["Barley_Label"]),
        transform=text_transform,
        ha="right",
        va="center",
        fontsize=11.5,
        fontweight="bold",
        fontstyle="italic",
        color="#3BA3E3",
        clip_on=False
    )

    # 第二行：小麦，橙色
    ax.text(
        -0.015,
        yy + 0.16,
        str(row["Wheat_Label"]),
        transform=text_transform,
        ha="right",
        va="center",
        fontsize=11.5,
        fontweight="bold",
        fontstyle="italic",
        color="#F28C28",
        clip_on=False
    )

    # 条形右侧数值
    value = row["Ka/Ks"]
    ax.text(
        value + 0.012,
        yy,
        f"{value:.3f}",
        ha="left",
        va="center",
        fontsize=10,
        fontweight="bold",
        color="#4D4D4D"
    )

# 图例
leg = ax.legend(
    loc="lower right",
    frameon=True,
    fontsize=9
)

leg.get_frame().set_facecolor("white")
leg.get_frame().set_edgecolor("#CCCCCC")

# 刻度
ax.tick_params(axis="x", labelsize=10, colors="#444444")
ax.tick_params(axis="y", length=0)

# 左侧给长 ID 留空间
plt.subplots_adjust(left=0.32, right=0.97, top=0.90, bottom=0.10)

# 保存
plt.savefig(output_png, dpi=300, bbox_inches="tight", facecolor="white")
plt.savefig(output_pdf, bbox_inches="tight", facecolor="white")
plt.close()

print("\n🎉 绘图完成！")
print(f"注释后表格: {output_csv}")
print(f"PNG 图片: {output_png}")
print(f"PDF 图片: {output_pdf}")