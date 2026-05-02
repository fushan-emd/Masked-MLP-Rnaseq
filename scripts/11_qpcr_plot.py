import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from pathlib import Path

# =========================
# 0. 全局绘图参数
# =========================
plt.rcParams["font.family"] = "Arial"
plt.rcParams["axes.linewidth"] = 1.0
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42

# =========================
# 1. 文件路径
# 改成你自己的 Excel 路径和输出路径
# =========================
file_path = Path(r"E:\github\project\bisheanalysis\finalpic\qpcr\qpcrdata.xlsx")
out_dir = Path(r"E:\github\project\bisheanalysis\finalpic\qpcr")
out_dir.mkdir(parents=True, exist_ok=True)

if not file_path.exists():
    raise FileNotFoundError(f"找不到 Excel 文件，请检查路径：{file_path}")

# =========================
# 2. 读取 Excel 原始 Ct 数据
# =========================
df_raw = pd.read_excel(file_path, sheet_name="Sheet1", header=None)

cp_table = pd.DataFrame(
    index=["A", "B", "C", "D", "E", "F"],
    columns=range(1, 13),
    dtype=float
)

for index, row in df_raw.iterrows():
    first_cell = str(row[0]).strip()

    if first_cell in ["A1", "B1", "C1", "D1", "E1", "F1"]:
        sample = first_cell[0]
        cp_table.loc[sample] = pd.to_numeric(
            df_raw.iloc[index + 1, :],
            errors="coerce"
        ).values

# =========================
# 3. 基因设计
# 1-4：大麦目标基因，用 11 列 Tubulin
# 5-10：小麦目标基因，用 12 列 Actin
# =========================
genes_design = {
    "Barley": {
        1: ("HvXB3", 11),
        2: ("HvSTI1", 11),
        3: ("HvUbiA", 11),
        4: ("HvCaM1", 11),
    },
    "Wheat": {
        5: ("TaSTI1-B", 12),
        6: ("TaDNAJ-A", 12),
        7: ("TaDDX-A", 12),
        8: ("TaVPS9-B", 12),
        9: ("TaSTK-B", 12),
        10: ("TaRPL32-D", 12),
    }
}

# =========================
# 4. 计算 2^-ΔΔCt
# 显著性：基于 ΔCt 做 Welch t-test
# =========================
summary_rows = []

for species, gene_dict in genes_design.items():
    for col, (gene_name, ref_col) in gene_dict.items():
        dct = (cp_table[col] - cp_table[ref_col]).astype(float)

        dct_ck = dct.loc[["A", "B", "C"]].dropna()
        dct_salt = dct.loc[["D", "E", "F"]].dropna()

        mean_dct_ck = dct_ck.mean()
        mean_dct_salt = dct_salt.mean()

        ddct = mean_dct_salt - mean_dct_ck
        fold_change = 2 ** (-ddct)

        # 处理组各重复相对于 CK 平均 ΔCt 的表达量
        salt_fc_each = 2 ** (-(dct_salt - mean_dct_ck))
        salt_fc_sd = salt_fc_each.std(ddof=1) if len(salt_fc_each) >= 2 else np.nan

        # Welch t-test：基于 ΔCt，而不是直接基于 fold change
        if len(dct_ck) >= 2 and len(dct_salt) >= 2:
            _, p_val = stats.ttest_ind(dct_ck, dct_salt, equal_var=False)
        else:
            p_val = np.nan

        if pd.isna(p_val):
            sig = ""
        elif p_val < 0.001:
            sig = "***"
        elif p_val < 0.01:
            sig = "**"
        elif p_val < 0.05:
            sig = "*"
        else:
            sig = "ns"

        summary_rows.append({
            "Species": species,
            "Gene": gene_name,
            "Target_col": col,
            "Reference_col": ref_col,
            "CK_mean_dCt": mean_dct_ck,
            "Salt_mean_dCt": mean_dct_salt,
            "DeltaDeltaCt": ddct,
            "FoldChange_2^-ddCt": fold_change,
            "Salt_FC_SD": salt_fc_sd,
            "P_value": p_val,
            "Significance": sig,
            "CK_n": len(dct_ck),
            "Salt_n": len(dct_salt),
        })

df_summary = pd.DataFrame(summary_rows)

summary_path = out_dir / "qpcr_final_summary.xlsx"
df_summary.to_excel(summary_path, index=False)

# =========================
# 5. 合并绘图函数：Barley + Wheat，标注 A/B
# =========================
def plot_combined_qpcr(barley_df, wheat_df, output_prefix):
    # 稍微深一点的纯色
    barley_color = "#3A8CC1"
    wheat_color = "#E08A24"

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(8.8, 4.2),
        dpi=300,
        gridspec_kw={"width_ratios": [1.0, 1.35], "wspace": 0.32}
    )

    plot_data = [
        (axes[0], barley_df, "Barley", barley_color, "A"),
        (axes[1], wheat_df, "Wheat", wheat_color, "B"),
    ]

    for ax, df_species, species_name, bar_color, panel_label in plot_data:
        genes = df_species["Gene"].tolist()
        means = df_species["FoldChange_2^-ddCt"].to_numpy(dtype=float)
        sds = df_species["Salt_FC_SD"].fillna(0).to_numpy(dtype=float)
        sigs = df_species["Significance"].tolist()
        ns = df_species["Salt_n"].tolist()

        x = np.arange(len(genes))
        bar_width = 0.48

        # 纯色实心柱
        ax.bar(
            x,
            means,
            width=bar_width,
            yerr=sds,
            capsize=3.5,
            color=bar_color,
            edgecolor=bar_color,
            linewidth=1.2,
            error_kw={
                "elinewidth": 1.1,
                "ecolor": "black",
                "capthick": 1.1,
            },
            zorder=3
        )

        # 横向网格线
        ax.yaxis.grid(
            True,
            linestyle="--",
            linewidth=0.6,
            alpha=0.22,
            zorder=1
        )
        ax.xaxis.grid(False)

        # y 轴范围
        ymax = np.nanmax(means + sds)

        if np.isnan(ymax) or ymax <= 0:
            ymax = 1.0

        if ymax < 0.08:
            ylim_top = ymax * 2.0 + 0.005
        elif ymax < 0.2:
            ylim_top = ymax * 1.8 + 0.01
        elif ymax < 1.0:
            ylim_top = ymax * 1.45 + 0.03
        else:
            ylim_top = ymax * 1.25 + 0.08

        ax.set_ylim(0, ylim_top)

        # 显著性标注
        for i, (mean, sd, sig, n) in enumerate(zip(means, sds, sigs, ns)):
            y_base = mean + (0 if np.isnan(sd) else sd)

            if sig not in ["", "ns"]:
                ax.text(
                    i,
                    y_base + ylim_top * 0.035,
                    sig,
                    ha="center",
                    va="bottom",
                    fontsize=12,
                    fontweight="bold",
                    color="black"
                )

            if n < 2:
                ax.text(
                    i,
                    y_base + ylim_top * 0.115,
                    f"n={n}",
                    ha="center",
                    va="bottom",
                    fontsize=8.5,
                    color="dimgray"
                )

        # 标题和坐标轴
        ax.set_title(
            f"{species_name} candidate genes",
            fontsize=12.5,
            fontweight="bold",
            pad=8
        )

        ax.set_ylabel(
            "Relative expression level (2$^{-\\Delta\\Delta Ct}$)",
            fontsize=11
        )

        ax.set_xlabel("Genes", fontsize=11)

        ax.set_xticks(x)
        ax.set_xticklabels(
            genes,
            rotation=28,
            ha="right",
            fontsize=10
        )

        # 基因名斜体，颜色和柱子一致
        for label in ax.get_xticklabels():
            label.set_fontstyle("italic")
            label.set_color(bar_color)

        # CK = 1
        ax.text(
            0.02,
            0.96,
            "CK = 1",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            color="dimgray"
        )

        # A / B 面板标注
        ax.text(
            -0.18,
            1.08,
            panel_label,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=17,
            fontweight="bold",
            color="black"
        )

        # 坐标轴美化
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        ax.spines["left"].set_linewidth(1.0)
        ax.spines["bottom"].set_linewidth(1.2)

        ax.tick_params(axis="x", colors=bar_color, length=0)
        ax.tick_params(axis="y", labelsize=10, length=3, width=1.0)

        ax.margins(x=0.06)

    plt.tight_layout(pad=0.8)

    png_path = out_dir / f"{output_prefix}.png"
    pdf_path = out_dir / f"{output_prefix}.pdf"

    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")

    plt.show()
    plt.close(fig)

    print(f"已保存: {png_path}")
    print(f"已保存: {pdf_path}")


# =========================
# 6. 合并作图
# =========================
barley_df = df_summary[df_summary["Species"] == "Barley"].copy()
wheat_df = df_summary[df_summary["Species"] == "Wheat"].copy()

plot_combined_qpcr(
    barley_df,
    wheat_df,
    "qpcr_barley_wheat_combined_AB_solid"
)

print("作图完成。")
print("计算结果表：", summary_path)
print("结果文件保存在：", out_dir)