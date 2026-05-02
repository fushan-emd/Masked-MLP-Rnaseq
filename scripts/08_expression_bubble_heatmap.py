import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns

import matplotlib as mpl
mpl.use("Agg")   # 必须放在 import matplotlib.pyplot as plt 前面

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


# ==========================================
# 1. 全局设置
# ==========================================
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42
mpl.rcParams["font.family"] = "sans-serif"
mpl.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]


def resolve_base_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    if "__file__" in globals():
        return Path(__file__).resolve().parent
    return Path.cwd()


base_dir = resolve_base_dir()


# ==========================================
# 2. 你指定的配色
#    负值：蓝色；0：暖白；正值：橙红
# ==========================================
pretty_cmap = mcolors.LinearSegmentedColormap.from_list(
    "paper_soft_div",
    [
        "#2E5E7E",  # deep blue
        "#79A9C7",  # soft blue
        "#F7F4EF",  # warm white
        "#F0B27A",  # soft orange
        "#C65D57",  # warm red
    ],
    N=256
)


# ==========================================
# 3. ID 标准化函数
# ==========================================
def normalize_barley_id(gene_id: str) -> str:
    """
    统一大麦 HORVU ID 格式。
    示例：
    HORVUMOREX.r3.1HG0123450 -> HORVUMOREXr3HG0123450
    HORVUMOREXr3HG0123450    -> HORVUMOREXr3HG0123450
    """
    if pd.isna(gene_id):
        return ""

    gene_id = str(gene_id).strip()

    if gene_id == "" or gene_id.lower() == "nan":
        return ""

    gene_id = gene_id.replace(" ", "")
    gene_id = gene_id.replace(".", "")

    # 处理 r3.1HG 去点后变成 r31HG 的情况
    gene_id = re.sub(r"r(\d+)1HG", r"r\1HG", gene_id, flags=re.IGNORECASE)

    return gene_id


def normalize_wheat_id(gene_id: str) -> str:
    if pd.isna(gene_id):
        return ""

    gene_id = str(gene_id).strip()

    if gene_id == "" or gene_id.lower() == "nan":
        return ""

    gene_id = gene_id.replace(" ", "")
    return gene_id


def normalize_gene_id(gene_id: str) -> str:
    if pd.isna(gene_id):
        return ""

    gene_id = str(gene_id).strip()

    if "HORVU" in gene_id.upper():
        return normalize_barley_id(gene_id)

    if "TRAES" in gene_id.upper():
        return normalize_wheat_id(gene_id)

    return gene_id


def compact_gene_id(gene_id: str) -> str:
    if not gene_id:
        return ""

    gene_id = str(gene_id).strip()

    m_barley = re.match(r"^HORVUMOREXr(\d+)HG(\d+)$", gene_id)
    if m_barley:
        return f"Hv{m_barley.group(1)}-{m_barley.group(2)[-6:]}"

    m_wheat = re.match(r"^TraesCS([0-9][ABD])02G(\d+)$", gene_id)
    if m_wheat:
        return f"Ta{m_wheat.group(1)}-{m_wheat.group(2)[-6:]}"

    return gene_id[:16] + ".." if len(gene_id) > 18 else gene_id


# ==========================================
# 4. 时间、数字、标签函数
# ==========================================
def time_to_hour(x) -> float:
    """
    把 1h、6h、24h、3d、30d、4w 转换成小时，用于排序。
    """
    x = str(x).strip().lower()

    if x in ["", "nan", "none", "unknown"]:
        return 0

    m = re.search(r"(\d+\.?\d*)", x)
    if not m:
        return 0

    v = float(m.group(1))

    if "min" in x:
        return v / 60
    if "h" in x:
        return v
    if "d" in x:
        return v * 24
    if "w" in x:
        return v * 24 * 7

    return v


def clean_number_str(x) -> str:
    x = str(x).strip()
    if x.endswith(".0"):
        x = x[:-2]
    return x


def extract_number(x) -> float:
    m = re.search(r"(\d+\.?\d*)", str(x))
    return float(m.group(1)) if m else 0.0


def pretty_sample_label(col: str) -> str:
    """
    salt|B4_350_24h -> B4\\n350-24h
    """
    label = str(col).split("|")[-1]
    parts = label.split("_")

    if len(parts) >= 3:
        batch = parts[0]
        conc = clean_number_str(parts[1])
        time = parts[2]
        return f"{batch}\n{conc}-{time}"

    return label


# ==========================================
# 5. Bubble heatmap 辅助函数
# ==========================================
def calc_species_vmax(df, q=90, min_v=0.5, max_v=2.0):
    """
    每个物种单独计算颜色范围
    """
    vals = pd.Series(df.to_numpy().ravel()).dropna()

    if len(vals) == 0:
        return 1.0

    vmax = np.nanpercentile(np.abs(vals), q)
    vmax = max(min_v, min(float(vmax), max_v))
    return vmax


def build_size_mapper(*dfs, q=95, smin=70, smax=750):
    """
    构建共享的圆点大小映射函数
    圆点面积和 |log2FC| 对应
    """
    arrs = []
    for df in dfs:
        arrs.append(df.to_numpy().ravel())

    vals = pd.Series(np.concatenate(arrs)).dropna().abs()

    if len(vals) == 0:
        ref = 1.0
    else:
        ref = float(np.nanpercentile(vals, q))
        ref = max(ref, 0.3)

    def mapper(v):
        a = np.abs(np.asarray(v, dtype=float))
        scaled = np.clip(a / ref, 0, 1)
        return smin + scaled * (smax - smin)

    return mapper, ref


def draw_dot_heatmap(ax, df, cmap, vmax, size_mapper, xtick_labels):
    """
    用 scatter 画圆形气泡热图
    """
    nrows, ncols = df.shape

    xs, ys, vals = [], [], []

    for i in range(nrows):
        for j in range(ncols):
            val = df.iat[i, j]
            if pd.notna(val):
                xs.append(j + 0.5)
                ys.append(i + 0.5)
                vals.append(val)

    vals = np.array(vals, dtype=float)
    sizes = size_mapper(vals)

    # 背景网格
    for x in range(ncols + 1):
        ax.axvline(x, color="#ECECEC", lw=0.8, zorder=0)
    for y in range(nrows + 1):
        ax.axhline(y, color="#ECECEC", lw=0.8, zorder=0)

    sc = ax.scatter(
        xs,
        ys,
        s=sizes,
        c=vals,
        cmap=cmap,
        vmin=-vmax,
        vmax=vmax,
        marker="o",
        edgecolors="white",
        linewidths=0.7,
        zorder=3,
    )

    ax.set_xlim(0, ncols)
    ax.set_ylim(nrows, 0)

    ax.set_xticks(np.arange(ncols) + 0.5)
    ax.set_xticklabels(
        xtick_labels,
        rotation=0,
        ha="center",
        fontsize=10,
        fontweight="bold",
    )

    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_facecolor("white")

    for spine in ax.spines.values():
        spine.set_visible(False)

    return sc


def add_size_legend(ax, size_mapper, ref_values=(0.5, 1.0, 1.5), title="|log2FC|"):
    """
    圆点大小图例
    """
    ax.axis("off")

    y_positions = [0.75, 0.50, 0.25]
    sizes = [float(size_mapper([v])[0]) for v in ref_values]

    ax.scatter(
        [0.35] * len(ref_values),
        y_positions,
        s=sizes,
        color="#BDBDBD",
        edgecolors="white",
        linewidths=0.8,
        transform=ax.transAxes,
        zorder=3,
    )

    for y, v in zip(y_positions, ref_values):
        ax.text(
            0.60,
            y,
            f"{v:.1f}",
            va="center",
            ha="left",
            fontsize=10,
            transform=ax.transAxes,
        )

    ax.text(
        0.5,
        0.95,
        title,
        ha="center",
        va="top",
        fontsize=11,
        fontweight="bold",
        transform=ax.transAxes,
    )


# ==========================================
# 6. 读取 rename.csv
# ==========================================
rename_map_path = base_dir / "rename.csv"

rename_map = {}
barley_genes_input = []
wheat_genes_input = []

try:
    df_rename = pd.read_csv(rename_map_path, header=None, dtype=str)

    for _, row in df_rename.iterrows():
        row_vals = [
            str(x).strip()
            for x in row.values
            if pd.notna(x) and str(x).strip() != ""
        ]

        if not row_vals:
            continue

        gid_raw = next(
            (
                val
                for val in row_vals
                if "HORVU" in val.upper() or "TRAES" in val.upper()
            ),
            None,
        )

        if not gid_raw:
            continue

        gid = normalize_gene_id(gid_raw)
        display_name = row_vals[-1].strip()

        if "HORVU" in display_name.upper() or "TRAES" in display_name.upper():
            display_name = compact_gene_id(gid)

        rename_map[gid] = display_name

        if "HORVU" in gid.upper() and gid not in barley_genes_input:
            barley_genes_input.append(gid)

        elif "TRAES" in gid.upper() and gid not in wheat_genes_input:
            wheat_genes_input.append(gid)

except Exception as e:
    print(f"❌ 读取 rename.csv 失败: {e}")
    sys.exit(1)


# ==========================================
# 7. 读取 ortholog_map.csv，并统一 ID
# ==========================================
ortho_map_path = base_dir / "ortholog_map.csv"

try:
    ortho_map = pd.read_csv(ortho_map_path, dtype=str).fillna("")
    ortho_map = ortho_map.apply(lambda x: x.astype(str).str.strip())

    for col in ortho_map.columns:
        col_series = ortho_map[col].astype(str)

        if col_series.str.contains("HORVU", case=False, na=False).any():
            ortho_map[col] = ortho_map[col].apply(normalize_barley_id)

        elif col_series.str.contains("TRAES", case=False, na=False).any():
            ortho_map[col] = ortho_map[col].apply(normalize_wheat_id)

except FileNotFoundError:
    print(f"❌ 找不到文件 {ortho_map_path}")
    sys.exit(1)

except Exception as e:
    print(f"❌ 读取 ortholog_map.csv 失败: {e}")
    sys.exit(1)


# ==========================================
# 8. 判断 ortholog_map 哪列是小麦，哪列是大麦
# ==========================================
col_0_is_wheat = ortho_map.iloc[:, 0].str.contains(
    "TRAES", case=False, na=False
).any()

w_col_idx, b_col_idx = (0, 1) if col_0_is_wheat else (1, 0)


# ==========================================
# 9. 构建同源基因对
# ==========================================
pairs = []
added_w = set()

for bg in barley_genes_input:
    w_matches = (
        ortho_map[ortho_map.iloc[:, b_col_idx] == bg]
        .iloc[:, w_col_idx]
        .replace("", np.nan)
        .replace("nan", np.nan)
        .dropna()
        .unique()
    )

    if len(w_matches) == 0:
        pairs.append({"Barley_Gene": bg, "Wheat_Gene": ""})
    else:
        for w in w_matches:
            w = normalize_wheat_id(w)
            pairs.append({"Barley_Gene": bg, "Wheat_Gene": w})
            added_w.add(w)

for wg in wheat_genes_input:
    wg = normalize_wheat_id(wg)

    if wg not in added_w:
        b_matches = (
            ortho_map[ortho_map.iloc[:, w_col_idx] == wg]
            .iloc[:, b_col_idx]
            .replace("", np.nan)
            .replace("nan", np.nan)
            .dropna()
            .unique()
        )

        if len(b_matches) == 0:
            pairs.append({"Barley_Gene": "", "Wheat_Gene": wg})
        else:
            for b in b_matches:
                b = normalize_barley_id(b)
                pairs.append({"Barley_Gene": b, "Wheat_Gene": wg})

pairs_df = pd.DataFrame(pairs).drop_duplicates().reset_index(drop=True)


# ==========================================
# 10. 读取注释表和表达矩阵
# ==========================================
try:
    anno_df = pd.read_csv(base_dir / "allanno.csv").fillna("Unknown")

    if "Sample_ID" in anno_df.columns:
        anno_df = anno_df.set_index("Sample_ID")

    anno_df.index = anno_df.index.astype(str).str.strip()

    anno_df["Condition"] = anno_df["Condition"].astype(str).str.strip().str.lower()

    ck_mask = (
        anno_df["Condition"].isin(
            ["0", "0mm", "0.0", "control", "ck", "none", "unknown", "0h", "0d"]
        )
        | anno_df["Condition"].str.contains("ck|control", regex=True, na=False)
    )

    anno_df["Treat_Class"] = np.where(ck_mask, "ck", "salt")

    anno_df["Prefix"] = anno_df["Species"].astype(str).str.lower().apply(
        lambda x: "B" if "barley" in x else ("W" if "wheat" in x else "")
    )

    clean_batch = (
        anno_df["Batch"]
        .astype(str)
        .str.upper()
        .str.replace("B", "", regex=False)
        .str.replace("W", "", regex=False)
        .str.strip()
    )

    anno_df["Batch"] = (
        anno_df["Prefix"]
        + pd.to_numeric(clean_batch, errors="coerce")
        .fillna(0)
        .astype(int)
        .astype(str)
    )

    if "Stress_Conc" not in anno_df.columns:
        anno_df["Stress_Conc"] = "0"

    if "Stress_Time" not in anno_df.columns:
        anno_df["Stress_Time"] = "0"

    wheat_exp = pd.read_csv(base_dir / "wheat_merged_tpm.csv", index_col=0).T
    barley_exp = pd.read_csv(base_dir / "barley_merged_tpm.csv", index_col=0).T

    wheat_exp.index = wheat_exp.index.astype(str).str.strip()
    barley_exp.index = barley_exp.index.astype(str).str.strip()

    wheat_exp.columns = [normalize_wheat_id(c) for c in wheat_exp.columns]
    barley_exp.columns = [normalize_barley_id(c) for c in barley_exp.columns]

    # 标准化后如果出现重复基因 ID，取均值
    wheat_exp = wheat_exp.T.groupby(level=0).mean(numeric_only=True).T
    barley_exp = barley_exp.T.groupby(level=0).mean(numeric_only=True).T

except Exception as e:
    print(f"❌ 数据读取失败: {e}")
    sys.exit(1)


# ==========================================
# 11. 先按 CK / SALT、Batch、浓度、时间聚合 TPM
# ==========================================
def process_unified_grouping(exp_df, ann_df, prefix):
    valid_samples = exp_df.index.intersection(ann_df[ann_df["Prefix"] == prefix].index)

    if len(valid_samples) == 0:
        print(f"⚠️ 警告：表达矩阵中没有找到 {prefix} 物种对应样本。")
        return pd.DataFrame()

    df = exp_df.loc[valid_samples].copy()
    ann = ann_df.loc[valid_samples].copy()

    df["Treat_Class"] = ann["Treat_Class"]
    df["Batch"] = ann["Batch"]
    df["Stress_Conc"] = ann["Stress_Conc"].astype(str).str.strip()
    df["Stress_Time"] = ann["Stress_Time"].astype(str).str.strip()

    grouped = (
        df.groupby(["Treat_Class", "Batch", "Stress_Conc", "Stress_Time"])
        .mean(numeric_only=True)
        .reset_index()
    )

    grouped["Sort_Key"] = grouped["Treat_Class"].map({"ck": 0, "salt": 1})
    grouped["Batch_Num"] = (
        grouped["Batch"].str.extract(r"(\d+)").astype(float).fillna(0)
    )
    grouped["Conc_Num"] = (
        grouped["Stress_Conc"]
        .astype(str)
        .str.extract(r"(\d+\.?\d*)")
        .astype(float)
        .fillna(0)
    )
    grouped["Time_Hour"] = grouped["Stress_Time"].apply(time_to_hour)

    grouped = grouped.sort_values(
        by=["Sort_Key", "Time_Hour", "Conc_Num", "Batch_Num"]
    )

    def make_col_id(r):
        tc = r["Treat_Class"]
        b = r["Batch"]
        c = clean_number_str(r["Stress_Conc"])
        t = str(r["Stress_Time"]).strip()

        if tc == "ck":
            return f"ck|{b}_CK"

        return f"salt|{b}_{c}_{t}"

    grouped["Col_ID"] = grouped.apply(make_col_id, axis=1)

    final_grouped = grouped.groupby("Col_ID").mean(numeric_only=True)
    final_grouped = final_grouped.loc[grouped["Col_ID"].drop_duplicates()]

    return final_grouped


b_grouped = process_unified_grouping(barley_exp, anno_df, "B")
w_grouped = process_unified_grouping(wheat_exp, anno_df, "W")


# ==========================================
# 12. 转换为同 Batch CK 配对 log2FC
# ==========================================
def make_batch_paired_log2fc(grouped_df, species_prefix):
    """
    输入：
        grouped_df: 行是 ck|B1_CK / salt|B4_350_24h，列是基因的 TPM 均值

    输出：
        行只保留 salt 条件，每个值是：
        log2(TPM_SALT + 1) - log2(TPM_同Batch_CK + 1)
    """
    if grouped_df.empty:
        return pd.DataFrame()

    meta_records = []

    for col_id in grouped_df.index:
        treat = str(col_id).split("|")[0]
        label = str(col_id).split("|")[-1]
        parts = label.split("_")

        batch = parts[0]

        if treat == "ck":
            conc = "0"
            time = "0"
        else:
            conc = parts[1] if len(parts) > 1 else "Unknown"
            time = parts[2] if len(parts) > 2 else "Unknown"

        meta_records.append(
            {
                "Col_ID": col_id,
                "Treat": treat,
                "Batch": batch,
                "Conc": conc,
                "Time": time,
                "Time_Hour": time_to_hour(time),
                "Conc_Num": extract_number(conc),
                "Batch_Num": extract_number(batch),
            }
        )

    meta_df = pd.DataFrame(meta_records).set_index("Col_ID")

    ck_ids = meta_df[meta_df["Treat"] == "ck"].index.tolist()
    salt_ids = meta_df[meta_df["Treat"] == "salt"].index.tolist()

    out_rows = []
    out_index = []
    missing_ck = []

    for salt_id in salt_ids:
        batch = meta_df.loc[salt_id, "Batch"]

        same_batch_cks = [
            ck_id for ck_id in ck_ids
            if meta_df.loc[ck_id, "Batch"] == batch
        ]

        if len(same_batch_cks) == 0:
            missing_ck.append(salt_id)
            continue

        ck_id = same_batch_cks[0]

        salt_log = np.log2(grouped_df.loc[salt_id] + 1)
        ck_log = np.log2(grouped_df.loc[ck_id] + 1)

        log2fc = salt_log - ck_log

        out_rows.append(log2fc)
        out_index.append(salt_id)

    if missing_ck:
        print(f"\n⚠️ {species_prefix} 有这些处理找不到同 Batch CK，已跳过：")
        for x in missing_ck:
            print("   ", x)

    if len(out_rows) == 0:
        print(f"❌ {species_prefix} 没有任何可用的 SALT-CK 配对。")
        return pd.DataFrame()

    fc_df = pd.DataFrame(out_rows, index=out_index)

    fc_meta = meta_df.loc[out_index].copy()
    fc_meta = fc_meta.sort_values(["Time_Hour", "Conc_Num", "Batch_Num"])

    fc_df = fc_df.loc[fc_meta.index]

    return fc_df


b_grouped_fc = make_batch_paired_log2fc(b_grouped, "Barley")
w_grouped_fc = make_batch_paired_log2fc(w_grouped, "Wheat")


# ==========================================
# 13. ID 匹配检查
# ==========================================
print("\n========== ID 匹配检查 ==========")
print(f"rename.csv 中大麦目标基因数: {len(barley_genes_input)}")
print(f"rename.csv 中小麦目标基因数: {len(wheat_genes_input)}")
print(f"构建出的同源基因对数: {len(pairs_df)}")

print("\n大麦目标基因在 barley/log2FC 矩阵中匹配情况：")
for g in barley_genes_input:
    print(f"{g}: {'✅' if g in b_grouped_fc.columns else '❌'}")

print("\n大麦目标基因在 ortholog_map.csv 中匹配情况：")
for g in barley_genes_input:
    ok = (ortho_map.iloc[:, b_col_idx] == g).any()
    print(f"{g}: {'✅' if ok else '❌'}")

print("\n小麦目标基因在 wheat/log2FC 矩阵中匹配情况：")
for g in wheat_genes_input:
    print(f"{g}: {'✅' if g in w_grouped_fc.columns else '❌'}")

print("\n小麦目标基因在 ortholog_map.csv 中匹配情况：")
for g in wheat_genes_input:
    ok = (ortho_map.iloc[:, w_col_idx] == g).any()
    print(f"{g}: {'✅' if ok else '❌'}")


# ==========================================
# 14. 构建热图矩阵
# ==========================================
b_hm_data = []
w_hm_data = []
b_names = []
w_names = []
target_info = []

for _, row in pairs_df.iterrows():
    bg = normalize_barley_id(row["Barley_Gene"])
    wg = normalize_wheat_id(row["Wheat_Gene"])

    b_names.append(rename_map.get(bg, compact_gene_id(bg)) if bg else "")
    w_names.append(rename_map.get(wg, compact_gene_id(wg)) if wg else "")

    target_info.append(
        (
            bg in barley_genes_input if bg else False,
            wg in wheat_genes_input if wg else False,
        )
    )

    if bg and bg in b_grouped_fc.columns:
        b_hm_data.append(b_grouped_fc[bg])
    else:
        b_hm_data.append(pd.Series(np.nan, index=b_grouped_fc.index))

    if wg and wg in w_grouped_fc.columns:
        w_hm_data.append(w_grouped_fc[wg])
    else:
        w_hm_data.append(pd.Series(np.nan, index=w_grouped_fc.index))

b_hm_df = pd.DataFrame(b_hm_data, index=range(len(b_names)))
w_hm_df = pd.DataFrame(w_hm_data, index=range(len(w_names)))


# ==========================================
# 15. 过滤左右两边都没有 log2FC 的行
# ==========================================
has_b_exp = b_hm_df.notna().any(axis=1)
has_w_exp = w_hm_df.notna().any(axis=1)

keep_rows = has_b_exp | has_w_exp
removed_n = int((~keep_rows).sum())

if removed_n > 0:
    print(f"\n⚠️ 已过滤左右两边都没有 log2FC 的同源对: {removed_n} 行")

b_hm_df = b_hm_df.loc[keep_rows].reset_index(drop=True)
w_hm_df = w_hm_df.loc[keep_rows].reset_index(drop=True)

b_names = [x for i, x in enumerate(b_names) if keep_rows.iloc[i]]
w_names = [x for i, x in enumerate(w_names) if keep_rows.iloc[i]]
target_info = [x for i, x in enumerate(target_info) if keep_rows.iloc[i]]

if len(b_names) == 0:
    print("❌ 没有任何可绘制的基因。请检查 ID 是否仍然无法匹配。")
    sys.exit(1)


# ==========================================
# 16. 按目标基因和平均 log2FC 排序
# ==========================================
with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=RuntimeWarning)
    b_fc_mean = b_hm_df.mean(axis=1, skipna=True)
    w_fc_mean = w_hm_df.mean(axis=1, skipna=True)

sort_df = pd.DataFrame(
    {
        "fc": b_fc_mean.fillna(0) + w_fc_mean.fillna(0),
        "has_b": [t[0] for t in target_info],
        "has_w": [t[1] for t in target_info],
    }
)

sort_df["group"] = 2
sort_df.loc[sort_df["has_w"], "group"] = 1
sort_df.loc[sort_df["has_b"], "group"] = 0

sorted_idx = sort_df.sort_values(
    ["group", "fc"], ascending=[True, False]
).index.tolist()

b_hm_df = b_hm_df.loc[sorted_idx].reset_index(drop=True)
w_hm_df = w_hm_df.loc[sorted_idx].reset_index(drop=True)

b_names = [b_names[i] for i in sorted_idx]
w_names = [w_names[i] for i in sorted_idx]
target_info = [target_info[i] for i in sorted_idx]
sort_df = sort_df.loc[sorted_idx].reset_index(drop=True)


# ==========================================
# 17. 设置颜色范围和圆点大小范围
# ==========================================
b_vmax = calc_species_vmax(b_hm_df, q=90, min_v=0.5, max_v=2.0)
w_vmax = calc_species_vmax(w_hm_df, q=90, min_v=0.5, max_v=2.0)

size_mapper, size_ref = build_size_mapper(
    b_hm_df, w_hm_df,
    q=95,
    smin=70,
    smax=750
)

print(f"\n🎨 Barley 色阶范围：{-b_vmax:.2f} 到 {b_vmax:.2f}")
print(f"🎨 Wheat  色阶范围：{-w_vmax:.2f} 到 {w_vmax:.2f}")
print(f"🔵 圆点大小参考上限(|log2FC| 95分位)：{size_ref:.2f}")

# ==========================================
# 18. 绘图（bubble heatmap）
# ==========================================
print("\n🎨 正在渲染 bubble heatmap...")

b_cols = b_hm_df.shape[1]
w_cols = w_hm_df.shape[1]
num_genes = len(b_names)

fig_height = max(6.5, num_genes * 0.42 + 2.8)
text_col_ratio = max(3.2, (b_cols + w_cols) * 0.22)
fig_height = max(6.3, num_genes * 0.42 + 2.2)
fig_width = max(17, (b_cols + w_cols) * 0.72 + text_col_ratio + 3)
fig = plt.figure(figsize=(fig_width, fig_height), facecolor="white")
gs = fig.add_gridspec(
    2,
    5,
    width_ratios=[
        max(b_cols, 1),
        0.30,
        text_col_ratio,
        max(w_cols, 1),
        0.30,
    ],
    height_ratios=[1.0, 0.09],
    wspace=0.12,
    hspace=0.10,
)

ax_b_hm = fig.add_subplot(gs[0, 0], facecolor="white")
ax_b_cbar = fig.add_subplot(gs[0, 1])

ax_text = fig.add_subplot(gs[0, 2])

ax_w_hm = fig.add_subplot(gs[0, 3], facecolor="white")
ax_w_cbar = fig.add_subplot(gs[0, 4])

# 底部图例横跨整张图
ax_size_legend = fig.add_subplot(gs[1, :])


# -------- 左边：Barley bubble heatmap --------
sc_b = draw_dot_heatmap(
    ax=ax_b_hm,
    df=b_hm_df,
    cmap=pretty_cmap,
    vmax=b_vmax,
    size_mapper=size_mapper,
    xtick_labels=[pretty_sample_label(c) for c in b_hm_df.columns],
)

ax_b_hm.set_title(
    "Barley\nlog2FC vs paired CK",
    fontsize=16,
    fontweight="bold",
    pad=15,
)

cbar_b = fig.colorbar(sc_b, cax=ax_b_cbar)
cbar_b.set_label("log2FC", fontsize=10, fontweight="bold")
cbar_b.ax.tick_params(labelsize=9)

for spine in ax_b_cbar.spines.values():
    spine.set_visible(False)


# -------- 右边：Wheat bubble heatmap --------
sc_w = draw_dot_heatmap(
    ax=ax_w_hm,
    df=w_hm_df,
    cmap=pretty_cmap,
    vmax=w_vmax,
    size_mapper=size_mapper,
    xtick_labels=[pretty_sample_label(c) for c in w_hm_df.columns],
)

ax_w_hm.set_title(
    "Wheat\nlog2FC vs paired CK",
    fontsize=16,
    fontweight="bold",
    pad=15,
)

cbar_w = fig.colorbar(sc_w, cax=ax_w_cbar)
cbar_w.set_label("log2FC", fontsize=10, fontweight="bold")
cbar_w.ax.tick_params(labelsize=9)

for spine in ax_w_cbar.spines.values():
    spine.set_visible(False)


# -------- 中间：同源基因名称 --------
ax_text.axis("off")

for i, (b_name, w_name, pair_target) in enumerate(
    zip(b_names, w_names, target_info)
):
    has_b, has_w = pair_target
    y_center = 1.0 - (i + 0.5) / num_genes

    if b_name:
        c_b = "#3A7CA5" if has_b else "#999999"
        w_b = "bold" if has_b else "normal"

        ax_text.text(
            0.45,
            y_center,
            b_name,
            ha="right",
            va="center",
            fontsize=12,
            fontweight=w_b,
            fontstyle="italic",
            color=c_b,
            transform=ax_text.transAxes,
        )

    if w_name:
        c_w = "#D0703C" if has_w else "#999999"
        w_w = "bold" if has_w else "normal"

        ax_text.text(
            0.55,
            y_center,
            w_name,
            ha="left",
            va="center",
            fontsize=12,
            fontweight=w_w,
            fontstyle="italic",
            color=c_w,
            transform=ax_text.transAxes,
        )

    if i < num_genes - 1:
        current_group = sort_df.loc[i, "group"]
        next_group = sort_df.loc[i + 1, "group"]

        if current_group != next_group:
            sep_y = 1.0 - (i + 1) / num_genes
            ax_text.axhline(
                sep_y,
                color="#BDBDBD",
                lw=1.0,
                linestyle="--",
                xmin=0.15,
                xmax=0.85,
            )

ax_text.text(
    0.5,
    1.02,
    "Orthologous Gene Pairs",
    ha="center",
    va="bottom",
    fontsize=14,
    fontweight="bold",
    transform=ax_text.transAxes,
)


# -------- 底部：圆点大小图例，横向排布，避免遮挡 --------
ax_size_legend.axis("off")

legend_values = (0.5, 1.0, 1.5)
legend_sizes = [float(size_mapper([v])[0]) for v in legend_values]
x_positions = [0.43, 0.50, 0.57]
y_pos = 0.45

ax_size_legend.text(
    0.34,
    y_pos,
    "Bubble size = |log2FC|",
    ha="right",
    va="center",
    fontsize=11,
    fontweight="bold",
    transform=ax_size_legend.transAxes,
)

for x, s, v in zip(x_positions, legend_sizes, legend_values):
    ax_size_legend.scatter(
        x,
        y_pos,
        s=s,
        color="#BDBDBD",
        edgecolors="white",
        linewidths=0.8,
        transform=ax_size_legend.transAxes,
        zorder=3,
    )

    ax_size_legend.text(
        x,
        0.08,
        f"{v:.1f}",
        ha="center",
        va="center",
        fontsize=10,
        transform=ax_size_legend.transAxes,
    )


plt.subplots_adjust(
    top=0.90,
    bottom=0.10,
    left=0.04,
    right=0.97,
)

plt.subplots_adjust(top=0.90, bottom=0.14)


# ==========================================
# 19. 保存
# ==========================================
save_png = base_dir / "Orthologs_BatchPaired_log2FC_bubbleHeatmap.png"
save_pdf = base_dir / "Orthologs_BatchPaired_log2FC_bubbleHeatmap.pdf"

plt.savefig(save_png, dpi=300, bbox_inches="tight")
plt.savefig(save_pdf, format="pdf", bbox_inches="tight")
plt.close()

print("\n🎉 成功！Bubble heatmap 已生成：")
print(f"   - {save_png}")
print(f"   - {save_pdf}")