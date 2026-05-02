import os
import pickle
import re
import sys
import warnings
from pathlib import Path

import matplotlib as mpl
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap

# 忽略 SHAP 自带的 tight_layout 警告
warnings.filterwarnings("ignore", message=".*tight_layout.*")

# 保持矢量图文字可编辑 (Type 42)
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42
mpl.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"]
mpl.rcParams["axes.unicode_minus"] = False

sns.set_theme(style="white", context="paper")


# ==================== 1. 工具函数与数据加载 ====================
def resolve_base_dir() -> Path:
    if getattr(sys, "frozen", False): return Path(sys.executable).resolve().parent
    if "__file__" in globals(): return Path(__file__).resolve().parent
    return Path.cwd()

def read_csv_strip(path: Path, **kwargs) -> pd.DataFrame:
    df = pd.read_csv(path, **kwargs)
    df.columns = [str(c).strip() for c in df.columns]
    return df

def load_rename_maps(path: Path):
    """读取注释映射，同时提取 MLP 排行信息"""
    rename_map = {}
    mlp_map = {}
    encodings = ("utf-8-sig", "gb18030", "gbk", "latin1")
    for enc in encodings:
        try:
            df = pd.read_csv(path, encoding=enc, engine="python")
            df.columns = [str(c).strip() for c in df.columns]
            if "GeneID" in df.columns and "Annoname" in df.columns:
                for _, row in df.dropna(subset=["GeneID", "Annoname"]).iterrows():
                    gid = str(row["GeneID"]).strip()
                    rename_map[gid] = str(row["Annoname"]).strip()
                    if "MLP impact ranks" in df.columns:
                        mlp_map[gid] = str(row["MLP impact ranks"]).strip()
                return rename_map, mlp_map
        except Exception:
            continue
    return rename_map, mlp_map

def compact_gene_id(gene_id: str) -> str:
    gene_id = str(gene_id).strip()
    m_barley = re.match(r"^HORVUMOREXr(\d+)HG(\d+)$", gene_id)
    if m_barley: return f"Hv{m_barley.group(1)}-{m_barley.group(2)[-4:]}"
    m_wheat = re.match(r"^TraesCS([0-9][ABD])02G(\d+)$", gene_id)
    if m_wheat: return f"Ta{m_wheat.group(1)}-{m_wheat.group(2)[-4:]}"
    return gene_id[:12] + "..." if len(gene_id) > 15 else gene_id

def prep_vote_df(vote_df: pd.DataFrame) -> tuple[pd.DataFrame, list, set]:
    vote_df = vote_df.copy()
    vote_df["GeneID"] = vote_df["GeneID"].astype(str).str.strip()
    vote_df["Total_Votes"] = pd.to_numeric(vote_df["Total_Votes"], errors="coerce").fillna(0)
    model_cols = [c for c in vote_df.columns if c not in {"GeneID", "Total_Votes"}]
    consensus_genes = set(vote_df[vote_df["Total_Votes"] > 0]["GeneID"])
    return vote_df, model_cols, consensus_genes


# ==================== 2. 自定义作图模块 ====================
def draw_upset_panel(fig: plt.Figure, subplot_spec, vote_df: pd.DataFrame, model_cols: list, species_title: str, accent_color: str) -> tuple:
    """绘制带渐变色的纯净版 UpSet 图（去除引线标注）"""
    gs_sub = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=subplot_spec, height_ratios=[2.2, 1.2], hspace=0.08)
    ax_bar = fig.add_subplot(gs_sub[0])
    ax_mat = fig.add_subplot(gs_sub[1], sharex=ax_bar)

    df_bin = vote_df.copy()
    for c in model_cols:
        df_bin[c] = pd.to_numeric(df_bin[c], errors="coerce").fillna(0).astype(int)
    
    df_bin_filtered = df_bin.loc[(df_bin[model_cols] != 0).any(axis=1)]
    inters = df_bin_filtered.groupby(model_cols).size().sort_values(ascending=False).head(15)

    x = np.arange(len(inters))
    
    votes_list = []
    for idx in inters.index:
        idx_tuple = (idx,) if isinstance(idx, (int, str)) else idx
        votes_list.append(sum(idx_tuple)) 
        
    max_possible_votes = len(model_cols)
    palette = sns.light_palette(accent_color, n_colors=max_possible_votes + 3)
    bar_colors = [palette[v + 2] for v in votes_list]

    ax_bar.bar(x, inters.values, color=bar_colors, edgecolor="none", width=0.6)
    
    ax_bar.set_ylabel("Intersection Size", fontsize=12, fontweight="bold")
    ax_bar.set_title(f"{species_title}: Models Consensus (UpSet)", fontsize=18, fontweight="bold", pad=15)
    
    ax_bar.spines['top'].set_visible(False)
    ax_bar.spines['right'].set_visible(False)
    ax_bar.tick_params(axis="x", bottom=False, labelbottom=False)
    ax_bar.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.4)
    
    max_inter_val = max(inters.values) if len(inters) > 0 else 1
    # 取消之前防爆顶的 1.5 倍扩宽，回归紧凑排版
    ax_bar.set_ylim(0, max_inter_val * 1.15)

    for i, (idx, val) in enumerate(inters.items()):
        # 1. 绘制柱子上方原本的交集数量数字（黑色）
        ax_bar.text(i, val + (max_inter_val * 0.02), str(val), 
                    ha='center', va='bottom', fontsize=11, fontweight='bold', color="#444444")

        # 2. 下方矩阵点阵绘制
        idx_tuple = (idx,) if isinstance(idx, (int, str)) else idx
        active_y = [j for j, is_in in enumerate(idx_tuple) if is_in == 1]
        inactive_y = [j for j, is_in in enumerate(idx_tuple) if is_in == 0]

        if len(active_y) > 1:
            ax_mat.plot([i, i], [min(active_y), max(active_y)], color='#B0B0B0', lw=2, zorder=1)
            
        ax_mat.scatter([i]*len(active_y), active_y, color=bar_colors[i], s=110, zorder=2)
        ax_mat.scatter([i]*len(inactive_y), inactive_y, color='#E0E0E0', s=50, zorder=2)

    ax_mat.set_yticks(np.arange(len(model_cols)))
    ax_mat.set_yticklabels(model_cols, fontsize=12, fontweight="bold")
    ax_mat.set_xlim(-0.6, len(inters) - 0.4)
    ax_mat.set_ylim(-0.5, len(model_cols) - 0.5)
    ax_mat.invert_yaxis()
    for spine in ax_mat.spines.values(): spine.set_visible(False)
    ax_mat.tick_params(axis="both", length=0)
    
    return ax_bar, ax_mat

def draw_shap_panel(ax_shap, shap_expl_path, shap_raw_path, x_test_path, target_genes, rename_map, species_title, accent_color):
    plt.sca(ax_shap)
    with open(shap_expl_path, "rb") as f: shap_expl = pickle.load(f)
    shap_values_raw = np.load(shap_raw_path)
    X_test = np.load(x_test_path)
    
    if shap_values_raw.ndim == 3: shap_values_raw = shap_values_raw[:, :, 1]
    
    original_features = []
    if hasattr(shap_expl, 'feature_names') and shap_expl.feature_names is not None:
        original_features = shap_expl.feature_names
    elif hasattr(shap_expl, 'columns'):
        original_features = shap_expl.columns
    elif isinstance(shap_expl, list) and hasattr(shap_expl[0], 'feature_names'):
        original_features = shap_expl[0].feature_names
        
    compact_features = [rename_map.get(str(g), compact_gene_id(str(g))) for g in original_features] if len(original_features) > 0 else None

    shap.summary_plot(
        shap_values=shap_values_raw, features=X_test, feature_names=compact_features,
        max_display=20, plot_size=None, show=False, plot_type="dot", color_bar=True
    )
    
    ax_shap.set_title(f"{species_title}: SHAP Global Importance", fontsize=18, fontweight="bold", pad=25)
    ax_shap.set_xlabel("SHAP Value (Impact on Model Output)", fontsize=12, fontweight="bold")
    ax_shap.grid(axis="x", linestyle="--", linewidth=0.8, alpha=0.35, color="#8A8F98")
    
    mapped_targets = {rename_map.get(g, compact_gene_id(g)) for g in target_genes}
    for label in ax_shap.get_yticklabels():
        if label.get_text() in mapped_targets:
            label.set_color(accent_color)
            label.set_fontweight("bold")
            

# ==================== 3. 🌟 升级：带列宽自适应的原始 ID 表格 🌟 ====================
def draw_table_panel(ax_table, vote_df: pd.DataFrame, rename_map: dict, mlp_map: dict, species_title: str, accent_color: str):
    ax_table.axis('off')
    vote_df = vote_df.copy()
    vote_df["Total_Votes"] = pd.to_numeric(vote_df["Total_Votes"], errors="coerce").fillna(0)
    
    target_genes = set(rename_map.keys())
    
    # 获取当前物种包含的所有目标基因
    species_targets = [g for g in target_genes if g in vote_df['GeneID'].values]
    
    # 获取按票数排列的高票基因
    df_top = vote_df[vote_df["Total_Votes"] > 0].sort_values("Total_Votes", ascending=False)
    
    # 强制让 target 基因先入表，再用其它高票基因补齐 15 席
    display_gids = species_targets.copy()
    for gid in df_top['GeneID']:
        if gid not in display_gids:
            display_gids.append(gid)
        if len(display_gids) >= 15:
            break
            
    table_data = []
    for gid in display_gids:
        # 获取票数
        row_data = vote_df[vote_df['GeneID'] == gid].iloc[0]
        votes = int(row_data['Total_Votes'])
        
        # 获取美化名
        anno = rename_map.get(gid, "-")
        
        # 获取 MLP 排名
        mlp_rank = mlp_map.get(gid, "-")
        if mlp_rank in ["/", "nan", ""]: 
            mlp_rank = "-"
            
        # 强制使用完整原始 ID
        table_data.append([gid, votes, mlp_rank, anno])
        
    # 表格重新排序：优先显示靶标基因（标红的），同类内部按票数降序
    table_data.sort(key=lambda x: (x[0] not in species_targets, -x[1]))
        
    col_labels = ["Original Gene ID", "Votes", "MLP Rank", "Annotation"]
    
    # 🌟 核心修复：强行干预列宽分配 (总和 1.0)
    # 分配策略：给第一列分配极其宽敞的 42% 空间，防止长 ID 超出格子
    custom_col_widths = [0.42, 0.12, 0.16, 0.30]
    
    table = ax_table.table(
        cellText=table_data, 
        colLabels=col_labels, 
        colWidths=custom_col_widths, 
        cellLoc='center', 
        loc='center'
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(9.5) 
    table.scale(1, 1.8)  
    
    for (i, j), cell in table.get_celld().items():
        cell.set_edgecolor('#DDDDDD')  
        if i == 0:
            cell.set_text_props(weight='bold', color='white', fontsize=11)
            cell.set_facecolor(accent_color)
        else:
            if i % 2 == 0:
                cell.set_facecolor('#F8F8F8')
            
            gene_id_in_row = table_data[i-1][0]
            # 如果是目标基因（包括 MLP 前三即使低票的），整行标红加粗
            if gene_id_in_row in target_genes:
                cell.get_text().set_color('#D62728')
                cell.get_text().set_fontweight('bold')
                
    ax_table.set_title(f"{species_title}: Top Hub Genes List", fontsize=18, fontweight="bold", pad=25)
    
    ax_table.text(0.5, -0.05, "★ Red/Bold rows: Hub Selected in Analysis", 
                  transform=ax_table.transAxes, ha='center', va='top', 
                  fontsize=11, fontweight='bold', color='#D62728')


# ==================== Main ====================
def main():
    base_dir = resolve_base_dir()

    # 读取重命名目标基因与 MLP 排名
    rename_map, mlp_map = load_rename_maps(base_dir / "rename.csv")
    target_genes = set(rename_map.keys())
    
    try:
        b_vote = pd.read_csv(base_dir / "B_Supplementary_Table_Gene_Matrix.csv")
        w_vote = pd.read_csv(base_dir / "W_Supplementary_Table_Gene_Matrix.csv")
    except:
        print("Error: Voting csv not found.")
        sys.exit(1)
        
    b_vote_proc, b_model_cols, _ = prep_vote_df(b_vote)
    w_vote_proc, w_model_cols, _ = prep_vote_df(w_vote)

    fig = plt.figure(figsize=(25, 14), facecolor="white")
    # 🌟 核心修复 2：进一步拉宽表格整体占地比例 (将 1.0 升至 1.25)
    gs = gridspec.GridSpec(2, 3, width_ratios=[1.0, 1.35, 1.25], hspace=0.35, wspace=0.25, 
                           left=0.04, right=0.96, top=0.92, bottom=0.08)
    
    # === 大麦 (Barley) Row ===
    # 注意：这里的参数已经移除了 rename_map
    ax_b_upset_bar, ax_b_upset_mat = draw_upset_panel(fig, gs[0, 0], b_vote_proc, b_model_cols, "Barley", "#3A7CA5")
    ax_b_shap = fig.add_subplot(gs[0, 1])
    draw_shap_panel(ax_b_shap, base_dir / "barley_shap_explanation.pkl", base_dir / "barley_shap_values_raw.npy", 
                    base_dir / "barley_X_test_scaled_data.npy", target_genes, rename_map, "Barley", "#3A7CA5")
    
    ax_b_table = fig.add_subplot(gs[0, 2])
    draw_table_panel(ax_b_table, b_vote, rename_map, mlp_map, "Barley", "#3A7CA5")

    # === 小麦 (Wheat) Row ===
    ax_w_upset_bar, ax_w_upset_mat = draw_upset_panel(fig, gs[1, 0], w_vote_proc, w_model_cols, "Wheat", "#D0703C")
    ax_w_shap = fig.add_subplot(gs[1, 1])
    draw_shap_panel(ax_w_shap, base_dir / "wheat_shap_explanation.pkl", base_dir / "wheat_shap_values_raw.npy", 
                    base_dir / "wheat_X_test_scaled_data.npy", target_genes, rename_map, "Wheat", "#D0703C")
    
    ax_w_table = fig.add_subplot(gs[1, 2])
    draw_table_panel(ax_w_table, w_vote, rename_map, mlp_map, "Wheat", "#D0703C")
    
    ax_b_upset_bar.text(-0.15, 1.15, "A", transform=ax_b_upset_bar.transAxes, fontsize=24, fontweight="bold")
    ax_b_shap.text(-0.25, 1.15, "B", transform=ax_b_shap.transAxes, fontsize=24, fontweight="bold")
    ax_b_table.text(-0.10, 1.15, "C", transform=ax_b_table.transAxes, fontsize=24, fontweight="bold")

    ax_w_upset_bar.text(-0.15, 1.15, "D", transform=ax_w_upset_bar.transAxes, fontsize=24, fontweight="bold")
    ax_w_shap.text(-0.25, 1.15, "E", transform=ax_w_shap.transAxes, fontsize=24, fontweight="bold")
    ax_w_table.text(-0.10, 1.15, "F", transform=ax_w_table.transAxes, fontsize=24, fontweight="bold")

    out_pdf = base_dir / "Figure3_UpSet_SHAP_Table_editable.pdf"
    out_png = base_dir / "Figure3_UpSet_SHAP_Table_700ppi.png"

    print("\n⏳ 正在渲染终极纯净版图表（去标注 UpSet + 定制宽列表格），请稍候...")
    fig.savefig(out_pdf, format="pdf", bbox_inches="tight", facecolor="white")
    fig.savefig(out_png, format="png", dpi=700, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print("[OK] Figure 3 generated successfully.")
    print(f"[PNG] {out_png}")

if __name__ == "__main__":
    main()