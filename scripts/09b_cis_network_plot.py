import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as mcolors
import numpy as np
import networkx as nx
import os
import matplotlib
import matplotlib.lines as mlines

# ================= 0. 全局字体与输出配置 =================
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
DPI_SETTING = 700

# ================= 1. 动态路径与配置区 =================
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()

INPUT_FILE = os.path.join(BASE_DIR, "plantCARE_output_PlantCARE_20190.tab")
RENAME_FILE = os.path.join(BASE_DIR, "rename.csv")
OUTPUT_CSV = os.path.join(BASE_DIR, "cis_elements.csv")
OUTPUT_DIST_PDF = os.path.join(BASE_DIR, "promoter_elements_beautified.pdf") 
OUTPUT_DIST_PNG = os.path.join(BASE_DIR, "promoter_elements_beautified.png") 
OUTPUT_NET_PNG = os.path.join(BASE_DIR, "Gene_Motif_Network.png")
OUTPUT_NET_PDF = os.path.join(BASE_DIR, "Gene_Motif_Network.pdf")

PROMOTER_LENGTH = 2000
BACKBONE_COLOR = "#333333"

# 🎯 扩充版核心过滤名单
TARGET_MOTIFS = [
    'ABRE', 'ABRE3a', 'ABRE4', 'AT~ABRE',
    'MBS', 'MYB', 'Myb', 'MYB recognition site', 'MYB-like sequence', 'Myb-binding site', 'MYC', 'Myc',
    'DRE', 'DRE core',
    'TC-rich repeats',
    'LTR',
    'CGTCA-motif', 'TGACG-motif',
    'TCA-element',
    'W box', 'W-box',
    'G-box', 'G-Box'
]

# 科研级配色方案
MOTIF_COLORS = {
    "ABRE": "#E64B35",
    "MYB/MBS/MYC": "#4DBBD5",
    "DRE": "#00A087",
    "LTR": "#3C5488",
    "TC-rich repeats": "#F39B7F",
    "W-box": "#8491B4",
    "G-box": "#91D1C2",
    "MeJA-motif": "#DC0000",
    "SA-motif": "#7E6148"
}
DEFAULT_COLOR = "#B09C85"

# ================= 辅助函数：生成颜色渐变 =================
def adjust_lightness(color, amount=0.5):
    try:
        c = mcolors.cnames[color]
    except:
        c = color
    c = mcolors.to_rgb(c)
    import colorsys
    r, g, b = colorsys.rgb_to_hls(*c)
    return colorsys.hls_to_rgb(r, max(0, min(1, g * amount)), b)

# ================= 2. 读取与清洗区 =================
rename_dict = {}
if os.path.exists(RENAME_FILE):
    try:
        rename_df = pd.read_csv(RENAME_FILE)
        if 'GeneID' in rename_df.columns and 'Annoname' in rename_df.columns:
            rename_dict = dict(zip(rename_df['GeneID'], rename_df['Annoname']))
        else:
            rename_dict = dict(zip(rename_df.iloc[:, 0], rename_df.iloc[:, -1]))
        print(f"📖 成功读取 {len(rename_dict)} 个基因的重命名映射规则。")
    except Exception as e:
        print(f"⚠️ 读取 {RENAME_FILE} 失败: {e}")

cleaned_data = []
if os.path.exists(INPUT_FILE):
    print(f"⏳ 正在读取数据、清洗过滤并构建关系网络...\n工作目录: {BASE_DIR}")
    with open(INPUT_FILE, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 6:
                gene_id = parts[0].strip()
                motif_name = parts[1].strip()
                
                gene_id = gene_id.replace('_promoter_2000bp', '')
                gene_id = rename_dict.get(gene_id, gene_id) 
                
                if motif_name in TARGET_MOTIFS:
                    try:
                        start = int(parts[3])
                        length = int(parts[4])
                        strand = parts[5].strip()
                        end = start + length
                        
                        display_name = motif_name
                        upper_motif = motif_name.upper()
                        
                        if 'MYB' in upper_motif or 'MBS' in upper_motif or 'MYC' in upper_motif:
                            display_name = 'MYB/MBS/MYC'
                        elif 'ABRE' in upper_motif:
                            display_name = 'ABRE'
                        elif 'DRE' in upper_motif:
                            display_name = 'DRE'
                        elif 'W BOX' in upper_motif or 'W-BOX' in upper_motif:
                            display_name = 'W-box'
                        elif 'G-BOX' in upper_motif:
                            display_name = 'G-box'
                        elif 'CGTCA' in upper_motif or 'TGACG' in upper_motif:
                            display_name = 'MeJA-motif'
                        elif 'TCA' in upper_motif:
                            display_name = 'SA-motif'

                        cleaned_data.append({
                            'Gene_ID': gene_id,
                            'Motif_Name': display_name,
                            'Start': start,
                            'End': end,
                            'Strand': strand
                        })
                    except ValueError:
                        continue

    df = pd.DataFrame(cleaned_data)
    if not df.empty:
        df = df.drop_duplicates().sort_values(by=['Gene_ID', 'Start'])
        df.to_csv(OUTPUT_CSV, index=False)
        print(f"✅ 数据清洗完成，共提取 {len(df)} 个抗逆元件，已保存至 {OUTPUT_CSV}。")
        
        # ----------------- 分类并排序基因 -----------------
        raw_genes = df['Gene_ID'].unique()
        barley_genes = sorted([g for g in raw_genes if str(g).startswith('HORVU') or str(g).startswith('Hv')])
        wheat_genes = sorted([g for g in raw_genes if str(g).startswith('Traes') or str(g).startswith('Ta')])
        other_genes = sorted([g for g in raw_genes if g not in barley_genes and g not in wheat_genes])
        
        genes = wheat_genes[::-1] + other_genes[::-1] + barley_genes[::-1]
        num_genes = len(genes)

        # ================= 3. 美化版：启动子元件分布图 =================
        row_height = 0.8
        fig_height = max(5, num_genes * row_height)
        
        # 【修改点 1】：缩小宽度，从 14 改为 10，使其更紧凑
        fig, ax = plt.subplots(figsize=(10, fig_height), facecolor='white')
        
        for spine in ax.spines.values(): spine.set_visible(False)
        ax.yaxis.set_ticks_position('none') 

        y_positions = np.arange(num_genes)

        v_ticks = np.arange(-PROMOTER_LENGTH, 1, 500)
        for vt in v_ticks:
            ax.axvline(x=vt, color='#E6E6E6', linestyle='-', linewidth=0.8, zorder=0)

        for i, gene in enumerate(genes):
            y = y_positions[i]
            ax.plot([-PROMOTER_LENGTH, 0], [y, y], color=BACKBONE_COLOR, linewidth=1.5, zorder=1)
            
            arrow_width = 0.1
            arrow_head_len = 50
            ax.arrow(0, y, arrow_head_len, 0, width=arrow_width, head_width=0.4, 
                     head_length=arrow_head_len, color='black', length_includes_head=True, zorder=3)
            # 因为变窄了，稍微调整 TSS 文字的距离，防止挤在一起
            ax.text(80, y, 'TSS', fontsize=11, fontweight='bold', va='center')
            
            gene_motifs = df[df['Gene_ID'] == gene]
            for _, row in gene_motifs.iterrows():
                motif = row['Motif_Name']
                start_pos = row['Start'] - PROMOTER_LENGTH
                width = row['End'] - row['Start']
                
                # 因为变窄了，视觉补偿也要相应缩小一点，否则色块会显得很宽
                if width < 12: width = 12
                strand = str(row['Strand']).strip()
                color = MOTIF_COLORS.get(motif, DEFAULT_COLOR)
                
                rect_height = 0.35
                y_bottom = y + 0.05 if strand == '+' else y - rect_height - 0.05
                gradient_color = adjust_lightness(color, 0.8) 
                
                rect = patches.FancyBboxPatch(
                    (start_pos, y_bottom), width, rect_height, 
                    boxstyle="round,pad=0,rounding_size=0",
                    facecolor=adjust_lightness(color, 1.1), 
                    edgecolor=gradient_color, 
                    linewidth=1.0, 
                    alpha=0.95, zorder=2
                )
                ax.add_patch(rect)

        ax.set_yticks(y_positions)
        ax.set_yticklabels(genes, fontsize=14, fontstyle='italic', fontweight='bold', ha='right')
        
        for tick_label in ax.get_yticklabels():
            gene_text = tick_label.get_text()
            if gene_text in barley_genes:
                tick_label.set_color('#5DA5DA') 
            elif gene_text in wheat_genes:
                tick_label.set_color('#FAA43A') 
            else:
                tick_label.set_color('#333333') 
                
        ax.set_xlabel(r"$\leftarrow$ Upstream ($\mathrm{bp}$)", fontsize=16, fontweight='bold', labelpad=15)
        ax.set_xticks(v_ticks)
        ax.set_xticklabels([f"{tick}" for tick in v_ticks], fontsize=13)
        
        # 【修改点 2】：收紧右侧多余的空间，从 250 改为 150
        ax.set_xlim(-PROMOTER_LENGTH - 50, 150)
        ax.set_ylim(-1, num_genes)

        unique_motifs_in_data = sorted(df['Motif_Name'].unique())
        legend_patches = [patches.Patch(facecolor=adjust_lightness(MOTIF_COLORS.get(m, DEFAULT_COLOR), 1.1), edgecolor=adjust_lightness(MOTIF_COLORS.get(m, DEFAULT_COLOR), 0.8), label=m) for m in unique_motifs_in_data]
        
        # 变窄后，图例最好分两行显示，避免拥挤，修改 ncol
        ax.legend(handles=legend_patches, loc='upper center', bbox_to_anchor=(0.5, -0.15), 
                  title="Cis-acting Elements", frameon=False, fontsize=12, title_fontsize=14, ncol=min(4, len(unique_motifs_in_data)))
        
        plt.title("Stress-Responsive Cis-acting Elements", fontsize=20, pad=20, fontweight='bold', color='#222222') 
        plt.tight_layout()

        plt.savefig(OUTPUT_DIST_PDF, format='pdf', dpi=DPI_SETTING, bbox_inches='tight')
        plt.savefig(OUTPUT_DIST_PNG, format='png', dpi=DPI_SETTING, bbox_inches='tight')
        plt.close()
        print(f"🎉 1. 紧凑美化版分布图绘制完成！(DPI: {DPI_SETTING}, 字体可编辑 PDF)")
