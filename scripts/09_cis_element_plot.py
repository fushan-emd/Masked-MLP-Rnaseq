import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import networkx as nx
import os
import matplotlib
import matplotlib.lines as mlines

# ================= 0. 全局字体与输出配置 =================
# 配置 PDF 保存为 Type 42 格式（保留字体，支持在 AI/PDF 编辑器中修改文字）
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
# 配置超高清图片分辨率 (PPI/DPI)
DPI_SETTING = 700

# ================= 1. 动态路径与配置区 =================
# 自动获取当前代码所在目录，确保所有输出文件都生成在代码同级目录下
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()

INPUT_FILE = os.path.join(BASE_DIR, "plantCARE_output_PlantCARE_20190.tab")
RENAME_FILE = os.path.join(BASE_DIR, "rename.csv")  # 👈 引用重命名映射文件
OUTPUT_CSV = os.path.join(BASE_DIR, "cis_elements.csv")
OUTPUT_DIST_PDF = os.path.join(BASE_DIR, "promoter_elements.pdf")
OUTPUT_DIST_PNG = os.path.join(BASE_DIR, "promoter_elements.png")
OUTPUT_NET_PNG = os.path.join(BASE_DIR, "Gene_Motif_Network.png")
OUTPUT_NET_PDF = os.path.join(BASE_DIR, "Gene_Motif_Network.pdf")

PROMOTER_LENGTH = 2000
LINE_COLOR = "black"

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
    "ABRE": "#E64B35",        # 红色 (ABA/盐响应)
    "MYB/MBS/MYC": "#4DBBD5", # 蓝色 (干旱/次生代谢)
    "DRE": "#00A087",         # 绿色 (核心脱水/高盐响应)
    "LTR": "#3C5488",         # 深蓝色 (低温响应)
    "TC-rich repeats": "#F39B7F", # 橙色 (防御响应)
    "W-box": "#8491B4",       # 灰蓝色 (WRKY结合)
    "G-box": "#91D1C2",       # 薄荷绿 (光/逆境)
    "MeJA-motif": "#DC0000",  # 深红色 (茉莉酸响应)
    "SA-motif": "#7E6148"     # 棕色 (水杨酸响应)
}
DEFAULT_COLOR = "#B09C85"

# ================= 2. 读取与清洗区 =================
# 📖 读取重命名映射
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
                
                # 净化基因名（去除多余后缀）
                gene_id = gene_id.replace('_promoter_2000bp', '')
                
                # 🔄 应用 rename.csv 中的重命名映射
                gene_id = rename_dict.get(gene_id, gene_id) 
                
                if motif_name in TARGET_MOTIFS:
                    try:
                        start = int(parts[3])
                        length = int(parts[4])
                        strand = parts[5].strip()
                        end = start + length
                        
                        # 同类元件合并命名
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
        
        genes = df['Gene_ID'].unique()
        num_genes = len(genes)

        # ================= 3. 图表一：高清矢量分布图 =================
        fig, ax = plt.subplots(figsize=(12, max(5, num_genes * 0.9)))
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)

        y_positions = np.arange(num_genes)

        for i, gene in enumerate(genes):
            y = y_positions[i]
            # 画启动子主干 (-2000 到 0)
            ax.plot([-PROMOTER_LENGTH, 0], [y, y], color=LINE_COLOR, linewidth=1.5, zorder=1)
            ax.plot([0, 0], [y-0.2, y+0.2], color='black', linewidth=2, zorder=1) # TSS 刻度
            
            gene_motifs = df[df['Gene_ID'] == gene]
            for _, row in gene_motifs.iterrows():
                motif = row['Motif_Name']
                start_pos = row['Start'] - PROMOTER_LENGTH
                width = row['End'] - row['Start']
                
                # 视觉补偿
                if width < 15: width = 15
                strand = str(row['Strand']).strip()
                color = MOTIF_COLORS.get(motif, DEFAULT_COLOR)
                
                rect_height = 0.4
                y_bottom = y if strand == '+' else y - rect_height
                    
                rect = patches.Rectangle(
                    (start_pos, y_bottom), width, rect_height, 
                    facecolor=color, edgecolor='black', linewidth=0.5, alpha=0.9, zorder=2
                )
                ax.add_patch(rect)

        ax.set_yticks(y_positions)
        ax.set_yticklabels(genes, fontsize=13, fontstyle='italic', fontweight='bold') # 放大基因名字号
        ax.set_xlabel("Position relative to Translation Start Site (bp)", fontsize=14, fontweight='bold') # 放大X轴字号
        ax.set_xlim(-PROMOTER_LENGTH - 50, 50)
        ax.set_ylim(-1, num_genes)

        unique_motifs_in_data = df['Motif_Name'].unique()
        legend_patches = [patches.Patch(color=MOTIF_COLORS.get(m, DEFAULT_COLOR), label=m) for m in unique_motifs_in_data]
        # 放大图例字号
        ax.legend(handles=legend_patches, loc='upper right', bbox_to_anchor=(1.25, 1), title="Cis-acting Elements", frameon=False, fontsize=12, title_fontsize=13)
        
        plt.title("Stress-Responsive Cis-acting Elements (Strand-Specific)", fontsize=18, pad=20, fontweight='bold') # 放大标题
        plt.tight_layout()

        plt.savefig(OUTPUT_DIST_PDF, format='pdf', dpi=DPI_SETTING, bbox_inches='tight')
        plt.savefig(OUTPUT_DIST_PNG, format='png', dpi=DPI_SETTING, bbox_inches='tight')
        plt.close()
        print(f"🎉 1. 分布图绘制完成！(DPI: {DPI_SETTING}, 字体可编辑 PDF)")

        # ================= 4. 图表二：双向流向图 (Bipartite / Sankey Style) =================
        import matplotlib.path as mpath  # 导入绘制 S型曲线所需的路径模块
        
        df_cleaned = df.dropna(subset=['Gene_ID', 'Motif_Name'])
        G = nx.Graph()
        
        for _, row in df_cleaned.iterrows():
            gene = row['Gene_ID']
            motif = row['Motif_Name']
            if G.has_edge(gene, motif):
                G[gene][motif]['weight'] += 1
            else:
                G.add_edge(gene, motif, weight=1)
                
        all_nodes = list(G.nodes())
        barley_nodes = sorted([n for n in all_nodes if str(n).startswith('HORVU') or str(n).startswith('Hv')])
        wheat_nodes = sorted([n for n in all_nodes if str(n).startswith('Traes') or str(n).startswith('Ta')])
        
        gene_nodes = barley_nodes + wheat_nodes
        motif_nodes = sorted([n for n in all_nodes if n not in gene_nodes])
        
        # ----------------- 坐标系布局 (左侧基因，右侧元件) -----------------
        pos = {}
        # 左侧基因坐标 (X = -1)
        y_genes = np.linspace(1, -1, max(len(gene_nodes), 1))
        for i, node in enumerate(gene_nodes):
            pos[node] = (-1, y_genes[i])
            
        # 右侧元件坐标 (X = 1)，为了美观，稍微向中间收拢一点点
        y_motifs = np.linspace(0.85, -0.85, max(len(motif_nodes), 1)) 
        for i, node in enumerate(motif_nodes):
            pos[node] = (1, y_motifs[i])
            
        # 动态调整画布高度：基因越多，画布自动拉长，绝对不会拥挤
        fig_height = max(12, len(gene_nodes) * 0.6) 
        fig, ax = plt.subplots(figsize=(14, fig_height), facecolor="#FFFFFF")
        
        max_w = max([G[u][v]['weight'] for u, v in G.edges()]) if G.edges() else 1
        
        # ----------------- 1. 绘制 S 型平滑连线 (Sankey 流向) -----------------
        for u, v in G.edges():
            weight = G[u][v]['weight']
            
            # 确保方向：连线始终从左侧(基因)画到右侧(元件)
            if u in gene_nodes:
                left_n, right_n = u, v
            else:
                left_n, right_n = v, u
                
            x0, y0 = pos[left_n]
            x1, y1 = pos[right_n]
            
            # 设定连线颜色
            edge_color = '#5DA5DA' if left_n in barley_nodes else '#FAA43A'
            
            # 构建贝塞尔曲线：水平向右拉伸控制点，形成完美的 S 型水流
            tension = 0.8  # 控制弯曲程度，0.8通常最优雅
            verts = [(x0, y0), (x0 + tension, y0), (x1 - tension, y1), (x1, y1)]
            codes = [mpath.Path.MOVETO, mpath.Path.CURVE4, mpath.Path.CURVE4, mpath.Path.CURVE4]
            path = mpath.Path(verts, codes)
            
            # 添加连线，zorder=1垫在最底层
            patch = patches.PathPatch(path, facecolor='none', edgecolor=edge_color, 
                                      alpha=0.35, lw=(weight / max_w) * 8 + 1, zorder=1)
            ax.add_patch(patch)


        # ----------------- 2. 绘制节点圆圈 -----------------
        # 绘制左侧大麦基因圆点
        if barley_nodes:
            nodes_barley = nx.draw_networkx_nodes(G, pos, ax=ax, nodelist=barley_nodes, 
                                                  node_color='#5DA5DA', node_size=150, node_shape='o', alpha=1)
            nodes_barley.set_zorder(2) # <--- 单独设置图层顺序
            
        # 绘制左侧小麦基因圆点
        if wheat_nodes:
            nodes_wheat = nx.draw_networkx_nodes(G, pos, ax=ax, nodelist=wheat_nodes, 
                                                 node_color='#FAA43A', node_size=150, node_shape='o', alpha=1)
            nodes_wheat.set_zorder(2)  # <--- 单独设置图层顺序
                               
        # 绘制右侧元件大圆 (包含保底大小，防遮挡)
        for motif in motif_nodes:
            color = MOTIF_COLORS.get(motif, '#F15854')
            node_size = 1200 + G.degree(motif) * 400 
            nodes_motif = nx.draw_networkx_nodes(G, pos, ax=ax, nodelist=[motif], node_color=color, 
                                                 node_size=node_size, node_shape='o', alpha=1.0, 
                                                 edgecolors='white', linewidths=3)
            nodes_motif.set_zorder(2)  # <--- 单独设置图层顺序
        # ----------------- 3. 添加文字标签 -----------------
        # 左侧：基因名称 (斜体 + 大麦蓝/小麦橙区分)
        for node in gene_nodes:
            x, y = pos[node]
            text_color = '#5DA5DA' if node in barley_nodes else '#FAA43A'
            # 文字放在圆点左侧 (x - 0.08)
            ax.text(x - 0.08, y, node, ha='right', va='center', fontsize=15, 
                    fontweight='bold', fontstyle='italic', color=text_color)
                    
        # 右侧：元件名称 (带白底黑框，放在圆点右侧)
        bbox_props = dict(boxstyle="round,pad=0.3", fc="white", ec="black", lw=1.2, alpha=0.9)
        for motif in motif_nodes:
            x, y = pos[motif]
            # 文字放在圆点右侧 (x + 0.12)
            ax.text(x + 0.12, y, motif, ha='left', va='center', fontsize=16, 
                    fontweight='bold', color='black', bbox=bbox_props)

        # ----------------- 4. 画布配置与导出 -----------------
        # 扩大 X 轴范围，给左右两边的文字留出足够的呼吸空间
        ax.set_xlim(-1.8, 1.8) 
        ax.set_ylim(-1.05, 1.05)
        plt.axis('off') # 隐藏坐标轴
        
        # 底部中央图例
        barley_line = mlines.Line2D([], [], color='#5DA5DA', linewidth=6, label='Barley Flux')
        wheat_line = mlines.Line2D([], [], color='#FAA43A', linewidth=6, label='Wheat Flux')
        plt.legend(handles=[barley_line, wheat_line], loc='lower center', bbox_to_anchor=(0.5, -0.05), 
                   fontsize=14, frameon=True, ncol=2, title="Interaction Flux", title_fontproperties={'weight':'bold', 'size':16})
        
        # 顶部标题
        plt.title('Bipartite Regulatory Network: Genes to Cis-Elements', 
                  fontsize=24, fontweight='bold', pad=20, color='#222222')
        
        plt.tight_layout()
        
        # 导出文件
        plt.savefig(OUTPUT_NET_PNG, format='png', dpi=DPI_SETTING, bbox_inches='tight')
        plt.savefig(OUTPUT_NET_PDF, format='pdf', dpi=DPI_SETTING, bbox_inches='tight')
        plt.close()
        print(f"🎉 2. 双向流向网络图绘制完成！已保存为：\n   - {OUTPUT_NET_PNG}\n   - {OUTPUT_NET_PDF}\n   (DPI: {DPI_SETTING}, 字体可编辑 PDF)")

    else:
        print("⚠️ 过滤后数据为空，请检查你的目标过滤名单。")
else:
    print(f"❌ 找不到文件 {INPUT_FILE}，请确认文件名和绝对路径。当前工作路径为: {BASE_DIR}")