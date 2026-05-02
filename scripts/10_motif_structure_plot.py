import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from Bio import Phylo
import xml.etree.ElementTree as ET
import matplotlib

# ================= 0. 全局配置与参数 =================
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
DPI_SETTING = 700

# 综合 InterPro 与 TOMTOM 数据库鉴定结果
MOTIF_ANNOTATIONS = {
    1: "Disordered",
    2: "Ankyrin",
    3: "RING-type",
    4: "Homeobox",
    9: "EF-hand",
    10: "Solanesyltrans"
}

COLORS = [
    "#E64B35", "#4DBBD5", "#00A087", "#3C5488", "#F39B7F", 
    "#8491B4", "#91D1C2", "#DC0000", "#7E6148", "#B09C85",
    "#FF9F40", "#9966FF", "#FF6384", "#36A2EB", "#4BC0C0"
]

# ================= 1. 动态路径配置 =================
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()

NWK_FILE = os.path.join(BASE_DIR, "tree.nwk")
MEME_XML = os.path.join(BASE_DIR, "meme.xml")
RENAME_FILE = os.path.join(BASE_DIR, "rename.csv")
OUTPUT_PDF = os.path.join(BASE_DIR, "Circular_Tree_and_Motifs.pdf")
OUTPUT_PNG = os.path.join(BASE_DIR, "Circular_Tree_and_Motifs.png")

print(f"⏳ 正在加载并解析文件...\n工作目录: {BASE_DIR}")

# ================= 2. 核心数据解析 =================
# 2.1 重命名映射
rename_dict = {}
if os.path.exists(RENAME_FILE):
    try:
        rename_df = pd.read_csv(RENAME_FILE)
        if 'GeneID' in rename_df.columns and 'Annoname' in rename_df.columns:
            rename_dict = dict(zip(rename_df['GeneID'], rename_df['Annoname']))
        else:
            rename_dict = dict(zip(rename_df.iloc[:, 0], rename_df.iloc[:, -1]))
    except Exception as e:
        pass

# 2.2 进化树解析
tree = Phylo.read(NWK_FILE, "newick")
terminals = tree.get_terminals()
num_leaves = len(terminals)

y_pos = {leaf: i for i, leaf in enumerate(terminals)}
for clade in tree.get_nonterminals(order='postorder'):
    y_pos[clade] = sum(y_pos[c] for c in clade.clades) / len(clade.clades)

depths = tree.depths()
max_depth = max(depths.values()) if depths else 1

# 2.3 MEME 解析
meme_tree = ET.parse(MEME_XML)
root = meme_tree.getroot()
seq_container = root.find('training_set') or root.find('sequences')

seq_info = {}
for seq in seq_container.findall('sequence'):
    raw_name = seq.get('name')
    if not raw_name: continue
    clean_name = raw_name.replace('_promoter_2000bp', '').strip()
    seq_info[seq.get('id')] = {
        'original_name': clean_name,
        'renamed': rename_dict.get(clean_name, clean_name),
        'length': int(seq.get('length', 0))
    }

motif_widths = {m.get('id'): int(m.get('width')) for m in root.find('motifs').findall('motif')}
motif_hits = {seq_data['renamed']: [] for seq_data in seq_info.values()}
scanned_sites = root.find('scanned_sites_summary')

if scanned_sites is not None:
    for scanned_seq in scanned_sites.findall('scanned_sites'):
        seq_id = scanned_seq.get('sequence_id')
        if seq_id not in seq_info: continue
        seq_renamed = seq_info[seq_id]['renamed']
        for hit in scanned_seq.findall('scanned_site'):
            m_id = hit.get('motif_id')
            m_num = int(m_id.replace('motif_', '')) if m_id.replace('motif_', '').isdigit() else 1
            motif_hits[seq_renamed].append({
                'motif_num': m_num,
                'start': int(hit.get('position', 0)),
                'width': motif_widths.get(m_id, 10)
            })

# ================= 3. 极坐标数学映射 =================
# 将线性 Y 轴转化为极坐标角度 (留出 10% 的开口，避免头尾相撞)
angle_start = 0.05 * math.pi
angle_end = 1.95 * math.pi
theta_range = angle_end - angle_start
angles = {node: angle_start + (y_pos[node] / (max(1, num_leaves - 1))) * theta_range for node in y_pos}

max_seq_length = max([info['length'] for info in seq_info.values()] + [1])

# ================= 4. 开始绘制圆环图 =================
print("🎨 正在构建极坐标系圆环图...")
# 开启极坐标投影 (Polar)
fig, ax = plt.subplots(figsize=(14, 14), subplot_kw={'projection': 'polar'})
ax.axis('off')

# ----------------- 4.1 绘制中心圆环树 -----------------
for clade in tree.get_nonterminals():
    r_parent = (depths[clade] / max_depth) * 100
    
    # 画圆弧连接子节点
    theta_min = min(angles[c] for c in clade.clades)
    theta_max = max(angles[c] for c in clade.clades)
    theta_vals = np.linspace(theta_min, theta_max, 50)
    ax.plot(theta_vals, [r_parent]*50, 'k-', lw=1.5)
    
    # 画放射状的分支
    for child in clade.clades:
        r_child = (depths[child] / max_depth) * 100
        theta_child = angles[child]
        ax.plot([theta_child, theta_child], [r_parent, r_child], 'k-', lw=1.5)

# ----------------- 4.2 绘制基因标签与 Motif 放射线 -----------------
renamed_labels = [(rename_dict.get(leaf.name.replace('_promoter_2000bp', '').strip(), leaf.name), angles[leaf], leaf) for leaf in terminals]

# 定义环的半径区间
R_TREE_MAX = 100
R_LABEL_POS = 108     # 名字所在位置
R_SEQ_START = 135     # Motif 序列开始位置
R_SEQ_RANGE = 100     # Motif 占据的宽度跨度

for label, theta, leaf in renamed_labels:
    # 引出虚线
    r_leaf = (depths[leaf] / max_depth) * 100
    ax.plot([theta, theta], [r_leaf, R_LABEL_POS - 2], color='gray', linestyle=':', lw=1, alpha=0.5)
    
    # 标签防倒置计算
    rot = np.degrees(theta)
    if 90 < rot < 270:
        rot -= 180
        ha = 'right'
    else:
        ha = 'left'
        
    # 物种颜色分配
    if str(label).startswith('Hv') or str(label).startswith('HORVU'):
        tc = '#5DA5DA'
    elif str(label).startswith('Ta') or str(label).startswith('Traes'):
        tc = '#FAA43A'
    else:
        tc = 'black'
        
    ax.text(theta, R_LABEL_POS, label, rotation=rot, rotation_mode='anchor', ha=ha, va='center', color=tc, fontsize=10, fontstyle='italic', fontweight='bold')

    # 画外围灰色主序列线
    seq_len = next((info['length'] for info in seq_info.values() if info['renamed'] == label), max_seq_length)
    r_seq_end = R_SEQ_START + (seq_len / max_seq_length) * R_SEQ_RANGE
    ax.plot([theta, theta], [R_SEQ_START, r_seq_end], color='grey', lw=1.5, zorder=1)
    
    # 画 Motif 色块
    if label in motif_hits:
        for hit in motif_hits[label]:
            start, width, m_num = hit['start'], hit['width'], hit['motif_num']
            color = COLORS[(m_num - 1) % len(COLORS)]
            
            m_start_r = R_SEQ_START + (start / max_seq_length) * R_SEQ_RANGE
            m_height_r = (width / max_seq_length) * R_SEQ_RANGE
            angular_width = (2 * math.pi / num_leaves) * 0.6 # 根据基因数量动态计算弧宽
            
            # 使用极坐标下的 bar 绘制完美贴合曲线的色块
            ax.bar(x=theta, height=m_height_r, bottom=m_start_r, width=angular_width, 
                   color=color, edgecolor='black', linewidth=0.5, alpha=0.9, zorder=2)

# ----------------- 4.3 构建图例 -----------------
unique_motifs_in_data = set(h['motif_num'] for hits in motif_hits.values() for h in hits)
legend_patches = []
for m_num in sorted(unique_motifs_in_data):
    color = COLORS[(m_num - 1) % len(COLORS)]
    label_text = f"Motif {m_num}: {MOTIF_ANNOTATIONS.get(m_num, 'Unknown')}"
    legend_patches.append(patches.Patch(color=color, label=label_text, ec='black'))

if legend_patches:
    ax.legend(handles=legend_patches, loc='upper right', bbox_to_anchor=(1.35, 1.1), 
              title="Motif Annotations", frameon=True, fontsize=10, title_fontsize=12)

# ================= 5. 保存输出 =================
plt.tight_layout()
plt.savefig(OUTPUT_PDF, format='pdf', dpi=DPI_SETTING, bbox_inches='tight')
plt.savefig(OUTPUT_PNG, format='png', dpi=DPI_SETTING, bbox_inches='tight')
plt.close()

print(f"🎉 酷炫的圆环图已生成！\n👉 结果已保存为：{OUTPUT_PNG} 和 {OUTPUT_PDF}")