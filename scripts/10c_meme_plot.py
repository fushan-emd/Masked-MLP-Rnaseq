import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from Bio import Phylo
import xml.etree.ElementTree as ET
import matplotlib

# ================= 0. 全局字体与输出配置 =================
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
DPI_SETTING = 700

# ================= 1. 动态路径配置 =================
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()

NWK_FILE = os.path.join(BASE_DIR, "tree.nwk")
MEME_XML = os.path.join(BASE_DIR, "meme.xml")
RENAME_FILE = os.path.join(BASE_DIR, "rename.csv")
OUTPUT_PDF = os.path.join(BASE_DIR, "Phylogenetic_Tree_and_Motifs.pdf")
OUTPUT_PNG = os.path.join(BASE_DIR, "Phylogenetic_Tree_and_Motifs.png")

# Motif 预设科研配色 (支持最多 15 个 Motif，超出会自动循环)
COLORS = [
    "#E64B35", "#4DBBD5", "#00A087", "#3C5488", "#F39B7F", 
    "#8491B4", "#91D1C2", "#DC0000", "#7E6148", "#B09C85",
    "#FF9F40", "#9966FF", "#FF6384", "#36A2EB", "#4BC0C0"
]

# ================= 1.5 Motif 功能注释配置 =================
# 根据 InterPro 和 TOMTOM 结果整理的注释映射
MOTIF_ANNOTATIONS = {
    1: "Disordered region",              # 来自 InterPro (MobiDBLite)
    2: "Ankyrin repeat",                 # 来自 InterPro (IPR036770)
    3: "Zinc finger RING-type",          # 来自 TOMTOM (PS00518)
    4: "Homeobox domain",                # 来自 TOMTOM (PS00033)
    9: "EF-hand domain",                 # 来自 InterPro (IPR002048)
    10: "Solanesyltransferase-like"      # 来自 InterPro (PTHR43009)
}


# ================= 2. 核心数据解析 =================
print(f"⏳ 正在加载并解析文件...\n工作目录: {BASE_DIR}")

# 2.1 加载重命名映射
rename_dict = {}
if os.path.exists(RENAME_FILE):
    try:
        rename_df = pd.read_csv(RENAME_FILE)
        if 'GeneID' in rename_df.columns and 'Annoname' in rename_df.columns:
            rename_dict = dict(zip(rename_df['GeneID'], rename_df['Annoname']))
        else:
            rename_dict = dict(zip(rename_df.iloc[:, 0], rename_df.iloc[:, -1]))
        print(f"📖 成功加载 {len(rename_dict)} 个重命名映射。")
    except Exception as e:
        print(f"⚠️ 读取重命名文件失败: {e}")

# 2.2 解析进化树 (Newick 格式)
if not os.path.exists(NWK_FILE):
    print(f"❌ 找不到进化树文件: {NWK_FILE}")
    exit()
tree = Phylo.read(NWK_FILE, "newick")
terminals = tree.get_terminals()
num_leaves = len(terminals)

# 计算树的 Y 轴坐标（为了保证叶子节点对齐）
y_pos = {leaf: i for i, leaf in enumerate(terminals)}
for clade in tree.get_nonterminals(order='postorder'):
    clade_y = sum(y_pos[c] for c in clade.clades) / len(clade.clades)
    y_pos[clade] = clade_y

# 计算树的 X 轴深度
depths = tree.depths()
max_depth = max(depths.values()) if depths else 1

# 2.3 解析 MEME XML 数据
if not os.path.exists(MEME_XML):
    print(f"❌ 找不到 MEME 文件: {MEME_XML}")
    exit()

meme_tree = ET.parse(MEME_XML)
root = meme_tree.getroot()

# 兼容不同 MEME 版本
seq_container = root.find('training_set')
if seq_container is None:
    seq_container = root.find('sequences')

if seq_container is None:
    print("❌ 无法在 meme.xml 中找到序列标签！请检查 XML 文件格式是否损坏。")
    exit()

# 解析序列信息 mapping
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

# 解析 Motif 宽度映射
motif_widths = {}
motifs_container = root.find('motifs')
if motifs_container is not None:
    for motif in motifs_container.findall('motif'):
        m_id = motif.get('id')
        width = motif.get('width')
        if m_id and width:
            motif_widths[m_id] = int(width)

# 解析扫描到的位点
motif_hits = {seq_data['renamed']: [] for seq_data in seq_info.values()}
scanned_sites = root.find('scanned_sites_summary')

if scanned_sites is not None:
    for scanned_seq in scanned_sites.findall('scanned_sites'):
        seq_id = scanned_seq.get('sequence_id')
        if seq_id not in seq_info: continue
            
        seq_renamed = seq_info[seq_id]['renamed']
        for hit in scanned_seq.findall('scanned_site'):
            m_id = hit.get('motif_id')
            m_num_str = m_id.replace('motif_', '')
            m_num = int(m_num_str) if m_num_str.isdigit() else 1
            start = int(hit.get('position', 0))
            width = motif_widths.get(m_id, 10)
            
            motif_hits[seq_renamed].append({
                'motif_num': m_num,
                'start': start,
                'width': width
            })

# ================= 3. 开始双轴联动绘图 =================
print("🎨 正在构建联动矢量图...")

# 设置整体画幅
fig, (ax_tree, ax_motif) = plt.subplots(1, 2, figsize=(15, max(5, num_leaves * 0.45)), gridspec_kw={'width_ratios': [1, 1.5]})

# ----------------- 3.1 绘制左侧：进化树 -----------------
ax_tree.spines['top'].set_visible(False)
ax_tree.spines['right'].set_visible(False)
ax_tree.spines['left'].set_visible(False)
ax_tree.spines['bottom'].set_visible(False)
ax_tree.set_xticks([])
ax_tree.set_yticks([])

# 手动绘制进化树线条
for clade in tree.find_clades(order='preorder'):
    x1 = depths[clade]
    y1 = y_pos[clade]
    for child in clade.clades:
        x2 = depths[child]
        y2 = y_pos[child]
        # 水平分支
        ax_tree.plot([x1, x2], [y2, y2], 'k-', lw=1.5)
        # 垂直连接线
        ax_tree.plot([x1, x1], [y1, y2], 'k-', lw=1.5)

# 提取并重命名叶子节点
renamed_labels = []
max_seq_length = 0
for leaf in terminals:
    raw_name = leaf.name
    if not raw_name: raw_name = "Unknown"
    clean_name = raw_name.replace('_promoter_2000bp', '').strip()
    renamed = rename_dict.get(clean_name, clean_name)
    renamed_labels.append((renamed, y_pos[leaf], leaf))
    
    if renamed in motif_hits:
        seq_len = next((info['length'] for info in seq_info.values() if info['renamed'] == renamed), 0)
        max_seq_length = max(max_seq_length, seq_len)

for label, y, leaf in renamed_labels:
    x_leaf = depths[leaf]
    
    # 画虚线引导线
    ax_tree.plot([x_leaf, max_depth * 1.1], [y, y], color='gray', linestyle=':', lw=1.2, alpha=0.6)
    
    # 🌟 核心修改：判断物种并分配字体颜色
    if str(label).startswith('Hv') or str(label).startswith('HORVU'):
        text_color = '#5DA5DA'  # 大麦 - 柔和蓝
    elif str(label).startswith('Ta') or str(label).startswith('Traes'):
        text_color = '#FAA43A'  # 小麦 - 活力橙
    else:
        text_color = 'black'    # 默认颜色
        
    # 放置名字（带上了特有的颜色）
    ax_tree.text(max_depth * 1.15, y, label, va='center', ha='left', fontsize=12, fontstyle='italic', fontweight='bold', color=text_color)

# 终极修复：把 X 轴极限值放大到 2.8 倍！强行撑大内部空间，绝不挡字！
ax_tree.set_xlim(0, max_depth * 2.8)
ax_tree.set_title("Phylogenetic Tree", fontsize=15, fontweight='bold', pad=15)

# ----------------- 3.2 绘制右侧：MEME Motifs -----------------
ax_motif.spines['top'].set_visible(False)
ax_motif.spines['right'].set_visible(False)
ax_motif.spines['left'].set_visible(False)
ax_motif.set_yticks([])

# 绘制每一条序列的主干和 Motif
for label, y, leaf in renamed_labels:
    seq_len = next((info['length'] for info in seq_info.values() if info['renamed'] == label), max_seq_length)
    if seq_len == 0: seq_len = max_seq_length if max_seq_length > 0 else 100
    
    # 画序列主线
    ax_motif.plot([0, seq_len], [y, y], color='grey', linewidth=2, zorder=1)
    
    # 画 Motifs
    if label in motif_hits:
        for hit in motif_hits[label]:
            start = hit['start']
            width = hit['width']
            m_num = hit['motif_num']
            color = COLORS[(m_num - 1) % len(COLORS)]
            
            rect_height = 0.5
            y_bottom = y - (rect_height / 2)
            
            rect = patches.Rectangle(
                (start, y_bottom), width, rect_height,
                facecolor=color, edgecolor='black', linewidth=0.8, alpha=0.9, zorder=2
            )
            ax_motif.add_patch(rect)

ax_motif.set_xlim(-10, max_seq_length + 50)
ax_motif.set_xlabel("Amino Acid Position (aa)", fontsize=12, fontweight='bold')
ax_motif.set_title("MEME Conserved Motifs", fontsize=15, fontweight='bold', pad=15)

# ----------------- 3.3 构建图例 (已整合注释) -----------------
unique_motifs_in_data = set()
for hits in motif_hits.values():
    for h in hits:
        unique_motifs_in_data.add(h['motif_num'])

legend_patches = []
for m_num in sorted(unique_motifs_in_data):
    color = COLORS[(m_num - 1) % len(COLORS)]
    
    # 🌟 核心修改：检查是否有对应的功能注释
    if m_num in MOTIF_ANNOTATIONS:
        label_text = f"Motif {m_num}: {MOTIF_ANNOTATIONS[m_num]}"
    else:
        label_text = f"Motif {m_num}"
        
    legend_patches.append(patches.Patch(color=color, label=label_text, ec='black'))

if legend_patches:
    # 适当增加 bbox_to_anchor 的宽度 (例如 1.3)，防止长注释超出画布
    ax_motif.legend(handles=legend_patches, loc='upper left', bbox_to_anchor=(1.02, 1), 
                    title="Motif Annotations", frameon=True, fontsize=10, title_fontsize=12)
# ================= 4. 保存输出 =================
plt.tight_layout()

# 调整子图间距为 0.05，让排版紧凑且自然
plt.subplots_adjust(wspace=0.05) 

plt.savefig(OUTPUT_PDF, format='pdf', dpi=DPI_SETTING, bbox_inches='tight')
plt.savefig(OUTPUT_PNG, format='png', dpi=DPI_SETTING, bbox_inches='tight')
plt.close()

print(f"🎉 结构图已优化完成！\n👉 大麦 ID 已设为蓝色，小麦 ID 已设为橙色。\n👉 已保存为：{OUTPUT_PNG} 和 {OUTPUT_PDF}")