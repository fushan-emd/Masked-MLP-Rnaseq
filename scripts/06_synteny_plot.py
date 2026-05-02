import re
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.path import Path as MplPath
import os
import sys
import numpy as np
from pathlib import Path

print("🎨 启动 Python SCI级高级共线性拉丝引擎 (全量绘制 + 漏填基因侦探版)...")

# ================= 0. 绝对路径地毯式搜索引擎 =================
def resolve_base_dir() -> Path:
    if getattr(sys, "frozen", False): return Path(sys.executable).resolve().parent
    if "__file__" in globals(): return Path(__file__).resolve().parent
    return Path.cwd()

script_dir = resolve_base_dir()
cwd_dir = Path.cwd()

def find_file(filenames):
    if isinstance(filenames, str):
        filenames = [filenames]
    for fn in filenames:
        p1 = script_dir / fn
        if p1.exists(): return p1
        p2 = cwd_dir / fn
        if p2.exists(): return p2
    return None

def get_clean_id(gid):
    return str(gid).strip().split('.')[0].upper()

def compact_gene_id(gene_id: str) -> str:
    if not gene_id or pd.isna(gene_id): return ""
    clean_id = str(gene_id).strip().split('.')[0]
    m_barley = re.match(r"^HORVUMOREXr(\d+)HG(\d+)$", clean_id, re.IGNORECASE)
    if m_barley: return f"Hv{m_barley.group(1)}-{m_barley.group(2)[-6:]}"
    m_wheat = re.match(r"^TraesCS([0-9][ABD])02G(\d+)$", clean_id, re.IGNORECASE)
    if m_wheat: return f"Ta{m_wheat.group(1)}-{m_wheat.group(2)[-6:]}"
    return clean_id[:16] + ".." if len(clean_id) > 18 else clean_id

# ================= 1. 读取自定义重命名映射 =================
rename_map = {}
rename_csv_path = find_file("rename.csv")

if rename_csv_path:
    try:
        df_rename = pd.read_csv(rename_csv_path, dtype=str) 
        col_id = df_rename.columns[0]
        col_name = df_rename.columns[3]
        for _, row in df_rename.iterrows():
            gid = str(row[col_id]).strip()
            new_name = str(row[col_name]).strip()
            if pd.notna(row[col_id]) and pd.notna(row[col_name]) and new_name != 'nan':
                clean_gid = get_clean_id(gid)
                rename_map[clean_gid] = new_name
        print(f"✅ 成功加载 {rename_csv_path.name}，共获取 {len(rename_map)} 个自定义基因名。")
    except Exception as e:
        print(f"⚠️ 读取 rename.csv 失败: {e}")

# ================= 2. 智能锁定配置文件路径 =================
kaks_file = find_file(["FINAL_KaKs_Results.csv", "ortholog_map.csv"])
barley_bed = find_file(["evolve/barley_positions.bed", "barley_positions.bed"])
wheat_bed = find_file(["evolve/wheat_positions.bed", "wheat_positions.bed"])

if not kaks_file or not barley_bed or not wheat_bed:
    print("❌ 致命错误：未找齐同源文件或 BED 文件！请检查文件位置。")
    sys.exit(1)

# ================= 3. 🎯 核心修改：移除过滤，全量读取同源对 =================
df_kaks = pd.read_csv(kaks_file)

if 'Barley_ID' in df_kaks.columns:
    b_col, w_col = 'Barley_ID', 'Wheat_ID'
elif 'Hordeum vulgare gene stable ID' in df_kaks.columns:
    b_col, w_col = 'Hordeum vulgare gene stable ID', 'Gene stable ID'
else:
    print(f"❌ 无法检测到列名!")
    sys.exit(1)

all_links = df_kaks[[b_col, w_col]].dropna().values.tolist()

links = []
kept_b_raw_ids = set()
kept_w_raw_ids = set()

# 【过滤网已拆除】直接读取 KaKs 表里的所有连线（比如全部 22 条）
for pair in all_links:
    b_raw, w_raw = pair
    if pd.isna(b_raw) and pd.isna(w_raw): continue
    links.append(pair)
    if pd.notna(b_raw): kept_b_raw_ids.add(b_raw)
    if pd.notna(w_raw): kept_w_raw_ids.add(w_raw)

print(f"🎯 已载入 {kaks_file.name} 中的全部 {len(links)} 对同源线，准备画图！")

def load_bed(bed_file, kept_set):
    bed_data = {}
    chr_len = {}
    with open(bed_file, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            parts = line.strip().split()
            if len(parts) >= 4:
                chrom, start, end, gene_id = parts[0], int(parts[1]), int(parts[2]), parts[3]
                clean_bed_id = get_clean_id(gene_id)
                for rid in kept_set:
                    if get_clean_id(rid) == clean_bed_id:
                        bed_data[rid] = {'chr': chrom, 'pos': (start + end) / 2}
                        break
                chr_len[chrom] = max(chr_len.get(chrom, 0), end)
    return bed_data, chr_len

barley_genes, barley_chrs = load_bed(barley_bed, kept_b_raw_ids)
wheat_genes, wheat_chrs = load_bed(wheat_bed, kept_w_raw_ids)

# ================= 4. 高级画板与 UI 设置 =================
fig, ax = plt.subplots(figsize=(18, 10), facecolor='#F8F9FA')
ax.set_facecolor('#F8F9FA')
ax.axis('off')

Y_BARLEY = 0.60 
Y_WHEAT = 0.40
CHR_HEIGHT = 0.025 

active_b_chrs = sorted(list(set([barley_genes[pair[0]]['chr'] for pair in links if pair[0] in barley_genes])))
active_w_chrs = sorted(list(set([wheat_genes[pair[1]]['chr'] for pair in links if pair[1] in wheat_genes])))

def draw_chromosomes(chr_list, chr_lengths, y_pos, face_color, edge_color):
    chr_x_positions = {}
    x_current = 0.15 
    spacing = 0.04   
    if not chr_list: return {}, x_current
    
    max_len = max([chr_lengths[c] for c in chr_list])
    scale_factor = 0.8 / (len(chr_list) * max_len) if chr_list else 1
    
    for chrom in chr_list:
        length = chr_lengths[chrom] * scale_factor
        rect = patches.FancyBboxPatch(
            (x_current, y_pos), length, CHR_HEIGHT, 
            boxstyle="round,pad=0.015,rounding_size=0.015", 
            edgecolor=edge_color, facecolor=face_color, linewidth=1.5, alpha=0.9, zorder=3
        )
        ax.add_patch(rect)
        short_chr = str(chrom).replace('chr', '').replace('Chr', '')
        label_y = y_pos - 0.04 if y_pos == Y_BARLEY else y_pos + CHR_HEIGHT + 0.04
        ax.text(x_current + length/2, label_y, short_chr, 
                ha='center', va='center', fontsize=12, fontweight='bold', color='#222222', zorder=4)
        chr_x_positions[chrom] = {'start_x': x_current, 'scale': scale_factor}
        x_current += length + spacing 
    return chr_x_positions, x_current

pos_B, max_x_B = draw_chromosomes(active_b_chrs, barley_chrs, Y_BARLEY, face_color='#D6EAF8', edge_color='#2E86C1')
pos_W, max_x_W = draw_chromosomes(active_w_chrs, wheat_chrs, Y_WHEAT, face_color='#FCF3CF', edge_color='#D68910')

canvas_max_width = max(max_x_B, max_x_W)
if canvas_max_width > 0: ax.set_xlim(0, canvas_max_width + 0.1)
ax.set_ylim(-0.1, 1.1) 

ax.text(0.04, Y_BARLEY + CHR_HEIGHT/2, 'Barley\n$(Diploid)$', fontsize=16, fontweight='bold', va='center', ha='center', color='#2E86C1', zorder=4)
ax.text(0.04, Y_WHEAT + CHR_HEIGHT/2, 'Wheat\n$(Hexaploid)$', fontsize=16, fontweight='bold', va='center', ha='center', color='#D68910', zorder=4)

# ================= 5. 绘制连线与记录漏掉的基因 =================
b_chr_labels = {}
w_chr_labels = {}

# 记录哪些基因没有在 rename.csv 里
missing_barley = set()
missing_wheat = set()

for b_id, w_id in links:
    if b_id in barley_genes and w_id in wheat_genes:
        b_info = barley_genes[b_id]
        w_info = wheat_genes[w_id]
        
        x1 = pos_B[b_info['chr']]['start_x'] + (b_info['pos'] * pos_B[b_info['chr']]['scale'])
        y1 = Y_BARLEY
        x2 = pos_W[w_info['chr']]['start_x'] + (w_info['pos'] * pos_W[w_info['chr']]['scale'])
        y2 = Y_WHEAT + CHR_HEIGHT
        
        verts = [(x1, y1), (x1, y1 - 0.08), (x2, y2 + 0.08), (x2, y2)]
        codes = [MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4]
        path = MplPath(verts, codes)
        
        patch_main = patches.PathPatch(path, facecolor='none', edgecolor='#C0392B', lw=1.5, alpha=0.60, zorder=2)
        ax.add_patch(patch_main)
        
        ax.plot(x1, y1, marker='o', color='#FFFFFF', markeredgecolor='#C0392B', markersize=4, markeredgewidth=1.0, zorder=5)
        ax.plot(x2, y2, marker='o', color='#FFFFFF', markeredgecolor='#C0392B', markersize=4, markeredgewidth=1.0, zorder=5)
        
        # 寻找大麦名字，找不到就记录下来
        clean_b = get_clean_id(b_id)
        if clean_b in rename_map:
            b_label = rename_map[clean_b]
        else:
            b_label = compact_gene_id(b_id)
            missing_barley.add(b_id)
            
        # 寻找小麦名字，找不到就记录下来
        clean_w = get_clean_id(w_id)
        if clean_w in rename_map:
            w_label = rename_map[clean_w]
        else:
            w_label = compact_gene_id(w_id)
            missing_wheat.add(w_id)
        
        if b_info['chr'] not in b_chr_labels: b_chr_labels[b_info['chr']] = []
        b_chr_labels[b_info['chr']].append({'actual_x': x1, 'label': b_label})
        
        if w_info['chr'] not in w_chr_labels: w_chr_labels[w_info['chr']] = []
        w_chr_labels[w_info['chr']].append({'actual_x': x2, 'label': w_label})

# ================= 6. 高级分散排版算法 =================
for chrom, items in b_chr_labels.items():
    items.sort(key=lambda x: x['actual_x']) 
    chr_start = pos_B[chrom]['start_x']
    chr_len = barley_chrs[chrom] * pos_B[chrom]['scale']
    if len(items) == 1: xs = [items[0]['actual_x']]
    else: xs = np.linspace(chr_start, chr_start + chr_len, len(items))
    for i, item in enumerate(items):
        lx, ly = xs[i], Y_BARLEY + CHR_HEIGHT + 0.08
        ax.plot([item['actual_x'], lx], [Y_BARLEY + CHR_HEIGHT, ly], color='#2980B9', lw=1.0, linestyle=':', alpha=0.8, zorder=1)
        ax.text(lx, ly + 0.01, item['label'], rotation=45, ha='left', va='bottom', fontsize=11, fontweight='bold', color='#1A5276', zorder=6)

for chrom, items in w_chr_labels.items():
    items.sort(key=lambda x: x['actual_x'])
    chr_start = pos_W[chrom]['start_x']
    chr_len = wheat_chrs[chrom] * pos_W[chrom]['scale']
    if len(items) == 1: xs = [items[0]['actual_x']]
    else: xs = np.linspace(chr_start, chr_start + chr_len, len(items))
    for i, item in enumerate(items):
        lx, ly = xs[i], Y_WHEAT - 0.08
        ax.plot([item['actual_x'], lx], [Y_WHEAT, ly], color='#D68910', lw=1.0, linestyle=':', alpha=0.8, zorder=1)
        ax.text(lx, ly - 0.01, item['label'], rotation=45, ha='right', va='top', fontsize=11, fontweight='bold', color='#935116', zorder=6)

# ================= 7. 自动居中标题 =================
center_x = (canvas_max_width + 0.1) / 2 if canvas_max_width > 0 else 0.5
plt.text(center_x, 1.05, "Syntenic Relationships of Target Genes", ha='center', va='center', fontsize=26, fontweight='bold', color='#222222')

# ================= 8. 保存并在终端汇报缺失情况 =================
output_fig = script_dir / "Synteny_Plot_All_Genes.png"
plt.savefig(output_fig, dpi=300, bbox_inches='tight', pad_inches=0.3)
print(f"🎉 连线已全部绘制完成：\n   -> {output_fig}")

print("\n" + "="*50)
print("🧐 漏填基因侦探报告：")
if missing_barley or missing_wheat:
    print("代码发现以下基因在 rename.csv 中没有对应的新名字，已使用默认缩写：")
    if missing_barley:
        print(f"\n🌾 【漏填的大麦基因】({len(missing_barley)}个):")
        for g in missing_barley: print(f"  - {g}")
    if missing_wheat:
        print(f"\n🌾 【漏填的小麦基因】({len(missing_wheat)}个):")
        for g in missing_wheat: print(f"  - {g}")
    print("\n💡 解决办法：请把上面的基因 ID 复制到你的 rename.csv 文件里，并在后面写上你想要的名字，再运行一次就完美了！")
else:
    print("✅ 完美！所有基因都在 rename.csv 中找到了名字！")
print("="*50 + "\n")