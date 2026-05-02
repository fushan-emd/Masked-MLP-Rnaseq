import re
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.path import Path as MplPath
from matplotlib.collections import LineCollection
import os
import sys
import numpy as np
import math
from pathlib import Path

print("🎨 启动 Python SCI级顶级共线性圆形引擎 (超大字号 + 双色标注版)...")

# ================= 0. 路径处理 =================
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

# ================= 1. 加载 rename 映射 =================
rename_map = {}
rename_csv_path = find_file("rename.csv")
if rename_csv_path:
    try:
        df_rename = pd.read_csv(rename_csv_path, dtype=str) 
        col_id, col_name = df_rename.columns[0], df_rename.columns[3]
        for _, row in df_rename.iterrows():
            if pd.notna(row[col_id]) and pd.notna(row[col_name]):
                rename_map[get_clean_id(row[col_id])] = str(row[col_name]).strip()
    except Exception as e: print(f"⚠️ 读取 rename.csv 失败: {e}")

# ================= 2. 锁定文件 =================
kaks_file = find_file(["FINAL_KaKs_Results.csv", "ortholog_map.csv"])
barley_bed = find_file(["evolve/barley_positions.bed", "barley_positions.bed"])
wheat_bed = find_file(["evolve/wheat_positions.bed", "wheat_positions.bed"])

if not kaks_file or not barley_bed or not wheat_bed:
    print("❌ 文件缺失！"); sys.exit(1)

# ================= 3. 数据处理 =================
df_kaks = pd.read_csv(kaks_file)
b_col, w_col = ('Barley_ID', 'Wheat_ID') if 'Barley_ID' in df_kaks.columns else ('Hordeum vulgare gene stable ID', 'Gene stable ID')
links = df_kaks[[b_col, w_col]].dropna().values.tolist()
kept_b_ids, kept_w_ids = set(df_kaks[b_col].dropna()), set(df_kaks[w_col].dropna())

def load_bed_all(bed_file):
    gene_dict, chr_len = {}, {}
    with open(bed_file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 4:
                chrom, start, end, gid = parts[0], int(parts[1]), int(parts[2]), parts[3]
                gene_dict[gid] = {'chr': chrom, 'pos': (start + end) / 2}
                chr_len[chrom] = max(chr_len.get(chrom, 0), end)
    return gene_dict, chr_len

barley_all, barley_chrs = load_bed_all(barley_bed)
wheat_all, wheat_chrs = load_bed_all(wheat_bed)

# ================= 4. 布局参数 (Circos 风格) =================
R_OUTER, R_INNER = 1.0, 0.90
R_MID = (R_OUTER + R_INNER) / 2

def assign_angles(chr_len_dict, start_angle, span_total):
    total_len = sum(chr_len_dict.values())
    chr_sorted = sorted(chr_len_dict.keys())
    gap = 2.5 # 加大染色体间距
    num_chrs = len(chr_sorted)
    effective_span = span_total - gap * (num_chrs - 1)
    current_angle, angles = start_angle, {}
    for chrom in chr_sorted:
        span = (chr_len_dict[chrom] / total_len) * effective_span
        angles[chrom] = {'theta1': current_angle, 'theta2': current_angle + span, 'mid': current_angle + span/2, 'scale': span/chr_len_dict[chrom]}
        current_angle += span + gap
    return angles

barley_angles = assign_angles(barley_chrs, 95, 170)
wheat_angles = assign_angles(wheat_chrs, -85, 170)

# ================= 5. 绘图 =================
fig, ax = plt.subplots(figsize=(26, 26), facecolor='white')
ax.set_aspect('equal')
ax.set_xlim(-2.0, 2.0) # 💥 扩展坐标轴空间给超大字体
ax.set_ylim(-2.0, 2.0)
ax.axis('off')

# --- 5.1 染色体环 ---
def draw_rings(angles_dict, f_color, e_color):
    for chrom, info in angles_dict.items():
        wedge = patches.Wedge((0,0), R_OUTER, info['theta1'], info['theta2'], width=R_OUTER-R_INNER, facecolor=f_color, edgecolor=e_color, lw=1.5, alpha=0.9, zorder=2)
        ax.add_patch(wedge)
        rad = math.radians(info['mid'])
        lx, ly = (R_OUTER + 0.06) * math.cos(rad), (R_OUTER + 0.06) * math.sin(rad)
        rot = info['mid'] + 180 if 90 < (info['mid'] % 360) < 270 else info['mid']
        ax.text(lx, ly, chrom.replace('chr','').replace('Chr',''), ha='center', va='center', rotation=rot, fontsize=18, fontweight='bold', zorder=5)

draw_rings(barley_angles, '#D6EAF8', '#2E86C1')
draw_rings(wheat_angles, '#FCF3CF', '#D68910')

# 物种标题 (💥 放大到 28)
ax.text(-1.5, 1.6, 'Barley (Diploid)', fontsize=28, fontweight='bold', color='#2E86C1')
ax.text(0.8, 1.6, 'Wheat (Hexaploid)', fontsize=28, fontweight='bold', color='#D68910')

# --- 5.2 基因角度映射 ---
def get_gene_angles(gene_dict, angles_dict):
    res = {}
    for gid, info in gene_dict.items():
        if info['chr'] in angles_dict:
            res[gid] = angles_dict[info['chr']]['theta1'] + info['pos'] * angles_dict[info['chr']]['scale']
    return res

barley_gene_angles = get_gene_angles(barley_all, barley_angles)
wheat_gene_angles = get_gene_angles(wheat_all, wheat_angles)

# --- 5.3 丝带连线 ---
print("🎨 绘制飘逸加粗丝带...")
for b_id, w_id in links:
    if b_id in barley_gene_angles and w_id in wheat_gene_angles:
        rb, rw = math.radians(barley_gene_angles[b_id]), math.radians(wheat_gene_angles[w_id])
        x1, y1 = R_INNER * math.cos(rb), R_INNER * math.sin(rb)
        x2, y2 = R_INNER * math.cos(rw), R_INNER * math.sin(rw)
        verts = [(x1, y1), (x1*0.35, y1*0.35), (x2*0.35, y2*0.35), (x2, y2)] # 💥 调整控制点让曲线更圆润
        path = MplPath(verts, [MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4])
        ax.add_patch(patches.PathPatch(path, facecolor='none', edgecolor='#C0392B', lw=2.5, alpha=0.6, zorder=3))

# --- 5.4 💥 超大基因标签 + 颜色区分 ---
def label_genes(gene_ids, angles, is_barley=True):
    # 配色方案
    color = '#1A5276' if is_barley else '#A04000' # 深蓝 vs 深橙
    for gid in gene_ids:
        if gid not in angles: continue
        label = rename_map.get(get_clean_id(gid), compact_gene_id(gid))
        ang_rad = math.radians(angles[gid])
        x0, y0 = R_OUTER * math.cos(ang_rad), R_OUTER * math.sin(ang_rad)
        
        # 💥 文字位置进一步外推，防止遮挡
        text_r = R_OUTER + 0.25
        x1, y1 = text_r * math.cos(ang_rad), text_r * math.sin(ang_rad)
        
        rot = angles[gid]
        if 90 < (rot % 360) < 270:
            rot += 180
            ha = 'right'
        else:
            ha = 'left'
        
        ax.plot([x0, x1*0.95], [y0, y1*0.95], color='black', lw=0.8, alpha=0.5, zorder=5)
        # 💥 fontsize=18，并应用对应颜色
        ax.text(x1, y1, label, ha=ha, va='center', rotation=rot, fontsize=18,
                fontweight='bold', color=color, zorder=6,
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.9, edgecolor='none'))

label_genes(kept_b_ids, barley_gene_angles, is_barley=True)
label_genes(kept_w_ids, wheat_gene_angles, is_barley=False)

# ================= 6. 保存 =================
output_path = script_dir / "Synteny_Circular_Final_Ultra.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0.5)
print(f"🎉 终极 SCI 级大图已生成：\n   -> {output_path}")