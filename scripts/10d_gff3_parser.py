import pandas as pd
import os

print("✂️ 正在从海量数据中‘抠出’目标基因的结构信息...")

# 1. 配置文件路径
kaks_file = "FINAL_KaKs_Results.csv"
BARLEY_GFF = "meme\Hordeum_vulgare.MorexV3_pseudomolecules_assembly.62.chr.gtf\Hordeum_vulgare.MorexV3_pseudomolecules_assembly.62.chr.gtf" # 替换为你的文件名
WHEAT_GFF = "meme\Triticum_aestivum.IWGSC.62.gtf\Triticum_aestivum.IWGSC.62.gtf"   # 替换为你的文件名
OUTPUT_GFF = "Target_Genes_Structure.gff3"

# 2. 获取目标 ID 列表
df = pd.read_csv(kaks_file)
targets = set(df['Barley_ID'].dropna()).union(set(df['Wheat_ID'].dropna()))

# 3. 提取逻辑
extracted_lines = []
def filter_gff(gff_path):
    if not os.path.exists(gff_path): return
    with open(gff_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.startswith('#'): continue
            # 只要这一行包含我们的目标 ID，就把它整行抓取
            for tid in targets:
                if tid in line:
                    extracted_lines.append(line)
                    break

print("⏳ 正在扫描大麦 GFF...")
filter_gff(BARLEY_GFF)
print("⏳ 正在扫描小麦 GFF...")
filter_gff(WHEAT_GFF)

with open(OUTPUT_GFF, 'w', encoding='utf-8') as f:
    f.writelines(extracted_lines)

print(f"✅ 完成！精简版结构文件已保存为: {OUTPUT_GFF}")