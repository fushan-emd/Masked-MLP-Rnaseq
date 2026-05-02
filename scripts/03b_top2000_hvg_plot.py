import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ================= 1. 路径配置 =================
input_file = r'E:\Local_project\bisheanalysis\finalpic\FIG1PCA\barley_Data_Combat_Corrected.csv'
output_dir = r'E:\Local_project\bisheanalysis\finalpic\FIG1PCA'

os.makedirs(output_dir, exist_ok=True)

# ================= 2. 读取数据 =================
print("📖 读取批次校正后的数据...")
target_df = pd.read_csv(input_file, index_col=0)

exclude = ['target', 'target_label', 'Batch', 'treatment', 'Treatment', 'sample_id']
candidates = [c for c in target_df.columns if c not in exclude]

print(f"✅ 成功加载数据，共检测到 {len(candidates)} 个候选基因。")

# ================= 3. 计算均值与方差 =================
print("🧮 正在计算表达均值与方差...")
means = target_df[candidates].mean()
variances = target_df[candidates].var()

# 提取 Top 2000 高变基因
top_genes = variances.nlargest(2000).index.tolist()

# 导出基因列表
list_save_path = os.path.join(output_dir, "Table_S1_Top2000_HVGs.csv")
pd.DataFrame(top_genes, columns=['Original_Gene_ID']).to_csv(list_save_path, index=False)
print(f"📄 Top 2000 基因列表已导出至: {list_save_path}")

# ================= 4. 绘制高变基因散点图 (分层高亮版) =================
print("📊 正在绘制均值-方差散点图...")

# 重新构建基础数据框 (刚才就是漏了这部分！)
plot_df = pd.DataFrame({'Mean': means, 'Variance': variances})
plot_df['Type'] = 'Other Genes'
valid_top_genes = [g for g in top_genes if g in plot_df.index]
plot_df.loc[valid_top_genes, 'Type'] = 'Top 2000 HVGs'

print(f"🔍 分类核查：成功识别红点数量 -> {(plot_df['Type'] == 'Top 2000 HVGs').sum()} 个")

# 将特征分为两组，方便分层画图
df_other = plot_df[plot_df['Type'] == 'Other Genes']
df_top = plot_df[plot_df['Type'] == 'Top 2000 HVGs']

# 设置绘图风格
sns.set_theme(style="whitegrid")
plt.figure(figsize=(8, 6))

# 1. 先画底层的灰点（缩小尺寸，增加透明度，作为背景）
plt.scatter(
    df_other['Mean'], df_other['Variance'], 
    color='#e0e0e0', s=10, alpha=0.4, label='Other Genes', zorder=1
)

# 2. 再画顶层的红点（显著放大尺寸，加上白色边框，彻底凸显）
plt.scatter(
    df_top['Mean'], df_top['Variance'], 
    color='#d62728', s=45, alpha=1.0, edgecolors='white', linewidths=0.5, 
    label='Top 2000 HVGs', zorder=5
)

# 画阈值虚线
threshold_var = variances.nlargest(2000).iloc[-1]
plt.axhline(y=threshold_var, color='black', linestyle='--', linewidth=1.5, zorder=2)

plt.title("Mean-Variance Scatter Plot for Gene Selection", fontsize=14, fontweight='bold')
plt.xlabel("Mean Expression", fontsize=12)
plt.ylabel("Expression Variance", fontsize=12)

# 将图例移到左上角，彻底避免遮挡右上角的红点
plt.legend(loc='upper left', frameon=True, shadow=True)
plt.tight_layout()

# 保存图片
plot_save_path_pdf = os.path.join(output_dir, "HVG_Selection_Plot.pdf")
plot_save_path_png = os.path.join(output_dir, "HVG_Selection_Plot.png")

plt.savefig(plot_save_path_pdf, format='pdf', bbox_inches='tight')
plt.savefig(plot_save_path_png, format='png', dpi=300, bbox_inches='tight')
plt.close()

print(f"🎉 绘图完成！红点已强力放大，图片已保存至:\n  1. {plot_save_path_pdf}\n  2. {plot_save_path_png}")