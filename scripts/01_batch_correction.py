import pandas as pd
import glob
import os
import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import warnings
import seaborn as sns
from combat.pycombat import pycombat

warnings.filterwarnings('ignore')

# 设置 Seaborn 的默认主题
sns.set_theme(style="whitegrid", palette="muted")

# ================= 1. 配置区域 =================
tpm_folder_path = '/root/nfsdata/project/bishe/wheattpm_each'      # 您的 TPM csv 文件夹
annotation_file = '/root/nfsdata/project/bishe/W_Annotation.csv' # 您的文件名

output_dir = 'W_pca' # 所有结果的输出文件夹
output_file = 'wheat_Data_Combat_Corrected.csv' # 输出的最终表达矩阵文件名

os.makedirs(output_dir, exist_ok=True)
print(f"📁 所有结果将统一保存至文件夹: ./{output_dir}/")
# ==============================================

# ==================== 2. 工具函数定义 ====================
def make_index_unique(index_list):
    seen = {}
    result = []
    for item in index_list:
        if item in seen:
            seen[item] += 1
            result.append(f"{item}_dup{seen[item]}")
        else:
            seen[item] = 0
            result.append(item)
    return result

def plot_and_export_pca(adata, raw_counts, out_dir):
    print("\n=== 7. 绘制 PCA 对比图并导出结果 ===")

    valid_samples = adata.obs_names
    valid_genes = adata.var_names

    X_before_df = raw_counts.T.loc[valid_samples, valid_genes] 
    X_before = np.log1p(X_before_df.values) 
    X_after = adata.X 

    pca = PCA(n_components=2)
    X_before = np.nan_to_num(X_before) 
    coords_before = pca.fit_transform(X_before)
    var_before = pca.explained_variance_ratio_

    X_after = np.nan_to_num(X_after)
    coords_after = pca.fit_transform(X_after)
    var_after = pca.explained_variance_ratio_

    batch_labels = adata.obs['Batch'].values
    possible_cols = ['treatment', 'Treatment', 'condition', 'Condition']
    treat_col = next((c for c in possible_cols if c in adata.obs.columns), None)
    treat_labels = adata.obs[treat_col].astype(str).values if treat_col else None

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    scatter_kws = {'s': 80, 'alpha': 0.8, 'edgecolor': 'w', 'linewidth': 0.3}

    # 左图
    sns.scatterplot(
        x=coords_before[:, 0], y=coords_before[:, 1],
        hue=batch_labels, style=treat_labels,
        palette="tab10", ax=axes[0], **scatter_kws
    )
    axes[0].set_title("Before Correction (Raw Log1p)", fontsize=14, fontweight='bold')
    axes[0].set_xlabel(f"PC1 ({var_before[0]:.1%})")
    axes[0].set_ylabel(f"PC2 ({var_before[1]:.1%})")
    axes[0].legend(bbox_to_anchor=(1.02, 1), loc='upper left', title="Batch", borderaxespad=0)

    # 右图
    sns.scatterplot(
        x=coords_after[:, 0], y=coords_after[:, 1],
        hue=batch_labels, style=treat_labels,
        palette="tab10", ax=axes[1], **scatter_kws
    )
    axes[1].set_title("After ComBat & Z-score", fontsize=14, fontweight='bold')
    axes[1].set_xlabel(f"PC1 ({var_after[0]:.1%})")
    axes[1].set_ylabel(f"PC2 ({var_after[1]:.1%})")
    axes[1].legend(bbox_to_anchor=(1.02, 1), loc='upper left', title="Batch", borderaxespad=0)

    plt.tight_layout()
    pdf_filename = os.path.join(out_dir, "PCA_Comparison_Plot.pdf")
    plt.savefig(pdf_filename, format='pdf', bbox_inches='tight') 
    print(f"📄 PCA 图片已保存为: {pdf_filename}")
    plt.close()

    data_export = {
        'SampleID': valid_samples, 'Batch': batch_labels,
        'PC1_Before': coords_before[:, 0], 'PC2_Before': coords_before[:, 1],
        'PC1_After': coords_after[:, 0], 'PC2_After': coords_after[:, 1]
    }
    if treat_labels is not None: data_export['Treatment'] = treat_labels
    pd.DataFrame(data_export).to_csv(os.path.join(out_dir, "PCA_Coordinates_Data.csv"), index=False)


def plot_boxplot_comparison(adata, raw_counts, out_dir, filename='Boxplot_Comparison_Plot.pdf'):
    """
    极致性能版：直接使用 NumPy 操作，不仅飞速出 PDF 矢量图，还附带完整作图数据导出。
    """
    print("\n=== 8. 绘制箱线图并导出作图数据 ===")
    
    valid_samples = adata.obs_names
    valid_genes = adata.var_names
    
    # 提取修正前的数据 (Log1p) 和 修正后的数据 (ComBat + Z-score)
    X_before_df = raw_counts.T.loc[valid_samples, valid_genes]
    X_before = np.log1p(X_before_df.values)
    X_after = adata.X
    
    # --- 【新增】1. 极速导出箱线图的底层作图数据 (CSV) ---
    print("⏳ 正在构建百万级作图数据表，请稍候...")
    n_samples, n_genes = X_before.shape
    # 使用 np.repeat 瞬间生成百万级标签
    sample_ids = np.repeat(valid_samples, n_genes)
    batches = np.repeat(adata.obs['Batch'].values, n_genes)
    
    df_export = pd.DataFrame({
        'Sample_ID': sample_ids,
        'Batch': batches,
        'Expression_Before_Log1p': X_before.flatten(),
        'Expression_After_ComBat_Zscore': X_after.flatten()
    })
    csv_path = os.path.join(out_dir, "Boxplot_Data.csv")
    df_export.to_csv(csv_path, index=False)
    print(f"📊 箱线图作图数据已保存为: {csv_path}")

    # --- 2. 使用 NumPy 极速绘制箱线图并导出 PDF ---
    batches_unique = sorted(adata.obs['Batch'].unique())
    data_before = []
    data_after = []
    
    for b in batches_unique:
        sample_indices = np.where(adata.obs['Batch'] == b)[0]
        data_before.append(X_before[sample_indices, :].flatten())
        data_after.append(X_after[sample_indices, :].flatten())

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    colors = sns.color_palette("Set2", len(batches_unique))
    
    # 左图
    bplot1 = axes[0].boxplot(data_before, labels=batches_unique, patch_artist=True, showfliers=False,
                             medianprops={'color': 'black', 'linewidth': 1.5})
    for patch, color in zip(bplot1['boxes'], colors):
        patch.set_facecolor(color)
        
    axes[0].set_title('Expression Distribution Before Correction (Log1p)', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Expression (log1p TPM)', fontsize=12)
    axes[0].grid(True, axis='y', linestyle='--', alpha=0.5)
    
    # 右图
    bplot2 = axes[1].boxplot(data_after, labels=batches_unique, patch_artist=True, showfliers=False,
                             medianprops={'color': 'black', 'linewidth': 1.5})
    for patch, color in zip(bplot2['boxes'], colors):
        patch.set_facecolor(color)
        
    axes[1].set_title('Expression Distribution After ComBat & Z-score', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('Expression (ComBat Corrected & Scaled)', fontsize=12)
    axes[1].grid(True, axis='y', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    full_path = os.path.join(out_dir, filename)
    plt.savefig(full_path, format='pdf', bbox_inches='tight') # 恢复保存为 PDF
    print(f"📄 箱线图 (PDF) 已极速保存为: {full_path}")
    plt.show()


# ==================== 3. 主流程 ====================

print("=== 1. 读取样本注释表 ===")
try:
    meta_df = pd.read_csv(annotation_file, sep=',')
    if len(meta_df.columns) < 2:
        meta_df = pd.read_csv(annotation_file, sep='\t')
except Exception as e:
    meta_df = pd.read_excel(annotation_file)

if 'Sample_ID' in meta_df.columns:
    meta_df['Sample_ID'] = meta_df['Sample_ID'].astype(str).str.strip()
    meta_df.set_index('Sample_ID', inplace=True)
else:
    raise ValueError("❌ 表格里没找到 'Sample_ID' 这一列！")

meta_df['Batch'] = meta_df['Batch'].astype(str)
print(f"✅ 注释表加载成功！共 {len(meta_df)} 个样本。")

print("\n=== 2. 读取并合并 TPM 数据 ===")
all_files = glob.glob(os.path.join(tpm_folder_path, '*.csv'))
df_list = []

for file in all_files:
    try:
        temp_df = pd.read_csv(file, sep=',', header=0)
        gene_col = temp_df.columns[0]
        temp_df = temp_df.dropna(subset=[gene_col]).dropna(how='all')
        genes = temp_df.iloc[:, 0].astype(str).tolist()
        temp_df.index = make_index_unique(genes)
        temp_df = temp_df.drop(columns=[gene_col])
        df_list.append(temp_df)
    except Exception as e:
        print(f"❌ 读取 {os.path.basename(file)} 失败: {e}")

raw_counts = pd.concat(df_list, axis=1, join='outer')
raw_counts.fillna(0, inplace=True)

common_samples = list(set(raw_counts.columns) & set(meta_df.index))
print(f"\n🔗 成功匹配到 {len(common_samples)} 个样本。")

raw_counts = raw_counts[common_samples]
meta_df = meta_df.loc[common_samples]

print("\n=== 3. 构建 AnnData 并预处理 ===")
adata = sc.AnnData(X=raw_counts.T.values.copy())
adata.obs = meta_df.copy()
adata.var_names = raw_counts.index.tolist()

sc.pp.filter_genes(adata, min_cells=3) 
sc.pp.log1p(adata)
print(f"基础过滤后基因数: {adata.n_vars}")

# ⚠️ 关键修改：把 ComBat 提前到 Z-score 之前
print("\n=== 4. 使用 Combat 去除批次效应 (Log1p 数据) ===")
expression_df = pd.DataFrame(adata.X.T, index=adata.var_names, columns=adata.obs_names) 
batch_info = adata.obs['Batch']

assert list(expression_df.columns) == list(batch_info.index), "样本顺序不一致"

try:
    corrected_expression = pycombat(expression_df, batch_info)
except Exception as e:
    raise RuntimeError(f"❌ ComBat 校正失败: {e}")

adata.X = corrected_expression.T.values 
print("✅ Combat 批次校正完成!")

# ⚠️ 关键修改：在 ComBat 校正完批次后，再执行 Z-score
print("\n=== 5. 对校正后的数据进行标准化 (Z-score) ===")
sc.pp.scale(adata, max_value=10)
print("✅ Z-score 标准化完成!")

# --- 绘图与导出 ---
plot_and_export_pca(adata, raw_counts, output_dir)
plot_boxplot_comparison(adata, raw_counts, output_dir)

print("\n=== 9. 保存最终校正数据矩阵 ===")
corrected_df = pd.DataFrame(
    adata.X, 
    index=adata.obs_names, 
    columns=adata.var_names
)

if 'treatment' in adata.obs.columns:
    corrected_df['target_label'] = adata.obs['treatment']
elif 'Treatment' in adata.obs.columns:
    corrected_df['target_label'] = adata.obs['Treatment']

final_output_path = os.path.join(output_dir, output_file)
corrected_df.to_csv(final_output_path)
print(f"✅ 最终表达矩阵(ComBat+Scaled)已保存: {final_output_path}")
print("🎉 全部流程运行完毕！去 pca 文件夹查看你的丰收成果吧！")