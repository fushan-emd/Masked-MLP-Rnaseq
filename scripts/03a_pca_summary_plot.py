import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import sys
import matplotlib.collections as mcoll
from matplotlib.colors import LinearSegmentedColormap

# ==========================================
# 0. 获取路径与全局设置
# ==========================================
if getattr(sys, 'frozen', False):
    script_dir = os.path.dirname(sys.executable)
elif '__file__' in globals():
    script_dir = os.path.dirname(os.path.abspath(__file__))
else:
    script_dir = os.getcwd()

sns.set_theme(style="whitegrid") 
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False 

plt.rcParams.update({
    'axes.titlesize': 22, 'axes.labelsize': 18,
    'xtick.labelsize': 14, 'ytick.labelsize': 14,
    'legend.fontsize': 14, 'legend.title_fontsize': 16,
    'figure.facecolor': 'white', 'axes.facecolor': 'white'
})

# ==========================================
# 1. 读取数据
# ==========================================
barley_boxplot_csv = r'finalpic\FIG1PCA\B_Boxplot_Data.csv'
barley_pca_csv = r'finalpic\FIG1PCA\B_PCA_Coordinates_Data.csv'
wheat_boxplot_csv = r'finalpic\FIG1PCA\W_Boxplot_Data.csv'
wheat_pca_csv = r'finalpic\FIG1PCA\W_PCA_Coordinates_Data.csv'

try:
    barley_boxplot_df = pd.read_csv(barley_boxplot_csv, sep=None, engine='python')
    barley_pca_df = pd.read_csv(barley_pca_csv, sep=None, engine='python')
    wheat_boxplot_df = pd.read_csv(wheat_boxplot_csv, sep=None, engine='python')
    wheat_pca_df = pd.read_csv(wheat_pca_csv, sep=None, engine='python')

    for df in [barley_boxplot_df, barley_pca_df, wheat_boxplot_df, wheat_pca_df]:
        df.columns = df.columns.str.strip()
    
    barley_pca_df['Treatment'] = barley_pca_df['Treatment'].str.strip()
    wheat_pca_df['Treatment'] = wheat_pca_df['Treatment'].str.strip()
except FileNotFoundError as e:
    print(f"\n[错误] 找不到数据文件！{e}")
    sys.exit(1)

barley_pca_df['Batch'] = barley_pca_df['Batch'].astype(str)
barley_boxplot_df['Batch'] = barley_boxplot_df['Batch'].astype(str)
wheat_pca_df['Batch'] = wheat_pca_df['Batch'].astype(str)
wheat_boxplot_df['Batch'] = wheat_boxplot_df['Batch'].astype(str)

# 获取正确的排序顺序 (1, 2, 3, 4...)
barley_batch_order = sorted(barley_boxplot_df['Batch'].unique(), key=float)
wheat_batch_order = sorted(wheat_boxplot_df['Batch'].unique(), key=float)

# ==========================================
# 2. 核心修复：建立绝对颜色绑定字典 (100% 杜绝颜色错乱)
# ==========================================
muted_colors = sns.color_palette("muted", 10)
# Barley 颜色字典 {'1': 蓝色, '2': 橙色, ...}
barley_palette = dict(zip(barley_batch_order, muted_colors[:len(barley_batch_order)]))
# Wheat 颜色字典
wheat_palette = dict(zip(wheat_batch_order, muted_colors[:len(wheat_batch_order)]))

# ==========================================
# 3. 辅助绘图函数
# ==========================================
def apply_gradient_bg(ax, color_bottom, color_top, alpha=0.15):
    cmap = LinearSegmentedColormap.from_list('grad', [color_bottom, color_top])
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    gradient = np.linspace(0, 1, 256).reshape(-1, 1)
    ax.imshow(gradient, aspect='auto', cmap=cmap, extent=[xlim[0], xlim[1], ylim[0], ylim[1]], 
              alpha=alpha, zorder=-10, origin='lower')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

def plot_true_raincloud(ax, df, x_col, y_col, order, palette_dict, title, ylabel):
    # 【注意这里】：强制使用 hue_order 和传入的颜色字典
    sns.violinplot(data=df, x=x_col, y=y_col, hue=x_col, hue_order=order, order=order, palette=palette_dict, 
                   ax=ax, split=False, inner="box", legend=False, cut=0, width=0.8, alpha=0.8)
    
    for collection in ax.collections:
        if isinstance(collection, mcoll.PolyCollection):
            for path in collection.get_paths():
                vertices = path.vertices
                centers = np.round(vertices[:, 0]) 
                vertices[:, 0] = np.clip(vertices[:, 0], centers, np.inf)
            collection.set_edgecolor('white')
            collection.set_linewidth(1.5)
            
    MAX_SCATTER_POINTS = 150 
    sampled_chunks = []
    for _, group in df.groupby(x_col):
        sampled_chunks.append(group.sample(n=min(len(group), MAX_SCATTER_POINTS), random_state=42))
    df_sampled = pd.concat(sampled_chunks, ignore_index=True)

    sns.stripplot(data=df_sampled, x=x_col, y=y_col, hue=x_col, hue_order=order, order=order, palette=palette_dict, 
                  ax=ax, alpha=0.6, zorder=0, jitter=0.1, legend=False, rasterized=True, size=4)
                  
    for collection in ax.collections:
        if isinstance(collection, mcoll.PathCollection):
            offsets = collection.get_offsets()
            if len(offsets) > 0:
                offsets[:, 0] -= 0.15 
                collection.set_offsets(offsets)
                
    for line in ax.lines:
        line.set_color('black')
        line.set_linewidth(1.5)
        
    ax.set_title(title, fontweight='bold', pad=15)
    ax.set_xlabel('Batch ID', fontweight='bold')
    ax.set_ylabel(ylabel, fontweight='bold')

# ==========================================
# 4. 绘图布局 
# ==========================================
fig, axes = plt.subplots(2, 4, figsize=(26, 12)) 

treatment_markers = {'ck': 'o', 'salt': 'X'} 
pca_kwargs = {
    'markers': treatment_markers, 's': 150, 'alpha': 0.8, 
    'edgecolors': 'white', 'linewidths': 1.5, 'rasterized': True  
}

# ==========================================
# 5. Barley 行
# ==========================================
# 【注意这里】：PCA 图强制加入 hue_order 和 palette_dict
sns.scatterplot(data=barley_pca_df, x='PC1_Before', y='PC2_Before', 
                hue='Batch', hue_order=barley_batch_order, style='Treatment', 
                palette=barley_palette, ax=axes[0,0], **pca_kwargs)
axes[0,0].set_title('Barley PCA (Before)', fontweight='bold', pad=15)
axes[0,0].set_xlabel('PC1 Before (39.7%)', fontweight='bold') 
axes[0,0].set_ylabel('PC2 Before (17.6%)', fontweight='bold')
handles_b, labels_b = axes[0,0].get_legend_handles_labels()
axes[0,0].get_legend().remove()

sns.scatterplot(data=barley_pca_df, x='PC1_After', y='PC2_After', 
                hue='Batch', hue_order=barley_batch_order, style='Treatment', 
                palette=barley_palette, ax=axes[0,1], legend=False, **pca_kwargs)
axes[0,1].set_title('Barley PCA (After ComBat)', fontweight='bold', pad=15)
axes[0,1].set_xlabel('PC1 After (35.0%)', fontweight='bold') 
axes[0,1].set_ylabel('PC2 After (9.2%)', fontweight='bold')

plot_true_raincloud(axes[0,2], barley_boxplot_df, 'Batch', 'Expression_Before_Log1p', 
                    barley_batch_order, barley_palette, 'Barley Expression (Before)', 'Expression (Log1p)')
plot_true_raincloud(axes[0,3], barley_boxplot_df, 'Batch', 'Expression_After_ComBat_Zscore', 
                    barley_batch_order, barley_palette, 'Barley Expression (After)', 'Expression (Z-score)')

leg_b = axes[0,3].legend(handles_b, labels_b, title='Barley Groups', 
                         bbox_to_anchor=(1.05, 1), loc='upper left', frameon=False, markerscale=1.2)
leg_b.get_title().set_fontweight('bold')

# ==========================================
# 6. Wheat 行
# ==========================================
sns.scatterplot(data=wheat_pca_df, x='PC1_Before', y='PC2_Before', 
                hue='Batch', hue_order=wheat_batch_order, style='Treatment', 
                palette=wheat_palette, ax=axes[1,0], **pca_kwargs)
axes[1,0].set_title('Wheat PCA (Before)', fontweight='bold', pad=15)
axes[1,0].set_xlabel('PC1 Before (35.6%)', fontweight='bold')
axes[1,0].set_ylabel('PC2 Before (19.8%)', fontweight='bold')
handles_w, labels_w = axes[1,0].get_legend_handles_labels()
axes[1,0].get_legend().remove()

sns.scatterplot(data=wheat_pca_df, x='PC1_After', y='PC2_After', 
                hue='Batch', hue_order=wheat_batch_order, style='Treatment', 
                palette=wheat_palette, ax=axes[1,1], legend=False, **pca_kwargs)
axes[1,1].set_title('Wheat PCA (After ComBat)', fontweight='bold', pad=15)
axes[1,1].set_xlabel('PC1 After (17.6%)', fontweight='bold')
axes[1,1].set_ylabel('PC2 After (8.9%)', fontweight='bold')

plot_true_raincloud(axes[1,2], wheat_boxplot_df, 'Batch', 'Expression_Before_Log1p', 
                    wheat_batch_order, wheat_palette, 'Wheat Expression (Before)', 'Expression (Log1p)')
plot_true_raincloud(axes[1,3], wheat_boxplot_df, 'Batch', 'Expression_After_ComBat_Zscore', 
                    wheat_batch_order, wheat_palette, 'Wheat Expression (After)', 'Expression (Z-score)')

leg_w = axes[1,3].legend(handles_w, labels_w, title='Wheat Groups', 
                         bbox_to_anchor=(1.05, 1), loc='upper left', frameon=False, markerscale=1.2)
leg_w.get_title().set_fontweight('bold')

# ==========================================
# 7. 统一格式化与保存
# ==========================================
for i in range(4):
    apply_gradient_bg(axes[0, i], color_bottom='#ffffff', color_top='#cce0ff', alpha=0.3)
    apply_gradient_bg(axes[1, i], color_bottom='#ffffff', color_top='#ffe5cc', alpha=0.3)

for ax in axes.flat:
    ax.grid(color='gray', linestyle='--', linewidth=1.0, alpha=0.3) 
    ax.spines['left'].set_color('black')
    ax.spines['left'].set_linewidth(2.0) 
    ax.spines['bottom'].set_color('black')
    ax.spines['bottom'].set_linewidth(2.0)
    ax.tick_params(axis='both', colors='black', width=2.0, length=6)
    
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight('bold')
    sns.despine(ax=ax)

plt.tight_layout(pad=2.0, w_pad=2.0, h_pad=2.5)

os.makedirs(script_dir, exist_ok=True)
png_output_path = os.path.join(script_dir, 'Figure1_Final_ColorFixed_700DPI.png')
pdf_output_path = os.path.join(script_dir, 'Figure1_Final_ColorFixed_AI.pdf')

print("\n⏳ 正在渲染图片，请稍候...")
fig.savefig(pdf_output_path, dpi=300, bbox_inches='tight', transparent=False, format='pdf')
fig.savefig(png_output_path, dpi=700, bbox_inches='tight', transparent=False)

print(f"✅ 颜色 100% 修复对应版已导出！")
print(f"🖼️ PNG 高清图路径: {png_output_path}")

plt.show()