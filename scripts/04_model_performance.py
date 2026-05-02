import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import sys
from scipy import stats
from matplotlib.patches import FancyBboxPatch, Rectangle
from matplotlib.colors import LinearSegmentedColormap

# ==========================================
# 0. 核心配置：确保导出可编辑的矢量字体
# ==========================================
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

if getattr(sys, 'frozen', False):
    script_dir = os.path.dirname(sys.executable)
elif '__file__' in globals():
    script_dir = os.path.dirname(os.path.abspath(__file__))
else:
    script_dir = os.getcwd()

# 加载数据
b_lc = pd.read_csv(os.path.join(script_dir, 'barley_LearningCurve_Data.csv'))
w_lc = pd.read_csv(os.path.join(script_dir, 'wheat_LearningCurve_Data.csv'))
b_cv = pd.read_csv(os.path.join(script_dir, 'barley_Algorithm_CV_Data.csv'))
w_cv = pd.read_csv(os.path.join(script_dir, 'wheat_Algorithm_CV_Data.csv'))
b_roc = pd.read_csv(os.path.join(script_dir, 'barley_ROC_Data.csv'))
w_roc = pd.read_csv(os.path.join(script_dir, 'wheat_ROC_Data.csv'))

# 样式设置
def set_style():
    plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False
    sns.set_theme(style="whitegrid", context="paper")
set_style()

# 右侧 ROC 曲线继续保持独立的高级定制色
MODEL_COLORS = {
    "MLP (Masked)": "#E15759",       
    "Random Forest": "#2A9D8F",      
    "XGBoost": "#F4A261",            
    "SVM": "#457B9D",                
    "Lasso (L1)": "#E9C46A",         
    "Univariate (ANOVA)": "#9CA3AF"  
}

def make_bars_rounded(ax, radius=0.08):
    for patch in ax.patches:
        if isinstance(patch, Rectangle):
            x, y = patch.get_xy()
            w, h = patch.get_width(), patch.get_height()
            if h == 0: continue
            new_patch = FancyBboxPatch((x, y), w, h,
                                       boxstyle=f"round,pad=0,rounding_size={radius}",
                                       ec="none", fc=patch.get_facecolor(),
                                       alpha=patch.get_alpha(), zorder=patch.get_zorder())
            patch.remove()
            ax.add_patch(new_patch)

# 🌟 核心修改 1：加深透明度 (alpha=0.4)
def apply_gradient_bg(ax, color_bottom, color_top, alpha=0.4):
    cmap = LinearSegmentedColormap.from_list('grad', [color_bottom, color_top])
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    gradient = np.linspace(0, 1, 256).reshape(-1, 1)
    ax.imshow(gradient, aspect='auto', cmap=cmap, 
              extent=[xlim[0], xlim[1], ylim[0], ylim[1]], 
              alpha=alpha, zorder=-10, origin='lower')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

# 统一轴线格式化函数 (纯黑化)
def format_black_axes(ax):
    # 保持网格线淡淡的灰色
    ax.grid(color='gray', linestyle='--', linewidth=1.0, alpha=0.3) 
    
    # 横纵坐标轴线变黑加粗
    ax.spines['left'].set_color('black')
    ax.spines['left'].set_linewidth(2.0) 
    ax.spines['bottom'].set_color('black')
    ax.spines['bottom'].set_linewidth(2.0)
    
    # 刻度线和数字统统变黑加粗
    ax.tick_params(axis='both', colors='black', width=2.0, length=6)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight('bold')

def add_stat_annotation(ax, x1, x2, y, h, p_value):
    if p_value < 0.001: 
        text = '***'
    elif p_value < 0.01: 
        text = '**'
    elif p_value < 0.05: 
        text = '*'
    else: 
        return  
        
    ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c='black') 
    ax.text((x1+x2)*.5, y+h+0.005, text, ha='center', va='bottom', color='black', fontsize=14, fontweight='bold')

fig = plt.figure(figsize=(24, 14)) 
gs = fig.add_gridspec(2, 3, wspace=0.25, hspace=0.6) 

# 🌟 去掉了 panel_labels 参数
def plot_performance_row(row_idx, species_name, n_val, lc_df, cv_df, roc_df, bg_color_top):
    # ---------------------------------------------------
    # 子图 1: Learning Curve
    # ---------------------------------------------------
    ax_lc = fig.add_subplot(gs[row_idx, 0])
    ax_lc.plot(lc_df['Epoch'], lc_df['train_loss'], label='Train Loss', color='#4E79A7', lw=3)
    ax_lc.plot(lc_df['Epoch'], lc_df['val_loss'], label='Val Loss', color='#E15759', lw=3, linestyle='--')
    
    # 🌟 核心修改 2：副标题变大 (fontsize=22)
    ax_lc.set_title(f"{species_name}: Training Dynamics\n(n={n_val})", fontsize=22, fontweight='bold', pad=15)
    ax_lc.set_xlabel("Epochs", fontsize=15, fontweight='bold')
    ax_lc.set_ylabel("Loss", fontsize=15, fontweight='bold')
    ax_lc.legend(frameon=True, fontsize=13, loc='upper right')
    ax_lc.tick_params(labelsize=13)
    sns.despine(ax=ax_lc)
    
    apply_gradient_bg(ax_lc, '#ffffff', bg_color_top)
    format_black_axes(ax_lc)

    # ---------------------------------------------------
    # 子图 2: Mean Performance 
    # ---------------------------------------------------
    ax_bar = fig.add_subplot(gs[row_idx, 1])
    means = cv_df.mean()
    stds = cv_df.std()
    
    order = means.sort_values(ascending=True).index
    means = means[order]
    stds = stds[order]
    
    colors = [ "#CCCCCC" if "MLP" not in idx else "#E15759" for idx in order]
    
    lower_err = stds.values
    upper_err = np.minimum(stds.values, 1.0 - means.values) 
    asymmetric_err = [lower_err, upper_err]
    
    bars = ax_bar.bar(range(len(means)), means, yerr=asymmetric_err, color=colors, 
                      capsize=6, error_kw={'elinewidth':1.5, 'alpha':0.8, 'ecolor':'black'}, alpha=0.9, width=0.6)
    
    ax_bar.set_xticks(range(len(means)))
    ax_bar.set_xticklabels(order, rotation=35, ha='right', fontsize=13)
    ax_bar.set_ylabel("Accuracy", fontsize=15, fontweight='bold')
    
    # 🌟 副标题变大 (fontsize=22)
    ax_bar.set_title(f"{species_name}: Model Accuracy\n(n={n_val})", fontsize=22, fontweight='bold', pad=15)
    ax_bar.set_ylim(0.80, 1.05) 
    
    mlp_idx = list(order).index("MLP (Masked)")
    target_baseline = "XGBoost" 
    if target_baseline in list(order):
        baseline_idx = list(order).index(target_baseline)
        t_stat, p_val = stats.ttest_rel(cv_df["MLP (Masked)"], cv_df[target_baseline])
        
        bracket_y = max(means.iloc[mlp_idx] + upper_err[mlp_idx], 
                        means.iloc[baseline_idx] + upper_err[baseline_idx]) + 0.015
        add_stat_annotation(ax_bar, min(mlp_idx, baseline_idx), max(mlp_idx, baseline_idx), bracket_y, 0.005, p_val)
    
    for i, v in enumerate(means):
        ax_bar.text(i, 0.82, f"{v*100:.1f}%", 
                    ha='center', fontsize=12, fontweight='bold', 
                    color='white' if colors[i]=="#E15759" else 'black')
    
    make_bars_rounded(ax_bar, radius=0.08)
    ax_bar.tick_params(labelsize=13)
    
    sns.despine(ax=ax_bar, bottom=False)
    apply_gradient_bg(ax_bar, '#ffffff', bg_color_top)
    format_black_axes(ax_bar)
    ax_bar.grid(False, axis='x')

    # ---------------------------------------------------
    # 子图 3: ROC Curve
    # ---------------------------------------------------
    ax_roc = fig.add_subplot(gs[row_idx, 2])
    fpr_cols = [c for c in roc_df.columns if '_FPR' in c and not c.endswith('.1') and not c.endswith('.2')]
    
    for fpr_col in fpr_cols:
        model_name = fpr_col.replace('_FPR', '')
        tpr_col = fpr_col.replace('_FPR', '_TPR')
        
        x = roc_df[fpr_col].dropna()
        y = roc_df[tpr_col].dropna()
        
        color = MODEL_COLORS.get(model_name, "#333333")
        lw = 3.5 if "MLP" in model_name else 2.0
        zorder = 10 if "MLP" in model_name else 5
        alpha = 1.0 if "MLP" in model_name else 0.8
        
        ax_roc.plot(x, y, label=model_name, color=color, lw=lw, zorder=zorder, alpha=alpha)
        
    ax_roc.plot([0, 1], [0, 1], 'k--', alpha=0.4, lw=1.5)
    
    # 🌟 副标题变大 (fontsize=22)
    ax_roc.set_title(f"{species_name}: ROC Curves\n(n={n_val})", fontsize=22, fontweight='bold', pad=15)
    ax_roc.set_xlabel("False Positive Rate", fontsize=15, fontweight='bold')
    ax_roc.set_ylabel("True Positive Rate", fontsize=15, fontweight='bold')
    ax_roc.legend(frameon=False, fontsize=12, loc='lower right')
    ax_roc.tick_params(labelsize=13)
    
    sns.despine(ax=ax_roc)
    apply_gradient_bg(ax_roc, '#ffffff', bg_color_top)
    format_black_axes(ax_roc)

# ==========================================
# 绘制
# ==========================================
n_barley = 246
n_wheat = 61  

# 🌟 核心修改 3：传入颜色更深的色号
plot_performance_row(0, "Barley", n_barley, b_lc, b_cv, b_roc, '#cce0ff') # 深一点的蓝色
plot_performance_row(1, "Wheat", n_wheat, w_lc, w_cv, w_roc, '#ffe5cc')   # 深一点的橙色

fig.suptitle('Machine Learning Model Performance & Robustness', 
             fontsize=34, fontweight='bold', y=1.06)

png_path = os.path.join(script_dir, 'Figure2_Performance_Final.png')
pdf_path = os.path.join(script_dir, 'Figure2_Performance_Final.pdf')

plt.savefig(png_path, dpi=700, bbox_inches='tight')
plt.savefig(pdf_path, dpi=700, bbox_inches='tight')
print(f"✅ 图表已导出！渐变更深、去除标号、副标题已放大。")

plt.show()