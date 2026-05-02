import os
import sys
import datetime
import copy
import matplotlib_venn
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Dataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib_venn import venn3
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, roc_curve, auc, roc_auc_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import xgboost as xgb
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline
from sklearn.inspection import permutation_importance
import shap
import warnings
warnings.filterwarnings('ignore')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================================================
# 📂 0. 自动归档系统
# =========================================================

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
output_dir = f"Final_Paper_MaskedMLP_{timestamp}"
os.makedirs(output_dir, exist_ok=True)
print(f"📁 [系统] 结果保存至: ./{output_dir}/")

def save_path(filename): return os.path.join(output_dir, filename)

# ---------------------------------------------------------
# 1. 数据准备 (保留原始基因 ID)
# ---------------------------------------------------------
print("📖 读取数据...")
input_file = '/root/nfsdata/project/bishe/B_pca/barley_Data_Combat_Corrected.csv'

if not os.path.exists(input_file):
    print(f"⚠️ 找不到 {input_file}，将生成模拟数据进行测试...")
    target_df = pd.DataFrame(np.random.randn(200, 2005), columns=[f'Gene_{i}' for i in range(2005)])
    target_df['target_label'] = np.random.choice(['ck', 'salt'], 200)
    target_df.index.name = 'sample_id'
else:
    print(f"✅ 成功加载上一步处理的数据: {input_file}")
    target_df = pd.read_csv(input_file, index_col=0)

# 自动排除非基因列，剩下的列名就是你的“原始基因 ID”
exclude = ['target', 'target_label', 'Batch', 'treatment', 'Treatment', 'sample_id']
candidates = [c for c in target_df.columns if c not in exclude]

# 筛选高变基因 (Top 2000) 用于输入模型
variances = target_df[candidates].var()
top_genes = variances.nlargest(2000).index.tolist()
# 这里的 top_genes 列表里存的就是如 HORVU.MOREX... 的原始 ID
pd.DataFrame(top_genes, columns=['Original_Gene_ID']).to_csv(save_path("Used_Feature_List.csv"), index=False)

X_raw = target_df[top_genes].values.astype(np.float32)

# 标签编码 (例如: ck -> 0, salt -> 1)
le = LabelEncoder()
y = le.fit_transform(target_df['target_label'].astype(str))
print(f"🏷️ 标签映射关系: {dict(zip(le.classes_, le.transform(le.classes_)))}")

X_train, X_test, y_train, y_test = train_test_split(X_raw, y, test_size=0.2, random_state=42, stratify=y)

# 标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 转 Tensor
X_train_t = torch.FloatTensor(X_train_scaled).to(device)
X_test_t = torch.FloatTensor(X_test_scaled).to(device)
y_train_float = torch.FloatTensor(y_train).unsqueeze(1).to(device)
y_test_float = torch.FloatTensor(y_test).unsqueeze(1).to(device)

print("✅ 数据准备完毕。")

# ---------------------------------------------------------
# 2. 定义 Optimized MLP (新增 Mask Gene 功能)
# ---------------------------------------------------------
class OptimizedMLP(nn.Module):
    def __init__(self, input_dim, mask_rate=0.0):
        super(OptimizedMLP, self).__init__()
        
        self.model = nn.Sequential(
            nn.Dropout(p=mask_rate), 
            
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),      

            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(32, 1)      
        )

    def forward(self, x):
        return self.model(x)

# ---------------------------------------------------------
# 3. Phase 2: 5折 CV (交叉验证)
# ---------------------------------------------------------
print("\n⚔️ [Phase 2] 开始 5-Fold CV 对比...")

ml_models = {
    "Univariate (ANOVA)": Pipeline([
        ('selector', SelectKBest(f_classif, k=20)), 
        ('clf', LogisticRegression(solver='liblinear'))
    ]),
    "Lasso (L1)": LogisticRegression(penalty='l1', solver='liblinear', C=0.5, max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "SVM": SVC(probability=True, kernel='linear', random_state=42),
    "XGBoost": xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', verbosity=0, random_state=42)
}

kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_results = {name: [] for name in ml_models.keys()}
cv_results["MLP (Masked)"] = [] 

X_all_scaled = scaler.fit_transform(X_raw)
y_all = y
CV_MASK_RATE = 0.15 

for fold, (train_idx, val_idx) in enumerate(kfold.split(X_train, y_train)):
    print(f"   Fold {fold+1}/5...", end="")
    
    # 1. 从切分好的训练集中取数据
    X_tr_raw, X_val_raw = X_train[train_idx], X_train[val_idx]
    y_tr, y_val = y_train[train_idx], y_train[val_idx]
    
    # 2. 核心修复：在每一折内部独立进行标准化，防止验证集信息泄露给训练集
    fold_scaler = StandardScaler()
    X_tr = fold_scaler.fit_transform(X_tr_raw)  # 仅使用当前折的训练集 fit
    X_val = fold_scaler.transform(X_val_raw)    # 验证集仅 transform

    # 跑 ML 模型
    for name, clf in ml_models.items():
        clf.fit(X_tr, y_tr)
        cv_results[name].append(accuracy_score(y_val, clf.predict(X_val)))
    
    # 跑 MLP
    Xt_tr = torch.FloatTensor(X_tr).to(device)
    Yt_tr = torch.FloatTensor(y_tr).unsqueeze(1).to(device)
    Xt_val = torch.FloatTensor(X_val).to(device)
    
    mlp = OptimizedMLP(input_dim=len(top_genes), mask_rate=CV_MASK_RATE).to(device)
    opt_mlp = optim.AdamW(mlp.parameters(), lr=0.001, weight_decay=1e-4)
    crit_mlp = nn.BCEWithLogitsLoss()
    
    mlp.train() 
    dataset = TensorDataset(Xt_tr, Yt_tr)
    dl = DataLoader(dataset, batch_size=32, shuffle=True)
    
    for ep in range(50):
        for xb, yb in dl:
            opt_mlp.zero_grad()
            out = mlp(xb)
            loss = crit_mlp(out, yb)
            loss.backward()
            opt_mlp.step()
    
    mlp.eval() 
    with torch.no_grad():
        logits = mlp(Xt_val)
        preds = (torch.sigmoid(logits) > 0.5).float().cpu().numpy()
    cv_results["MLP (Masked)"].append(accuracy_score(y_val, preds))
    print(" Done.")

# ---------------------------------------------------------
# 4. Phase 3: 最终微调 (MLP) - 全程 100 Epochs
# ---------------------------------------------------------
print("\n🚀 [Phase 3] 最终微调 MLP (Gene Masking Enabled) - 全程 100 Epochs...")
# 🌟 核心修复 1：从 X_train 中再切分出 15% 作为验证集，保护 X_test 绝对不可见
X_train_sub, X_val_sub, y_train_sub, y_val_sub = train_test_split(
    X_train, y_train, test_size=0.15, random_state=42, stratify=y_train
)

# 🌟 核心修复 2：严格的标准化逻辑，只在纯训练集上 fit
# 严格的标准化逻辑
scaler_final = StandardScaler()
X_train_sub_scaled = scaler_final.fit_transform(X_train_sub)
X_val_sub_scaled = scaler_final.transform(X_val_sub)
X_test_scaled = scaler_final.transform(X_test) # 测试集仅 transform

# 转 Tensor
X_tr_t = torch.FloatTensor(X_train_sub_scaled).to(device)
y_tr_f = torch.FloatTensor(y_train_sub).unsqueeze(1).to(device)
X_val_t = torch.FloatTensor(X_val_sub_scaled).to(device)
y_val_f = torch.FloatTensor(y_val_sub).unsqueeze(1).to(device)
X_test_t = torch.FloatTensor(X_test_scaled).to(device)
y_test_f = torch.FloatTensor(y_test).unsqueeze(1).to(device)

FINAL_MASK_RATE = 0.15
print(f"🧬 基因掩码率 (Mask Rate): {FINAL_MASK_RATE}")

final_model = OptimizedMLP(input_dim=len(top_genes), mask_rate=FINAL_MASK_RATE).to(device)

optimizer = optim.AdamW(final_model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
criterion = nn.BCEWithLogitsLoss()

# DataLoader 仅使用切分后的训练子集
train_dl = DataLoader(TensorDataset(X_tr_t, y_tr_f), batch_size=32, shuffle=True)
history = {'train_loss': [], 'val_loss': []}

best_metric = 0 
best_model_wts = copy.deepcopy(final_model.state_dict())

# 完整跑完 100 轮
for epoch in range(100):
    final_model.train() 
    losses = []
    for xb, yb in train_dl:
        optimizer.zero_grad()
        out = final_model(xb)
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    scheduler.step()
    
    final_model.eval()
    with torch.no_grad():
        v_out = final_model(X_val_t)
        v_loss = criterion(v_out, y_val_f).item()
        probs_val = torch.sigmoid(v_out).cpu().detach().numpy()
        
        # 🌟 修复点 2：使用正确的 y_val_sub 并捕获真实异常
        try:
            current_auc = roc_auc_score(y_val_sub, probs_val.flatten())
        except Exception as e: 
            print(f"⚠️ 第 {epoch+1} 轮 AUC 计算异常: {e}") 
            current_auc = 0.5

    history['train_loss'].append(np.mean(losses))
    history['val_loss'].append(v_loss)
    
    if (epoch+1) % 10 == 0:
        print(f"  Epoch {epoch+1}: Train Loss={np.mean(losses):.4f} | Val AUC={current_auc:.4f}")

    if current_auc > best_metric:
        best_metric = current_auc
        best_model_wts = copy.deepcopy(final_model.state_dict())
        if (epoch+1) % 10 != 0: 
            print(f"    ⭐ Epoch {epoch+1}: 发现更高验证集 AUC: {best_metric:.4f}")

final_model.load_state_dict(best_model_wts)
print(f"\n✅ 训练完成！已加载最优模型权重 (Best AUC: {best_metric:.4f})")

final_model.eval()
with torch.no_grad():
    probs_v_opt = torch.sigmoid(final_model(X_val_t)).cpu().numpy().flatten()
fpr_v, tpr_v, thresholds_v = roc_curve(y_val_sub, probs_v_opt)

J = tpr_v - fpr_v
ix = np.argmax(J)
best_thresh = thresholds_v[ix]
print(f"🔥 基于验证集确定的最佳区分阈值: {best_thresh:.4f}")

# 🌟 终极测试：将选定的模型和阈值，应用于【完全未见过的测试集】
with torch.no_grad():
    probs_t = torch.sigmoid(final_model(X_test_t)).cpu().numpy().flatten()
fpr_t, tpr_t, _ = roc_curve(y_test, probs_t)
auc_t = auc(fpr_t, tpr_t)

preds_t_opt = (probs_t >= best_thresh).astype(int)
final_acc_t = accuracy_score(y_test, preds_t_opt)
print(f"🏆 MLP 独立测试集最终表现: AUC={auc_t:.4f}, Accuracy={final_acc_t*100:.2f}%")

# ---------------------------------------------------------
# 5. 统一训练并提取所有模型的 Top 30 原始基因
# ---------------------------------------------------------
print("\n🔄 [Phase 5] 正在统一各模型训练步调 (使用子训练集) 并提取特征...")

# 初始化用于存储混淆矩阵和 ROC 的字典
all_preds = {}
all_probs = {}
gene_sets = {}
ml_results_roc = []
# --- 5.1 评估 MLP (直接使用 Phase 4 训练好的 final_model) ---
final_model.eval()
with torch.no_grad():
    mlp_probs = torch.sigmoid(final_model(X_test_t)).cpu().numpy().flatten()
    mlp_preds = (mlp_probs >= best_thresh).astype(int)
    all_probs["MLP (Masked)"] = mlp_probs
    all_preds["MLP (Masked)"] = mlp_preds
fpr_mlp, tpr_mlp, _ = roc_curve(y_test, mlp_probs)
ml_results_roc.append({
    "name": "MLP (Masked)", 
    "fpr": fpr_mlp, 
    "tpr": tpr_mlp, 
    "auc": auc(fpr_mlp, tpr_mlp), 
    "color": "red", 
    "lw": 3.0
})
print("\n🧠 正在使用 SHAP (DeepExplainer) 计算 MLP 特征重要性 (这可能需要一小会)...")

# 1. 准备背景数据集 (Background Data)
# DeepExplainer 需要一个背景数据集来做基准期望值。为了计算效率，通常从训练集中随机采样 100-200 个样本。
X_train_final_scaled = scaler_final.transform(X_train)
np.random.seed(42) # 保证每次跑出来的 SHAP 抽样一致
bg_samples_idx = np.random.choice(X_train_final_scaled.shape[0], min(200, X_train_final_scaled.shape[0]), replace=False)
background_tensor = torch.FloatTensor(X_train_final_scaled[bg_samples_idx]).to(device)

# 2. 准备要解释的数据 (使用独立的测试集来评估特征重要性更加客观)
test_tensor = torch.FloatTensor(X_test_scaled).to(device)

# 3. 初始化 DeepExplainer 并计算 SHAP 值
explainer = shap.DeepExplainer(final_model, background_tensor)
shap_values = explainer.shap_values(test_tensor)

# 兼容处理：有些 SHAP 版本返回 list，有些返回 tensor
if isinstance(shap_values, list):
    shap_values = shap_values[0]

# 确保转换为 NumPy 数组
shap_values = np.array(shap_values)

# 4. 计算全局特征重要性
shap_importance = np.abs(shap_values).mean(axis=0)

# 🌟 核心修复：彻底展平为 1D 数组 (2000,)，消除多余维度
shap_importance = shap_importance.flatten()

# 5. 提取 Top 30 基因并存入集合
top_idx_mlp = np.argsort(shap_importance)[-30:]
# 为了极致的安全，强制把 i 转成标准的 Python 整数
genes_mlp = set([top_genes[int(i)] for i in top_idx_mlp])
gene_sets["MLP (Masked)"] = genes_mlp

# 6. 导出 MLP Top 30 基因 CSV
df_mlp_genes = pd.DataFrame({
    'Original_Gene_ID': top_genes, 
    'Importance_SHAP': shap_importance
}).sort_values('Importance_SHAP', ascending=False)

df_mlp_genes.head(30).to_csv(save_path("Top30_Genes_MLP_Masked_SHAP.csv"), index=False)
print("💾 已导出 MLP 的 Top 30 基因 (基于 SHAP 算法)。")

# --- 5.2 重新训练传统 ML 模型并提取重要性 ---
for name, clf in ml_models.items():
    # 🌟 核心：统一使用子训练集，保持与 MLP 训练范围一致
    clf.fit(X_train_sub_scaled, y_train_sub)
    
    # 预测并存储结果 (用于后续混淆矩阵)
    if hasattr(clf, "predict_proba"):
        probs = clf.predict_proba(X_test_scaled)[:, 1]
    else:
        probs = clf.decision_function(X_test_scaled)
    
    all_probs[name] = probs
    all_preds[name] = clf.predict(X_test_scaled)
    
    # 更新 ROC 数据
    fpr, tpr, _ = roc_curve(y_test, probs)
    ml_results_roc.append({"name": name, "fpr": fpr, "tpr": tpr, "auc": auc(fpr, tpr), "color": None, "lw": 1.5})
    
    # 提取特征重要性并导出
    try:
        imp = None
        if name == "Univariate (ANOVA)":
            imp = clf.named_steps['selector'].scores_ 
        elif hasattr(clf, 'feature_importances_'): 
            imp = clf.feature_importances_ 
        elif hasattr(clf, 'coef_'): 
            imp = np.abs(clf.coef_[0]) 
        else: 
            imp = permutation_importance(clf, X_test_scaled, y_test, n_repeats=5).importances_mean
            
        if imp is not None:
            top_idx = np.argsort(imp)[-30:]
            gene_sets[name] = set([top_genes[i] for i in top_idx])
            
            safe_name = name.replace(' ', '_').replace('(', '').replace(')', '')
            df_ml_genes = pd.DataFrame({'Original_Gene_ID': top_genes, 'Importance': imp}).sort_values('Importance', ascending=False)
            df_ml_genes.head(30).to_csv(save_path(f"Top30_Genes_{safe_name}.csv"), index=False)
            print(f"💾 已导出 {name} 的 Top 30 基因。")
            
    except Exception as e: 
        print(f"⚠️ 提取 {name} 重要性失败: {e}")

print("✅ 所有模型统一训练及特征导出完成。")

# =========================================================
# 🎨 6. 绘图
# =========================================================
print("\n🎨 正在生成最终综合可视化大图...")
plt.figure(figsize=(20, 12))

# A. CV Accuracy
plt.subplot(2, 3, 1)
df_cv = pd.DataFrame(cv_results).melt(var_name='Algorithm', value_name='Accuracy')
sns.boxplot(data=df_cv, x='Algorithm', y='Accuracy', palette="Set3")
plt.xticks(rotation=45, ha='right')
plt.title('A. Algorithm Comparison (5-Fold CV)', fontsize=14, fontweight='bold')
plt.ylabel('CV Accuracy')

# B. Loss
plt.subplot(2, 3, 2)
plt.plot(history['train_loss'], label='Train Loss (Masked)', lw=2)
plt.plot(history['val_loss'], label='Val Loss (Full)', lw=2)
plt.title('B. MLP Learning Curve', fontsize=14, fontweight='bold')
plt.xlabel('Epoch')
plt.ylabel('BCE Loss')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)

# C. ROC
plt.subplot(2, 3, 3)
for res in ml_results_roc:
    plt.plot(res['fpr'], res['tpr'], label=f"{res['name']} (AUC: {res['auc']:.3f})", color=res['color'], linewidth=res['lw'])
plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
plt.title('C. ROC Curves on Test Set', fontsize=14, fontweight='bold')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right", fontsize=9)
plt.grid(True, linestyle='--', alpha=0.6)

# D. Heatmap
plt.subplot(2, 3, 4)
names = list(gene_sets.keys())
mat = np.zeros((len(names), len(names)))
for i, m1 in enumerate(names):
    for j, m2 in enumerate(names):
        set1, set2 = gene_sets.get(m1, set()), gene_sets.get(m2, set())
        if len(set1) > 0 and len(set2) > 0:
            mat[i, j] = len(set1 & set2) / len(set1 | set2) 

sns.heatmap(mat, xticklabels=names, yticklabels=names, annot=True, fmt=".2f", cmap="YlGnBu", cbar_kws={'label': 'Jaccard Similarity'})
plt.title('D. Top 30 Genes Similarity (Jaccard)', fontsize=14, fontweight='bold')
plt.xticks(rotation=45, ha='right')

# E. Venn
plt.subplot(2, 3, 5)
try:
    venn_names = ['MLP (Masked)', 'Lasso (L1)', 'Random Forest'] 
    venn_sets = [gene_sets.get(n, set()) for n in venn_names]
    
    if all(len(s) > 0 for s in venn_sets):
        venn3(venn_sets, set_labels=venn_names)
        plt.title('E. Top 30 Genes Overlap', fontsize=14, fontweight='bold')
    else:
        plt.text(0.5, 0.5, 'Insufficient data for Venn Diagram', ha='center', va='center')
except Exception as e: 
    plt.text(0.5, 0.5, 'Venn Diagram Plot Failed', ha='center', va='center')

plt.tight_layout()
final_fig_path = save_path("Final_Full_Figure_Masked.png")
plt.savefig(final_fig_path, dpi=300, bbox_inches='tight')
plt.show()

print(f"\n🎉 运行结束！")
print(f"📊 所有的 Top 30 基因列表 (.csv) 和综合对比图已保存至文件夹: {output_dir}/")