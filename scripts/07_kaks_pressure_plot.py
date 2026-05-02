import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

input_csv = 'FINAL_KaKs_Results.csv'

if not os.path.exists(input_csv):
    print(f"❌ 找不到文件: {input_csv}。请确保文件放在代码同级目录下。")
else:
    print("⏳ 正在生成高清 Ka/Ks 进化压力图 (修复字体显示问题)...")
    
    # 1. 加载并清洗数据
    df = pd.read_csv(input_csv)
    df = df.drop_duplicates().copy()

    # 2. 配置模型选出的核心基因名单
    target_genes = [
        "HORVU.MOREX.r3.1HG0005750", "HORVU.MOREX.r3.5HG0495580", 
        "HORVU.MOREX.r3.1HG0080260", "HORVU.MOREX.r3.1HG0069750",
        "TraesCS5B02G296200", "TraesCS2A02G284200", 
        "TraesCS3A02G122100", "TraesCS4B02G215900", "TraesCS2B02G364400", "TraesCS5D02G286200"
    ]

    def format_id(gene_id):
        # 【修改点】：换成了标准且 100% 兼容的普通星号 (*)
        if gene_id in target_genes:
            return f"{gene_id} (*)"
        return gene_id

    # 3. 保留完整前缀并打上标准的星号标签
    df['Formatted_Barley'] = df['Barley_ID'].apply(format_id)
    df['Formatted_Wheat'] = df['Wheat_ID'].apply(format_id)

    # 上下拼接大麦和小麦ID
    df['Pair_Label'] = df['Formatted_Barley'] + '\nvs\n' + df['Formatted_Wheat']

    # 按 Ka/Ks 值降序排列
    df = df.sort_values(by='Ka/Ks', ascending=False)

    # 4. 样式与画板设置
    sns.set_theme(style="whitegrid", rc={"axes.facecolor": "#F4F6F9", "figure.facecolor": "#F4F6F9"})
    plt.figure(figsize=(18, 10), facecolor='#F4F6F9')

    # 绘制带有色彩映射的柱状图
    ax = sns.barplot(x='Pair_Label', y='Ka/Ks', data=df, palette='Spectral')

    # 画一条 Ka/Ks = 1.0 的红色警戒线
    plt.axhline(y=1.0, color='#E74C3C', linestyle='--', linewidth=2.5, label='Ka/Ks = 1.0 (Neutral Selection)')

    # 在每根柱子上方精准打上具体数值
    for p in ax.patches:
        ax.annotate(format(p.get_height(), '.3f'), 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha = 'center', va = 'center', 
                    xytext = (0, 9), 
                    textcoords = 'offset points',
                    fontsize=10, fontweight='bold', color='#333333')

    # 5. 坐标轴与字体美化
    plt.xticks(rotation=45, ha='right', fontsize=10, fontweight='bold', color='#333333')
    plt.yticks(fontsize=12, fontweight='bold', color='#333333')
    
    # 【修改点】：底部注释里的星号也同步修改
    plt.xlabel('Homologous Gene Pairs (Barley vs Wheat)\n[ * Indicates core target genes selected by the model ]', 
               fontsize=14, fontweight='bold', color='#2C3E50', labelpad=15)
    plt.ylabel('Ka/Ks Ratio (Evolutionary Rate)', fontsize=14, fontweight='bold', color='#2C3E50', labelpad=15)
    plt.title('Evolutionary Selection Pressure (Ka/Ks) of Target Genes', fontsize=22, fontweight='bold', color='#2C3E50', pad=20)

    plt.ylim(0, 1.15)

    plt.legend(fontsize=12, loc='upper right', frameon=True, shadow=True, title="Selection Type")
    plt.tight_layout()
    
    # 6. 保存双格式文件
    output_png = 'KaKs_Evolutionary_Pressure_Marked.png'
    output_pdf = 'KaKs_Evolutionary_Pressure_Marked.pdf'
    plt.savefig(output_png, dpi=300, bbox_inches='tight')
    plt.savefig(output_pdf, dpi=300, bbox_inches='tight')
    
    print(f"🎉 带星号标记的高清图已完美保存为：\n - {output_png}\n - {output_pdf}")