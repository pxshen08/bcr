import os
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 设置Matplotlib后端
import matplotlib

matplotlib.use('Agg')

# 路径设置
path = r"/home/mist/projects/Wang2023/data/BCR_embed/dataa/"
file = "cdr3aa_antiBERTy_umap.pkl"
bcell_type_csv = r"/home/mist/projects/Wang2023/data/Csv/cdr3cn0729.csv"


# def classify_b_cell(label):
#     immature_labels = ['IGHM', 'IGHD']
#     Mature_labels = ['IGHE', 'IGHA','IGHG']
#     if label in Mature_labels:
#         return 'Mature_B_Cell'
#     elif label in immature_labels:
#         return 'Immature_B_Cell'
#     else:
#         return 'None'


# 读取UMAP降维后的数据
with open(os.path.join(path, file), 'rb') as f:
    umap_coordinates = pickle.load(f)

# 读取B细胞类型数据
bcell_types = pd.read_csv(bcell_type_csv)
'''
desired_shape = (858682, 9)
# 使用 .iloc 进行裁剪
umap_coordinates= umap_coordinates.iloc[:desired_shape[0], :]
# 检查数据是否匹配
assert len(umap_coordinates) == len(bcell_types) "UMAP数据与B细胞类型数据长度不匹配"
'''
# 裁剪使两个列表长度相同
if umap_coordinates.shape[0] < bcell_types.shape[0]:
    bcell_types = bcell_types.iloc[:umap_coordinates.shape[0], :]
else:
    umap_coordinates = umap_coordinates.head(bcell_types.shape[0])

# 将坐标和B细胞类型合并
umap_df = pd.DataFrame(umap_coordinates, columns=['UMAP1', 'UMAP2'])
umap_df['UMAP1'] = umap_coordinates[0]
umap_df['UMAP2'] = umap_coordinates[1]

label_total = bcell_types.columns.values.tolist()
label_6 = label_total[1:4]
bcell_types_label_6 = bcell_types[label_6]
hebing = pd.concat([umap_df, bcell_types_label_6], axis=1)
# 将label的数据，非零值都设置为1
# hebing['mu_freq_light'] = hebing['mu_freq_light'].apply(lambda x: 1 if x != 0 else 0)
# hebing['junction_aa_length_light'] = hebing['junction_aa_length_light'].fillna(value=161)
# # 0-19是第一类
# hebing['junction_aa_length_light'] = hebing['junction_aa_length_light'] // 20 + 1

# #umap_df['label'] = bcell_types['label'] # 这里填多的那一列的名字

def myplot(label, n_col):
    # 绘制图像
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x='UMAP1', y='UMAP2', hue=label, palette='bright', data=hebing, legend='full', s=2)
    plt.title('UMAP Clustering of B Cells')
    plt.xlabel('UMAP1')
    plt.ylabel('UMAP2')
    plt.legend(bbox_to_anchor=(1.05, 0), loc=3, title=label, ncol=n_col)

    # 保存图像
    output_file = "{}+{}_2.png".format(file, l)  # 这个名字帮我改成“模型_2.png"
    plt.savefig(os.path.join(path, output_file), dpi=300, bbox_inches='tight')
    plt.show()

# 单个图片测试
# label_6 = "mu_freq_heavy"
for l in label_6:
    # 统计当前label下，有多少种分类，用于计算图例个数
    len_label = len(hebing[l].value_counts().index)
    # 选择一列显示20个图例，计算需要多少列
    n_col = int(np.ceil(len_label / 20))
    myplot(l, n_col)
    print("{}画完了".format(l))


# # 统计当前label下，有多少种分类，用于计算图例个数
# len_label = len(hebing[l].value_counts().index)
# # 选择一列显示20个图例，计算需要多少列
# n_col = int(np.ceil(len_label / 20))
#
# # 绘制图像
# plt.figure(figsize=(10, 8))
# sns.scatterplot(x='UMAP1', y='UMAP2', hue=l, palette='bright', data=hebing, legend='full')
# plt.title('UMAP Clustering of B Cells')
# plt.xlabel('UMAP1')
# plt.ylabel('UMAP2')
# # plt.legend(loc='best', title='B Cell Types')
# plt.legend(bbox_to_anchor=(1.05, 0), loc=3, title=l, ncol=n_col)
# # 保存图像
# output_file = "{}+{}_2.png".format(file, l)   # 这个名字帮我改成“模型_2.png"
# plt.show()
# plt.savefig(os.path.join(path, output_file), dpi=300, bbox_inches='tight')


# bcell_types['B_Cell_Status'] = bcell_types['isotype_heavy'].apply(classify_b_cell)
#
# # 创建UMAP DataFrame
# umap_df = pd.DataFrame(umap_coordinates, columns=['UMAP1', 'UMAP2'])
# umap_df['UMAP1'] = umap_coordinates[0]
# umap_df['UMAP2'] = umap_coordinates[1]
# umap_df['B_Cell_Status'] = bcell_types['B_Cell_Status']
#
# # 绘制UMAP散点图，按B细胞状态着色
# plt.figure(figsize=(10, 8))
# sns.scatterplot(data=umap_df, x='UMAP1', y='UMAP2', hue='B_Cell_Status',
#                 palette={'Immature_B_Cell': 'blue', 'Mature_B_Cell': 'green', 'None': 'red'})
# plt.title('UMAP of B Cells by Maturity Status')
# plt.legend(title='B Cell Status')
# output_file = 'HC_2.png'
# plt.savefig(os.path.join(path, output_file), dpi=300)
# plt.show()
