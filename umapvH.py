import os
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 设置Matplotlib后端
import matplotlib

matplotlib.use('Agg')

# 路径设置
path = r"/home/mist/migrate/Csv/"
file = "combined_distinct_light_antiBERTy_umap.pkl"
bcell_type_csv = r"/home/mist/migrate/Csv/specificityg.csv"


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
bcell_types = pd.read_csv(os.path.join(path, bcell_type_csv))
# Drop the first row
umap_coordinates = umap_coordinates.drop(0)

# Reset the index if needed
umap_coordinates = umap_coordinates.reset_index(drop=True)
# desired_shape = (858682, 9)
# 使用 .iloc 进行裁剪
# umap_coordinates= umap_coordinates.iloc[:desired_shape[0], :]
# 检查数据是否匹配
print(len(umap_coordinates),len(bcell_types))
if umap_coordinates.shape[0] < bcell_types.shape[0]:
    # bcell_types = bcell_types[:umap_coordinates.shape[0], :]
    bcell_types = bcell_types.head(umap_coordinates.shape[0])
else:
    umap_coordinates = umap_coordinates.head(bcell_types.shape[0])
assert len(umap_coordinates) == len(bcell_types), "UMAP数据与B细胞类型数据长度不匹配"

# 将坐标和B细胞类型合并
umap_df = pd.DataFrame(umap_coordinates, columns=['UMAP1', 'UMAP2'])
umap_df['UMAP1'] = umap_coordinates[0]
umap_df['UMAP2'] = umap_coordinates[1]
umap_df['label'] = bcell_types['label']#这里填多的那一列的名字

# 绘制图像
plt.figure(figsize=(10, 8))
# sns.scatterplot(x='UMAP1', y='UMAP2', hue='label', palette='viridis', data=umap_df, legend='full')
sns.scatterplot(x='UMAP1', y='UMAP2', hue='label', palette='bright', data=umap_df, legend='full', s=2)
plt.title('UMAP Clustering of B Cells')
plt.xlabel('UMAP1')
plt.ylabel('UMAP2')
plt.legend(loc='best', title='V_gene Types')

# 保存图像
output_file = 'umapldistinct_clustering-s.png'#这个名字帮我改成“模型_2.png"
plt.savefig(os.path.join(path, output_file), dpi=300)
plt.show()

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
