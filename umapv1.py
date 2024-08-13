import os
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 设置Matplotlib后端
import matplotlib
matplotlib.use('Agg')

# 路径设置
path = '/home/mist/migrate/BCR_embed/dataa/'
file = 'cdr3seq_antiBERTy_umap.pkl'
bcell_type_csv = '/home/mist/projects/Wang2023/data/Csv/cdr3cn0729.csv'

def classify_b_cell(label):
    immature_labels = ['immature_b_cell', 'transitional_b_cell']
    if label in immature_labels:
        return 'Immature_B_Cell'
    else:
        return 'Mature_B_Cell'

# 读取UMAP降维后的数据
with open(os.path.join(path, file), 'rb') as f:
    umap_coordinates = pickle.load(f)

# 读取B细胞类型数据
bcell_types = pd.read_csv(os.path.join(path, bcell_type_csv))

# 检查数据是否匹配
assert len(umap_coordinates) == len(bcell_types), "UMAP数据与B细胞类型数据长度不匹配"

# 将坐标和B细胞类型合并
# umap_df = pd.DataFrame(umap_coordinates, columns=['UMAP1', 'UMAP2'])
# umap_df['UMAP1'] = umap_coordinates[0]
# umap_df['UMAP2'] = umap_coordinates[1]
# umap_df['label'] = bcell_types['label']

# 绘制图像
# plt.figure(figsize=(10, 8))
# sns.scatterplot(x='UMAP1', y='UMAP2', hue='label', palette='viridis', data=umap_df, legend='full')
# plt.title('UMAP Clustering of B Cells')
# plt.xlabel('UMAP1')
# plt.ylabel('UMAP2')
# plt.legend(loc='best', title='B Cell Types')
#
# # 保存图像
# output_file = 'Bcell_umap_clustering.png'
# plt.savefig(os.path.join(path, output_file), dpi=300)
# plt.show()

# bcell_types['B_Cell_Status'] = bcell_types['label'].apply(classify_b_cell)
bcell_types['B_Cell_Status'] = bcell_types['label']

# 创建UMAP DataFrame
umap_df = pd.DataFrame(umap_coordinates, columns=['UMAP1', 'UMAP2'])
umap_df['UMAP1'] = umap_coordinates[0]
umap_df['UMAP2'] = umap_coordinates[1]
umap_df['B_Cell_Status'] = bcell_types['B_Cell_Status']

# 绘制UMAP散点图，按B细胞状态着色
plt.figure(figsize=(10, 8))
# sns.scatterplot(data=umap_df, x='UMAP1', y='UMAP2', hue='B_Cell_Status', palette={'immature_b_cell': 'blue', 'mature_b_cell': 'green',
#                                                                                   'transitional_b_cell': 'pink', 'memory_IgD+': 'yellow',
#                                                                                   'memory_IgD-': 'purple', 'plasmacytes_PC': 'red'})
sns.scatterplot(data=umap_df, x='UMAP1', y='UMAP2', hue='B_Cell_Status', palette={'ASC': 'blue', 'Memory': 'green',
                                                                                  'Naive': 'orange'})
plt.title('UMAP of B Cells by Maturity Status')
plt.legend(title='B Cell Status')
output_file = 'Bcell_3seq.png'
plt.savefig(os.path.join(path, output_file), dpi=300)
plt.show()