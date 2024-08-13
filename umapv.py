import os
import pickle
# import cv2
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib
from matplotlib.colors import ListedColormap
#
# matplotlib.use('TkAgg')

#
# matplotlib.use('TkAgg')
path = "/home/mist/projects/Wang2023/data/BCR_embed/datai/"

file_list = {}
file= 'Bcell_ablang_umap.pkl'
f = open(os.path.join(path, file), 'rb')
f = pickle.load(f)
file_list[file.split('-')[-1].split('.')[0]] = f

# 调用 KMeans 算法进行聚类
kmeans = KMeans(n_clusters=6)  # 设置聚类簇数
labels = kmeans.fit_predict(f)

# 获得聚类中心
cluster_centers = kmeans.cluster_centers_

# 使用不同的颜色映射为不同的聚类结果
cmap = plt.cm.get_cmap('viridis', len(set(labels)))  # 使用 'viridis' 颜色映射，设置颜色映射数量为聚类结果数量

# 绘制散点图，根据聚类标签设置颜色
plt.scatter(f.iloc[:, 0], f.iloc[:, 1], c=labels, cmap=cmap)

# 绘制聚类中心点
plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], c='black', marker='X', s=150)

# 添加标题和轴标签
plt.title("UMAP Visualization with Clustering")
plt.xlabel("UMAP Dimension 1")
plt.ylabel("UMAP Dimension 2")

# 显示图像
plt.show()

print(file_list)

# #20240201代码可以优化
# df = pd.DataFrame(file_list[file.split('.')[0]]).T.astype(float)
# df1=df.T
# # 创建散点图
# plt.scatter(df1[0], df1[1], alpha=0.5)
# plt.title('Scatter Plot of '+file.split('.')[0])
# plt.xlabel('X-axis')
# plt.ylabel('Y-axis')
plt.savefig(path + file.split('.')[0]+'1.png')
plt.show()
