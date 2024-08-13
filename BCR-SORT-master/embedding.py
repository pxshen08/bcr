import umap
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# 假设我们有一个数据加载器 data_loader 和已经训练好的模型 model
model = load_model().to(device)
model.eval()

all_encoder_outputs = []
all_labels = []

with torch.no_grad():
    for inputs, lengths, labels in data_loader:
        sequence, vgene, jgene, isotype = inputs
        sequence = sequence.type(torch.cuda.LongTensor).to(device)
        vgene = vgene.type(torch.cuda.LongTensor).to(device)
        jgene = jgene.type(torch.cuda.LongTensor).to(device)
        isotype = isotype.type(torch.cuda.LongTensor).to(device)

        encoder_output = model(sequence, vgene, jgene, isotype, lengths, return_encoder_output=True)
        all_encoder_outputs.append(encoder_output.cpu().numpy())
        all_labels.append(labels.cpu().numpy())

all_encoder_outputs = np.concatenate(all_encoder_outputs, axis=0)
all_labels = np.concatenate(all_labels, axis=0)

# 使用UMAP进行降维
reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='correlation')
embedding = reducer.fit_transform(all_encoder_outputs)

# 可视化
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 8))
plt.scatter(embedding[:, 0], embedding[:, 1], c=np.argmax(all_labels, axis=1), cmap='Spectral', s=1)
plt.colorbar()
plt.title('UMAP projection of BCRSORT encoder output')
plt.xlabel('UMAP1')
plt.ylabel('UMAP2')
plt.show()
