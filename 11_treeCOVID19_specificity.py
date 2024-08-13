import pickle
import torch

from sklearn.model_selection import StratifiedGroupKFold, GridSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.metrics import matthews_corrcoef, f1_score, r2_score, mean_squared_error, roc_auc_score, balanced_accuracy_score
from sklearn.svm import SVC
from collections import Counter
from sklearn.decomposition import PCA
import re
import time
import torch.nn as nn
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold
import numpy as np
import pandas as pd
import argparse
from distree import DisTree
import torch.optim as optim

def parse_args():
 parser = argparse.ArgumentParser(description="Gene usage tasks")
 parser.add_argument("--embedding", default="antiBERTy_CDR3",type=str, help="Type of embedding (TFIDF, immune2vec, esm1b)")
 parser.add_argument("--model", default="H",type=str, help="Type of model (HL, H)")
 parser.add_argument("--random", type=bool, help="Shuffle the data matrix", default=True)
 args = parser.parse_args()
 print(args)
 return args
args=parse_args()#原先没有20240126

BASE_DIR = "../data/BCR_embed/"

def load_data(embedding, model = "HL"):
    if "FULL" in embedding:
        print(f"Loading full length data...")
        anno = "specificity.anno"
        prefix_H = "combined_distinct_heavy"
        prefix_L = "combined_distinct_light"
    else:
        print(f"Loading CDR3 data...")
        anno = "cdr3_specificity.anno"
        prefix_H = "combined_cdr3_heavy"
        prefix_L = "combined_cdr3_light"
    y = pd.read_table("../data/Annotations/" + anno)
    if re.match('esm2|antiBERTy', embedding):
        suffix = ""
        da = "datae/"
        if "3B" in embedding:
           suffix = "_3B"
           da="datae/",
        if "antiBERTy" in embedding:
            suffix = "_antiBERTy"
            da="dataa/"
        esm_H = torch.load(BASE_DIR + da + prefix_H + suffix + ".pt",
                          map_location=torch.device('cpu')).numpy()
        esm_L = torch.load(BASE_DIR + da + prefix_L + suffix + ".pt",
                          map_location=torch.device('cpu')).numpy()
        X = esm_H[y.heavy_id.astype(int)-1,:]    
        if model == "HL":
            remove = np.isnan(y.light_id)
            y = y.loc[~remove,:]
            X = X[~remove,:]
            X_L = esm_L[y.light_id-1,:]
            X = np.hstack([X, X_L])          
    else:
        if re.match("immune2vec", embedding):
            emb_H = pd.read_pickle(BASE_DIR + prefix_H + "_" + embedding + ".pkl")
            emb_L = pd.read_pickle(BASE_DIR + prefix_L + "_" + re.sub("H", "L", embedding) + ".pkl")
        else:
            embedding_name = re.match("frequency|ProtT5|physicochemical", embedding).group()
            if "frequency" in embedding:
                da = "dataf/"
            if "ProtT5" in embedding:
                da = "datap/"
            if "physicochemical" in embedding:
                da = "datapp/"
            emb_H = pd.read_pickle(BASE_DIR + da+prefix_H + "_" + embedding_name + ".pkl")
            emb_L = pd.read_pickle(BASE_DIR + da+prefix_L + "_" + embedding_name + ".pkl")
        emb_H.index = pd.Series([int(x) for x in emb_H.index.values])
        emb_L.index = pd.Series([int(x) for x in emb_L.index.values])
        # intersect the dataset in case of some failures in embedding
        y_H_overlap = np.isin(y.heavy_id, emb_H.index)
        y = y.loc[y_H_overlap,:]
        idx_H = y.heavy_id[y_H_overlap]
        X = emb_H.loc[idx_H,:]
        if model == "H":
            X = X.values       
        elif model == "HL":
            remove = np.isnan(y.light_id)
            y = y.loc[~remove,:] # ordering kept in y_train
            y_L_overlap = np.isin(y.light_id, emb_L.index)
            idx_L = y.light_id[y_L_overlap]
            y = y.loc[y_L_overlap,:]
            X_L = emb_L.loc[idx_L,:]
            X = emb_H.loc[y.heavy_id,:]
            X = np.hstack([X, X_L])
    y_groups = y.subject.values
    y = np.isin(y.label.values, ["S+", "S1+", "S2+"])    
    assert X.shape[0] == len(y)   
    return X, y, y_groups

# 定义神经网络模型类
class BCRNet(nn.Module):
    def __init__(self, embedding_dim, num_classes):
        super(BCRNet, self).__init__()
        # Task A 输出层
        self.fc_A = nn.Linear(embedding_dim, num_classes)
        # Task B 输出层
        self.fc_B = nn.Linear(embedding_dim, 1)

    def forward(self, x):
        # Task A 输出
        output_A = nn.functional.softmax(self.fc_A(x), dim=1)
        # Task B 输出
        output_B = self.fc_B(x)
        return output_A, output_B

# 计算欧几里得距离函数
def euclidean_distance(x1, x2):
    return torch.norm(x1 - x2, p=2, dim=1)

def compute_loss_taskB(batch_embeddings, batch_labels, tree):
    """
    计算 Task B 的损失函数

    参数：
    - batch_embeddings: 当前批次的 BCR embeddings
    - batch_labels: 当前批次的标签
    - tree: 树的结构和权重信息，字典形式，键为节点，值为该节点的父节点和权重的元组

    返回值：
    - loss: Task B 的损失值
    """

    # 获取当前批次中的 BCR 嵌入和标签
    embeddings_BCR1 = batch_embeddings[:, :embedding_dim // 2]
    embeddings_BCR2 = batch_embeddings[:, embedding_dim // 2:]
    labels_BCR = batch_labels

    # 计算每对 BCR 的树上距离
    distances = []
    for i in range(len(embeddings_BCR1)):
        distance = DisTree(embeddings_BCR1[i], embeddings_BCR2[i], tree)
        distances.append(distance)
    distances = torch.tensor(distances)

    # 计算损失，可以使用均方差损失函数
    loss = nn.functional.mse_loss(distances, labels_BCR)
    return loss

# 模型参数
embedding_dim = 100
num_classes = 3  # 假设有3个类别
lr = 0.001
alpha = 0.5  # Task A 权重
beta = 0.5   # Task B 权重
# 1. Load embeddings and labels
X, y, y_groups = load_data(args.embedding, args.model)
# group the entries with < 100 together
y_group_counts = Counter(y_groups)
small_groups = np.array(list(y_group_counts.keys()))[np.array(list(y_group_counts.values())) < 100]
y_groups[np.isin(y_groups, small_groups)] = "small"
print(f"Class size: {Counter(np.sort(y)).most_common()}")
    
print(f"In total, {len(y)} sequences from {len(np.unique(y_groups))} donors/studies.")
print(f"Class size: {Counter(np.sort(y)).most_common()}")

if args.random:
    print(f"Shuffling the embedding...")
    X = X[np.random.permutation(range(X.shape[0])),:][:,np.random.permutation(range(X.shape[1]))]

# 3. Nested cross validation (combine all data)
n_splits_outer = 4
n_splits_inner = 3
outer_cv = StratifiedGroupKFold(n_splits=n_splits_outer, shuffle=True, random_state=0)
inner_cv = StratifiedGroupKFold(n_splits=n_splits_inner, shuffle=True, random_state=1)

p_grid = {"C": [1e-2, 1e-1, 10, 100]}

outer_cv_w_groups = outer_cv.split(X, y, y_groups)
f1_scores = []
mcc_scores = []
acc_scores = []
auroc_scores = []
i = 1
svc = SVC(kernel = "rbf", class_weight = "balanced",  random_state=1, probability = False)#20240323

for train_index, test_index in outer_cv_w_groups:
    print(f"##### Outer fold {i} #####")
    # 获取当前外部交叉验证折的训练和测试数据
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    print(f"Train size: {len(train_index)}, test size: {len(test_index)}")
    print(f"% positive train: {np.mean(y_train)}, % positive test: {np.mean(y_test)}")
    # 内部交叉验证
    cur_time = time.time()
    search = GridSearchCV(estimator=svc,
                          param_grid=p_grid,
                          cv=inner_cv,
                          scoring="f1_weighted",
                          n_jobs=-1,
                          pre_dispatch="1*n_jobs")
    search.fit(X_train, y_train, groups=y_groups[train_index])
    print(f"[Time (Outer fold {i})]: {time.time() - cur_time}")
    prediction = search.predict(X_test)
    f1 = f1_score(y_test, prediction, average="weighted")
    mcc = matthews_corrcoef(y_test, prediction)
    acc = balanced_accuracy_score(y_test, prediction)
    auroc = roc_auc_score(y_test, prediction)

    # 计算 Task B 的损失
    embeddings_BCR1 = X_test[:, :embedding_dim // 2]
    embeddings_BCR2 = X_test[:, embedding_dim // 2:]
    labels_BCR = compute_labels_BCR(embeddings_BCR1, embeddings_BCR2, tree)
    loss_taskB = compute_loss_taskB(embeddings_BCR1, embeddings_BCR2, labels_BCR)

    # 创建模型
    model = BCRNet(embedding_dim, num_classes)
    criterion_A = nn.CrossEntropyLoss()  # Task A 的损失函数
    criterion_B = nn.MSELoss()  # Task B 的损失函数
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 模型训练
    num_epochs = 10
    batch_size = 32
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        # 随机抽取一组 BCR 数据
        indices = np.random.choice(len(X_train), batch_size, replace=False)
        x_batch = torch.Tensor(X_train[indices])
        label_batch_A = torch.LongTensor(y_train[indices])
        tree_distance_batch = torch.Tensor(tree_distance_data[indices])
        # 前向传播
        output_A, output_B = model(x_batch)
        # 计算 Task A 的损失
        loss_A = criterion_A(output_A, label_batch_A)
        # 计算 Euclidean distance
        euclidean_dist_batch = euclidean_distance(x_batch[0], x_batch[1])
        # 计算 Task B 的损失
        loss_B = criterion_B(output_B.squeeze(), tree_distance_batch)
        # 总损失
        loss_total = alpha * loss_A + beta * loss_B
        # 反向传播及优化
        loss_total.backward()
        optimizer.step()

        # 打印损失
        print(f'Epoch [{epoch + 1}/{num_epochs}], Task A Loss: {loss_A.item():.4f}, Task B Loss: {loss_B.item():.4f}')

    # 在这里计算 Task A 和 Task B 的评价指标
    print(f"[Time (Outer fold {i})]: {time.time() - cur_time} seconds")
    prediction = search.predict(X_test)
    f1 = f1_score(y_test, prediction, average="weighted")
    mcc = matthews_corrcoef(y_test, prediction)
    acc = balanced_accuracy_score(y_test, prediction)
    auroc = roc_auc_score(y_test, prediction)

    print(f"[F1 score (Outer fold {i})]: {f1}")
    print(f"[MCC score (Outer fold {i})]: {mcc}")
    print(f"[ACC score (Outer fold {i})]: {acc}")
    print(f"[AUROC score (Outer fold {i})]: {auroc}")
    f1_scores.append(f1)
    mcc_scores.append(mcc)
    acc_scores.append(acc)
    auroc_scores.append(auroc)
    i += 1

# 将结果保存到文件
out_score = pd.DataFrame({"Fold": range(n_splits_outer),
                          "F1": f1_scores,
                          "MCC": mcc_scores,
                          "ACC": acc_scores,
                          "AUROC": auroc_scores})

if args.random:
    is_random = "_random"
else:
    is_random = ""

filename = "../data/results/specificity_" + re.sub("_H", "", args.embedding) + "_" + args.model + is_random + ".csv"
out_score.to_csv(filename)
print("Results saved at: " + filename)

f1_scores = np.array(f1_scores)
mcc_scores = np.array(mcc_scores)
acc_scores = np.array(acc_scores)
auroc_scores = np.array(auroc_scores)

print(f"[Mean F1]: {f1_scores.mean()}")
print(f"[SD F1]: {f1_scores.std()}")
print(f"[Mean MCC]: {mcc_scores.mean()}")
print(f"[SD ACC]: {mcc_scores.std()}")
print(f"[Mean ACC]: {acc_scores.mean()}")
print(f"[SD ACC]: {acc_scores.std()}")
print(f"[Mean AUROC]: {auroc_scores.mean()}")
print(f"[SD AUROC]: {auroc_scores.std()}")

