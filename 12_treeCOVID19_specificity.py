from antiberty import AntiBERTyRunner
from torch.nn.utils.rnn import pad_sequence
import torch
from sklearn.model_selection import StratifiedGroupKFold, GridSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.metrics import matthews_corrcoef, f1_score, r2_score, mean_squared_error, roc_auc_score, balanced_accuracy_score,precision_recall_fscore_support
from collections import Counter
import re
import torch.nn as nn
from sklearn.model_selection import StratifiedGroupKFold
import numpy as np
import pandas as pd
import argparse
from distree import DisTree
import torch.optim as optim
import math
import random
from arch import BCRNet
from distance import euclidean_distance,mahalanobis_distance,covariance
from sklearn.model_selection import GroupKFold


# ARCH_NAMES = arch.__all__
def parse_args():
 parser = argparse.ArgumentParser(description="Gene usage tasks")
 parser.add_argument("--embedding", default="antiBERTy_Bcell_1",type=str, help="Type of embedding (TFIDF, immune2vec, esm1b)")
 parser.add_argument("--model", default="H",type=str, help="Type of model (HL, H)")
 parser.add_argument("--fasta_file",default="/home/mist/projects/Wang2023/data/FASTA1/Bcell_1.fasta",type=str, help="Path to the fasta file")
 parser.add_argument("--random", type=bool, help="Shuffle the data matrix", default=True)
 args = parser.parse_args()
 print(args)
 return args
args=parse_args()#原先没有20240126

BASE_DIR = "../data/BCR_embed/"
##读antibert
def read_fasta(input_fasta):
    sequences_with_id = []
    with open(input_fasta, 'r') as f:
        current_id = None
        current_sequence = ""
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if current_id is not None:
                    sequences_with_id.append((current_id, current_sequence))
                current_id = line[1:]
                current_sequence = ""
            else:
                current_sequence += line
        # Add the last sequence
        if current_id is not None:
            sequences_with_id.append((current_id, current_sequence))
    return sequences_with_id

def load_data(embedding, model):
    if "FULL" in embedding:
        print(f"Loading full length data...")
        anno = "specificity.anno"
        prefix_H = "combined_distinct_heavy"
        prefix_L = "combined_distinct_light"

    else:
        if "ELL" in embedding:
            print(f"Loading ellbedy data...")
            anno = "cdr3_specificity_ellebedy.anno"
            prefix_H = "combined_cdr3_heavy"
            prefix_L = "combined_cdr3_light"
            y = pd.read_table("../data/Annotations/" + anno)
            # prefix_H = "ellebedy_heavy"#不能用， index 124287 is out of bounds for axis 0 with size 1957
            # prefix_L = "ellebedy_light"
        elif "Bcell_1" in embedding:
            print(f"Loading Bcell data...")
            anno = "Bcell_1.csv"
            prefix_H = "Bcell"
            prefix_L = "Bcell"
            y = pd.read_csv("../data/Csv/" + anno)
        else:
           print(f"Loading CDR3 data...")
           anno = "cdr3_specificity.anno"
           prefix_H = "combined_cdr3_heavy"
           prefix_L = "combined_cdr3_light"
           y = pd.read_table("../data/Annotations/" + anno)
    y1=y
    leny1=max_sequence_length = y1.apply(lambda x: len(x)).max()
    if re.match('esm2|antiBERTy', embedding):
        suffix = ""
        da = "datae/"
        if "3B" in embedding:
           suffix = "_3B"
           da="datae/",
        if "antiBERTy" in embedding:
            suffix = "_antiBERTy"
            da="dataa/"
        # esm_Hx=torch.load(BASE_DIR + da + prefix_H + suffix + ".pt",
        #                   map_location=torch.device('cpu')).numpy()
        esm_H = torch.load(BASE_DIR + da + prefix_H + suffix + ".pt",
                          map_location=torch.device('cpu')).numpy()
        esm_L = torch.load(BASE_DIR + da + prefix_L + suffix + ".pt",
                          map_location=torch.device('cpu')).numpy()
        X = esm_H[y.seq_index.astype(int) - 1, :]  # X = esm_H[y.heavy_id.astype(int)-1,:]
        # X = esm_H[y.heavy_id.astype(int)-1,:]#(1985, 512)
        sequences = read_fasta(args.fasta_file)#找embedding输入前input
        matched_sequences = []
        for idx in y.seq_index.astype(int):#y.heavy_id
            for seq_id, seq in sequences:
                if seq_id == 'seq'+str(idx):
                    matched_sequences.append(seq)
                    break
        # print(matched_sequences)
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
        y_H_overlap = np.isin(y.heavy_id, emb_H.index)#检查 y 数据集中的 heavy_id 列是否存在于 emb_H 数据集的索引中。
        # 这将返回一个布尔数组，指示哪些样本在 emb_H 中能够找到匹配。
        sequences = read_fasta(args.fasta_file)
        s_H_overlap= np.isin(y.heavy_id, sequences.index)
        y = y.loc[y_H_overlap,:]#将筛选出在 emb_H 中能够找到匹配的样本，并更新 y 数据集。
        idx_H = y.heavy_id[y_H_overlap]#筛选后的 y 数据集中提取符合条件的 heavy_id 值。
        X = emb_H.loc[idx_H,:]#根据提取的符合条件的 heavy_id 值，从 emb_H 数据集中选择相应的嵌入向量作为特征向量 X
        seq=sequences.loc[idx_H,:]
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
    # y_groups = y.subject.values
    y_groups = y.label.values
    # y = np.isin(y.label.values, ["S+", "S1+", "S2+"])
    label_map = {"plasmacytes_PC": 0, "memory_IgD-": 1, "memory_IgD+": 2,"mature_b_cell":3,"transitional_b_cell":4,"immature_b_cell":5}
    y = np.array([label_map[label] for label in y.label.values])
    assert X.shape[0] == len(y)   
    return X, y, y_groups,y1,matched_sequences

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

# 1. Load embeddings and labels in TASK A
X, y, y_groups,y1,sequences = load_data(args.embedding, args.model)
antiberty = AntiBERTyRunner()
# group the entries with < 100 together
y_group_counts = Counter(y_groups)
small_groups = np.array(list(y_group_counts.keys()))[np.array(list(y_group_counts.values()))<10]
y_groups[np.isin(y_groups, small_groups)] = "small"
print(f"In total, {len(y)} sequences from {len(np.unique(y_groups))} donors/studies.")
print(f"Class size: {Counter(np.sort(y)).most_common()}")
# load tree
filename = "/home/mist/ClonalTree/Examples/output/Bcell_1.abRT.nk.csv"
tree_graph = pd.read_csv(filename, header=None, names=['source', 'target', 'weight'])
tree = tree_graph.values.tolist()
#tree_distance_data=DisTree().sumoneOfDistancesInTree(len(y), tree).cuda()#这里有问题是算一个点到所有点之和，还是一个矩阵

if args.random:
    print(f"Shuffling the embedding...")
    X = X[np.random.permutation(range(X.shape[0])),:][:,np.random.permutation(range(X.shape[1]))]
# 模型参数
embedding_dim = 512
num_classes = 6  # 假设有2个类别
lr = 0.001
alpha = 0.4  # Task A 权重
beta = 0.3# Task B 权重
xigma=0.3# Task A input和output 权重
n_seqs = len(sequences)
embeddings = torch.empty((n_seqs, embedding_dim))
# 训练模型
p_grid = {"C": [1e-2, 1e-1, 10, 100]}
mname=math.floor(1e5 * random.random())
print(mname)
num_epochs =20
batch_size = 32
# 初始化最佳 F1 分数
best_f1_A = 0.0
best_model_state = None
print(embedding_dim,batch_size,lr,alpha,beta,xigma)
ie=1
# 创建模型
model = BCRNet(batch_size, num_classes).cuda()
criterion_A = nn.CrossEntropyLoss()  # Task A 的损失函数
criterion_B = nn.MSELoss()  # Task B 的损失函数
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.001)
# svc = SVC(kernel = "rbf", class_weight = "balanced",  random_state=1, probability = False)#20240323
outer_cv_w_groups = GroupKFold(n_splits=num_epochs)
# outer_cv_w_groups=StratifiedGroupKFold(n_splits=num_epochs, shuffle=True, random_state=0).split(X, y, y_groups)
# train_indices, val_indices = train_test_split(np.arange(num_samples), test_size=0.2, random_state=42, shuffle=False)
# indices = np.random.choice(num_samples, batch_size, replace=False)
for train_index, test_indexs in outer_cv_w_groups:
    f1_scores = []
    mcc_scores = []
    acc_scores = []
    auroc_scores = []
    optimizer.zero_grad()
    # 随机抽取一组 BCR 数据
    integers= train_index
    # print(integers)
    print(f"Train size: {len(train_index)}, test size: {len(test_indexs)}")
    print(f"% positive train: {np.mean(y[train_index])}, % positive test: {np.mean(y[test_indexs])}")
    np.random.shuffle(integers)#打乱顺序
    # 初始化索引
    index = 0
    index1=0
    # 计算需要的循环次数
    num_loops =len(integers) // batch_size
    inu=1
    # 循环读取整数
    for _ in range(num_loops):
        print('-------------------------------train------------------------------')
        indices= []
        test_index=[]
        for _ in range(batch_size):
            if index >=len(integers):
                index = 0  # 如果列表读完，重新从开头读取
            indices.append(integers[index])
            if index1 >=len(test_indexs):
                index1 = 0  # 如果列表读完，重新从开头读取
            test_index.append(test_indexs[index1])
            index += 1
            index1 += 1
        # print("Read integers:", indices,"Read test:", test_index)
        sequence = [sequences[idx2] for idx2 in indices]#得到batch对应sequenceinput
        max_length = max(len(string) for string in sequence)
        # 将每个字符串转换为其 ASCII 值，并进行填充并归一化处理
        padded_ascii_list = [torch.tensor([ord(char) / 255.0 for char in string] + [0.0] * (max_length - len(string)),
                                          dtype=torch.float) for string in sequence]
        # 构建张量
        b1 = pad_sequence(padded_ascii_list, batch_first=True)
        b2 = nn.Linear(b1.size(1), 1)(b1).cuda()
        xa1 = antiberty.embed(sequence)
        xa2 = [a.mean(axis=0) for a in xa1]
        x_batch = torch.stack(xa2).cuda()
        xa4 = nn.Linear(512, 1).cuda()(x_batch)#这里重新训练antibert,用不到
        # embeddings[indices] = xa3
        # x_batch = torch.Tensor(X[indices]).cuda()
        # x_batch= torch.transpose(x_batch, 0, 1)#512,4
        label_batch_A = torch.LongTensor(y[indices]).cuda()
        y2 = y1.seq_index[indices].values  # 现在是heavyid 要改的20240403y1.heavy_id
        # 计算两两之间的距离
        tree_distance_batch = np.zeros((batch_size, batch_size))
        for it in range(batch_size):
            for jt in range(it + 1, batch_size):
                # 这里可以用任何您想要的距离度量方式，这里简单地使用绝对值
                # print(y2[it],y2[jt])
                distance = DisTree().sumOfDistancesInTree(tree, 'seq' + str(y2[it]), 'seq' + str(y2[jt]))
                # print(distance)
                tree_distance_batch[it][jt] = distance
                tree_distance_batch[jt][it] = distance
        tree_distance_batch = torch.Tensor(tree_distance_batch).cuda()
        # 前向传播
        output_A, output_B = model(x_batch)  # 'tuple' object has no attribute 'cuda'20240403
        # 计算 Task A 的损失
        CUDA_LAUNCH_BLOCKING = 0
        loss_A = criterion_A(output_A,label_batch_A).cuda()  # tensor(0.8620, device='cuda:0', grad_fn=<NllLossBackward0>)
        print("lossa",loss_A)
        xT = x_batch.t()
        D = torch.cov(xT)
        invD = torch.inverse(D)
        # 计算 Euclidean distance这个没用到
        euclidean_dist_batch = np.zeros((batch_size, batch_size))
        for id in range(batch_size):
            for jd in range(id + 1, batch_size):
                invD = torch.inverse(D)
                tp = x_batch[id]- x_batch[jd]
                # distance = torch.sqrt(torch.dot(tp, torch.mv(invD, tp)))
                distance = euclidean_distance(x_batch[id], x_batch[jd])
                # print(distance)
                euclidean_dist_batch[id][jd] = distance
                euclidean_dist_batch[jd][id] = distance
        euclidean_dist_batch = torch.Tensor(euclidean_dist_batch).cuda()
        print(euclidean_dist_batch)
        # euclidean_dist_batch = euclidean_distance(x_batch)
        # 计算 Task B 的损失
        tree1 = nn.Linear(batch_size, 1).to("cuda")(tree_distance_batch).squeeze()
        # print(tree1)  # 这里是有问题的
        # loss_B2 =  torch.mean((euclidean_dist_batch - tree_distance_batch) ** 2)
        # print("lossb2",loss_B2)
        loss_B = criterion_B(output_B.squeeze(), tree1)/batch_size
        print("lossb",loss_B)
        loss_B1 = criterion_B(euclidean_dist_batch, tree_distance_batch)/batch_size
        print("lossb1",loss_B1)
        loss_A1 = criterion_B(b2, xa4).cuda()#input 和output
        print("lossa1", loss_A1)
        # 总损失
        loss_total = alpha * loss_A + beta * loss_B1+xigma*loss_A1
        # 反向传播及优化
        loss_total.backward()
        optimizer.step()
        print("losstotal",loss_total)
        predictiont = np.argmax(output_A.cpu().detach(), axis=1)
        f1t = f1_score(y[indices].astype(int), predictiont, average="weighted")
        mcct = matthews_corrcoef(y[indices].astype(int), predictiont)
        acct = balanced_accuracy_score(y[indices].astype(int), predictiont)
        print(precision_recall_fscore_support(y[indices].astype(int), predictiont))
        print(f"[F1 score (Outer fold {ie})]: {f1t}")
        print(f"[MCC score (Outer fold {ie})]: {mcct}")
        print(f"[ACC score (Outer fold {ie})]: {acct}")
        ##########val#################
        print("-------------val--------------")
        model.eval()
        x_testbatch = torch.Tensor(X[test_index]).cuda()
        with torch.no_grad():
            output_A1, output_B1 = model(x_testbatch)
            # 将输出转换为类别
            # prediction = (output_A > 0.45).int()
            # prediction=prediction.cpu().numpy()[:, 1]
            prediction=np.argmax(output_A1.cpu(), axis=1)
        # print("real",y[test_index].astype(int),"pre",prediction)
        f1 = f1_score(y[test_index].astype(int), prediction, average="weighted")
        mcc = matthews_corrcoef(y[test_index].astype(int), prediction)
        acc = balanced_accuracy_score(y[test_index].astype(int), prediction)
        auroc = acc
        # roc_auc_score(y[test_index].astype(int), prediction)
        print(f"Epoch [{inu}/{num_loops}],[F1 score (Outer fold {ie})]: {f1}")
        print(f"[MCC score (Outer fold {ie})]: {mcc}")
        print(f"[ACC score (Outer fold {ie})]: {acc}")
        print(f"[AUROC score (Outer fold {ie})]: {auroc}")
        inu +=1
        f1_scores.append(f1)
        mcc_scores.append(mcc)
        acc_scores.append(acc)
        auroc_scores.append(auroc)
        # 打印损失
        # print(f'Epoch [{ie}/{num_epochs}], Task A Loss: {loss_A.item():.4f}, Task B Loss: {loss_B.item():.4f}')
    f1_scores1 = np.array(f1_scores)
    mcc_scores1 = np.array(mcc_scores)
    acc_scores1 = np.array(acc_scores)
    auroc_scores1 = np.array(auroc_scores)
    print('--------------------------------------------------------------------------')
    print(f'Epoch [{ie}/{num_epochs}], Task A f1 M: {f1_scores1.mean():.4f},Task A f1 S: {f1_scores1.std():.4f}'
          f'Task A mcc M: {mcc_scores1.mean():.4f},Task A mcc S: {mcc_scores1.std():.4f}'
          f'Task A acc M: {acc_scores1.mean():.4f},Task A acc S: {acc_scores1.std():.4f}'
          f'Task A auroc M: {auroc_scores1.mean():.4f},Task A auroc S: {auroc_scores1.std():.4f}')
    if f1_scores1.mean() > best_f1_A:
        best_f1_A = max(f1_scores1.mean(), best_f1_A)
        best_model_state = model.state_dict()
        torch.save(best_model_state, str(mname) + 'best_model.pth')
        print('save' + str(mname) + 'best_model.pth')
    else:
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
    ie += 1
# # 模型保存
# torch.save(model.state_dict(), 'bcr_model_taskAB.pth')
