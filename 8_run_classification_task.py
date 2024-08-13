import pickle
import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import matthews_corrcoef, f1_score
from collections import Counter
import argparse
import time
import os

def parse_args():#原先是没有def的，直接由bash调用20240126
 parser = argparse.ArgumentParser(description="Gene usage tasks")
 parser.add_argument('--gene', default='VH',type=str, help="Gene type (VH, VL, JH, JL, isoH, isoL)")
 parser.add_argument('--embedding',default='physicochemical', type=str, help="Type of embedding (immune2vec, esm2, ProtT5)")
 parser.add_argument('--random', type=bool, help="Shuffle the data matrix", default=True)
 args = parser.parse_args()
 print(args)
 return args

BASE_DIR = "/home/mist/projects/Wang2023/data/Annotations/"
args=parse_args()#原先没有20240126

COL_MAP = {"VH": "v_call_family_heavy",
           "VL": "v_call_family_light",
           "JH": "j_call_family_heavy",
           "JL": "j_call_family_light",
           "isoH": "isotype_heavy",
           "isoL": "isotype_light"}
#定义基因类型列的映射，并设置最小/最大类大小。
MIN_CLASS_SIZE = 100
MAX_CLASS_SIZE = 5000

def load_data(gene, embedding):
    if "H" in gene:
        data_prefix = "combined_distinct_heavy"
    elif "L" in gene:
        data_prefix = "combined_distinct_light"
    y = pd.read_table(BASE_DIR + data_prefix + ".anno")
    
    # remove sequences with empty labels or labeled as Bulk
    gene_col = y.loc[:, COL_MAP[gene]]
    y = y.loc[~(np.isin(gene_col, ["Bulk"]) | gene_col.isna()),:]
    y = y.set_index('id')
    
    if embedding == 'esm2' or embedding == 'esm2_3B':
        suffix = embedding.replace('esm2', '')
        X = torch.load("/home/mist/projects/Wang2023/data/BCR_embed/datae/" + data_prefix + suffix + ".pt", map_location=torch.device('cpu')).numpy()
        X = X[y.index-1,:]
        y_groups = y.subject.values
        y = y.loc[:, COL_MAP[gene]].values

    elif embedding == 'antiBERTy' or embedding =='ablang':
        X = torch.load("/home/mist/projects/Wang2023/data/BCR_embed/datai/" + data_prefix + "_ablang.pt", map_location=torch.device('cpu')).numpy()
        print(X)
        y20240126=(y.index).to_numpy(dtype=int, na_value=-1)
        X = X[y20240126-1,:]
        y_groups = y.subject.values
        y = y.loc[:, COL_MAP[gene]].values
    
    elif ("immune2vec" in embedding) or (np.isin(embedding, ["physicochemical", "frequency", "ProtT5"])):
        X = pd.read_pickle("/home/mist/projects/Wang2023/data/BCR_embed/datab/" + data_prefix + "_" + embedding + ".pkl")
        X.index = pd.Series([int(x) for x in X.index.values])
        
        # intersect the dataset (in case of failures in embedding)
        X_idx = np.array(set(X.index) & set(y.index))
        X = X.loc[X_idx,:].values
        y_groups = y.loc[X_idx, "subject"].values
        y = y.loc[X_idx, COL_MAP[gene]].values 
    
    assert X.shape[0] == len(y)
    
    return X, y, y_groups

# 1. Load embeddings and labels
# X：特征矩阵，包含每个序列的嵌入。
# y：目标变量，表示基因类型（例如，VH、VL、JH、JL、isoH、isoL）。
# y_groups：其他分组信息，可能指示与序列相关的主题或研究。
X, y, y_groups = load_data(args.gene, args.embedding)

# 2. Remove class with too few examples
class_sizes = Counter(y)
target_class = np.array(list(class_sizes.keys()))[np.array(list(class_sizes.values())) > MIN_CLASS_SIZE]
include = np.isin(y, target_class)
X, y, y_groups = X[include,:], y[include], y_groups[include]

# 3. Downsample class too large in sizes 
def subsample(index, class_size):
    classes = np.unique(index)
    subsampled = []
    np.random.seed(0)
    for i in classes:
        is_class = np.where(index == i)[0]
        if len(is_class) <= class_size:
            subsampled.append(is_class)
        else:
            subsampled.append(np.random.choice(is_class, size = class_size, replace = False))
    subsampled = np.hstack(subsampled)
    return subsampled

downsample = True
if downsample:
    include = subsample(y, MAX_CLASS_SIZE)
    X, y, y_groups = X[include,:], y[include], y_groups[include]
    print(f"Downsampling classes to at most {MAX_CLASS_SIZE} sequences for {len(target_class)} classes with size > {MIN_CLASS_SIZE}.")
print(f"In total, {len(y)} sequences from {len(np.unique(y_groups))} donors/studies.")
print(f"Class size: {Counter(np.sort(y)).most_common()}")

#20240126转gpu
if torch.cuda.is_available():
    # 获取 GPU 设备的数量
    device_count = torch.cuda.device_count()
    print(f"Found {device_count} GPU(s) available.")
else:
    print("No GPU available, using CPU.")

# 4. Random baseline
if args.random:
    print(f"Shuffling the embedding...")
    X = X[np.random.permutation(range(X.shape[0])),:][:,np.random.permutation(range(X.shape[1]))]

# 5. Nested cross validation 使用 设置嵌套交叉验证。StratifiedGroupKFold
# 定义 SVM 的超参数网格。
# 执行嵌套交叉验证、超参数搜索并评估模型。
n_splits_outer = 5
n_splits_inner = 3
outer_cv = StratifiedGroupKFold(n_splits=n_splits_outer, shuffle=True, random_state=0)
inner_cv = StratifiedGroupKFold(n_splits=n_splits_inner, shuffle=True, random_state=1)

p_grid = {"C": [1e-2, 1e-1, 10, 100]}

outer_cv_w_groups = outer_cv.split(X, y, y_groups)
f1_scores = []
mcc_scores = []
acc_scores = []
i = 1
svc = SVC(kernel = "rbf", class_weight = "balanced", 
          random_state = 1, probability = False)
for train_index, test_index in outer_cv_w_groups:
    print(f"##### Outer fold {i} #####")
    # get the cross validation score on the test
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    print(f"Train size: {len(train_index)}, test size: {len(test_index)}")
    # inner loop
    # hyperparameter search
    cur_time = time.time()
    search = GridSearchCV(estimator = svc, 
               param_grid = p_grid, 
               cv = inner_cv, scoring = "f1_weighted", 
               n_jobs = -1, 
               pre_dispatch = "1*n_jobs")
    search.fit(X_train, y_train, groups = y_groups[train_index])
    print(f"[Time (Outer fold {i})]: {time.time() - cur_time} seconds")
    prediction = search.predict(X_test)
    f1 = f1_score(y_test, prediction, average="weighted")
    print(f"[F1 (Outer fold {i})]: {f1}")
    mcc = matthews_corrcoef(y_test, prediction)
    print(f"[MCC (Outer fold {i})]: {mcc}")
    acc = np.mean(y_test.reshape(-1) == prediction.reshape(-1))
    print(f"[ACC (Outer fold {i})]: {acc}")
    print()
    f1_scores.append(f1)
    mcc_scores.append(mcc)
    acc_scores.append(acc)
    i += 1

# Save results as a table
out_score = pd.DataFrame({"Fold": range(n_splits_outer), 
                          "F1": f1_scores,
                          "MCC": mcc_scores,
                          "ACC": acc_scores})

if args.random:
    is_random = "_random"
else:
    is_random = ""
filename = "/home/mist/projects/Wang2023/data/BCR_embed/datai/" + args.embedding + "_" + args.gene + is_random + ".csv"#这里文件地方应该要改一下20240126
out_score.to_csv(filename)
print("Results saved at: " + filename)

f1_scores = np.array(f1_scores)
mcc_scores = np.array(mcc_scores)
acc_scores = np.array(acc_scores)

print(f"[Mean F1]: {f1_scores.mean()}")
print(f"[SD F1]: {f1_scores.std()}")
print(f"[Mean MCC]: {mcc_scores.mean()}")
print(f"[SD MCC]: {mcc_scores.std()}")
print(f"[Mean ACC]: {acc_scores.mean()}")
print(f"[SD ACC]: {acc_scores.std()}")


# 3. Split one dataset as the training and rest as validation
# gss = GroupShuffleSplit(n_splits=1, train_size=.9, random_state=1)
# splits = gss.split(X = X_train, groups = y_groups)
# train_idx, val_idx = next(splits)
# X_train, y_train, X_val, y_val = X_train[train_idx,:], y_train[train_idx], X_train[val_idx,:], y_train[val_idx]


# # 5. Hyperparameter search 
# # tuning for the regularization parameter 
# def lasso_objective(trial):  
#     global clf
#     c = trial.suggest_loguniform('alpha', 1e-1, 1e3)

#     clf = SVC(C = c, kernel = "rbf", class_weight = "balanced", 
#                   random_state = 1, probability = False)
#     clf.fit(X_train,y_train)
    
#     print("Regularization: {}".format(c))
    
#     y_pred_train = clf.predict(X_train)
#     mcc = matthews_corrcoef(y_train, y_pred_train)
#     f1 = f1_score(y_train, y_pred_train, average = "weighted")
#     acc = np.mean(y_train.reshape(-1) == y_pred_train.reshape(-1))
#     print("train mcc: {}".format(mcc))
#     print("train acc: {}".format(acc))
#     print("train f1: {}".format(f1))
    
    
#     y_pred_lasso = clf.predict(X_val)
#     mcc = matthews_corrcoef(y_val, y_pred_lasso) # balanced classes
#     f1 = f1_score(y_val, y_pred_lasso, average = "weighted")
#     acc = np.mean(y_val.reshape(-1) == y_pred_lasso.reshape(-1))
    
#     print("validation mcc: {}".format(mcc))
#     print("validation acc: {}".format(acc))
#     print("validation f1: {}".format(f1))
    
#     loss = 1 - f1
            
#     return loss
        
# def callback(study,trial):
#     global best_lasso_model
#     if study.best_trial == trial:
#         best_lasso_model = clf

# best_lasso_model = None
# clf = None
# study = optuna.create_study(pruner=None)
# study.optimize(lasso_objective, n_trials=5, callbacks=[callback])

# # 5. Retrain with best model on train + validation
# best_lasso_model.fit(np.vstack([X_train, X_val]), 
#                      np.hstack([y_train, y_val]))

# # 6. Evaluate on test
# y_pred_test = best_lasso_model.predict(X_test)
# acc_test = np.mean(y_test.reshape(-1) == y_pred_test.reshape(-1))
# mcc_test = matthews_corrcoef(y_test, y_pred_test)
# f1_test = f1_score(y_test, y_pred_test, average = "weighted")
# print("Test accuracy: " + str(acc_test))
# print("Test MCC: " + str(mcc_test))
# print("Test F1: " + str(f1_test))

# # 7. Save the best model for further evaluation
# out_dir = "/gpfs/gibbs/pi/kleinstein/mw957/BCR_embed/model/" + args.embedding + "/"
# filename = out_dir + args.gene + ".sav"
# with open(filename, 'wb') as f:
#     pickle.dump(best_lasso_model, f)
# print("Best model saved at: " + filename)