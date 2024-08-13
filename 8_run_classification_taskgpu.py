import pickle
import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold, GridSearchCV
from sklearn.metrics import matthews_corrcoef, f1_score
from collections import Counter
import argparse
import time
import os
from cuml.svm import SVC as cuSVC  # Import cuML's SVM


def parse_args():
    parser = argparse.ArgumentParser(description="Gene usage tasks")
    parser.add_argument('--gene', default='isoH', type=str, help="Gene type (VH, VL, JH, JL, isoH, isoL)")
    parser.add_argument('--embedding', default='ablang', type=str, help="Type of embedding (immune2vec, esm2, ProtT5)")
    parser.add_argument('--random', type=bool, help="Shuffle the data matrix", default=False)
    args = parser.parse_args()
    print(args)
    return args


BASE_DIR = "/home/mist/projects/Wang2023/data/Annotations/"
args = parse_args()

COL_MAP = {"VH": "v_call_family_heavy",
           "VL": "v_call_family_light",
           "JH": "j_call_family_heavy",
           "JL": "j_call_family_light",
           "isoH": "isotype_heavy",
           "isoL": "isotype_light"}

MIN_CLASS_SIZE = 100
MAX_CLASS_SIZE = 5000


def load_data(gene, embedding):
    if "H" in gene:
        data_prefix = "combined_distinct_heavy"
    elif "L" in gene:
        data_prefix = "combined_distinct_light"
    y = pd.read_table(BASE_DIR + data_prefix + ".anno")

    gene_col = y.loc[:, COL_MAP[gene]]
    y = y.loc[~(np.isin(gene_col, ["Bulk"]) | gene_col.isna()), :]
    y = y.set_index('id')

    if embedding == 'esm2' or embedding == 'esm2_3B':
        suffix = embedding.replace('esm2', '')
        X = torch.load("/home/mist/projects/Wang2023/data/BCR_embed/datae/" + data_prefix + suffix + ".pt",
                       map_location=torch.device('cpu')).numpy()
        X = X[y.index - 1, :]
        y_groups = y.subject.values
        y = y.loc[:, COL_MAP[gene]].values

    elif embedding == 'antiBERTy' or embedding == 'ablang':
        X = torch.load("/home/mist/projects/Wang2023/data/BCR_embed/datai/" + data_prefix + "_ablang.pt",
                       map_location=torch.device('cpu')).numpy()
        y20240126 = (y.index).to_numpy(dtype=int, na_value=-1)
        X = X[y20240126 - 1, :]
        y_groups = y.subject.values
        y = y.loc[:, COL_MAP[gene]].values

    elif ("immune2vec" in embedding) or (np.isin(embedding, ["physicochemical", "frequency", "ProtT5"])):
        X = pd.read_pickle("/home/mist/projects/Wang2023/data/BCR_embed/" + data_prefix + "_" + embedding + ".pkl")
        X.index = pd.Series([int(x) for x in X.index.values])

        X_idx = np.array(set(X.index) & set(y.index))
        X = X.loc[X_idx, :].values
        y_groups = y.loc[X_idx, "subject"].values
        y = y.loc[X_idx, COL_MAP[gene]].values

    assert X.shape[0] == len(y)

    return X, y, y_groups


X, y, y_groups = load_data(args.gene, args.embedding)

class_sizes = Counter(y)
target_class = np.array(list(class_sizes.keys()))[np.array(list(class_sizes.values())) > MIN_CLASS_SIZE]
include = np.isin(y, target_class)
X, y, y_groups = X[include, :], y[include], y_groups[include]


def subsample(index, class_size):
    classes = np.unique(index)
    subsampled = []
    np.random.seed(0)
    for i in classes:
        is_class = np.where(index == i)[0]
        if len(is_class) <= class_size:
            subsampled.append(is_class)
        else:
            subsampled.append(np.random.choice(is_class, size=class_size, replace=False))
    subsampled = np.hstack(subsampled)
    return subsampled


downsample = True
if downsample:
    include = subsample(y, MAX_CLASS_SIZE)
    X, y, y_groups = X[include, :], y[include], y_groups[include]
    print(
        f"Downsampling classes to at most {MAX_CLASS_SIZE} sequences for {len(target_class)} classes with size > {MIN_CLASS_SIZE}.")
print(f"In total, {len(y)} sequences from {len(np.unique(y_groups))} donors/studies.")
print(f"Class size: {Counter(np.sort(y)).most_common()}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

if args.random:
    print(f"Shuffling the embedding...")
    X = X[np.random.permutation(range(X.shape[0])), :][:, np.random.permutation(range(X.shape[1]))]

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
svc = cuSVC(kernel="rbf", class_weight="balanced", random_state=1, probability=False)

for train_index, test_index in outer_cv_w_groups:
    print(f"##### Outer fold {i} #####")
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    print(f"Train size: {len(train_index)}, test size: {len(test_index)}")
    cur_time = time.time()
    search = GridSearchCV(estimator=svc, param_grid=p_grid, cv=inner_cv, scoring="f1_weighted", n_jobs=-1,
                          pre_dispatch="1*n_jobs")
    search.fit(X_train, y_train, groups=y_groups[train_index])
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

out_score = pd.DataFrame({"Fold": range(n_splits_outer),
                          "F1": f1_scores,
                          "MCC": mcc_scores,
                          "ACC": acc_scores})

if args.random:
    is_random = "_random"
else:
    is_random = ""
filename = "/home/mist/projects/Wang2023/data/BCR_embed/datai/" + args.embedding + "_" + args.gene + is_random + ".csv"
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
