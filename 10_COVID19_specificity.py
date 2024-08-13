import pickle
import torch

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold, GridSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.metrics import matthews_corrcoef, f1_score, r2_score, mean_squared_error, roc_auc_score, balanced_accuracy_score
from sklearn.svm import SVC
from collections import Counter
from sklearn.decomposition import PCA
import re
import time
import argparse

# def parse_args():
#  parser = argparse.ArgumentParser(description="Gene usage tasks")
#  parser.add_argument("--embedding", default="antiBERTy_Bcell_1",type=str, help="Type of embedding (TFIDF, immune2vec, esm1b)")
#  parser.add_argument("--model", default="H",type=str, help="Type of model (HL, H)")
#  parser.add_argument("--random", type=bool, help="Shuffle the data matrix", default=True)
#  args = parser.parse_args()
#  print(args)
#  return args
# args=parse_args()#原先没有20240126

parser = argparse.ArgumentParser(description="Gene usage tasks")
parser.add_argument("embedding", type=str, help="Type of embedding (TFIDF, immune2vec, esm1b)")
parser.add_argument("model", type=str, help="Type of model (HL, H)")
parser.add_argument("--random", type=bool, help="Shuffle the data matrix", default=False)
args = parser.parse_args()
BASE_DIR = "../data/BCR_embed/"

def load_data(embedding, model = "HL"):
    if "FULL" in embedding:
        print(f"Loading full length data...")
        anno = "specificity.anno"
        prefix_H = "combined_distinct_heavy"
        prefix_L = "combined_distinct_light"
        y = pd.read_table("../data/Annotations/" + anno)

    else:
        if "ELL" in embedding:
            print(f"Loading ellbedy data...")
            anno = "Bcell.anno"
            prefix_H = "combined_cdr3_heavy"
            prefix_L = "combined_cdr3_light"
            y = pd.read_table("../data/Annotations/" + anno)
            # prefix_H = "ellebedy_heavy"
            # prefix_L = "ellebedy_light"
        elif "Bcell_1" in embedding:
            print(f"Loading Bcell data...")
            anno = "Bcell_2.csv"
            prefix_H="Bcell"
            prefix_L = "Bcell"
            y = pd.read_csv("../data/Csv/" + anno)
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
        X = esm_H[y.seq_index.astype(int)-1,:]#X = esm_H[y.heavy_id.astype(int)-1,:]
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
        y_H_overlap = np.isin(y.seq_index, emb_H.index) #np.isin(y.heavy_id, emb_H.index)
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
    y_groups = y.label.values
    y = np.isin(y.label.values, ["plasmacytes_PC","memory_IgD-","memory_IgD+"])#, "memory_IgD-","mature_b_cell","plasmacytes_PC","memory_IgD-","transitional_b_cell""immature_b_cell"
    assert X.shape[0] == len(y)
    return X, y,y_groups

# 1. Load embeddings and labels
X, y, y_groups = load_data(args.embedding, args.model)
# X, y = load_data(args.embedding, args.model)
# group the entries with < 100 together
y_group_counts = Counter(y_groups)
small_groups = np.array(list(y_group_counts.keys()))[np.array(list(y_group_counts.values())) < 30]
y_groups[np.isin(y_groups, small_groups)] = "small"
print(f"Class size: {Counter(np.sort(y)).most_common()}")
    
print(f"In total, {len(y)} sequences from {len(np.unique(y_groups))} donors/studies.")

if args.random:
    print(f"Shuffling the embedding...")
    X = X[np.random.permutation(range(X.shape[0])),:][:,np.random.permutation(range(X.shape[1]))]

# 3. Nested cross validation (combine all data)
n_splits_outer = 4
n_splits_inner = 3
outer_cv = StratifiedGroupKFold(n_splits=n_splits_outer, shuffle=True, random_state=0)
inner_cv = StratifiedGroupKFold(n_splits=n_splits_inner, shuffle=True, random_state=1)

p_grid = {"C": [1e-2, 1e-1, 10, 100]}

outer_cv_w_groups = outer_cv.split(X, y,y_groups)
f1_scores = []
mcc_scores = []
acc_scores = []
auroc_scores = []
i = 1
svc = SVC(kernel = "rbf", class_weight = "balanced",  random_state=1, probability = False)#20240323

for train_index, test_index in outer_cv_w_groups:
    print(f"##### Outer fold {i} #####")
    # get the cross validation score on the test
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    print(f"Train size: {len(train_index)}, test size: {len(test_index)}")
    print(f"% positive train: {np.mean(y_train)}, % positive test: {np.mean(y_test)}")
    print(y_train,y_test)
    # inner loop
    cur_time = time.time()
    search = GridSearchCV(estimator = svc, 
               param_grid = p_grid, 
               cv = inner_cv, scoring = "f1_weighted", 
               n_jobs = -1,
               pre_dispatch = "1*n_jobs")
    search.fit(X_train, y_train, groups = y_groups[train_index])
    # search.fit(X_train, y_train)
    print(f"[Time (Outer fold {i})]: {time.time() - cur_time} seconds")
    prediction = search.predict(X_test)
    f1 = f1_score(y_test, prediction, average = "weighted")
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
