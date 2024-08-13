import pickle
import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.svm import SVR
from sklearn.metrics import matthews_corrcoef, f1_score, r2_score, mean_squared_error, mean_absolute_error
from collections import Counter
import time
import argparse

parser = argparse.ArgumentParser(description="Gene usage tasks")
parser.add_argument("target", type=str, help="Variable to predict")
parser.add_argument("embedding", type=str, help="Type of embedding (immune2vec, esm2, ProtT5)")
parser.add_argument("--random", type=bool, help="Shuffle the data matrix", default=True)
args = parser.parse_args()

BASE_DIR = "../data/embeddings/"

COL_MAP = {"mu_H": "mu_freq_heavy",
           "mu_L": "mu_freq_light",
           "Jlen_H": "junction_aa_length_heavy",
           "Jlen_L": "junction_aa_length_light"}

MIN_CLASS_SIZE = 100
MAX_CLASS_SIZE = 5000

def load_data(gene, embedding):
    if "H" in gene:
        data_prefix = "combined_distinct_heavy"
    elif "L" in gene:
        data_prefix = "combined_distinct_light"
    y = pd.read_table(BASE_DIR + data_prefix + ".anno")
    
    # remove sequences with empty labels
    y = y.loc[~np.isnan(y.loc[:,COL_MAP[gene]]),:]    
    y = y.set_index('id')

    if embedding == 'esm2' or embedding == 'esm2_3B':
        suffix = embedding.replace('esm2', '')
        X = torch.load(BASE_DIR + data_prefix + suffix + ".pt", map_location=torch.device('cpu')).numpy()
        X = X[y.index-1,:]
        y_groups = y.subject.values
        y = y.loc[:, COL_MAP[gene]].values
        
    elif embedding == 'antiBERTy':
        X = torch.load(BASE_DIR + data_prefix + "_antiBERTy.pt", map_location=torch.device('cpu')).numpy()
        X = X[y.index-1,:]
        y_groups = y.subject.values
        y = y.loc[:, COL_MAP[gene]].values
        
    elif ("immune2vec" in embedding) or np.isin(embedding, ["physicochemical", "frequency", "ProtT5"]):
        X = pd.read_pickle(BASE_DIR + data_prefix + "_" + embedding + ".pkl")
        X.index = pd.Series([int(x) for x in X.index.values])
        
        # intersect the dataset (in case of failures in embedding)
        X_idx = np.array(set(X.index) & set(y.index))
        X = X.loc[X_idx,:].values
        y_groups = y.loc[X_idx, "subject"].values
        y = y.loc[X_idx, COL_MAP[gene]].values
    
    assert X.shape[0] == len(y)
    
    return X, y, y_groups

# 1. Load embeddings and labels
X, y, y_groups = load_data(args.target, args.embedding)
    
# 2. Down-sample overrepresented values
def subsample(data, label, class_size, group = None):
    data = pd.DataFrame(data)
    data.loc[:,"label"] = label
    if group is not None:
        data.loc[:,"group"] = group
    counts, bins = np.histogram(label, bins = 10)

    subsampled = []
    for i in range(len(bins)-1):
        if i == len(bins)-2:
        # last bin is inclusive on both sides
            section = data.loc[(label>=bins[i]) & (label<=bins[i+1]),:]
        else:
            section = data.loc[(label>=bins[i]) & (label<bins[i+1]),:]
        sub_section = section.sample(min(class_size, section.shape[0]), replace=False)
        subsampled.append(sub_section)
    balanced = pd.concat(subsampled)
    
    return balanced

include = subsample(X, y, MAX_CLASS_SIZE, y_groups)
X, y, y_groups = include.iloc[:,:-2].values, include.label.values, include.group.values
print(f"Downsampling each bin to at most {MAX_CLASS_SIZE} sequences for training.")

# 3. Random baseline
if args.random:
    print(f"Shuffling the embedding...")
    X = X[np.random.permutation(range(X.shape[0])),:][:,np.random.permutation(range(X.shape[1]))]

# 4. Nested cross validation
n_splits_outer = 5
n_splits_inner = 3
outer_cv = GroupKFold(n_splits=n_splits_outer)
inner_cv = GroupKFold(n_splits=n_splits_inner)

p_grid = {"alpha": [1e-6, 1e-5, 1e-4, 1e-3]} # {"C": [1e-2, 1e-1, 10, 100]}

outer_cv_w_groups = outer_cv.split(X, y, y_groups)
rmse_scores = []
r2_scores = []
mae_scores = []
i = 1
model = Lasso(random_state = 1, max_iter = int(1e4))
#model = SVR(kernel = "linear", max_iter = int(1e4))

for train_index, test_index in outer_cv_w_groups:
    print(f"##### Outer fold {i} #####")
    # get the cross validation score on the test
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    # inner loop
    # hyperparameter search
    cur_time = time.time()
    search = GridSearchCV(estimator = model, 
               param_grid = p_grid, 
               cv = inner_cv, 
               scoring = "neg_root_mean_squared_error", 
               n_jobs = -1,
               pre_dispatch = "1*n_jobs")
    search.fit(X_train, y_train, groups = y_groups[train_index])
    print(f"[Time (Outer fold {i})]: {time.time() - cur_time} seconds")
    prediction = search.predict(X_test)
    rmse = mean_squared_error(y_test, prediction, squared = False)
    r2 = r2_score(y_test, prediction)
    mae = mean_absolute_error(y_test, prediction)
    n = X_test.shape[0]
    p = X_test.shape[1]
    adjusted_r2 = 1-(1-r2)*(n-1)/(n-p-1)
    print(f"[RMSE (Outer fold {i})]: {rmse}")
    print(f"[Adjusted R2 (Outer fold {i})]: {r2}")
    print(f"[MAE (Outer fold {i})]: {mae}")
    print()
    rmse_scores.append(rmse)
    r2_scores.append(r2)
    mae_scores.append(mae)
    i += 1

# Save results as a table
out_score = pd.DataFrame({"Fold": range(n_splits_outer), 
                          "RMSE": rmse_scores,
                          "R2": r2_scores,
                          "MAE": mae_scores})

if args.random:
    is_random = "_random"
else:
    is_random = ""
filename = "../data/results/" + args.embedding + "_" + args.target + is_random + ".csv"
out_score.to_csv(filename)
print("Results saved at: " + filename)

rmse_scores = np.array(rmse_scores)
r2_scores = np.array(r2_scores)
mae_scores = np.array(mae_scores)

print(f"[Mean RMSE]: {rmse_scores.mean()}")
print(f"[SD RMSE]: {rmse_scores.std()}")
print(f"[Mean R2]: {r2_scores.mean()}")
print(f"[SD R2]: {r2_scores.std()}")
print(f"[Mean MAE]: {mae_scores.mean()}")
print(f"[SD MAE]: {mae_scores.std()}")

# # 3. Split one dataset as the training and rest as validation
# gss = GroupShuffleSplit(n_splits=1, train_size=.9, random_state=1)
# splits = gss.split(X = X_train, groups = y_groups)
# train_idx, val_idx = next(splits)
# X_train, y_train, X_val, y_val = X_train[train_idx,:], y_train[train_idx], X_train[val_idx,:], y_train[val_idx]

# print(f"Using {len(y_train)} training sequences, {len(y_val)} validation sequences, {len(y_test)} test sequences.")

# # 4. Hyperparameter search 
# # tuning for the regularization parameter 
# def lasso_objective(trial):  
#     global LR_clf
#     a_min = 1e-6
#     a_max = 1e-3        
#     a = trial.suggest_loguniform('alpha', a_min, a_max)

#     LR_clf = Lasso(alpha=a, random_state = 1, max_iter = int(1e4))
#     LR_clf.fit(X_train,y_train)

#     print("Regularization: {}".format(a))
    
#     y_pred_lasso = LR_clf.predict(X_train)
#     rmse = mean_squared_error(y_train, y_pred_lasso, squared = False)
#     r2 = r2_score(y_train, y_pred_lasso)
#     n = X_train.shape[0]
#     p = X_train.shape[1]
#     adjusted_r2 = 1-(1-r2)*(n-1)/(n-p-1)

#     print("Train rmse: {}".format(rmse))
#     print("Train adjusted r2: {}".format(adjusted_r2))
    
#     y_pred_lasso = LR_clf.predict(X_val)
#     rmse = mean_squared_error(y_val, y_pred_lasso, squared = False)
#     r2 = r2_score(y_val, y_pred_lasso)
#     n = X_val.shape[0]
#     p = X_val.shape[1]
#     adjusted_r2 = 1-(1-r2)*(n-1)/(n-p-1)
   
#     print("validation rmse: {}".format(rmse))
#     print("validation adjusted r2: {}".format(adjusted_r2))
    
#     loss = rmse**2
            
#     return loss
        
# def callback(study,trial):
#     global best_lasso_model
#     if study.best_trial == trial:
#         best_lasso_model = LR_clf

# best_lasso_model = None
# LR_clf = None
# study = optuna.create_study(pruner=None)
# study.optimize(lasso_objective, n_trials=5, callbacks=[callback])

# # 5. Retrain with best model on train + validation
# best_lasso_model.fit(np.vstack([X_train, X_val]), 
#                      np.hstack([y_train, y_val]))

# # 6. Evaluate on test
# y_pred_test = best_lasso_model.predict(X_test)
# rmse_test = mean_squared_error(y_test, y_pred_test, squared = False)
# r2_test = r2_score(y_test, y_pred_test)
# n = X_test.shape[0]
# p = X_test.shape[1]
# adjusted_r2_test = 1-(1-r2_test)*(n-1)/(n-p-1)
# print("Test RMSE: " + str(rmse_test))
# print("Test adjusted R2: " + str(adjusted_r2_test))

# # 7. Save the best model for further evaluation
# out_dir = "/gpfs/gibbs/pi/kleinstein/mw957/BCR_embed/model/" + args.embedding + "/"
# filename = out_dir + args.target + ".sav"
# with open(filename, 'wb') as f:
#     pickle.dump(best_lasso_model, f)
# print("Best model saved at: " + filename)