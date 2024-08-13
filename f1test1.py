from sklearn.metrics import matthews_corrcoef, f1_score
import numpy as np


y_true = [0, 1, 2, 0, 1, 2]
y_pred = [0, 2, 1, 0, 0, 1]
print(f1_score(y_true, y_pred, average='macro'))  # 0.26666666666666666
print(f1_score(y_true, y_pred, average='micro'))  # 0.3333333333333333
print(f1_score(y_true, y_pred, average='weighted'))  # 0.26666666666666666
print(f1_score(y_true, y_pred, average=None))  # [0.8 0.  0. ]
print(matthews_corrcoef(y_true, y_pred))
acc = np.mean(y_true.reshape(-1) == y_pred.reshape(-1))
