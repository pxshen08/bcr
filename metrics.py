def p_r_f1_a(acts, pres):
    TP, FP, TN, FN = 0, 0, 0, 0
    for i in range(len(acts)):
        if acts[i] == 1 and pres[i] == 1:
            TP += 1
        if acts[i] == 0 and pres[i] == 1:
            FP += 1
        if acts[i] == 1 and pres[i] == 0:
            FN += 1
        if acts[i] == 0 and pres[i] == 0:
            TN += 1

            # 精确率Precision
    P = TP / (TP + FP)

    # 召回率Recall
    R = TP / (TP + FN)

    # F1
    F1 = 2 / (1 / P + 1 / R)

    # 准确率Accuracy
    A = (TP + TN) / (TP + FP + FN + TN)

    return P, R, F1, A

from sklearn.metrics import f1_score
y_true = [0, 1, 1, 0, 1, 1, 0, 1]
y_pred = [0, 1, 0, 0, 1, 1, 1, 1]
f1 = f1_score(y_true, y_pred)
f2=p_r_f1_a(y_true, y_pred)
print('F1-score:', f1,f2)