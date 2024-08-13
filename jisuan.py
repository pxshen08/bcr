from sklearn.metrics import precision_recall_fscore_support

# 真实标签
y_true = [0, 1, 2, 0, 1, 2]
# 预测标签
y_pred = [0, 2, 1, 0, 0, 1]

# 计算精确率、召回率、F1 分数和支持度
precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred)
print(f1)
# 打印每个类别的精确率、召回率、F1 分数和支持度
for i in range(len(precision)):
    print(f"类别 {i}: 精确率={precision[i]}, 召回率={recall[i]}, F1 分数={f1[i]}, 支持度={support[i]}")

# 计算宏平均的精确率、召回率和 F1 分数
macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro')
print(f"宏平均精确率: {macro_precision}, 宏平均召回率: {macro_recall}, 宏平均F1分数: {macro_f1}")
