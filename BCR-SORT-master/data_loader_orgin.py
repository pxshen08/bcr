import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from data_utilsori import encode_sequence, encode_categorical, generate_dict, split_train_test
from sklearn.preprocessing import LabelEncoder

class DatasetBCRSORT(Dataset):
    def __init__(self, data_in):
        # encoding sequence
        data_in.loc[:, 'seq'] = data_in.apply(lambda row: encode_sequence(row['seq']), axis=1)

        # encoding categorical variables
        encoding_dict = generate_dict()
        categorical_variable = ['V_gene', 'J_gene', 'isotype']
        if 'label' in data_in.columns.values.tolist():
            categorical_variable += ['label']

        for variable in categorical_variable:
            data_in.loc[:, variable] = data_in.apply(lambda row: encode_categorical(row[variable], encoding_dict[variable]), axis=1)

        if 'label' in data_in.columns.values.tolist():
            # 初始化LabelEncoder
            label_encoder = LabelEncoder()

            # 将字符串标签转换为数值标签
            data_in['label_encoded'] = label_encoder.fit_transform(data_in['label'])

            # 检查标签的范围
            print(f"Unique encoded labels: {data_in['label_encoded'].unique()}")

            # 将数值标签转换为NumPy数组
            labels_np = data_in['label_encoded'].values

            # 将NumPy数组转换为PyTorch张量
            labels_tensor = torch.from_numpy(labels_np).type(torch.LongTensor)

            # 获取类别数
            num_classes = len(label_encoder.classes_)

            # 检查类别数
            print(f"Number of classes: {num_classes}")

            # 进行one-hot编码
            self.label = torch.nn.functional.one_hot(labels_tensor, num_classes=num_classes)

            # 检查one-hot编码的结果
            print(f"One-hot encoded labels: {self.label}")

            # 删除原始的`label`列和中间生成的`label_encoded`列，仅保留其他数据
            self.data = data_in.drop(['label', 'label_encoded'], axis=1, inplace=False)
        else:
            self.label = None
            self.data = data_in

        self.dataset_size = data_in.shape[0]

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        data = self.data.iloc[idx].values.tolist()
        seq = data[0]
        features = data[1:]
        data = torch.from_numpy(np.concatenate([seq, features]))

        if self.label is None:
            label = None
        else:
            label = self.label[idx]

        return data, label


def load_dataset(file_in, mode, test_ratio=None):
    if isinstance(file_in, str):
        data_in = pd.read_csv(file_in)
    elif isinstance(file_in, list):
        data_in = pd.concat((pd.read_csv(f) for f in file_in))

    if mode == 'predict':
        col_requirement = ['seq', 'V_gene', 'J_gene', 'isotype']
    elif mode == 'train':
        col_requirement = ['seq', 'V_gene', 'J_gene', 'isotype', 'label']
    else:
        raise Exception("Invalid mode to load dataset")

    col_input = data_in.columns.values
    for col in col_requirement:
        if col not in col_input:
            raise Exception("No %s feature in the input data" % col)

    data_in = data_in.loc[:, col_requirement]
    data_in.drop_duplicates(subset=col_requirement, inplace=True)

    if test_ratio is None:
        return DatasetBCRSORT(data_in=data_in)
    else:
        data_train, data_val, data_test = split_train_test(data_in, test_ratio, seed=12345)
        TrainDatasetBCR = DatasetBCRSORT(data_in=data_train)
        ValDatasetBCR = DatasetBCRSORT(data_in=data_val)
        TestDatasetBCR = DatasetBCRSORT(data_in=data_test)

        return TrainDatasetBCR, ValDatasetBCR, TestDatasetBCR
