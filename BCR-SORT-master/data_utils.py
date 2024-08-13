import torch
import random
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import math
import numpy as np

amino_acid_dict = {'G': 1, 'A': 2, 'S': 3, 'T': 4, 'C': 5, 'V': 6, 'L': 7, 'I': 8, 'M': 9, 'P': 10, 'F': 11, 'Y': 12,
                   'W': 13, 'D': 14, 'E': 15, 'N': 16, 'Q': 17, 'H': 18, 'K': 19, 'R': 20, '_': 21}
# amino_acid_dict = {'A': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8, 'K': 9, 'L': 10, 'M': 11, 'N': 12,
#                    'P': 13, 'Q': 14, 'R': 15, 'S': 16, 'T': 17, 'V': 18, 'W': 19, 'Y': 20, '_': 21}


def sequence_padding(data_in, mode):
    afeature_list = ['V_gene', 'J_gene', 'isotype']
    annotation_feature_length = len(afeature_list)

    data_in.sort(key=lambda x: len(x[0][0]), reverse=True)

    sequence = []
    annotation_feature = torch.Tensor()
    data_length = []
    for dataset in data_in:
        sample, label = dataset
        features = sample[1:]#20240721
        features = torch.from_numpy(np.concatenate([features]))#20240721
        sequence.append(sample[:len(sample)-annotation_feature_length])
        # annotation_feature = torch.cat([annotation_feature, sample[-annotation_feature_length:]], dim=0)#截取最后 annotation_feature_length 个元素，
        annotation_feature = torch.cat([annotation_feature, features], dim=0)
        # 并将其与 annotation_feature 进行拼接。torch.cat 函数用于在指定维度上连接张量，这里是在维度 0 上进行连接。
        data_length.append(len(sample[0]))
        # data_length.append(len(sample[:len(sample)-annotation_feature_length]))#计算从 sample 中截取的
        # 前 len(sample) - annotation_feature_length 个元素的长度，并将其添加到 data_length 列表中

    # sequence_padded = pad_sequence(sequence, batch_first=True, padding_value=0)
    # data_padded = torch.cat([sequence_padded[i].reshape(1, -1) for i, _ in enumerate(data_in)], dim=0)

    annotation_feature = annotation_feature.view((len(data_in), annotation_feature_length))
    # data_out = (data_padded,
    #             annotation_feature[:, 0].unsqueeze(dim=1),
    #             annotation_feature[:, 1].unsqueeze(dim=1),
    #             annotation_feature[:, 2].unsqueeze(dim=1))
    data_out = (sequence,
                annotation_feature[:, 0].unsqueeze(dim=1),
                annotation_feature[:, 1].unsqueeze(dim=1),
                annotation_feature[:, 2].unsqueeze(dim=1))
    if mode == 'train':
        data_label = torch.Tensor()
        for i, dataset in enumerate(data_in):
            _, label = dataset
            data_label = torch.cat([data_label, torch.unsqueeze(label, 0)], dim=0)

    elif mode == 'predict':
        data_label = None

    else:
        raise Exception("Invalid mode to load dataset")

    return data_out, data_length, data_label


def split_train_test(data_in, test_ratio, seed=12345):
    val_ratio = test_ratio

    ntotal = len(data_in)
    nval = int(ntotal * val_ratio)
    ntest = int(ntotal * test_ratio)

    index_list = list(range(ntotal))
    random.seed(seed)
    random.shuffle(index_list)

    test_idx = index_list[:ntest]
    val_idx = index_list[ntest:ntest+nval]
    train_idx = index_list[ntest+nval:]

    data_train = data_in.iloc[train_idx].copy()
    data_val = data_in.iloc[val_idx].copy()
    data_test = data_in.iloc[test_idx].copy()

    return data_train, data_val, data_test


def generate_seq_mask(sequence, lengths):
    token_aa_mask = 21

    pos_mask = torch.Tensor([random.randrange(n) for n in lengths]).type(torch.cuda.LongTensor)
    mask_seq = F.one_hot(pos_mask, max(lengths))
    mask_seq = (mask_seq == torch.zeros_like(mask_seq)).type(torch.cuda.LongTensor)
    label_aux = torch.Tensor([seq[pos].item() for pos, seq in zip(pos_mask, sequence)]).type(torch.cuda.LongTensor)

    return sequence.masked_fill_(mask_seq == 0, token_aa_mask), label_aux


def generate_seq_mask1(sequence, lengths):
    token_aa_mask = '_'
    num_classes = len(amino_acid_dict)+1  # 22 classes including 'X'

    # 生成随机位置掩码
    pos_mask = torch.Tensor([random.randint(0, n - 1) for n in lengths]).type(torch.LongTensor)

    # 生成 one-hot 编码掩码
    max_len = max(lengths)
    mask_seq = F.one_hot(pos_mask, num_classes=max_len)
    mask_seq = (mask_seq == 0).type(torch.FloatTensor)

    # 生成辅助标签
    label_aux = [seq[0][pos.item()] for pos, seq in zip(pos_mask, sequence)]

    # 使用掩码填充序列
    sequence_masked = []
    for seq, mask in zip(sequence, mask_seq):
        masked_seq = ''.join([token_aa_mask if mask[i].item() == 0 else seq[0][i] for i in range(len(seq[0]))])
        sequence_masked.append(masked_seq)

    label_aux_one_hot = torch.zeros((len(label_aux), num_classes))

    for i, aa in enumerate(label_aux):
        index = amino_acid_dict[aa]
        label_aux_one_hot[i, index-1] = 1

    return sequence_masked, label_aux_one_hot

def encode_sequence(seq):
    # print(seq)

    seq = character_tokenizer(seq)
    # print(seq)
    if seq is None:
        print("Skipping sequence due to NaN or empty value")
        return []  # 或者根据需要返回其他内容，如 None
    return [amino_acid_dict[char] for char in seq]


def decode_sequence(seq_idx):
    data_out = []
    for idx in seq_idx:
        seq = ''
        for i in idx:
            if i != 0:
                seq += [k for k, v in amino_acid_dict.items() if v == i][0]
            if i == 0:
                break
        data_out.append(seq)

    return data_out


def character_tokenizer(x):
   try:
       if isinstance(x, float) and (math.isnan(x) or x == 0):
           return None  # 返回 None 以表示跳过
       if isinstance(x, str) and not x.strip():
           return None  # 跳过空字符串
       if isinstance(x, float):
           x = str(x)
       return list(x)
   except:
       print("wrong",x)
       return None


def encode_categorical(x, encoding_dict):
    return encoding_dict.index(x)


def decode_categorical(x, encoding_dict):
    return encoding_dict[x]


def read_dict_file(file_category, baseline_token=True):
    encoding_list = []
    for line in open(file_category):
        encoding_list.append(line.replace('\n', ''))

    if baseline_token:
        encoding_list.append('baseline')

    return encoding_list


def generate_dict(baseline_token=True):
    file_dict_vgene = './dictionary/IGHV_functional.tab'
    file_dict_jgene = './dictionary/IGHJ_functional.tab'
    file_dict_isotype = './dictionary/IGHC_functional.tab'
    file_label = './dictionary/cell_subset_label.tab'

    encoding_dict = {'V_gene': read_dict_file(file_dict_vgene, baseline_token),
                     'J_gene': read_dict_file(file_dict_jgene, baseline_token),
                     'isotype': read_dict_file(file_dict_isotype, baseline_token),
                     'label': read_dict_file(file_label, baseline_token)}

    return encoding_dict
