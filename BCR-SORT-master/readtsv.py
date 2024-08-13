# import pandas as pd
# # train=pd.read_csv('/home/mist/BCR-SORT/data/vdjserver.tsv', sep='\t', header=0)
# # print(train)
# # 读取 TSV 文件
# df = pd.read_csv('/home/mist/BCR-SORT/data/naive/vdjserver.tsv', sep='\t', header=0)
#
# # 选择 'v_call' 和 'j_call' 列
# selected_columns = df[['sequence_alignment_aa','v_call', 'j_call','c_call','locus','d_call','junction','cdr3_aa']]
#
# # 保存选择的列到 CSV 文件
# output_file_path = '/home/mist/BCR-SORT/data/vdjserver-naive.csv'  # 请替换为你想保存的文件路径
# selected_columns.to_csv(output_file_path, index=False)
#
# print(f"Selected columns saved to {output_file_path}")

import pandas as pd


def split_tsv(file_path, output_prefix, chunk_size):
    # 尝试读取前几行以确定列名
    sample_df = pd.read_csv(file_path, sep='\t', nrows=10)
    columns = sample_df.columns

    # 读取大型 TSV 文件时显式指定数据类型为字符串，并设置 low_memory=False
    chunks = pd.read_csv(file_path, sep='\t', chunksize=chunk_size, dtype=str, low_memory=False)

    # 遍历每个 chunk 并保存到新的文件中
    for i, chunk in enumerate(chunks):
        # 处理布尔列的 NA 值，将其转换为 False 或其他默认值
        for col in chunk.columns:
            if chunk[col].dtype == 'bool':
                chunk[col] = chunk[col].fillna(False).astype(str)

        output_file = f"{output_prefix}_part_{i+1}.csv"
        chunk.to_csv(output_file, index=False)  # 默认保存为 CSV 格式
        print(f"Saved {output_file}")

# 示例使用
file_path = '/home/mist/BCR-SORT/data/naive/vdjserver.tsv'
output_prefix = '/home/mist/BCR-SORT/data/naive/vdj'
chunk_size = 100000  # 每个小文件的行数

split_tsv(file_path, output_prefix, chunk_size)
