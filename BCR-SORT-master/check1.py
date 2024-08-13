import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

def process_chunk(args):
    chunk, c_values = args
    for c_value in c_values:
        mask = chunk['SequenceID'].str.contains(c_value, na=False)
        chunk.loc[mask, 'D'] = c_value
    return chunk

def match_and_copy_to_d_column(file_path, num_workers=8):
    try:
        # 使用 on_bad_lines='skip' 来跳过错误行
        df = pd.read_csv(file_path, on_bad_lines='skip', sep=',')
    except pd.errors.ParserError as e:
        print(f"Error reading the CSV file: {e}")
        return

    print(df.head())

    # 检查是否存在 'SequenceID.1' 和 'SequenceID' 列
    if 'SequenceID.1' not in df.columns or 'SequenceID' not in df.columns:
        print("The required columns 'SequenceID.1' and 'SequenceID' are not in the dataframe")
        return

    # 确保 'D' 列存在
    if 'D' not in df.columns:
        df['D'] = None

    # 获取唯一的非空 SequenceID.1 值
    c_values = df['SequenceID.1'].dropna().unique()

    # 将数据分块以便并行处理
    chunk_size = len(df) // num_workers
    chunks = [df.iloc[i:i + chunk_size].copy() for i in range(0, len(df), chunk_size)]

    # 创建参数列表，包含每个块和c_values
    tasks = [(chunk, c_values) for chunk in chunks]

    # 使用 ProcessPoolExecutor 进行并行处理
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = list(tqdm(executor.map(process_chunk, tasks), total=len(tasks), desc="Processing"))

    # 合并结果
    df_updated = pd.concat(results)

    # 保存更新后的 DataFrame 到 Excel 文件
    df_updated.to_excel('/home/mist/BCR-SORT/data/output2.xlsx', index=False)
    print("Output saved to '/home/mist/BCR-SORT/data/output2.xlsx'")

# 示例用法
excel_file_path = '/home/mist/BCR-SORT/data/combined_distinct_heavy1check.csv'  # 请替换为你的 CSV 文件路径
match_and_copy_to_d_column(excel_file_path)
