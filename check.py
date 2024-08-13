import pandas as pd
from tqdm import tqdm

def match_and_copy_to_d_column(file_path):
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

    i = 0
    # 使用矢量化操作匹配并更新 'D' 列，并显示进度条
    for c_value in tqdm(df['SequenceID.1'].dropna().unique(), desc="Processing"):

        mask = df['SequenceID'].str.contains(c_value, na=False)
        df.loc[mask, 'D'] = c_value
        i += 1
        if i % 10 == 0:
            matched_df = df.dropna(subset=['D'])
            matched_df.to_excel('/home/mist/BCR-SORT/data/outputmcheck2.xlsx', index=False)

    # 保存更新后的 DataFrame 到 Excel 文件
    df.to_excel('/home/mist/BCR-SORT/data/outputcheck2-1.xlsx', index=False)
    print("Output saved to '/home/mist/BCR-SORT/data/outputcheck2.xlsx'")

# 示例用法
excel_file_path = '/home/mist/BCR-SORT/data/check2.csv'  # 请替换为你的 CSV 文件路径
match_and_copy_to_d_column(excel_file_path)
