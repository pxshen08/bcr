import pandas as pd


def copy_isotype_from_second_to_first(csv1_path, csv2_path):
    # 读取第一个 CSV 文件
    df1 = pd.read_csv(csv1_path)

    # 读取第二个 CSV 文件
    df2 = pd.read_csv(csv2_path)

    # 将第二个 CSV 文件中的 isotype_light 列和 id 列合并到一个 DataFrame 中
    merged_df = pd.merge(df1, df2[['id', 'isotype_light']], left_on='light_id', right_on='id', how='left')

    # 更新第一个 CSV 文件中的 isotype 列
    df1['isotype'] = merged_df['isotype_light']

    # 将更新后的数据保存回第一个 CSV 文件
    df1.to_csv(csv1_path, index=False)

    print("isotype_light 已成功复制到 isotype 列中。")


# 调用函数，传入两个 Excel 文件的路径
excel2_path = '/home/mist/projects/Wang2023/data/Csv/combined_distinct_light.csv'  # 替换为第一个 Excel 文件的路径
excel1_path = '/home/mist/projects/Wang2023/data/Csv/specificity.csv'  # 替换为第二个 Excel 文件的路径
copy_isotype_from_second_to_first(excel1_path, excel2_path)
