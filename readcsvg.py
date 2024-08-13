import os
import pandas as pd


def delete_first_row_csv_files(folder_path):
    # 获取文件夹下所有文件
    files = os.listdir(folder_path)
    print (files)
    # 遍历文件夹下的每个文件
    for file in files:
        # 确保文件是 CSV 文件
        if file.endswith(".csv"):
            # 构建文件的完整路径
            file_path = os.path.join(folder_path, file)

            # 读取 CSV 文件
            df = pd.read_csv(file_path)

            # 删除第一行
            df = df.iloc[1:]

            # 保存修改后的 CSV 文件
            df.to_csv(file_path, index=False)


# 调用函数删除指定文件夹下所有 CSV 文件的第一行
delete_first_row_csv_files("/home/mist/ClonalTree/Examples/input1/")
