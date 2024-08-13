import pandas as pd
df1 = pd.read_csv("/home/mist/ClonalTree/Examples/input/specificity.csv")
df2 = pd.read_csv("/home/mist/projects/Wang2023/data/Csv/combined_distinct_light1.csv")

# 按照 Subject 进行分组统计第一个表
grouped = df1.groupby("source")

# 遍历每个 Subject
for subject, group in grouped:
    # 筛选第二个表中对应 Subject 的数据
    #merged_df = pd.merge(group, df2, left_on="id", right_on="Meta")
    filtered_df2 = df2[df2["Meta"].isin(group["light_id"])]
    s2=subject.split(" ")[0]
    # 生成对应 Subject 的文件名
    filename = f"{s2}.csv"
    if filtered_df2["Meta"].shape ==(0,):
        continue
    else:
       filtered_df2["Meta"] = "seq" + filtered_df2["Meta"].astype(str)
       # 将数据保存为 CSV 文件
       filtered_df2.to_csv("/home/mist/ClonalTree/Examples/sl1/"+filename, index=False)

