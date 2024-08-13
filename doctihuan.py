# # 显示分开，v,j,vl
# import pandas as pd
# import graphviz
# # 读取两个 CSV 文件
# # df1 = pd.read_csv("/home/mist/ClonalTree/Examples/output1/Kaminski_TP9.abRT.nk.csv", delimiter='\t', header=None, names=['parent', 'child'])
# df1= pd.read_csv("/home/mist/ClonalTree/Examples/output1/Kaminski_TP8.abRT.nk.csv",header=None)
# df2 = pd.read_csv("/home/mist/ClonalTree/Examples/input/combined_distinct_lightg.csv")
# # filepath = "/home/mist/ClonalTree/Examples/output1/iR_Patient_1.abRT.nk.csv"
# # with open(filepath) as csvfile:
# #     reader = csv.DictReader(csvfile)
# #     edges = [(row['parent'], row['child'], row['distance']) for row in reader]
# # 将第一个文件中含有'seq'的单元格替换为对应的v_call_family_light
# df1[0] =df1[0].apply(lambda x: df2.loc[df2['seqid'].str.contains(x), 'v_call_family_light'].values[0] if 'seq' in x else x)
# df1[1] = df1[1].apply(lambda x: df2.loc[df2['seqid'].str.contains(x), 'v_call_family_light'].values[0] if 'seq' in x else x)
#
# # 输出替换后的结果
# print(df1)
# dfl=df1.values.tolist()#转list
# graph = graphviz.Digraph(format='png')
# for edge in dfl:
#     parent, child, weight = edge
#     weight = str(weight)
#     graph.edge(parent, child, label=weight)
# graph.render("/home/mist/ClonalTree/Examples/output1/Kaminski_TP8", view=True)

import pandas as pd
import graphviz

# 读取两个CSV文件
df_graph = pd.read_csv("/home/mist/migrate/Csv/Bcell_1.abRT.nk.csv", header=None, names=['source', 'target', 'weight'])
df_info = pd.read_csv("/home/mist/migrate/Csv/Bcell.csv")

# 创建一个字典，将seqid映射到v_call_family_light
# seqid_to_v_call_family_light = dict(zip(df_info['seq_index'], df_info['label']))

# 在第一个文件中的每一行中匹配seqid，并将v_call_family_light添加到对应的节点ID中
for index, row in df_graph.iterrows():
    if 'seq' in row['source']:
        # seqid = 'seq' + row['source'].split('seq')[1]  # 提取出seqid
        seqid =int(row['source'].split('seq')[1] ) # 提取出seqid
        if seqid in df_info.seq_index:
            row_index = df_info[df_info['seq_index'] == seqid].index[0]
            # print(df_info.label[row_index])
        # if seqid in seqid_to_v_call_family_light:
            # df_graph.at[index, 'source'] += ' (' + seqid_to_v_call_family_light[seqid].split(" ")[0] + ')'
            df_graph.at[index, 'source'] += ' (' + str(df_info.label[row_index]) + ')'
    if 'seq' in row['target']:
        # seqid = 'seq' + row['target'].split('seq')[1]
        seqid2 = int(row['target'].split('seq')[1]) # 提取出seqid
        # if seqid in seqid_to_v_call_family_light:
        if seqid2 in df_info.seq_index:
            row_index = df_info[df_info['seq_index'] == seqid2].index[0]
            df_graph.at[index, 'target'] += ' (' + str(df_info.label[row_index]) + ')'

# 输出结果
print(df_graph)

# 转换为列表
dfl = df_graph.values.tolist()

# 创建有向图
graph = graphviz.Digraph(format='pdf')

color_list = ['#FF0000', '#00FF00', '#0000FF', '#CCCC00', '#FF00FF', '#00FFFF', '#800000', '#008000', '#000080','#003366',
                 '#808000', '#800080', '#008080', '#808080', '#C0C0C0', '#FFA500', '#FF007F','#FFC0CB', '#A52A2A', '#FFFF80',
                 '#80FF00', '#80FFFF', '#FF80FF', '#FF8000', '#8000FF', '#8080FF', '#FF0080', '#FF8080', '#80FF80', '#FFD700']

# 定义颜色映射
color_map = {}

# 添加边，并根据不同的v_call_family设置不同颜色的节点文字
for edge in dfl:
    parent, child, weight = edge
    weight = str(weight)
    graph.edge(parent, child, label=weight)
    # 检查孩子节点的v_call_family是否已存在于颜色映射中，若不存在则为其分配一个新颜色
    child_v_call_family = child.split('(')[-1].split(')')[0].strip() if '(' in child else None
    if child_v_call_family not in color_map:
        color_map[child_v_call_family] = color_list[len(color_map) % len(color_list)]  # 使用循环的方式从颜色列表中选择颜色
        print(color_map)
    # 检查父节点的v_call_family是否已存在于颜色映射中，若不存在则为其分配一个新颜色
    parent_v_call_family = parent.split('(')[-1].split(')')[0].strip() if '(' in parent else None
    if parent_v_call_family not in color_map:
        available_colors = [color for color in color_list if color not in color_map.values()]  # 排除已分配的颜色
        if available_colors:
            color_map[parent_v_call_family] = available_colors[0]  # 使用未分配的颜色
        else:
            color_map[parent_v_call_family] = color_list[
                len(color_map) % len(color_list)]  # 如果没有可用颜色，则使用循环的方式从颜色列表中选择颜色

    # 设置根节点的颜色
    root_node = dfl[0][0]
    #print(root_node)# 根节点为第一个节点的父节点
    # root_node_v_call_family = root_node.split('(')[-1].split(')')[0].strip() if '(' in root_node else None
    # # if  root_node_v_call_family not in color_map:
    # #     color_map[root_node_v_call_family] = color_list[0]  # 为根节点设置第一个颜色
    # if root_node_v_call_family not in color_map:
    #     available_colors = [color for color in color_list if color not in color_map.values()]  # 排除已分配的颜色
    #     if available_colors:
    #         color_map[root_node_v_call_family] = available_colors[0]  # 使用未分配的颜色
    #     else:
    #         color_map[root_node_v_call_family] = color_list[
    #             len(color_map) % len(color_list)]  # 如果没有可用颜色，则使用循环的方式从颜色列表中选择颜色
    # node_color = color_map[child_v_call_family] if child_v_call_family else color_map[root_node_v_call_family]
    node_color = color_map[child_v_call_family]
    # # print(color_map)
    # # print(color_map[root_node_v_call_family])
    # graph.node(root_node, fontcolor=color_map[root_node_v_call_family])
    graph.node(child,fontcolor = node_color)


# 渲染并保存图像
graph.render("/home/mist/migrate/Csv/aab_label")

# #简单输出，没有颜色
# import pandas as pd
# import graphviz
# # 读取两个 CSV 文件
# # df1 = pd.read_csv("/home/mist/ClonalTree/Examples/output1/Kaminski_TP9.abRT.nk.csv", delimiter='\t', header=None, names=['parent', 'child'])
# df1= pd.read_csv("/home/mist/ClonalTree/Examples/crd3l1/368-01a.abRT.nk.csv",header=None)
# # filepath = "/home/mist/ClonalTree/Examples/output1/iR_Patient_1.abRT.nk.csv"
# # with open(filepath) as csvfile:
# #     reader = csv.DictReader(csvfile)
# #     edges = [(row['parent'], row['child'], row['distance']) for row in reader]
# # 将第一个文件中含有'seq'的单元格替换为对应的v_call_family_light
# # 输出替换后的结果
# print(df1)
# dfl=df1.values.tolist()#转list
# graph = graphviz.Digraph(format='pdf')
# for edge in dfl:
#     parent, child, weight = edge
#     weight = str(weight)
#     graph.edge(parent, child, label=weight)
# graph.render("/home/mist/ClonalTree/Examples/crd3l1/368-01a", view=True)
