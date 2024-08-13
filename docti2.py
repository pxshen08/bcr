import pandas as pd
import graphviz

# 读取两个CSV文件
df_graph = pd.read_csv("/home/mist/ClonalTree/Examples/output/Bcell_1.abRT.nk.csv", header=None, names=['source', 'target', 'weight'])
df_info = pd.read_csv("/home/mist/projects/Wang2023/data/Csv/Bcell.csv")

# 创建一个字典，将seqid映射到v_call_family_light
seqid_to_v_call_family_light = dict(zip(df_info['seq_index'], df_info['label']))

# 在第一个文件中的每一行中匹配seqid，并将v_call_family_light添加到对应的节点ID中
for index, row in df_graph.iterrows():
    if 'seq' in row['source']:
        # seqid = 'seq' + row['source'].split('seq')[1]  # 提取出seqid
        seqid =int(row['source'].split('seq')[1] ) # 提取出seqid
        if seqid in seqid_to_v_call_family_light:
            # df_graph.at[index, 'source'] += ' (' + seqid_to_v_call_family_light[seqid].split(" ")[0] + ')'
            df_graph.at[index, 'source'] += ' (' + str(seqid_to_v_call_family_light[seqid]) + ')'
    if 'seq' in row['target']:
        # seqid = 'seq' + row['target'].split('seq')[1]
        seqid = int(row['target'].split('seq')[1]) # 提取出seqid
        if seqid in seqid_to_v_call_family_light:
            df_graph.at[index, 'target'] += ' (' + str(seqid_to_v_call_family_light[seqid]) + ')'

# 输出结果
print(df_graph)

# 转换为列表
dfl = df_graph.values.tolist()

# 定义细胞类型进化顺序
evolution_order = {
    'immature_b_cell': 1,
    'transitional_b_cell': 2,
    'mature_b_cell': 3,
    'memory_IgD+': 4,
    'memory_IgD-': 5,
    'plasmacytes_PC': 6
}

# 根据进化顺序检查并交换父子关系
def get_evolution_rank(cell_label):
    return evolution_order.get(cell_label, float('inf'))

# 迭代交换父子关系函数
def iterative_swap(nodes):
    stack = nodes.copy()
    while stack:
        parent, child, weight = stack.pop()
        parent_label = parent.split('(')[-1].split(')')[0].strip() if '(' in parent else None
        child_label = child.split('(')[-1].split(')')[0].strip() if '(' in child else None

        parent_rank = get_evolution_rank(parent_label)
        child_rank = get_evolution_rank(child_label)

        # 如果父节点的进化顺序大于子节点的进化顺序，交换父子关系
        if parent_rank > child_rank:
            parent, child = child, parent
            # 找到当前子节点的父节点
            for edge in nodes:
                if edge[1] == parent:
                    stack.append((edge[0], parent, edge[2]))
                    break
        modified_edges.append([parent, child, weight])

# 初始化修改后的边列表
modified_edges = []
iterative_swap(dfl)

# 创建有向图
graph = graphviz.Digraph(format='pdf')

color_list = ['#FF0000', '#00FF00', '#0000FF', '#CCCC00', '#FF00FF', '#00FFFF', '#800000', '#008000', '#000080','#003366',
                 '#808000', '#800080', '#008080', '#808080', '#C0C0C0', '#FFA500', '#FF007F','#FFC0CB', '#A52A2A', '#FFFF80',
                 '#80FF00', '#80FFFF', '#FF80FF', '#FF8000', '#8000FF', '#8080FF', '#FF0080', '#FF8080', '#80FF80', '#FFD700']

# 定义颜色映射
color_map = {}

# 添加边，并根据不同的v_call_family设置不同颜色的节点文字
for edge in modified_edges:
    parent, child, weight = edge
    weight = str(weight)
    graph.edge(parent, child, label=weight)

    # 检查孩子节点的v_call_family是否已存在于颜色映射中，若不存在则为其分配一个新颜色
    child_v_call_family = child.split('(')[-1].split(')')[0].strip() if '(' in child else None
    if child_v_call_family not in color_map:
        color_map[child_v_call_family] = color_list[len(color_map) % len(color_list)]  # 使用循环的方式从颜色列表中选择颜色

    # 检查父节点的v_call_family是否已存在于颜色映射中，若不存在则为其分配一个新颜色
    parent_v_call_family = parent.split('(')[-1].split(')')[0].strip() if '(' in parent else None
    if parent_v_call_family not in color_map:
        available_colors = [color for color in color_list if color not in color_map.values()]  # 排除已分配的颜色
        if available_colors:
            color_map[parent_v_call_family] = available_colors[0]  # 使用未分配的颜色
        else:
            color_map[parent_v_call_family] = color_list[
                len(color_map) % len(color_list)]  # 如果没有可用颜色，则使用循环的方式从颜色列表中选择颜色

    # 设置节点的颜色
    node_color = color_map[child_v_call_family]
    graph.node(child, fontcolor=node_color)

# 渲染并保存图像
graph.render("/home/mist/ClonalTree/Examples/output1/c1")
print(f"图像已保存为PDF文件：/home/mist/ClonalTree/Examples/output1/c1.pdf")