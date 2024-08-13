# from ete3 import Tree
# import pandas as pd
#
# # 读取Newick格式的树文件
# with open('/home/mist/BCR-SORT-master/data/aatree.nk', 'r') as file:
#     newick_str = file.read()
#
# # 创建树对象
# tree = Tree(newick_str)
#
# # 提取节点信息
# data = []
# for node in tree.traverse():
#     node_name = node.name if node.name else "Unnamed_Node"
#     parent_name = node.up.name if node.up and node.up.name else "Root"
#     support = node.support if hasattr(node, 'support') else None
#     data.append([node_name, parent_name, support])
#
#     # 调试输出，检查每个节点的信息
#     print(f"Node: {node_name}, Parent: {parent_name}, Support: {support}")
#
# # 创建DataFrame
# df = pd.DataFrame(data, columns=["Node", "Parent", "Support"])
#
# # 保存为CSV文件
# output_path = '/home/mist/BCR-SORT-master/data/aatree_nodes.csv'
# df.to_csv(output_path, index=False)
#
# print(f"Newick树的节点信息已保存到 {output_path}")
from Bio import Phylo
import pandas as pd

# 读取Newick文件
tree = Phylo.read('/home/mist/BCR-SORT/data/aatree.nk', 'newick')

# 提取树的节点信息
nodes = []
for clade in tree.find_clades(order='level'):
    # 获取节点名称，如果没有则使用一个占位符
    name = clade.name if clade.name else 'Unnamed'
    # 获取父节点
    path = tree.get_path(clade)
    parent = tree.root if clade == tree.root else path[-2].name if len(path) > 1 else None
    nodes.append({'name': name, 'branch_length': clade.branch_length, 'parent': parent})
# for clade in tree.find_clades():
#     if clade.name:
#         nodes.append({'name': clade.name, 'branch_length': clade.branch_length})
# for clade in tree.find_clades(order='level'):
#     if clade.name:
#         parent = tree.root if clade == tree.root else next(tree.get_path(clade)[-2].name)
#         nodes.append({'name': clade.name, 'branch_length': clade.branch_length, 'parent': parent})

# 转换为DataFrame
df = pd.DataFrame(nodes)
# 保存为CSV文件
df.to_csv('/home/mist/BCR-SORT/data/outputaa1.csv', index=False)