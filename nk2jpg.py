# import subprocess
# from PIL import Image
#
#
# def render_nuke_to_image(nk_file, output_jpg):
#     # 使用 Nuke 的命令行模式来执行渲染
#     cmd = f"nuke -t -xi -V 2 -x {nk_file}"
#     subprocess.run(cmd, shell=True)
#
#     # 读取渲染结果并保存为 .jpg
#     img = Image.open("output.exr")
#     img.save(output_jpg, "JPEG")
#
#     # 删除临时文件
#     subprocess.run("rm output.exr", shell=True)
# # 使用示例
# nk_file = "/home/mist/ClonalTree/Examples/output/clonalTree.abRT.nk"
# output_jpg = "/home/mist/ClonalTree/Examples/output/output.jpg"
# render_nuke_to_image(nk_file, output_jpg)

# from PIL import Image
#
# nk_file_path = "/home/mist/ClonalTree/Examples/output/clonalTree.abRT.nk"
# jpg_file_path = "/home/mist/ClonalTree/Examples/output/output.jpg"
#
# # Open the .nk file
# image = Image.open(nk_file_path)
#
# # Save the opened image as a .jpg file
# image.save(jpg_file_path, "JPEG")

# import pandas as pd
# import graphviz
#
# # 读取CSV数据
# # data = pd.read_csv("/home/mist/ClonalTree/Examples/output/clonalTree.abRT.nk.csv", delimiter='\t')
#
# parent = pd.read_csv("/home/mist/ClonalTree/Examples/output/clonalTree.abRT.nk.csv",sep=',',header='infer',usecols=[0])
# child = pd.read_csv("/home/mist/ClonalTree/Examples/output/clonalTree.abRT.nk.csv",sep=',',header='infer',usecols=[1])
# distance = pd.read_csv("/home/mist/ClonalTree/Examples/output/clonalTree.abRT.nk.csv",sep=',',header='infer',usecols=[2])
# # 创建一个有向图
# graph = graphviz.Digraph(format='png')
# parent1=
# # 根据CSV数据创建二叉树
# for i in len(parent):
#     parent1 = parent[i]
#     child1 = child[i]
#     graph.edge(parent1, child1)
#
# # 保存图像
# graph.render("/home/mist/ClonalTree/Examples/output/binary_tree", view=True)


#20230308这个能用但是会出现AttributeError: 'PosixPath' object has no attribute 'endswith'
import csv
# import graphviz
# import pandas as pd
#
# filepath = "/home/mist/ClonalTree/Examples/output1/Kaminski_TP9.abRT.nk.csv"
# df1= pd.read_csv("/home/mist/ClonalTree/Examples/output1/Kaminski_TP9.abRT.nk.csv",header=None)
# dfl=df1.values.tolist()#转list
# # with open(filepath) as csvfile:
# #     reader = csv.DictReader(csvfile,header=None)
# #     edges = [(row['parent'], row['child'], row['distance']) for row in reader]
#
# graph = graphviz.Digraph(format='png')
#
# for edge in dfl:
#     parent, child, weight = edge
#     weight = str(weight)
#     graph.edge(parent, child, label=weight)
#
# graph.render("/home/mist/ClonalTree/Examples/output/Kaminski_TP9", view=True)

# #20240308这个好用但是显示不太清楚
# from Bio import Phylo
# # 读取Newick格式的树文件
# tree_file = "/home/mist/BCR-SORT-master/data/aatree.nk"
# tree = Phylo.read(tree_file, "newick")
# # 显示树
# Phylo.draw(tree)
