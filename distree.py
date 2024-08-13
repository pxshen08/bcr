import numpy as np
import torch
import pandas as pd

class DisTree:
    class Node:
        def __init__(self):
            self.next = []  # 存储孩子节点的索引
            self.weights = []  # 存储与孩子节点对应的边的权重
            self.nums = 0

    def get_index(self, target_node):
        for index, (node_id, node) in enumerate(self.nodes.items()):
            if node_id == target_node:
                return index
        return None

    def __init__(self):
        self.nodes = {}  # 使用字典存储节点信息
        self.dp = {}  # 使用字典存储动态规划数组，记录每个节点到其它节点的距离之和
        self.counts = {}  # 使用字典存储节点的子树节点数
        self.ans = {}  # 使用字典存储结果矩阵

    def dfs(self, father, current_node, target_node):
        # 检查查询的节点是否存在
        if current_node not in self.nodes:
            return -1
        # 这里修改为只计算从节点 3 到目标节点 0 的距离
        if current_node == target_node:
            return 0

        result = float('inf')
        for i, (son_node, weight) in enumerate(zip(self.nodes[current_node].next, self.nodes[current_node].weights)):
            if son_node != father:
                distance = self.dfs(current_node, son_node, target_node)
                if distance != -1:
                    result = min(result, distance + weight)

        if result == float('inf'):
            return -1
        return result

    def dfs3(self, father, current_node, target_node):
        # 检查查询的节点是否存在
        if current_node not in self.nodes:
            return -1
        # 这里修改为只计算从节点 3 到目标节点 0 的距离
        if current_node == target_node:
            return 0

        result = float('inf')
        for i, (son_node, weight) in enumerate(zip(self.nodes[current_node].next, self.nodes[current_node].weights)):
            if son_node != father:
                distance = self.dfs(current_node, son_node, target_node)
                if distance != -1:
                    result = min(result, distance + weight)

        if result == float('inf'):
            return -1
        return result

    def sumOfDistancesInTree(self, edges, source_node, target_node):
        # if n == 1:#没用
        #     return 0

        # 构建树结构
        for x, y, w in edges:
            if x not in self.nodes:
                self.nodes[x] = self.Node()  # 如果节点 x 不存在，则创建一个新的节点对象
            if y not in self.nodes:
                self.nodes[y] = self.Node()  # 如果节点 y 不存在，则创建一个新的节点对象

            self.nodes[x].nums += 1
            self.nodes[x].next.append(y)
            self.nodes[x].weights.append(w)
            self.nodes[y].nums += 1
            self.nodes[y].next.append(x)
            self.nodes[y].weights.append(w)

        # 使用 DFS 计算距离
        return self.dfs(-1, source_node, target_node)

    def dfs1(self, father, current_node):
        # 第一次遍历，计算每个节点到其它节点的距离之和
        self.dp[current_node] = 0
        self.counts[current_node] = 1
        for i, son_node in enumerate(self.nodes[current_node].next):
            if son_node != father:
                self.dfs1(current_node, son_node)
                # 更新距离之和
                self.dp[current_node] += self.dp[son_node] + self.counts[son_node] * self.nodes[current_node].weights[i]
                # 更新子树节点数
                self.counts[current_node] += self.counts[son_node]

    def dfs2(self, oldroot, current_root):
        # 第二次遍历，更新结果
        self.ans[current_root] = self.dp[current_root]
        for i, next_root in enumerate(self.nodes[current_root].next):
            if next_root != oldroot:
                a = self.dp[current_root]
                b = self.dp[next_root]
                c = self.counts[current_root]
                d = self.counts[next_root]
                # 交换节点，更新距离之和和节点数
                self.dp[current_root] = self.dp[current_root] - self.dp[next_root] - self.counts[next_root] * \
                                        self.nodes[current_root].weights[i]
                self.counts[current_root] -= self.counts[next_root]
                self.dp[next_root] = self.dp[next_root] + self.dp[current_root] + self.counts[current_root] * \
                                     self.nodes[current_root].weights[i]
                self.counts[next_root] += self.counts[current_root]

                self.dfs2(current_root, next_root)

                # 回溯
                self.dp[current_root] = a
                self.dp[next_root] = b
                self.counts[current_root] = c
                self.counts[next_root] = d

    def sumallOfDistancesInTree(self, n, edges):
        self.dp = {}  # 使用字典初始化 self.dp
        self.counts = {}  # 使用字典初始化 self.counts
        self.ans = {}

        if n == 1:
            return [0]

        # 构建树结构
        for x, y, w in edges:
            if x not in self.nodes:
                self.nodes[x] = self.Node()  # 如果节点 x 不存在，则创建一个新的节点对象
            if y not in self.nodes:
                self.nodes[y] = self.Node()  # 如果节点 y 不存在，则创建一个新的节点对象

            self.nodes[x].nums += 1
            self.nodes[x].next.append(y)
            self.nodes[x].weights.append(w)
            self.nodes[y].nums += 1
            self.nodes[y].next.append(x)
            self.nodes[y].weights.append(w)

        # 第一次遍历，计算距离之和
        self.dfs1(-1, list(self.nodes.keys())[0])  # 从第一个节点开始遍历
        # 第二次遍历，更新结果
        self.dfs2(-1, list(self.nodes.keys())[0])  # 从第一个节点开始遍历

        return self.ans

    def sumoneOfDistancesInTree(self, n, edges):
        # 构建树结构
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(torch.cuda.is_available())

        # # Move data to GPU
        # # Convert string numbers to float
        # data = [
        #     [float(item) if isinstance(item, str) and item.replace('.', '', 1).isdigit() else item for item in sublist]
        #     for sublist in edges]
        #
        # # Convert to torch tensor
        # tensor_data = torch.tensor(data)
        # print(tensor_data)

        for x, y, w in edges:
            if x not in self.nodes:
                self.nodes[x] = self.Node()  # 如果节点 x 不存在，则创建一个新的节点对象
            if y not in self.nodes:
                self.nodes[y] = self.Node()  # 如果节点 y 不存在，则创建一个新的节点对象

            self.nodes[x].nums += 1
            self.nodes[x].next.append(y)
            self.nodes[x].weights.append(w)
            self.nodes[y].nums += 1
            self.nodes[y].next.append(x)
            self.nodes[y].weights.append(w)

        # 初始化结果矩阵
        for i in range(n):
            self.ans[i] = {}

        # 使用 DFS 计算距离
        for source_node in self.nodes.keys():
            for target_node in self.nodes.keys():
                self.ans[self.get_index(source_node)][self.get_index(target_node)] = self.dfs3(-1, source_node, target_node)
        # distance_matrix = np.array([[self.ans[i][j] for j in range(6)] for i in range(6)])
        distance_matrix = np.array([[self.ans[i].get(j, np.inf) for j in range(6)] for i in range(6)])

        print(distance_matrix)
        return distance_matrix


# 测试代码
if __name__ == "__main__":
    solution = DisTree()
    n = 6
    edges = [['seq0', 'seq1', 2.0], ['seq0', 'seq2', 3.0], ['seq2', 'seq3', 4.0], ['seq2', 'seq4', 5.0], ['seq2', 'seq5', 6.0]]
    source_node = 'seq2'
    target_node = 'seq5'
    # filename = "/home/mist/ClonalTree/Examples/cdr3l1/ellebedy.abRT.nk.csv"
    # tree_graph = pd.read_csv(filename, header=None, names=['source', 'target', 'weight'])
    # tree = tree_graph.values.tolist()
    # tree_distance_data = DisTree().sumoneOfDistancesInTree(len(y), tree)
    print(solution.sumOfDistancesInTree(edges, source_node, target_node))
    # #print(solution.sumallOfDistancesInTree(n,edges))
    # print(solution.sumoneOfDistancesInTree(n, edges))
