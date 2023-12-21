#Attention Mechanism -> 
#Communication phase in Transformer
#communication phase -> Multi-Headed Attention
#computation phas -> Multi layer perceptron


#This code basically simulates the one round of communication as a message passing scheme using 
#  a directed graph
import numpy as np
import matplotlib.pyplot as plt
#Node class that stores whole data
class Node:
    def __init__(self):
        #private information that is stored in individual node
        self.data = np.random.rand(5)
        #weights governing how this node interacts with other nodes
        #key: what are the things that I am looking for
        self.wkey = np.random.rand(5, 5)
        #query: what are the things that I have
        self.wquery = np.random.rand(5, 5)
        #value: what are the things that I want to communicate
        self.wvalue = np.random.rand(5, 5)

    def key(self):
        return self.wkey @ self.data
    def query(self):
        return self.wquery @ self.data
    def value(self):
        return self.wvalue @ self.data


class Graph:
    def __init__(self):
        #making 10 nodes
        self.nodes = [Node() for _ in range(10)]
        for node in self.nodes:
            print(node.data) 
        # 람다로 만든 함수는 그래프의 노드 중 하나를 무작위로 선택할 때 사용됩니다
        randi = lambda: np.random.randint(len(self.nodes))
        # 튜플리스트로 40개의 edge들이 형성됨
        self.edges = [(randi(), randi()) for _ in range(40)]

    def run(self):
        updates = []
        attention_weights = [[] for _ in range(len(self.nodes))]  # 각 노드별로 빈 리스트를 생성
        for i, n in enumerate(self.nodes):
            q = n.query()
            inputs = [self.nodes[ifrom] for ifrom, ito in self.edges if ito == i]
            if len(inputs) == 0:
                continue
            keys = [m.key() for m in inputs]
            scores = [np.dot(k, q) for k in keys]
            scores = np.exp(scores) / sum(np.exp(scores))
            attention_weights.append(scores)  # 가중치를 저장
            
            values = [m.value() for m in inputs]
            update = sum([s * v for s, v in zip(scores, values)])
            updates.append(update)
        
        for n, u in zip(self.nodes, updates):
            n.data = n.data + u  # Residual connection
        return attention_weights  # 어텐션 가중치를 반환
# 사용 예시
graph = Graph()
attention_weights = graph.run()
print("attention weights is ")
print(attention_weights)


def visualize_graph(graph, attention_weights):
    fig, ax = plt.subplots()

    # Plot nodes
    for node in graph.nodes:
        ax.plot(node.data[0], node.data[1], 'bo')  # Blue dots for nodes

    # Plot edges
    for edge_idx, (from_idx, to_idx) in enumerate(graph.edges):
        from_node = graph.nodes[from_idx]
        to_node = graph.nodes[to_idx]
        # Check if attention_weights is not empty and has weights for this edge
        if len(attention_weights) > edge_idx and len(attention_weights[edge_idx]) > 0:
            weight = np.mean(attention_weights[edge_idx])
            # Ensure there's at least a minimum line width
            line_width = max(weight * 10, 0.1)
        else:
            line_width = 0.1  # Default line width
        # Draw the edge
        ax.plot([from_node.data[0], to_node.data[0]], 
                [from_node.data[1], to_node.data[1]], 
                'k-', 
                linewidth=line_width)  # 'k-' for black lines
    ax.set_xlabel('Data Dimension 1')
    ax.set_ylabel('Data Dimension 2')
    plt.show()
# Run the visualization after running the graph
visualize_graph(graph, attention_weights)


# As a result we can see that 10 nodes were plotted and 