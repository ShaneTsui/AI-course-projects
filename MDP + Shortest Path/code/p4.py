from collections import defaultdict

import heapq
import os.path
import numpy as np


def load_data(input_file):
    '''
    Read deterministic shortest path specification
    '''
    with np.load(input_file) as data:
        n = data["number_of_nodes"]
        s = data["start_node"]
        t = data["goal_node"]
        C = data["cost_matrix"]
    return n, s, t, C


def plot_graph(C, path_nodes, output_file):
    '''
    Plot a graph with edge weights sepcified in matrix C.
    Saves the output to output_file.
    '''
    from graphviz import Digraph

    G = Digraph(filename=output_file, format='pdf', engine='neato')
    G.attr('node', colorscheme='accent3', color='1', shape='oval', style="filled", label="")

    # Normalize the edge weights to [1,11] to fit the colorscheme
    maxC = np.max(C[np.isfinite(C)])
    minC = np.min(C)
    norC = 10*np.nan_to_num((C-minC)/(maxC-minC))+1

    # Add edges with non-infinite cost to the graph
    for i in range(C.shape[0]):
        for j in range(C.shape[1]):
            if C[i,j] < np.inf:
                G.edge(str(i), str(j), colorscheme="rdylbu11", color="{:d}".format(int(norC[i,j])))

    # Display path
    for n in path_nodes:
        G.node(str(n), str(n), colorscheme='accent3', color='3', shape='oval', style="filled")

    G.view()


def save_results(path, cost, output_file):
    '''
    write the path and cost arrays to a text file
    '''
    with open(output_file, 'w') as fp:
        for i in range(len(path)):
            fp.write('%d ' % path[i])
        fp.write('\n')
        for i in range(len(cost)):
            fp.write('%.2f ' % cost[i])


def dijkstra(start, target, costs):
    parents = {node:node for node in range(costs.shape[0])}
    graph = defaultdict(list)
    heap = [(0, start)]
    min_distance = {node: float('inf') for node in range(costs.shape[0])}
    min_distance[start] = 0

    for s in range(costs.shape[0]):
        for t in range(costs.shape[1]):
            if costs[s][t] != float('inf'):
                graph[s].append((t, costs[s][t]))

    while heap:
        prev_dis, prev_node = heapq.heappop(heap)
        # screen out the old nodes with updated shorter distance
        if prev_dis == min_distance[prev_node]:
            for cur_node, cur_dis in graph[prev_node]:
                dis = prev_dis + cur_dis
                if dis < min_distance[cur_node]:
                    min_distance[cur_node] = dis
                    heapq.heappush(heap, (dis, cur_node))
                    parents[cur_node] = prev_node

    path, costs = [target], [min_distance[target]]
    while parents[target] != target:
        target = parents[target]
        path.append(target)
        costs.append(min_distance[target])

    return np.array(path[::-1]), np.array(costs)


def shortest_path(input_file = '../data/problem1.npz'):

    file_name = os.path.splitext(input_file)[0]

    # Load data
    n,s,t,C = load_data(input_file)

    print(f"start: {s}, end: {t}")

    path, cost = dijkstra(int(s), int(t), C)

    # Visualize (requires: pip install graphviz --user)
    plot_graph(C, path, file_name)

    # print("Floyd cost = ", floyd(int(s), int(t), C))
    # Save the results
    save_results(path, cost, file_name+"_results.txt")


if __name__ == "__main__":

    for idx in range(1, 7):
        shortest_path(f'../data/problem{idx}.npz')