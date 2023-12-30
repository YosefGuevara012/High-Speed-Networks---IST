paths.py
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 14:11:53 2022

@author: Jose Ribeiro
"""


import numpy as np
import copy
import itertools



# Class to represent a graph
class Graph:

    # A utility function to find the
    # vertex with minimum dist value, from
    # the set of vertices still in queue
    def minDistance(self, dist, queue):
        # Initialize min value and min_index as -1
        minimum = float("Inf")
        min_index = -1

        # from the dist array,pick one which
        # has min value and is till in queue
        for i in range(len(dist)):
            if dist[i] < minimum and i in queue:
                minimum = dist[i]
                min_index = i
        return min_index

    # Function to print shortest path
    # from source to j
    # using parent array
    def printPath(self, parent, j):
        path = []
        # Base Case : If j is source
        if parent[j] == -1:
            # print (j+1)
            return [j + 1]

        path.extend(self.printPath(parent, parent[j]))
        # print (j+1)
        path.append(j + 1)
        return path

        # A utility function to print

    # the constructed distance
    # array
    def printSolution(self, src, dist, parent):
        paths = []
        # print("Vertex \t\tDistance from Source\tPath")
        for i in range(0, len(dist)):
            # print("\n%d --> %d \t\t%d \t\t\t\t\t" % (src+1, i+1, dist[i])),
            path = self.printPath(parent, i)
            paths.append({
                "source": src + 1,
                "destination": i + 1,
                "distance": dist[i],
                "path": path
            })
        return paths

    '''Function that implements Dijkstra's single source shortest path
    algorithm for a graph represented using adjacency matrix
    representation'''

    def dijkstra(self, graph, src):

        row = len(graph)
        col = len(graph[0])

        # The output array. dist[i] will hold
        # the shortest distance from src to i
        # Initialize all distances as INFINITE
        dist = [float("Inf")] * row

        # Parent array to store
        # shortest path tree
        parent = [-1] * row

        # Distance of source vertex
        # from itself is always 0
        dist[src] = 0

        # Add all vertices in queue
        queue = []
        for i in range(row):
            queue.append(i)

        # Find shortest path for all vertices
        while queue:

            # Pick the minimum dist vertex
            # from the set of vertices
            # still in queue
            u = self.minDistance(dist, queue)

            # remove min element
            if u != -1:
                queue.remove(u)

                # Update dist value and parent
                # index of the adjacent vertices of
                # the picked vertex. Consider only
                # those vertices which are still in
                # queue
                for i in range(col):
                    '''Update dist[i] only if it is in queue, there is
                    an edge from u to i, and total weight of path from
                    src to i through u is smaller than current value of
                    dist[i]'''
                    if graph[u][i] and i in queue:
                        if dist[u] + graph[u][i] < dist[i]:
                            dist[i] = dist[u] + graph[u][i]
                            parent[i] = u
            else:
                queue.clear()

        # print the constructed distance array
        return self.printSolution(src, dist, parent)


def getPaths(graph: Graph, matrix: list):
    patths = []
    # Print the solution
    for i in range(len(matrix)):
        patths.append(graph.dijkstra(matrix, i))

    return patths


def shortestPaths(graph: Graph, matrix: list):
    pairs = []
    paths = []

    count = min([len([i for i in row if i > 0]) for row in matrix])

    for i in range(0, len(matrix)):
        for j in range(i, len(matrix)):
            if matrix[i][j] != 0:
                pairs.append(f'{i + 1}-{j + 1}')

    for i in range(len(matrix)):
        if i > count:
            break
        combinations = list(itertools.combinations(pairs, i))

        for comb in combinations:
            aux_matrix = copy.deepcopy(matrix)

            for pair in comb:
                row, column = pair.split('-')
                aux_matrix[int(row) - 1][int(column) - 1] = 0
                aux_matrix[int(column) - 1][int(row) - 1] = 0

            # print(f'Pair removed: {comb}')
            # for row in range(len(np.array(aux_matrix))):
            # print(aux_matrix[row])

            if not (~np.array(aux_matrix).any(axis=0)).any():
                aux_paths = getPaths(graph, aux_matrix)

                if len(paths) == 0:
                    paths = aux_paths
                    for path in paths:
                        for p in path:
                            p["path"] = [p["path"]]
                else:
                    for a, path in enumerate(paths):
                        for b, p in enumerate(path):
                            if aux_paths[a][b]["distance"] == p["distance"] and aux_paths[a][b]["path"] not in p[
                                    "path"]:
                                p["path"].append(aux_paths[a][b]["path"])
    return paths

def countHops(paths: list):
    hops_matrix = []

    for path in paths:
        hops_row = []
        for p in path:
            hops_row.append(len(min(p["path"], key=len)) - 1)
        hops_matrix.append(hops_row)

    return hops_matrix



def create_traffic_matrix(matrix, traffic):
    matrix_size = len(matrix)
    if traffic == None or len(traffic) != matrix_size:
        a = np.ones((matrix_size, matrix_size), int)
        np.fill_diagonal(a, 0)
        return a.tolist()
    else:
        return traffic
    
def countConects(paths: list, traffic_matrix: list, order="Shortest"):
    orderpath = []
    auxOrderPath = []

    for path in paths:
        for p in path:
            for i in range(0, len(p["path"])):
                oxy = p["path"].copy()
                oxy[0], oxy[i] = oxy[i], oxy[0]
                auxOrderPath.append(oxy)
    if order == "longest":
        # auxOrderPath.sort(key=lambda x: len(x[0]), reverse=True)
        auxOrderPath = sorted(auxOrderPath, key=lambda x: (
            len(x[0]), -len(x)), reverse=True)
    else:
        # auxOrderPath.sort(key=lambda x: len(x[0]))
        auxOrderPath = sorted(auxOrderPath, key=lambda x: (len(x[0]), len(x)))
    minPaths = len(auxOrderPath[0])
    maxlenght = len(auxOrderPath[0][0])

    while auxOrderPath:
        if order == "longest" and len(auxOrderPath[0][0]) < maxlenght:
            maxlenght = len(auxOrderPath[0][0])
            minPaths = len(auxOrderPath[0])
        elif order != "longest" and len(auxOrderPath[0][0]) > maxlenght:
            maxlenght = len(auxOrderPath[0][0])
            minPaths = len(auxOrderPath[0])
        if maxlenght == 1:
            auxOrderPath.pop(0)
            continue
        aux = [a for a in auxOrderPath if len(
            a) == minPaths and len(a[0]) == maxlenght]
        for a in aux:
            orderpath.append(a)
            auxOrderPath.remove(a)
        minPaths += 1

    # print (orderpath)
    new_order_path = []

    if order != "largest":
        for i, path in enumerate(orderpath):
            start_node = path[0][0]
            end_node = path[0][-1]
            for j in range(traffic_matrix[start_node - 1][end_node - 1]):
                new_order_path.append(path)
    else:
        aux_traffic_matrix = copy.deepcopy(traffic_matrix)
        while np.array(aux_traffic_matrix).max() > 0:
            a = np.array(aux_traffic_matrix)
            (start_node, end_node) = np.unravel_index(
                np.argmax(a, axis=None), a.shape)
            for path in orderpath:
                if path[0][0] == start_node+1 and path[0][-1] == end_node+1:
                    for j in range(aux_traffic_matrix[start_node][end_node]):
                        new_order_path.append(path)
                    aux_traffic_matrix[start_node][end_node] = 0
                    break

    pairs = {}
    paths_counter = {}
    pairs_removed = []
    for pairs_list in new_order_path:
        if len(pairs_list) == 1:
            path_key = "-".join(map(str, pairs_list[0]))
            if path_key not in paths_counter.keys():
                paths_counter[path_key] = 1
            else:
                paths_counter[path_key] += 1
            for i in range(len(pairs_list[0])):
                if i == len(pairs_list[0]) - 1:
                    break
                pair = f"{pairs_list[0][i]}-{pairs_list[0][i + 1]}"
                
                if pair not in pairs.keys():
                    pairs[pair] = 1
                else:
                    pairs[pair] += 1
                    
                
        else:
            if pairs_list[0] in pairs_removed:
                continue
            aux_pairs = pairs.copy()
            aux_paths_counter = paths_counter.copy()
            flag = True
            
            path_key = "-".join(map(str, pairs_list[0]))
            if path_key not in aux_paths_counter.keys():
                aux_paths_counter[path_key] = 1
            else:
                aux_paths_counter[path_key] += 1

            # calcula as ligacoes do primeiro path
            for i in range(len(pairs_list[0])):
                if i == len(pairs_list[0]) - 1:
                    break
                pair = f"{pairs_list[0][i]}-{pairs_list[0][i + 1]}"
                
                if pair not in aux_pairs.keys():
                    aux_pairs[pair] = 1
                else:
                    aux_pairs[pair] += 1
                    
                
            max_weight = max([n for n in aux_pairs.values()])

            list_indices = [0]

            for indice, p in enumerate(pairs_list[1:]):  # compara
                aux_pairs2 = pairs.copy()
                aux_paths_counter2 = paths_counter.copy()
                path_key = "-".join(map(str, p))
                
                if path_key not in aux_paths_counter2.keys():
                    aux_paths_counter2[path_key] = 1
                else:
                    aux_paths_counter2[path_key] += 1
                    
                for i in range(len(p)):
                    if i == len(p) - 1:
                        break
                    pair = f"{p[i]}-{p[i + 1]}"
                    
                    if pair not in aux_pairs2.keys():
                        aux_pairs2[pair] = 1
                    else:
                        aux_pairs2[pair] += 1

                if max([n for n in aux_pairs2.values()]) < max_weight:
                    flag = False
                    break
                elif max([n for n in aux_pairs2.values()]) == max_weight:
                    list_indices.append(indice + 1)
            if flag:
                pairs_lenght = [len(pairs_list[k]) for k in list_indices]
                if len(pairs_list[0]) == min(pairs_lenght):
                    pairs = aux_pairs.copy()
                    paths_counter = aux_paths_counter.copy()
                    for k in list_indices[1:]:
                        pairs_removed.append(pairs_list[k])

    return pairs, paths_counter
    
    
    
    


graph = Graph()
average_node_degree = []
traffic = []
number_of_hops_per_demand = []
diameter = []



#matrix = [[0,315.98,226.07,0,0,0,0,0,0,0,0,0],[315.98,0,334.40,0,0,0,0,253.07,425.25,0,0,0],[226.07,334.40,0,188.81,238.98,261.66,274.08,0,0,0,0,0],[0,0,188.81,0,192.83,0,0,0,0,0,0,0],[0,0,238.98,192.83,0,0,224.10,0,0,0,0,0],[0,0,261.66,0,0,0,51.73,0,0,0,0,0],[0,0,274.08,0,224.10,51.73,0,0,0,330.72,0,0],[0,253.07,0,0,0,0,0,0,208.90,0,0,0],[0,425.25,0,0,0,0,0,208.90,0,173.75,207.25,378.51],[0,0,0,0,0,0,330.72,0,173.75,0,136.94,212.79],[0,0,0,0,0,0,0,0,207.25,136.94,0,0],[0,0,0,0,0,0,0,0,378.51,212.79,0,0]]
matrix = [[0,1,1,0,0,0,0,0,0,0,0,0],[1,0,1,0,0,0,0,1,1,0,0,0],[1,1,0,1,1,1,1,0,0,0,0,0],[0,0,1,0,1,0,0,0,0,0,0,0],[0,0,1,1,0,0,1,0,0,0,0,0],[0,0,1,0,0,0,1,0,0,0,0,0],[0,0,1,0,1,1,0,0,0,1,0,0],[0,1,0,0,0,0,0,0,1,0,0,0],[0,1,0,0,0,0,0,1,0,1,1,1],[0,0,0,0,0,0,1,0,1,0,1,1],[0,0,0,0,0,0,0,0,1,1,0,0],[0,0,0,0,0,0,0,0,1,1,0,0]]


paths = shortestPaths(graph, matrix)
hop_matrix = countHops(paths)
average_node_degree.append(np.count_nonzero(matrix)/len(matrix))
traffic = create_traffic_matrix(matrix, None)
number_of_hops_per_demand.append(np.matrix(hop_matrix).sum() / np.matrix(traffic).sum())
diameter.append(np.matrix(hop_matrix).max())
ligacoes, paths_counter_shortest = countConects(paths, traffic)

print(type(hop_matrix))

import csv
file = open('Paths.csv', 'w+', newline ='')
 
# writing the data into the file
with file:   
    write = csv.writer(file)
    write.writerows(paths)