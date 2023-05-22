# prend en paramètre en ligne de commande le nom d’une
# instance inst-n-p.txt dans le même dossier
# ,qui trouve la solution optimale
# via la méthode des chemins augmentants
# sans utiliser de solveur externe
# et qui la stock dans un fichier model-n-p.path
import os
import time
from collections import deque

import numpy as np
# from loguru import logger
# from line_profiler_pycharm import profile

def check_minimum_cut(graph, capacities, source, total_flow, optimal):
    ###
    # The min-cut represents the minimum capacity needed to
    # disconnect the source node from the sink node in the graph.
    # The max-flow value obtained from the algorithm will always be
    # equal to the capacity of the min-cut.
    # here
    ###
    # find all nodes reachable from source
    queue = deque([source])
    visited = set()
    while queue:
        current = queue.popleft()
        visited.add(current)
        for neighbor, capacity in enumerate(capacities[current]):
            if capacity > 0 and neighbor not in visited:
                queue.append(neighbor)
    reachable_nodes = visited
    # find all edges with one node in reachable_nodes and the other not
    min_cut_capacity = 0
    for node in reachable_nodes:
        for neighbor in graph[node]:
            if neighbor not in reachable_nodes:
                min_cut_capacity += capacities[neighbor][node]
    if total_flow == min_cut_capacity:
        # logger.info("The solution is optimal!")
        optimal = True
    return optimal

def update(values_matrix, current_node, neighbor, quantity):
    values_matrix[current_node][neighbor] -= quantity
    values_matrix[neighbor][current_node] += quantity
# @profile
def find_augmenting_path(graph, flow, capacities, source, sink):
    # Recherche d'un chemin augmentant dans le graphe
    queue = deque([source])
    visited = set()
    parent = {}
    found_path = False
    # Parcours en largeur jusqu'à trouver le puits ou épuiser tous les chemins
    while queue and not found_path:
        current = queue.popleft()
        visited.add(current)
        for neighbor in graph[current]:
            if capacities[current][neighbor] > 0 and neighbor not in \
                    visited:
                parent[neighbor] = current
                if neighbor == sink:
                    found_path = True
                    break
                queue.append(neighbor)
    # Si un chemin augmentant a été trouvé, le construire et renvoyer son flux maximum
    if found_path:
        # Construire le chemin à partir du puits dans dequeue pour avoir le chemin dans le bon ordre
        path = deque([sink])
        current = sink
        while current != source:
            current = parent[current]
            path.appendleft(current)
        # path_flow est le minimum des capacités des arcs du chemin
        path_flow = min(capacities[path[i]][path[i+1]] for i in range(len(path)-1))
        for i in range(len(path)-1):
            flow[path[i]][path[i+1]] += path_flow
            update(capacities, path[i], path[i+1], path_flow)
        return path_flow
    # Sinon, renvoyer 0 pour indiquer que le flot maximum a été atteint
    else:
        return 0

# @profile
def edmonds_karp(graph, capacities, source, sink, flow,
                 nodes_quantity, arcs_quantity):
    # Trouver un chemin augmentant jusqu'à ce qu'il n'y en ait plus
    total_flow = 0
    optimal = False
    while not optimal:
        path_flow = find_augmenting_path(graph, flow, capacities,
                                         source, sink)
        if path_flow == 0:
            break
        total_flow += path_flow
        optimal = check_minimum_cut(graph, capacities, source,
                                    total_flow, optimal)
    return total_flow

def relabel_to_front(capacities
                     , flow
                     , source
                     , sink
                     , nodes_quantity
                     ):
    height = [0] * nodes_quantity  # height of node
    excess = [0] * nodes_quantity  # flow into node minus flow from node
    seen   = [0] * nodes_quantity  # neighbours seen since last relabel
    # node "queue"
    excess_nodes = [i for i in range(nodes_quantity) if i != source and i != sink]
    def push(u, v):
        send = min(excess[u], capacities[u][v] - flow[u][v])
        update(flow, v, u, send)
        excess[u] -= send
        excess[v] += send


    def relabel(u):
        # Find smallest new height making a push possible,
        # if such a push is possible at all.
        min_height = float('inf')
        for v in range(nodes_quantity):
            if capacities[u][v] - flow[u][v] > 0:
                min_height = min(min_height, height[v])
                height[u] = min_height + 1
    def discharge(u):
        while excess[u] > 0:
            if seen[u] < nodes_quantity:  # check next neighbour
                v = seen[u]
                if capacities[u][v] - flow[u][v] > 0 and height[u] > height[v]:
                    push(u, v)
                    if excess[
                        sink] == 0:  # Check if maximum flow reached
                        return True
                else:
                    seen[u] += 1
            else:  # we have checked all neighbours. must relabel
                relabel(u)
                seen[u] = 0

    height[source] = nodes_quantity  # longest path from source to sink is less than n long
    excess[source] = float('inf')
    # send as much flow as possible to source neighbours
    for v in range(nodes_quantity):
        push(source, v)

    p = 0
    while p < len(excess_nodes):
        u = excess_nodes[p]
        old_height = height[u]
        # discharge(u)
        max_flow_reached = discharge(u)
        if max_flow_reached:
            break  # Terminate the loop if maximum flow reached

        if height[u] > old_height:
            excess_nodes.insert(0, excess_nodes.pop(p))  # move to front of list
            p = 0  # start from front of list
        else:
            p += 1

    return sum(flow[source])

def read_file(file_path):
    # Read input origin_file
    with open(file_path, 'r') as f:
        lines = f.readlines()
    # Parse input origin_file
    nodes_quantity = int(lines[0].split()[1])
    source = int(lines[1].split()[1])
    sink = int(lines[2].split()[1])
    arcs_quantity = int(lines[3].split()[1])
    arcs_data = [tuple(map(int, line.split())) for line in lines[4:]]

    return nodes_quantity, source, sink, arcs_quantity, arcs_data

def solve_max_flow_augmenting_paths(file_path, algorithm=None):
    nodes_quantity, source, sink, arcs_quantity, arcs_data = read_file(file_path)

    # Initialize graph and values_matrix matrices
    graph = [[] for _ in range(nodes_quantity)]
    capacities = np.zeros((nodes_quantity, nodes_quantity))
    flow = np.zeros((nodes_quantity, nodes_quantity))

    for arc in arcs_data:
        # Add arcs to the graph
        graph[arc[0]].append(arc[1])
        graph[arc[1]].append(arc[0])
        # Set capacities
        capacities[arc[0], arc[1]] = arc[2]
    # GraphVisualizer(graph).draw()

    # Compute max flow
    if algorithm == 'edmonds_karp':
        max_flow = edmonds_karp(graph
                                , capacities
                                , source
                                , sink
                                , flow
                                , nodes_quantity
                                , arcs_quantity
                                )

    elif algorithm == 'relabel_to_front':
        max_flow = relabel_to_front(capacities
                                    , flow
                                    , source
                                    , sink
                                    , nodes_quantity
                                    )

    else:
        if nodes_quantity < arcs_quantity:
            print('Using O(V²E) relabel_to_front algorithm')
            algorithm = 'relabel_to_front'
            max_flow = relabel_to_front(capacities
                                        , flow
                                        , source
                                        , sink
                                        , nodes_quantity
                                        )
        else:
            # sparse graph
            print('Using O(VE²) edmonds_karp algorithm')
            algorithm = 'edmonds_karp'
            max_flow = edmonds_karp(graph
                                    , capacities
                                    , source
                                    , sink
                                    , flow
                                    , nodes_quantity
                                    , arcs_quantity
                                    )
    max_flow = int(max_flow)
    print(f'file_path: {file_path}')
    print(f'Max flow: {max_flow}')

    write_output_file_to(algorithm, file_path, nodes_quantity
                         , max_flow, flow)


# @profile
def write_output_file_to(path, file_path, nodes, max_flow, flow):
    if not os.path.exists(path):
        # logger.debug(f'Creating directory {path}')
        os.makedirs(path)
    with open(path + "/" + file_path.replace('Instances',
                                             'Path').replace('inst',
                                                             'model').replace('.txt',
                                                                              '.path'), 'w') as f:
        # logger.info(f'{max_flow}\n')
        f.write(f'{max_flow}\n')
        lines = []
        for i in range(nodes):
            for j in range(nodes):
                if flow[i][j] > 0:
                    lines.append(f'{i} {j} {int(flow[i][j])}\n')
        f.writelines(lines)
if __name__ == '__main__':
    import sys

    log_option = False

    if 1 == len(sys.argv):
        print("Usage: python chemin_augmentant.py file_path <algorithm>")
        instance = "inst-2-0.25.txt"
        # instance = "inst-3-0.2.txt"
        # instance = "inst-3-0.22.txt"
        # instance = "inst-3-0.3.txt"
        # instance = "inst-4-0.25.txt"
        instance = "inst-100-0.1.txt"
        # instance = "inst-100-0.2.txt"
        # instance = "inst-500-0.1.txt"

        algorithm = None
        # algorithm = "distance"
        # algorithm = "edmonds_karp"
        # algorithm = "relabel_to_front"
        log_option = True
        # log_option = False
        # print("Example: "
        #       "python chemin_augmentant.py {0} {1}".format(instance,
        #                                                    algorithm))
        print("Example: "
              "python chemin_augmentant.py {0} {1}".format(instance,
                                                           algorithm))
        print("Result: ")

    elif 2 < len(sys.argv):
            instance = os.path.join(sys.argv[1])
            algorithm = sys.argv[2]
            if "debug" in sys.argv:
                log_option = True

    print(f'instance: {instance}')
    # if not log_option:
        # logger.info("logger debug mode disabled")
        # logger.remove()

    # timer
    start_time = time.time()
    solve_max_flow_augmenting_paths(instance, algorithm)
    end_time = time.time()
    print(f'Execution time: {end_time - start_time} seconds')
