# prend en paramètre en ligne de commande le nom d’une
# instance inst-n-p.txt dans le même dossier
# ,qui trouve la solution optimale
# via la méthode des chemins augmentants
# sans utiliser de solveur externe
# et qui la stock dans un fichier model-n-p.path
import os
import time
from collections import deque

from loguru import logger
from graph_visualizer import GraphVisualizer

def check_minimum_cut():
    pass # TODO
def find_augmenting_path(graph, capacities, source, sink):
    logger.debug(f'find_augmenting_path(graph, capacities, {source}, {sink})')
    # logger.debug(f'Finding augmenting path from {source} to {sink}')
    # Recherche d'un chemin augmentant dans le graphe
    queue = deque([source])
    visited = set()
    parent = {}
    found_path = False

    # Parcours en largeur jusqu'à trouver le puits ou épuiser tous les chemins
    while queue and not found_path:
        logger.debug(f'queue: {queue}')
        logger.debug(f'visited: {visited}')
        current = queue.popleft()
        logger.debug(f'current: {current}')
        visited.add(current)
        for neighbor, capacity in enumerate(capacities[current]):
            # todo : choose max capacity neighbor
            if capacity > 0 and neighbor not in visited:
                parent[neighbor] = current
                if neighbor == sink:
                    found_path = True
                    break
                queue.append(neighbor)

    # Si un chemin augmentant a été trouvé, le construire et renvoyer son flux maximum
    if found_path:
        path = [sink]
        current = sink
        while current != source:
            current = parent[current]
            path.append(current)
        # Le chemin est construit à l'envers, le retourner
        path.reverse()
        logger.debug(f'Augmenting path: {path}')
        # max_flow est le minimum des capacités des arcs du chemin
        max_flow = min(capacities[path[i]][path[i+1]] for i in range(len(path)-1))
        logger.debug(f'Max flow: {max_flow}')
        for i in range(len(path)-1):
            logger.debug(f'Updating capacities[{path[i]}][{path[i+1]}] from '
                         f'{capacities[path[i]][path[i+1]]} to '
                         f'{capacities[path[i]][path[i+1]] - max_flow}')
            capacities[path[i]][path[i+1]] -= max_flow
            capacities[path[i+1]][path[i]] += max_flow
        return max_flow

    # Sinon, renvoyer 0 pour indiquer que le flot maximum a été atteint
    else:
        return 0

def max_flow_edmonds_karp(graph, capacities, source, sink):
    # Trouver un chemin augmentant jusqu'à ce qu'il n'y en ait plus
    total_flow = 0
    while True:
        flow = find_augmenting_path(graph, capacities, source, sink)
        if flow == 0:
            break
        total_flow += flow
    return total_flow


def distance_directed_augmenting_path(graph, capacities, source, sink):
    nodes_quantity = len(graph)
    heights = [0] * nodes_quantity    # Heights of nodes in the residual graph
    excess_flow = [0] * nodes_quantity    # Excess flow at each node
    flow = [[0] * nodes_quantity for _ in range(nodes_quantity)]    # Flow matrix
    distance = [0] * nodes_quantity    # Distance labels for nodes

    logger.debug(f'graph: {graph}')
    logger.debug(f'capacities: {capacities}')
    logger.debug(f'source: {source}')
    logger.debug(f'sink: {sink}')
    logger.debug(f'nodes_quantity: {nodes_quantity}')
    logger.debug(f'heights: {heights}')
    logger.debug(f'excess_flow: {excess_flow}')
    logger.debug(f'flow: {flow}')
    logger.debug(f'distance: {distance}')

    # Initialize source preflow
    logger.debug(f'Initialize source preflow on graph {graph}')
    heights[source] = nodes_quantity    # Set height of source node to the number of nodes
    excess_flow[source] = float('inf')    # Set excess flow at source
    # node to infinity
    logger.debug(f'heights: {heights}')
    logger.debug(f'excess_flow: {excess_flow}')

    for neighbor in graph[source]:
        capacity = capacities[source][neighbor]
        logger.debug(f'neighbor: {neighbor}')
        logger.debug(f'capacity from {source} to {neighbor}: {capacity}')
        # if capacity > 0:
        flow[source][neighbor] = capacity    # Initialize flow from source to its neighbors
        # Initialize flow from neighbors to source (negative)
        # flow[neighbor][source] = -capacity
        flow[neighbor][source] -= capacity
        excess_flow[neighbor] = capacity    # Set excess flow at neighbors to their capacity

    def push_flow(u, v):
        logger.debug(f'push_flow({u}, {v})')
        logger.debug(f'flow: {flow}')
        # Push flow from node u to node v
        logger.debug(f'excess_flow[u]: {excess_flow[u]}')
        logger.debug(f'capacities[u][v]: {capacities[u][v]}')
        logger.debug(f'flow[u][v]: {flow[u][v]}')
        delta = min(excess_flow[u], capacities[u][v] - flow[u][v])    # Compute the maximum amount of flow that can be pushed
        logger.debug(f'delta: {delta}')
        flow[u][v] += delta    # Increase the flow from u to v
        flow[v][u] -= delta    # Decrease the flow from v to u (negative flow)
        excess_flow[u] -= delta    # Update the excess flow at node u
        excess_flow[v] += delta    # Update the excess flow at node v
        logger.debug(f'flow: {flow}')

    def relabel_node(u):
        logger.debug(f'relabel_node({u})')
        logger.debug(f'heights: {heights}')
        # Relabel the height of node u to signify
        min_height = float('inf')    # Initialize the minimum height to infinity
        for v in range(nodes_quantity):
            logger.debug(f'capacities[{u}][{v}] : {capacities[u][v]}')
            logger.debug(f'flow[{u}][{v}] : {flow[u][v]}')
            if capacities[u][v] - flow[u][v] > 0:
                logger.debug(f'min_height: {min_height}')
                logger.debug(f'heights[{v}] : {heights[v]}')

                min_height = min(min_height, heights[v])    # Find the minimum height of neighboring nodes with available capacity
        heights[u] = min_height + 1    # Set the new height of node u to the minimum height + 1
        logger.debug(f'heights: {heights}')

    def discharge_node(u):
        logger.debug(f'discharge_node({u})')
        # Discharge excess flow from node u
        logger.debug(f'excess_flow[{u}]: {excess_flow[u]}')
        while excess_flow[u] > 0:
            v = 0    # Initialize the index of the neighbor to push flow to
            for i in range(nodes_quantity):
                if capacities[u][i] - flow[u][i] > 0 and heights[u] == heights[i] + 1:
                    v = i    # Find a neighbor with available capacity and height difference of 1
                    # height difference must be 1 and only 1 because
                    # todo
                    break
            if v > 0:
                push_flow(u, v)    # Push flow from node u to node v
            else:
                relabel_node(u)    # If no valid neighbor found, relabel the height of node u and break the loop

    # Initialize distance labels
    for v in range(nodes_quantity):
        distance[v] = heights[v]    # Set the distance label of each node equal to its height
        logger.debug(f'distance[{v}]: {distance[v]}')

    # Main loop
    while excess_flow[source] > 0:
        logger.debug(f'excess_flow[{source}]: {excess_flow[source]} '
                     f'still greater than 0, continue')
        u = -1    # Initialize the index of the node to discharge
        for v in range(nodes_quantity):
            logger.debug(f'excess_flow[{v}]: {excess_flow[v]}')
            if excess_flow[v] > 0 and (u == -1 or distance[v] < distance[u]):
                logger.debug(f'v: {v}')

                u = v    # Find a node with excess flow and the minimum distance label
        if u == -1:
            logger.debug(f'u: {u}')
            break    # If no node found, exit
        discharge_node(u)
        for v in range(nodes_quantity):
            if capacities[u][v] - flow[u][v] > 0:
                distance[u] = min(distance[u], heights[v] + 1)

    return sum(flow[source])

def read_file(file_path):
    # Read input file
    with open(file_path, 'r') as f:
        lines = f.readlines()

    # Parse input file
    nodes = int(lines[0].split()[1])
    source = int(lines[1].split()[1])
    sink = int(lines[2].split()[1])
    arcs = int(lines[3].split()[1])
    arcs_data = [tuple(map(int, line.split())) for line in lines[4:]]

    return nodes, source, sink, arcs, arcs_data

def solve_max_flow_augmenting_paths(file_path, algorithm):
    logger.info(f'Solving max flow problem with augmenting paths for {file_path}')
    logger.info(f'Algorithm: {algorithm}')

    nodes, source, sink, arcs, arcs_data = read_file(file_path)

    # Build graph
    graph = [[] for _ in range(nodes)]
    # capacities[i][j] = capacity of the arc from i to j
    capacities = [[0 for _ in range(nodes)] for _ in range(nodes)]
    for arc in arcs_data:
        graph[arc[0]].append(arc[1])
        graph[arc[1]].append(arc[0])
        capacities[arc[0]][arc[1]] = arc[2]
    logger.debug(f'graph: {graph}')
    logger.debug(f'capacities: {capacities}')

    # GraphVisualizer(graph).draw()

    # Compute max flow
    max_flow = 0
    if algorithm == 'edmonds_karp':
        max_flow = max_flow_edmonds_karp(graph, capacities, source, sink)
    elif algorithm == 'distance':
        max_flow = distance_directed_augmenting_path(graph, capacities, source, sink)
    # elif algorithm == 'bfs':
    #     max_flow = bfs_directed_augmenting_path(graph, capacities, source, sink)
    # elif algorithm == 'dfs':
    #     max_flow = dfs_directed_augmenting_path(graph, capacities, source, sink)

    print(f'Max flow: {max_flow}')

    # Write output file
    write_output_file_to(".", file_path, nodes, capacities, max_flow)
    write_output_file_to("Path", file_path, nodes, capacities, max_flow)



def write_output_file_to(path, file_path, nodes, capacities, max_flow):
    if not os.path.exists(path):
        logger.debug(f'Creating directory {path}')
        os.makedirs(path)
    with open(path + "/" + file_path.replace('Instances',
                                             'Path').replace('inst',
                                                             'model').replace('.txt',
                                                                              '.path'), 'w') as f:
        f.write(f'{max_flow}\n')
        for i in range(nodes):
            for j in range(nodes):
                if capacities[i][j] > 0:
                    f.write(f'{i} {j} {capacities[i][j]}\n')

if __name__ == '__main__':
    import sys
    # print("len(sys.argv): ", len(sys.argv))
    if 1 == len(sys.argv):
        print("Usage: python chemin_augmentant.py <file_path> <algorithm>")
        print("Example: "
              "python chemin_augmentant.py inst-100-0.1.txt distance")
        print("Result: ")
        # solve_max_flow_augmenting_paths("inst-100-0.1.txt", "distance")
        solve_max_flow_augmenting_paths("inst-2-0.25.txt", "distance")
        exit(1)

    else:
        if 2 < len(sys.argv):
            if "debug" not in sys.argv:
                # logger disable
                logger.remove()
        solve_max_flow_augmenting_paths(os.path.join(sys.argv[1]), sys.argv[2])
# test
