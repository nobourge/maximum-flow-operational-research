# prend en paramètre en ligne de commande le nom d’une
# instance inst-n-p.txt dans le même dossier
# ,qui trouve la solution optimale
# via la méthode des chemins augmentants
# sans utiliser de solveur externe
# et qui la stock dans un fichier model-n-p.path
import os
from collections import deque, defaultdict
from timeit import timeit

import numpy as np
from loguru import logger
from line_profiler_pycharm import profile

def check_minimum_cut(capacities, source, total_flow):
    ###
    # The min-cut represents the minimum capacity needed to
    # disconnect the source node from the sink node in the graph.
    # The max-flow value obtained from the algorithm will always be
    # equal to the capacity of the min-cut.
    # here
    ###
    # logger.debug(f'check_minimum_cut(graph, to_update, {source}, {sink}, {total_flow})')

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
    # logger.info(f"reachable_nodes: {reachable_nodes}")

    # find all edges with one node in reachable_nodes and the other not
    min_cut_capacity = 0
    for node in reachable_nodes:
        for neighbor in capacities[node].keys():
            if neighbor not in reachable_nodes:
                min_cut_capacity += capacities[neighbor][node]

    logger.info(f"total_flow: {total_flow}")
    logger.info(f"min_cut_capacity: {min_cut_capacity}")
    if total_flow == min_cut_capacity:
        logger.info("The solution is optimal!")
    else:
        logger.info("The solution is not optimal.")

def update(to_update, complete, current_node, neighbor, quantity):
    logger.debug(f'update(capacities, {current_node}, {neighbor}, {quantity})')

    if not complete:
        # Set flow value in the flow dictionary
        if current_node not in to_update:
            to_update[current_node] = {}
        # if arc[1] not in to_update[current_node]:
        #     to_update[current_node][arc[1]] = 0

        # If the graph is undirected, you can add the reverse arc as well
        if neighbor not in to_update:
            to_update[neighbor] = {}
        # if current_node not in to_update[arc[1]]:
        #     to_update[arc[1]][current_node] = 0
    logger.info(f'Updating capacities[{current_node}][{neighbor}] from '
                f'{to_update[current_node][neighbor]} to '
                f'{to_update[current_node][neighbor] - quantity}')
    to_update[current_node][neighbor] -= quantity

    logger.info(f'Updating capacities[{neighbor}][{current_node}] from '
                f'{to_update[neighbor][current_node]} to '
                f'{to_update[neighbor][current_node] + quantity}')
    to_update[neighbor][current_node] += quantity


def find_path(flow, capacities, source, sink, queue, visited, parent):

    found_path = False

    # Parcours en largeur jusqu'à trouver le puits ou épuiser tous les chemins
    while queue and not found_path:
        # logger.debug(f'path: {path}')
        # logger.debug(f'parent: {parent}')
        # logger.debug(f'queue: {queue}')
        # logger.debug(f'visited: {visited}')
        current = queue.popleft()
        # current = queue.pop()
        # logger.debug(f'current: {current}')
        visited.add(current)
        # logger.debug(f"to_update: {to_update}")
        # logger.debug(f"to_update[current]: {to_update[current]}")

        # logger.debug(f"to_update[current] reversed: {list(reversed(to_update[current]))}")

        # added_current_to_path = False
        # for neighbor, capacity in enumerate(to_update[current]):
        # for neighbor in graph[current]:
        for neighbor in capacities[current]:
            # todo : choose max capacity neighbor
            if capacities[current][neighbor] > 0 and neighbor not in \
                    visited:
                # logger.debug(
                #     f'to_update[current]: {to_update[current]}')
                # logger.debug(f'neighbor: {neighbor}')
                # logger.debug(f'capacity: {capacity}')
                parent[neighbor] = current
                # logger.debug(f'parent[{neighbor}] = {current}')
                # if path[-1] != current:
                #     path.append(current)
                #     added_current_to_path = True
                #     logger.debug(f'path: {path}')
                if neighbor == sink:
                    # path.append(neighbor)
                    # added_current_to_path = True

                    found_path = True
                    # logger.debug(f'found_path: {found_path}')
                    # logger.debug(f'path: {path}')
                    break
                queue.append(neighbor)
                # logger.debug(f'queue: {queue}')
        # if not added_current_to_path and current != path[-1]:
        #     path.pop()
        #     logger.debug(f'path: {path}')
    return parent
@profile
def find_augmenting_path(flow, capacities, source, sink):
    # logger.debug(f'find_augmenting_path(graph, to_update, {source}, {sink})')
    # Recherche d'un chemin augmentant dans le graphe
    queue = deque([source])
    visited = set()
    parent = {}


    # Si un chemin augmentant a été trouvé, le construire et renvoyer son flux maximum
    if find_path(flow, capacities, source, sink, queue, visited, parent):
        parent_path = [sink]
        current = sink
        # logger.debug(f'current: {current}')
        # logger.debug(f'parent: {parent}')
        while current != source:
            logger.debug(f'current: {current}')
            logger.debug(f'parent: {parent}')
            current = parent[current]
            parent_path.append(current)
            # logger.debug(f'current: {current}')
            # logger.debug(f'parent_path: {parent_path}')
        # Le chemin est construit à l'envers, le retourner
        path = list(reversed(parent_path))
        # logger.debug(f'Augmenting path: {path}')
        # max_flow est le minimum des capacités des arcs du chemin
        path_flow = min(capacities[path[i]][path[i+1]] for i in range(len(path)-1))
        # logger.debug(f'path flow: {path_flow}')
        for i in range(len(path)-1):
            # if flow[path[i]].get(path[i+1]) is None:
            #     flow[path[i]][path[i+1]] = 0
            if path[i] not in flow:
                flow[path[i]] = {}
            if path[i+1] not in flow[path[i]]:
                flow[path[i]][path[i+1]] = path_flow
            else:
                flow[path[i]][path[i+1]] += path_flow
            logger.debug(
                f'Updating flow[{path[i]}][{path[i + 1]}] from '
                f'{flow[path[i]][path[i + 1]]} to '
                f'{flow[path[i]][path[i + 1]] + path_flow}')

            update(capacities, True, path[i], path[i+1], \
                path_flow)
        # logger.debug(f'New to_update: {to_update}')
        return path_flow

    # Sinon, renvoyer 0 pour indiquer que le flot maximum a été atteint
    else:
        return 0

def ford_fulkerson(capacities, source, sink, flow, nodes_quantity,
                   arcs_quantity):
    # Trouver un chemin augmentant jusqu'à ce qu'il n'y en ait plus
    total_flow = 0
    optimal = False
    while not optimal:
        path_flow = find_augmenting_path(flow, capacities, source, sink)
        if path_flow == 0:
            # logger.debug("flow == 0")
            # logger.info("No more augmenting paths.")
            break
        total_flow += path_flow
        optimal = check_minimum_cut(capacities, source, total_flow)
    return total_flow

@profile
def edmonds_karp(capacities, source, sink, flow, nodes_quantity,
                 arcs_quantity):
    # Trouver un chemin augmentant jusqu'à ce qu'il n'y en ait plus
    total_flow = 0
    optimal = False
    while not optimal:
        path_flow = find_augmenting_path(flow, capacities, source, sink)
        if path_flow == 0:
            # logger.debug("flow == 0")
            # logger.info("No more augmenting paths.")
            break
        total_flow += path_flow
        optimal = check_minimum_cut(capacities, source, total_flow)
    return total_flow

@profile
def push(u, v, C, F, excess):
    # logger.debug(f'Pushing {u} to {v}')
    # logger.debug(f'excess[{u}] = {excess[u]}')
    # logger.debug(f'C[{node_to_discharge}][{v}] = {C[node_to_discharge][v]}')
    # logger.debug(f'F[{u}][{v}] = {F[u][v]}')
    # logger.debug(f'C[{node_to_discharge}][{v}] = {C[node_to_discharge][v]}')

    capacity = C[u][v]
    # if node_to_discharge != v:
    #     capacity = C[node_to_discharge][v]
    # else:
    #     capacity = 0
    send = min(excess[u], capacity - F[u][v])
    # logger.debug(f'send = {send}')
    # logger.debug(f'Updating F[{u}][{v}] from '
    #              f'{F[u][v]} to '
    #              f'{F[u][v] + send}')
    F[u][v] += send
    F[v][u] -= send
    excess[u] -= send
    # logger.debug(f'excess[{u}] = {excess[u]}')
    excess[v] += send
    # logger.debug(f'excess[{v}] = {excess[v]}')

@profile
def relabel(u, C, F, height, neighbor_keys):
    # logger.debug(f'Relabeling {u}')
    # Find smallest new height making a push possible,
    # if such a push is possible at all.
    min_height = float('inf')
    # for v in range(n):
    for v in neighbor_keys:
        if C[u][v] - F[u][v] > 0:
            min_height = min(min_height, height[v])
            # height[u] = min_height + 1
    height[u] = min_height + 1
    # logger.debug(f'height[{u}] = {height[u]}')
@profile
def discharge(node_to_discharge, C, F, excess, seen, height, neighbors):
    # logger.debug(f'Discharging {node_to_discharge}')
    # neighbor_keys = [key for key, value in C[node_to_discharge].items() if value > 0]
    # neighbor_keys = [key for key, value in C[
    #     node_to_discharge].items()]
    # neighbor_keys = list(C[node_to_discharge].keys()) # type: list
    node_neighbor_list = neighbors[node_to_discharge]
    # logger.debug(f'Neighbours of {node_to_discharge}: {node_neighbor_list}')
    neighbor_quantity = len(node_neighbor_list)
    while excess[node_to_discharge] > 0:
        # logger.debug(f'Excess[{node_to_discharge}] = {excess[node_to_discharge]}')
        # logger.debug(f'Seen[{node_to_discharge}] = {seen[node_to_discharge]}')
        if seen[node_to_discharge] < neighbor_quantity:
            # check next neighbour
            v = seen[node_to_discharge]
            v = node_neighbor_list[v]
            # logger.debug(f'Checking neighbour {v}')
            # if node_to_discharge != v:

            # logger.debug(f'C[{node_to_discharge}][{v}] = {C[node_to_discharge][v]}')
            c = C[node_to_discharge][v]
            # logger.debug(f'F[{node_to_discharge}] = {F[node_to_discharge]}')
            # logger.debug(f'F[{node_to_discharge}][{v}] = {F[node_to_discharge][v]}')
            f = F[node_to_discharge][v]
            # logger.debug(f'height[{node_to_discharge}] = {height[node_to_discharge]}')
            # logger.debug(f'height[{v}] = {height[v]}')
            if c - f > 0 \
                    and height[node_to_discharge] > height[v]:
                push(node_to_discharge, v, C, F, excess)
            else:
                seen[node_to_discharge] += 1
        else:  # we have checked all neighbours. must relabel
            relabel(node_to_discharge, C, F, height, node_neighbor_list)
            seen[node_to_discharge] = 0
@profile
def relabel_to_front(C
                     , source: int
                     , sink: int
                     , F
                     , nodes_quantity
                     , neighbors
                     ) -> int:
    # logger.debug(f'Capacity matrix: {C}')
    # logger.info("Relabel to front algorithm")
    # n = len(C)  # C is the capacity matrix
    n = nodes_quantity  # C is the capacity matrix
    # F = [[0] * n for _ in range(n)]
    # F = defaultdict(lambda: defaultdict(int))
    # residual capacity from node_to_discharge to v is C[node_to_discharge][v] - F[node_to_discharge][v]

    height = {key : 0 for key in range(n)}
    excess = {key : 0 for key in range(n)}
    seen = {key : 0 for key in range(n)}
    # node "queue"
    nodelist = [i for i in range(n) if i != source and i != sink]

    height[source] = n  # longest path from source to sink is less than n long
    excess[source] = float('inf')  # send as much flow as possible to
    # neighbours
    # of source

    # for v in range(n):
    #     push(source, v)

    if source == 0:
        for neighbor in neighbors[source]:
            # logger.debug(f'Pushing {source} to {neighbor}')
            push(source, neighbor, C, F, excess)
    else:
        # logger.warning(f'source = {source}')
        for v in range(n): # n is the number of nodes
            if C[source][v] > 0:
                # logger.debug(f'Pushing {source} to {v}')
                push(source, v, C, F, excess)

    p = 0
    # while there is excess flow at any non-sink node
    while p < len(nodelist):
        u = nodelist[p]
        old_height = height[u]
        discharge(u, C, F, excess, seen, height, neighbors)
        if height[u] > old_height:
            # node_to_discharge was relabeled
            nodelist.insert(0, nodelist.pop(p))  # move to front of list
            p = 0  # start from front of list
        else:
            p += 1
    return sum(F[source])

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
    capacities = {}
    flow = {}
    neighbors = {}

    # Add neighbors for each node
    for arc in arcs_data:
        node1, node2 = arc[0], arc[1]

        if node1 not in neighbors:
            neighbors[node1] = []
        if node2 not in neighbors:
            neighbors[node2] = []

        if node2 not in neighbors[node1]:
            neighbors[node1].append(node2)
        if node1 not in neighbors[node2]:
            neighbors[node2].append(node1)

        # Set capacity value in the capacities dictionary
        if arc[0] not in capacities:
            capacities[arc[0]] = {}
        capacities[arc[0]][arc[1]] = arc[2]

        # # If the graph is undirected, you can add the reverse arc as well
        if arc[1] not in capacities:
            capacities[arc[1]] = {}
        if capacities[arc[1]].get(arc[0]) is None:
            capacities[arc[1]][arc[0]] = 0

        # Set flow value in the flow dictionary
        if arc[0] not in flow:
            flow[arc[0]] = {}
        if arc[1] not in flow[arc[0]]:
            flow[arc[0]][arc[1]] = 0

        # If the graph is undirected, you can add the reverse arc as well
        if arc[1] not in flow:
            flow[arc[1]] = {}
        if arc[0] not in flow[arc[1]]:
            flow[arc[1]][arc[0]] = 0

    # logger.debug(f'flow[6] = {flow[6]}')

    # Compute max flow
    if algorithm == 'edmonds_karp':
        max_flow = edmonds_karp(capacities, source, sink, flow,
                                nodes_quantity, arcs_quantity)

    elif algorithm == 'relabel_to_front':
        max_flow = relabel_to_front(
            capacities
            , source
            , sink
            , flow
            , nodes_quantity
            , neighbors
        )

    else:
        if nodes_quantity < arcs_quantity:
            # dense graph
            algorithm = 'push_relabel'
            # max_flow = push_relabel(graph, capacities, source, sink,
            #                         flow, nodes_quantity, arcs_quantity)
            max_flow = relabel_to_front(
                capacities
                , source
                , sink
                , flow
                , nodes_quantity
                , neighbors
            )
        else:
            # sparse graph
            algorithm = 'edmonds_karp'
            max_flow = edmonds_karp(capacities
                                    , source
                                    , sink
                                    , flow
                                    , nodes_quantity
                                    , arcs_quantity)
    max_flow = int(max_flow)
    logger.info(f'Max flow: {max_flow}')
    print(f'file_path: {file_path}')
    print(f'Max flow: {max_flow}')

    # solution_graph = [[] for _ in range(nodes_quantity)]
    # for i in range(nodes_quantity):

    # Write output origin_file
    # write_output_file_to(".", file_path, nodes_quantity, to_update,
    #                      max_flow, flow)
    write_output_file_to(algorithm, file_path, nodes_quantity,
                         capacities, max_flow, flow, neighbors)


@profile
def write_output_file_to(path, file_path, nodes, capacities,
                         max_flow, flow, neighbors):
    # logger.info(f'Writing output origin_file to {path}')
    # logger.debug(f'file_path: {file_path}')
    # logger.debug(f'nodes: {nodes}')
    # logger.debug(f'capacities: {capacities}')
    # logger.debug(f'max_flow: {max_flow}')

    if not os.path.exists(path):
        logger.debug(f'Creating directory {path}')
        os.makedirs(path)
    with open(path + "/" + file_path.replace('Instances',
                                             'Path').replace('inst',
                                                             'model').replace('.txt',
                                                                              '.path'), 'w') as f:
        # logger.info(f'{max_flow}\n')
        f.write(f'{max_flow}\n')
        logger.debug(f'flow: {flow}')
        for i in range(nodes):
            for j in neighbors[i]:
                if flow[i][j] > 0:
                    logger.info(f'{i} {j} {flow[i][j]}\n')
                    f.write(f'{i} {j} {flow[i][j]}\n')

if __name__ == '__main__':
    import sys

    log_option = False

    if 1 == len(sys.argv):
        print("Usage: python chemin_augmentant.py file_path <algorithm>")
        # instance = "inst-2-0.25.txt"
        instance = "inst-3-0.2.txt"
        # instance = "inst-3-0.22.txt"
        # instance = "inst-3-0.3.txt"
        # instance = "inst-4-0.25.txt"
        # instance = "inst-100-0.1.txt"
        # instance = "inst-100-0.3.txt"
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

    logger.info(f'instance: {instance}')
    if not log_option:
        logger.info("logger debug mode disabled")
        logger.remove()

    solve_max_flow_augmenting_paths(instance, algorithm)
