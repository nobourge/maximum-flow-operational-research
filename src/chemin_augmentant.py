# prend en paramètre en ligne de commande le nom d’une
# instance inst-n-p.txt dans le même dossier
# ,qui trouve la solution optimale
# via la méthode des chemins augmentants
# sans utiliser de solveur externe
# et qui la stock dans un fichier model-n-p.path
import os
from collections import deque

from loguru import logger

def check_minimum_cut(graph, capacities, source, sink, total_flow):
    ###
    # The min-cut represents the minimum capacity needed to
    # disconnect the source node from the sink node in the graph.
    # The max-flow value obtained from the algorithm will always be
    # equal to the capacity of the min-cut.
    # here
    ###
    logger.debug(f'check_minimum_cut(graph, capacities, {source}, {sink}, {total_flow})')

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
    logger.info(f"reachable_nodes: {reachable_nodes}")

    # find all edges with one node in reachable_nodes and the other not
    min_cut_capacity = 0
    for node in reachable_nodes:
        for neighbor in graph[node]:
            if neighbor not in reachable_nodes:
                # min_cut_capacity += capacities[node][neighbor]
                min_cut_capacity += capacities[neighbor][node]

    logger.info(f"total_flow: {total_flow}")
    logger.info(f"min_cut_capacity: {min_cut_capacity}")
    if total_flow == min_cut_capacity:
        logger.info("The solution is optimal!")
    else:
        logger.info("The solution is not optimal.")


def find_augmenting_path(flow, capacities, source, sink):
    logger.debug(f'find_augmenting_path(graph, capacities, {source}, {sink})')
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
        logger.debug("capacities:", capacities)
        for neighbor, capacity in enumerate(capacities[current]):
            logger.debug(f'capacities[current]: {capacities[current]}')
            logger.debug(f'neighbor: {neighbor}')
            logger.debug(f'capacity: {capacity}')
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
        path_flow = min(capacities[path[i]][path[i+1]] for i in range(len(path)-1))
        logger.debug(f'Max flow: {path_flow}')
        for i in range(len(path)-1):
            logger.debug(f'Updating flow[{path[i]}][{path[i+1]}] from '
                            f'{flow[path[i]][path[i+1]]} to '
                            f'{flow[path[i]][path[i+1]] + path_flow}')
            flow[path[i]][path[i+1]] += path_flow
            logger.debug(f'Updating capacities[{path[i]}][{path[i+1]}] from '
                         f'{capacities[path[i]][path[i+1]]} to '
                         f'{capacities[path[i]][path[i+1]] - path_flow}')
            capacities[path[i]][path[i+1]] -= path_flow
            logger.debug(f'Updating capacities[{path[i+1]}][{path[i]}] from '
                            f'{capacities[path[i+1]][path[i]]} to '
                            f'{capacities[path[i+1]][path[i]] + path_flow}')
            capacities[path[i+1]][path[i]] += path_flow
        logger.debug(f'New capacities: {capacities}')
        return path_flow

    # Sinon, renvoyer 0 pour indiquer que le flot maximum a été atteint
    else:
        return 0

def edmonds_karp(graph, capacities, source, sink, flow):
    # Trouver un chemin augmentant jusqu'à ce qu'il n'y en ait plus
    total_flow = 0
    optimal = False
    while not optimal:
        path_flow = find_augmenting_path(flow, capacities, source, sink)
        if path_flow == 0:
            logger.debug("flow == 0")
            logger.info("No more augmenting paths.")
            break
        total_flow += path_flow
        optimal = check_minimum_cut(graph, capacities, source, sink, total_flow)
    return total_flow

def push_relabel(graph, capacities, source, sink):
    # Initialize preflow
    heights = [0] * len(graph)    # Heights of nodes in the residual graph
    excess_flow = [0] * len(graph)    # Excess flow at each node
    flow = [[0] * len(graph) for _ in range(len(graph))]    # Flow matrix
    for edge in graph[source]:
        excess_flow[edge.to_node] = edge.capacity
        capacities[source][edge.to_node] -= edge.capacity
        capacities[edge.to_node][source] += edge.capacity
        flow[source][edge.to_node] = edge.capacity
        flow[edge.to_node][source] = -edge.capacity

    # Initialize distance labels
    distance = [0] * len(graph)    # Distance labels for nodes
    queue = deque([node for node in graph if node != source and node != sink])
    while queue:
        current = queue.popleft()
        distance[current] = min(distance[neighbor] + 1 for neighbor, capacity in enumerate(capacities[current])
                                if capacity > 0)
    distance[source] = len(graph)

    # Main loop
    while True:
        # Find a node with excess flow
        current = None
        for node in graph:
            if node != source and node != sink and excess_flow[node] > 0:
                current = node
                break
        if current is None:
            break

        # Push flow to a neighbor
        pushed = False
        for neighbor, capacity in enumerate(capacities[current]):
            if capacity > 0 and distance[current] == distance[neighbor] + 1:
                pushed = True
                flow_to_push = min(excess_flow[current], capacity)
                logger.debug(f'Pushing {flow_to_push} from {current} to {neighbor}')
                capacities[current][neighbor] -= flow_to_push
                capacities[neighbor][current] += flow_to_push
                flow[current][neighbor] += flow_to_push
                flow[neighbor][current] -= flow_to_push
                excess_flow[current] -= flow_to_push
                excess_flow[neighbor] += flow_to_push
                break

        # If no push was possible, relabel the node
        if not pushed:
            logger.debug(f'Relabeling {current}')
            distance[current] = min(distance[neighbor] + 1 for neighbor, capacity in enumerate(capacities[current])
                                    if capacity > 0)

    return sum(flow[source])

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
        logger.debug(f'flow: {flow}')

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
                logger.debug(f'capacities[{u}][{i}] : {capacities[u][i]}')
                logger.debug(f'flow[{u}][{i}] : {flow[u][i]}')
                logger.debug(f'heights[{u}] : {heights[u]}')
                logger.debug(f'heights[{i}] : {heights[i]}')
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

                u = v    # Find a node with excess flow and the minimum distance label
            logger.debug(f'v: {v}')

        if u == -1:
            logger.debug(f'u: {u}')
            break    # If no node found, exit
        discharge_node(u) # Discharge excess flow from node u
        # Update distance labels
        for v in range(nodes_quantity):
            if capacities[u][v] - flow[u][v] > 0:
                distance[u] = min(distance[u], heights[v] + 1)
                logger.debug(f'distance[{u}]: {distance[u]}')
                # distance[v] = min(distance[v], heights[u] + 1)

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
    logger.info(f'Solving max flow problem with augmenting paths for {file_path}')
    logger.info(f'Algorithm: {algorithm}')

    nodes_quantity, source, sink, arcs_quantity, arcs_data = read_file(file_path)

    # Build graph
    graph = [[] for _ in range(nodes_quantity)]
    # capacities[i][j] = capacity of the arc from i to j
    capacities = [[0 for _ in range(nodes_quantity)] for _ in range(nodes_quantity)]
    flow = [[0 for _ in range(nodes_quantity)] for _ in range(nodes_quantity)]
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
        max_flow = edmonds_karp(graph, capacities, source, sink)
    elif algorithm == 'distance':
        max_flow = distance_directed_augmenting_path(graph, capacities, source, sink)

    else:
        if nodes_quantity < arcs_quantity:
            # dense graph
            algorithm = 'push_relabel'
            max_flow = push_relabel(graph, capacities, source, sink)
        else:
            # sparse graph
            algorithm = 'edmonds_karp'
            max_flow = edmonds_karp(graph, capacities, source, sink, flow)
    logger.info(f'Max flow: {max_flow}')

    # solution_graph = [[] for _ in range(nodes_quantity)]
    # for i in range(nodes_quantity):

    # Write output origin_file
    write_output_file_to(".", file_path, nodes_quantity, capacities,
                         max_flow, flow)
    write_output_file_to(algorithm, file_path, nodes_quantity,
                         capacities, max_flow, flow)



def write_output_file_to(path, file_path, nodes, capacities, max_flow, flow):
    logger.info(f'Writing output origin_file to {path}')
    logger.debug(f'file_path: {file_path}')
    logger.debug(f'nodes: {nodes}')
    logger.debug(f'capacities: {capacities}')
    logger.debug(f'max_flow: {max_flow}')

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
                if flow[i][j] > 0:
                    f.write(f'{i} {j} {flow[i][j]}\n')

if __name__ == '__main__':
    import sys

    log_option = False

    # print("len(sys.argv): ", len(sys.argv))
    if 1 == len(sys.argv):
        print("Usage: python chemin_augmentant.py file_path <algorithm>")
        # instance = "inst-2-0.25.txt"
        instance = "inst-3-0.2.txt"
        # instance = "inst-100-0.1.txt"
        # algorithm = "distance"
        # algorithm = "edmonds_karp"
        algorithm = None
        log_option = True
        # print("Example: "
        #       "python chemin_augmentant.py {0} {1}".format(instance,
        #                                                    algorithm))
        print("Example: "
              "python chemin_augmentant.py {0}".format(instance))
        print("Result: ")

    elif 2 < len(sys.argv):
            instance = os.path.join(sys.argv[1])
            algorithm = sys.argv[2]
            if "debug" in sys.argv:
                log_option = True

    if not log_option:
        logger.info("logger debug mode disabled")
        logger.remove()

    print("instance: ", instance)
    solve_max_flow_augmenting_paths(instance, algorithm)
