# prend en paramètre en ligne de commande le nom d’une
# instance inst-n-p.txt dans le même dossier
# ,qui trouve la solution optimale
# via la méthode des chemins augmentants
# sans utiliser de solveur externe
# et qui la stock dans un fichier model-n-p.path
import os
from collections import deque, defaultdict

import networkx as nx
import numpy as np
from loguru import logger
from line_profiler_pycharm import profile

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




def find_augmenting_path(graph, residual_capacities, source, sink,
                         flow):
    queue = deque([source])
    visited = set()
    parent = {}
    found_path = False

    while queue and not found_path:
        current = queue.popleft()
        visited.add(current)

        for neighbor, capacity in enumerate(
                residual_capacities[current]):
            if capacity > 0 and neighbor not in visited:
                parent[neighbor] = current
                if neighbor == sink:
                    found_path = True
                    break
                queue.append(neighbor)

    if found_path:
        path = deque([sink])
        current = sink

        while current != source:
            current = parent[current]
            path.appendleft(current)

        path_flow = min(
            residual_capacities[path[i]][path[i + 1]] for i in
            range(len(path) - 1))

        for i in range(len(path) - 1):
            residual_capacities[path[i]][path[i + 1]] -= path_flow
            residual_capacities[path[i + 1]][path[i]] += path_flow

            flow[(path[i], path[i + 1])] = flow.get(
                (path[i], path[i + 1]), 0) + path_flow

        return path_flow
    else:
        return 0


def edmonds_karp(graph, capacities, source, sink, nodes_quantity,
                    arcs_quantity):
    graph = [[] for _ in range(nodes_quantity)]
    capacities = np.zeros((nodes_quantity, nodes_quantity))
    flow = {}
    total_flow = 0
    optimal = False

    while not optimal:
        residual_capacities = [
            [capacities[i][j] - flow.get((i, j), 0) for j in
             range(nodes_quantity)] for i in range(nodes_quantity)]
        path_flow = find_augmenting_path(graph, residual_capacities,
                                         source, sink, flow)

        if path_flow == 0:
            break

        total_flow += path_flow

        visited = set()
        queue = deque([source])
        while queue:
            current = queue.popleft()
            visited.add(current)
            for neighbor, capacity in enumerate(
                    residual_capacities[current]):
                if capacity > 0 and neighbor not in visited:
                    queue.append(neighbor)

        min_cut_capacity = sum(
            capacities[i][j] for i in range(nodes_quantity) for j in
            range(nodes_quantity) if
            capacities[i][j] > 0 and i in visited and j not in visited)

        if total_flow == min_cut_capacity:
            optimal = True

    return total_flow


def update(values_dict, current_node, neighbor, quantity):
    values_dict[(current_node, neighbor)] -= quantity
    values_dict[(neighbor, current_node)] += quantity

def relabel_to_front(graph
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
        if u == v:
            return
        logger.info(f"push {u} {v}")
        logger.info(f"excess {excess}")
        logger.info(f"height {height}")
        logger.info(f"seen {seen}")
        logger.info(f"excess_nodes {excess_nodes}")
        logger.info(f"graph.edges[u, v]['capacity'] {graph.edges[u, v]['capacity']}")
        send = min(excess[u], graph.edges[u, v]['capacity'] - graph.edges[u, v]['flow'])
        # update(flow, v, u, send)
        graph.edges[u, v]['flow'] += send
        graph.edges[v, u]['flow'] -= send
        excess[u] -= send
        excess[v] += send


    def relabel(u):
        # Find smallest new height making a push possible,
        # if such a push is possible at all.
        min_height = float('inf')
        for v in range(nodes_quantity):
            # if graph[u][v]['capacity'] - graph[u][v]['flow'] > 0:
            if graph.edges[u, v]['capacity'] - graph.edges[u, v]['flow'] > 0:
                min_height = min(min_height, height[v])
                height[u] = min_height + 1
    def discharge(u):
        while excess[u] > 0:
            if seen[u] < nodes_quantity:  # check next neighbour
                v = seen[u]
                # if graph[u][v]['capacity'] - graph[u][v]['flow'] > 0 \
                if graph.edges[u, v]['capacity'] - graph.edges[u, v]['flow'] > 0 \
                        and height[u] > height[v]:
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
    # for v in range(nodes_quantity):
    for v in graph[source]:
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

    return sum(graph[source][i]['flow'] for i in graph[source])

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

# def highest_label(graph, source, sink):
#     # Initialize the labels and excess flow
#     labels = {node: 0 for node in graph.nodes()}
#     excess = {node: 0 for node in graph.nodes()}
#     labels[source] = graph.number_of_nodes()
#
#     # Initialize the preflow by saturating outgoing edges from the source
#     for neighbor in graph.neighbors(source):
#         # if 'capacity' in graph[source][neighbor]:
#         logger.debug('Saturating edge: ({}, {})'.format(source, neighbor))
#         graph[source][neighbor].update({'flow': graph[source][neighbor]['capacity']})
#         graph[neighbor][source].update({'flow': -graph[source][neighbor]['capacity']})
#         excess[neighbor] = graph[source][neighbor]['capacity']
#
#     # Create the highest-label-first list
#     hlf_list = []
#
#     # for neighbor in graph.neighbors(source):
#     #     # if 'capacity' in graph[source][neighbor]:
#     #     if neighbor != sink:
#     #         hlf_list.append(neighbor)
#     # hlf_list = [node for node in graph.neighbors(source) if node != sink]
#
#     # create hlf_list: list of nodes with excess flow
#     hlf_list = [node for node in graph.neighbors(source) if node != source]
#
#
#     # Main loop
#     while hlf_list:
#         logger.debug(f'hlflist: {hlf_list}')
#         logger.debug(f'labels: {labels}')
#         logger.debug(f'excess: {excess}')
#         logger.debug(f'flow: {nx.get_edge_attributes(graph, "flow")}')
#         node = hlf_list[-1]
#         for neighbor in graph.neighbors(node):
#             logger.debug(f'neighbor: {neighbor}')
#             logger.debug(f'capacity: {graph[node][neighbor]["capacity"]}')
#
#             if labels[node] > labels[neighbor] and excess[
#                 node] > 0:
#                 capacity = graph[node][neighbor]['capacity']
#                 flow = graph[node][neighbor]['flow']
#                 residual_capacity = capacity - flow
#
#                 if residual_capacity > 0:
#                     delta = min(excess[node], residual_capacity)
#                     graph[node][neighbor]['flow'] += delta
#                     graph[neighbor][node]['flow'] -= delta
#                     excess[node] -= delta
#                     excess[neighbor] += delta
#
#                     # if neighbor != source and neighbor != sink and excess[neighbor] > 0:
#                     # if neighbor != sink and excess[neighbor] > 0:
#                     if neighbor != source and excess[neighbor] > 0:
#                         hlf_list.append(neighbor)
#
#         if excess[node] == 0:
#             # if node == sink:
#             #     break
#             hlf_list.pop()
#
#         # Relabel the node if there is excess flow remaining
#         if excess[node] > 0:
#             min_label = float('inf')
#             for neighbor in graph.neighbors(node):
#                 # if 'capacity' in graph[node][neighbor]:
#                 logger.debug(f"node: {node}, neighbor: {neighbor}" )
#                 capacity = graph[node][neighbor]['capacity']
#                 logger.debug(f"capacity: {capacity}")
#                 flow = graph[node][neighbor]['flow']
#                 logger.debug(f"flow: {flow}")
#                 residual_capacity = capacity - flow
#
#                 if residual_capacity > 0:
#                     min_label = min(min_label, labels[neighbor])
#
#             labels[node] = min_label + 1
#             logger.debug(f"labels: {labels}")
#
#     # Compute the maximum flow
#     max_flow = sum(graph[source][neighbor]['flow'] for neighbor in
#                    graph.neighbors(source)
#                    # if graph[source][
#                    #     neighbor]['capacity'] > 0
#                    )
#
#     return max_flow



def push(graph, source, target, residual_capacity):
    # Calculate the amount of flow to push
    logger.debug(f'Pushing flow from {source} to {target}')
    logger.debug(f'Residual capacity: {residual_capacity}')
    logger.debug(f'Excess flow of {source}: {graph.nodes[source]["excess_flow"]}')
    flow = min(residual_capacity, graph.nodes[source]['excess_flow'])

    # Update the flow and residual capacity of the edge
    graph.edges[source, target]['flow'] += flow
    logger.debug(f'Flow of edge ({source}, {target}): {graph.edges[source, target]["flow"]}')
    graph.edges[source, target]['capacity'] -= flow
    logger.debug(f'Pushing {flow} units of flow from {source} to {target}')

    # Update the residual capacity of the reverse edge
    graph.edges[target, source]['capacity'] += flow


    # Update the excess flow of the nodes
    graph.nodes[source]['excess_flow'] -= flow
    graph.nodes[target]['excess_flow'] += flow
    logger.debug(f'Excess flow of {source} is now {graph.nodes[source]["excess_flow"]}')
    logger.debug(f'Excess flow of {target} is now {graph.nodes[target]["excess_flow"]}')

    # Check if the target node becomes active and add it to the list of active nodes
    if graph.nodes[target]['excess_flow'] > 0 and target not in \
            graph.nodes[target]['active_nodes']:
        logger.debug(f'Adding {target} to the list of active nodes')
        graph.nodes[target]['active_nodes'].append(target)


def relabel(graph, node):
    # Find the minimum label value among the neighbors
    min_label = min(
        graph.nodes[n]['label'] for n in graph.neighbors(node))

    # Assign the minimum label value + 1 to the node
    graph.nodes[node]['label'] = min_label + 1

    # Check if the node becomes active and add it to the list of active nodes
    if graph.nodes[node]['excess_flow'] > 0 and node not in \
            graph.nodes[node]['active_nodes']:
        graph.nodes[node]['active_nodes'].append(node)
        logger.debug(f'Adding {node} to the list of active nodes')

    logger.debug(f'Relabeling {node} to {graph.nodes[node]["label"]}')


def highest_label(graph, source, target):
    # Initialize variables and data structures
    for node in graph.nodes:
        graph.nodes[node]['excess_flow'] = 0
        graph.nodes[node]['label'] = 0
        graph.nodes[node]['active_nodes'] = []


    # Assign the label value to the source node
    graph.nodes[source]['label'] = len(graph.nodes)
    logger.debug(f'Label of {source} is {graph.nodes[source]["label"]}')

    # Assign the excess flow to the source node
    graph.nodes[source]['excess_flow'] = float('inf')
    logger.debug(f'Excess flow of {source} is {graph.nodes[source]["excess_flow"]}')


    # Perform initial push operations from the source node
    for neighbor in graph.neighbors(source):
        capacity = graph.edges[source, neighbor]['capacity']
        if capacity > 0:
            push(graph, source, neighbor, capacity)

    # List of label values in descending order
    label_values = list(range(1, len(graph.nodes)))
    logger.debug(f'Label values: {label_values}')

    # Main loop of the push-relabel algorithm
    while label_values:
        logger.debug(f'Label values: {label_values}')
        # Select the label value with the highest index
        current_label = label_values[-1]
        logger.debug(f'Current label: {current_label}')

        # Get the list of active nodes for the current label value
        # active_nodes = graph.nodes[source]['active_nodes']
        active_nodes = graph.nodes[current_label]['active_nodes']
        logger.debug(f'Active nodes: {active_nodes}')

        # If there are no active nodes, move to the next label value
        if not active_nodes:
            logger.debug(f'No active nodes for label {current_label}')
            label_values.pop()
            continue

        # Process the active nodes
        for node in active_nodes:
            neighbors = graph.neighbors(node)

            # Iterate through the neighbors of the active node
            for neighbor in neighbors:
                logger.debug(f'Neighbor: {neighbor}')
                # Perform a push operation if the conditions are met
                if graph.nodes[node]['label'] > graph.nodes[neighbor][
                    'label']:
                    logger.debug(f'node label: {graph.nodes[node]["label"]}')
                    logger.debug(f'neighbor label: {graph.nodes[neighbor]["label"]}')
                    residual_capacity = graph.edges[node, neighbor][
                        'capacity']
                    logger.debug(f'Residual capacity: {residual_capacity}')
                    if residual_capacity > 0:
                        push(graph, node, neighbor, residual_capacity)

                        # Check if the excess flow becomes 0 and remove the node from the active nodes list
                        if graph.nodes[node]['excess_flow'] == 0:
                            graph.nodes[node]['active_nodes'].remove(
                                node)

            # Perform a relabel operation if the excess flow is still positive
            if graph.nodes[node]['excess_flow'] > 0:
                relabel(graph, node)

            # Check if the excess flow becomes 0 and remove the node from the active nodes list
            if graph.nodes[node]['excess_flow'] == 0:
                graph.nodes[node]['active_nodes'].remove(node)

    # Return the maximum flow value
    # return graph.edges[source, target]['flow']
    return sum(graph[source][i]['flow'] for i in graph[source])


def solve_max_flow_augmenting_paths(file_path, algorithm=None):
    nodes_quantity, source, sink, arcs_quantity, arcs_data = read_file(file_path)

    # Create a directed graph
    graph = nx.DiGraph()

    # Add nodes to the graph
    graph.add_nodes_from(range(nodes_quantity))
    for arc in arcs_data:
        # Add arcs to the graph
        graph.add_edge(arc[0], arc[1], capacity=arc[2], flow=0)
        graph.add_edge(arc[1], arc[0], capacity=0, flow=0)
    # GraphVisualizer(graph).draw()

    # Compute max flow
    if algorithm == 'edmonds_karp':
        max_flow = edmonds_karp(graph
                                # , capacities
                                , source
                                , sink
                                # , flow
                                , nodes_quantity
                                , arcs_quantity
                                )

    elif algorithm == 'relabel_to_front':
        max_flow = relabel_to_front(graph
                                    # , flow
                                    , source
                                    , sink
                                    , nodes_quantity
                                    )
    elif algorithm == 'highest_label':
        max_flow = highest_label(graph
                                 # , capacities
                                 , source
                                 , sink
                                 # , flow
                                 # , nodes_quantity
                                 # , arcs_quantity
                                 )

    else:
        if nodes_quantity < arcs_quantity:
            # dense graph
            algorithm = 'relabel_to_front'
            max_flow = relabel_to_front(graph
                                        # , flow
                                        , source
                                        , sink
                                        , nodes_quantity
                                        )
        else:
            # sparse graph
            algorithm = 'edmonds_karp'
            max_flow = edmonds_karp(graph
                                    , source
                                    , sink
                                    , nodes_quantity
                                    , arcs_quantity
                                    )
    max_flow = int(max_flow)
    print(f'file_path: {file_path}')
    print(f'Max flow: {max_flow}')

    write_output_file_to(algorithm
                         , file_path
                         , graph
                         , nodes_quantity
                         , max_flow
                         )


@profile
def write_output_file_to(path
                         , file_path
                         , graph
                         , nodes_quantity
                         , max_flow
                         ):
    logger.info(f'Writing output origin_file to {path}')
    logger.debug(f'file_path: {file_path}')
    logger.debug(f'max_flow: {max_flow}')

    if not os.path.exists(path):
        logger.debug(f'Creating directory {path}')
        os.makedirs(path)
    with open(path + "/" + file_path.replace('Instances',
                                             'Path').replace('inst',
                                                             'model').replace('.txt',
                                                                              '.path'), 'w') as f:
        logger.info(f'{max_flow}\n')
        f.write(f'{max_flow}\n')
        lines = []
        for node in graph.nodes:
            for neighbor in graph.neighbors(node):
                if graph[node][neighbor]['flow'] > 0:
                    logger.info(f'{node} {neighbor} {graph[node][neighbor]["flow"]}\n')
                    lines += [f'{node} {neighbor} {graph[node][neighbor]["flow"]}\n']
        lines.sort()
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
        # instance = "inst-500-0.1.txt"

        algorithm = None
        # algorithm = "distance"
        # algorithm = "edmonds_karp"
        algorithm = "relabel_to_front"
        # algorithm = "highest_label"
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
