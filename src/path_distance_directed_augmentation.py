from collections import deque

def distance_directed_augmenting_path(graph, capacities, source, sink):
    num_nodes = len(graph)
    heights = [0] * num_nodes    # Heights of nodes in the residual graph
    excess_flow = [0] * num_nodes    # Excess flow at each node
    flow = [[0] * num_nodes for _ in range(num_nodes)]    # Flow matrix
    distance = [0] * num_nodes    # Distance labels for nodes

    # Initialize preflow
    heights[source] = num_nodes    # Set height of source node to the number of nodes
    excess_flow[source] = float('inf')    # Set excess flow at source node to infinity
    for neighbor, capacity in enumerate(graph[source]):
        flow[source][neighbor] = capacity    # Initialize flow from source to its neighbors
        flow[neighbor][source] = -capacity    # Initialize flow from neighbors to source (negative)
        excess_flow[neighbor] = capacity    # Set excess flow at neighbors to their capacity

    def push_flow(u, v):
        # Push flow from node u to node v
        delta = min(excess_flow[u], capacities[u][v] - flow[u][v])    # Compute the maximum amount of flow that can be pushed
        flow[u][v] += delta    # Increase the flow from u to v
        flow[v][u] -= delta    # Decrease the flow from v to u (negative flow)
        excess_flow[u] -= delta    # Update the excess flow at node u
        excess_flow[v] += delta    # Update the excess flow at node v

    def relabel_node(u):
        # Relabel the height of node u
        min_height = float('inf')    # Initialize the minimum height to infinity
        for v in range(num_nodes):
            if capacities[u][v] - flow[u][v] > 0:
                min_height = min(min_height, heights[v])    # Find the minimum height of neighboring nodes with available capacity
        heights[u] = min_height + 1    # Set the new height of node u to the minimum height + 1

    def discharge_node(u):
        # Discharge excess flow from node u
        while excess_flow[u] > 0:
            v = 0    # Initialize the index of the neighbor to push flow to
            for i in range(num_nodes):
                if capacities[u][i] - flow[u][i] > 0 and heights[u] == heights[i] + 1:
                    v = i    # Find a neighbor with available capacity and height difference of 1
                    break
            if v > 0:
                push_flow(u, v)    # Push flow from node u to node v
            else:
                relabel_node(u)    # If no valid neighbor found, relabel the height of node u and break the loop

    # Initialize distance labels
    for v in range(num_nodes):
        distance[v] = heights[v]    # Set the distance label of each node equal to its height

    # Main loop
    while excess_flow[source] > 0:
        u = -1    # Initialize the index of the node to discharge
        for v in range(num_nodes):
            if excess_flow[v] > 0 and (u == -1 or distance[v] < distance[u]):
                u = v    # Find a node with excess flow and the minimum distance label
        if u == -1:
            break    # If no node found, exit
        discharge_node(u)
        for v in range(num_nodes):
            if capacities[u][v] - flow[u][v] > 0:
                distance[u] = min(distance[u], heights[v] + 1)

    return sum(flow[source])


# # Example usage:
# graph = [
#     [0, 16, 13, 0, 0, 0],
#     [0, 0, 10, 12, 0, 0],
#     [0, 4, 0, 0, 14, 0],
#     [0, 0, 9, 0, 0, 20],
#     [0, 0, 0, 7, 0, 4],
#     [0, 0, 0, 0, 0, 0]
# ]
#
# capacities = [
#     [0, 3, 3, 0, 0, 0],
#     [0, 0, 2, 3, 0, 0],
#     [0, 0, 0, 0, 2, 0],
#     [0, 0, 0, 0, 0, 2],
#     [0, 0, 0,
