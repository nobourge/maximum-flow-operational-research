from collections import deque


class MaxFlowSolver:
    class Node:
        def __init__(self):
            self.edges = []  # List of edges connected to this node
            self.index = 0  # Index in nodes array
            self.dist = 0  # PushRelabel, Dinic, and AhujaOrlin
            self.mindj = 0  # AhujaOrlin
            self.currentarc = 0  # AhujaOrlin
            self.minCapacity = 0  # FordFulkerson, AhujaOrlin

    class Edge:
        def __init__(self, s, d, c, f):
            self.forward = f  # True: edge is in the original graph, False: edge is a backward edge
            self.from_node = s  # Starting node
            self.to_node = d  # Ending node
            self.flow = 0  # Current flow
            self.capacity = c  # Capacity of the edge
            self.dual = None  # Reference to this edge's dual
            self.cost = 0  # Only used for MinCost

        def remaining(self):
            return self.capacity - self.flow  # Remaining capacity

        def add_flow(self, amount):
            self.flow += amount  # Increase flow on this edge
            self.dual.flow -= amount  # Adjust flow on the dual edge

    def __init__(self, n=0):
        self.nodes = []
        for _ in range(n):
            self.add_node()

    def link(self, n1, n2, capacity, cost=1):
        e12 = self.Edge(n1, n2, capacity, True)  # Create a forward edge
        e21 = self.Edge(n2, n1, 0,
                        False)  # Create a dual edge with capacity 0
        e12.dual = e21
        e21.dual = e12
        n1.edges.append(
            e12)  # Add the forward edge to the starting node
        n2.edges.append(e21)  # Add the dual edge to the ending node
        e12.cost = cost  # Set the cost of the forward edge (unused)
        e21.cost = -cost  # Set the cost of the dual edge (unused)

    def add_node(self):
        n = self.Node()
        n.index = len(self.nodes)
        self.nodes.append(n)
        return n

    def get_max_flow(self, src, snk):

        # 			 * INITIALIZE. Perform a (reverse) breadth-first search of the
        # 			 * residual network starting from the sink node to compute distance
        # 			 * labels d(i).
        n = len(self.nodes)
        total_flow = 0

        # Initialize distances and other variables
        for u in self.nodes:
            u.dist = -1  # Set distance to -1 (unvisited)
            u.mindj = n  # Set minimum distance to the sink to the maximum value
            u.currentarc = 0  # Initialize the current edge index

        count = [0] * (
                    n + 1)  # Array to count nodes at each distance level
        count[0] = 1  # There is one node at distance 0 (the sink)

        Q = deque()
        Q.append(snk)  # Start BFS from the sink node
        snk.dist = 0  # Distance to the sink is 0

        # Perform BFS to find
        # the shortest paths from the sink
        while Q:
            x = Q.popleft()
            for e in x.edges:
                if e.to_node.dist == -1: # If the node has not been visited
                    e.to_node.dist = e.from_node.dist + 1 # Set its distance
                    count[e.to_node.dist] += 1 # Increment the count for that distance
                    Q.append(e.to_node) # Add it to the BFS queue

        if src.dist == -1:
            return 0  # If the source is unreachable, return 0

        src.min_capacity = float(
            'inf')  # Set the minimum capacity of the source node to infinity
        pred = [
                   None] * n  # Array to store the predecessors in the augmenting path
        i = src  # Start from the source node

        # Augment the flow until the sink is reached
        while src.dist < n:
            # If the residual network contains an admissible arc (i, j),
            # then set pred(j) := i If j = t then go to AUGMENT; otherwise,
            # replace i by j and repeat ADVANCE(i).
            augment = False

            # Find an augmenting path using DFS
            for j in range(i.currentarc, len(i.edges)):
                e = i.edges[j]
                if e.remaining() == 0:  # If the edge has no remaining capacity, skip it
                    continue

                if i.dist == e.to_node.dist + 1:  # If the node is one level deeper in the BFS tree
                    pred[e.to_node.index] = e  # Set its predecessor
                    e.to_node.min_capacity = min(i.min_capacity,
                                                 e.remaining())  # Update the minimum capacity
                    if e.to_node == snk:
                        augment = True  # If the augmenting path reaches the sink, set the flag
                        break
                    else:
                        i = e.to_node  # Move to the next node in the augmenting path
                        continue

            if not augment:
                # RETREAT(i).
                # If there is no admissible edge leaving i, then
                # relabel i: d(i) := min{d(j): (i, j) \in E and r(i, j) > 0} + 1.
                # Set i = j and go to ADVANCE(i).
                if count[
                    i.dist] == 0 and i.dist < src.dist:  # If there are no more nodes at the current distance
                    break  # and the distance is smaller than the source distance

                i.dist = i.mindj  # Update the distance to the minimum distance to the sink
                count[
                    i.dist] += 1  # Increment the count for the new distance

                i.currentarc = 0  # Start from the first edge of the node in the next iteration
                i.mindj = n  # Reset the minimum distance to the maximum value

                if i != src:
                    i = pred[
                        i.index].from_node  # Move back to the previous node in the augmenting path
            else:
                # AUGMENT. Let sigma: = min{ri: (i, j) \in P}.
                # Augment sigma units of flow along P.
                # Set i = s and go to
                # ADVANCE(i).
                added_flow = snk.min_capacity  # Determine the flow to be added to the augmenting path
                edge = pred[snk.index]
                while edge:
                    edge.add_flow(
                        added_flow)  # Update the flow on each edge in the augmenting path
                    edge = pred[edge.dual.to_node.index]

                total_flow += added_flow
                i = src  # Start from the source node for the next iteration

        return total_flow
