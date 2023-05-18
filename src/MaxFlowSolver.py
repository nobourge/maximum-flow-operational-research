from collections import deque

from networkx import Graph


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
        self.start_node = s  # Starting node
        self.end_node = d  # Ending node
        self.flow = 0  # Current flow
        self.capacity = c  # Capacity of the edge
        self.cost = 0  # Only used for MinCost

    def remaining(self):
        return self.capacity - self.flow  # Remaining capacity

    def add_flow(self, amount):
        self.flow += amount  # Increase flow on this edge


class MaxFlowSolver:


    def __init__(self, n=0):
        self.sink = None
        self.src = None
        self.nodes = []
        for _ in range(n):
            self.add_node()

    def link(self, n1, n2, capacity, cost=1):
        e12 = Edge(n1, n2, capacity, True)  # Create a forward edge
        e21 = Edge(n2, n1, 0,
                        False)  # Create a dual edge with capacity 0
        e12.dual = e21
        e21.dual = e12
        n1.edges.append(
            e12)  # Add the forward edge to the starting node
        n2.edges.append(e21)  # Add the dual edge to the ending node
        e12.cost = cost  # Set the cost of the forward edge (unused)
        e21.cost = -cost  # Set the cost of the dual edge (unused)

    def add_node(self):
        n = Node()
        n.index = len(self.nodes)
        self.nodes.append(n)
        return n

    def initialize(self):
        # INITIALIZE.
        # Perform a (reverse) breadth-first search
        # of the residual network
        # starting from the sink node to compute
        # distance labels d(i).
        # Let P = p and i = s.
        # Go to ADVANCE(i).
        n = len(self.nodes)

        # Initialize distances and other variables
        for u in self.nodes:
            u.dist = -1  # Set distance to -1 (unvisited)
            u.mindj = n  # Set minimum distance to the sink to the maximum value
            u.currentarc = 0  # Initialize the current edge index

        self.count = [0] * (
                n + 1)  # Array to count nodes at each distance level
        self.count[0] = 1  # There is one node at distance 0 (the sink)

        Q = deque()
        Q.append(self.sink)  # Start BFS from the sink node
        self.sink.dist = 0  # Distance to the sink is 0

        # Perform BFS to find
        # the shortest paths from the sink
        while Q:
            node = Q.popleft()
            for e in node.edges:
                if e.end_node.dist == -1:  # If the node has not been visited
                    e.end_node.dist = e.start_node.dist + 1  # Set its distance
                    self.count[
                        e.end_node.dist] += 1  # Increment the count for that distance
                    Q.append(e.end_node)  # Add it to the BFS queue

        if self.src.dist == -1:
            return 0  # If the source is unreachable, return 0

        # Set the minimum capacity of the source node to infinity
        self.src.min_capacity = float('inf')
        # Array to store the predecessors in the augmenting path
        self.predecessors = [None] * n
        i = self.src  # Start from the source node

        self.advance(i)

    def advance(self, i):
        # ADVANCE(i).
        # If the residual network contains no admissible arc (i, j),
        # then go to RETREAT(i).

        # If the residual network contains an admissible arc (i, j),
        # then set predecessors(j): = i and P: = P U {(i,j)}.
        # If j = t then go to AUGMENT;
        # otherwise, replace i by j
        # and repeat ADVANCE(i).

        while self.src.dist < n:
            augment = False
            # Find an augmenting path using DFS
            for j in range(i.currentarc, len(i.edges)):
                e = i.edges[j]
                if e.remaining() == 0:  # If the edge has no remaining capacity, skip it
                    continue

                if i.dist == e.end_node.dist + 1:  # If the node is one level deeper in the BFS tree
                    # Set its predecessor
                    # self.predecessors[e.end_node.index] = e
                    self.predecessors[e.end_node.index] = e.start_node
                    e.end_node.min_capacity = min(i.min_capacity,
                                                  e.remaining())  # Update the minimum capacity
                    if e.end_node == self.sink:
                        augment = True  # If the augmenting path reaches the sink, set the flag
                        break
                    else:
                        i = e.end_node  # Move to the next node in the augmenting path
                        continue
            if not augment:
                if self.count[i.dist] == 0 and i.dist < self.src.dist:
                    # there are no more nodes at the current distance
                    # and the distance is smaller than the source distance
                    break
                i = self.retreat(i)
            else:
                i = self.augment(i)

    def retreat(self, i):
        # RETREAT(i).
        # Update d(i): = min{d(j) + 1: r_ij > 0 and (i, j) in A(i)}.

        # If d(s) >= n, then STOP.

        # Otherwise, if i = s then go to
        # ADVANCE(i);
        # else delete (predecessors(i), i) from P, replace i by
        # predecessors(i) and go to ADVANCE(i).

        i.dist = i.mindj  # Update the distance to the minimum distance to the sink
        self.count[i.dist] += 1  # Increment the count for the new distance

        i.currentarc = 0  # Start from the first edge of the node in the next iteration
        i.mindj = n  # Reset the minimum distance to the maximum value

        if i != self.src:
            # Move back to the previous node in the augmenting path
            i = self.predecessors[i.index]

        return i
    def augment(self, i):
        # AUGMENT.
        # Let delta: = min{r_ij: (i, j) in P}.
        # Augment delta units of flow along P.
        # Set P: = phi,
        # i: = s and go to ADVANCE(i)
        # Augment the flow until the sink is reached
        added_flow = self.sink.min_capacity  # Determine the flow to be
        # added to the augmenting path
        edge = self.predecessors[self.sink.index]
        while edge:
            edge.add_flow(
                added_flow)  # Update the flow on each edge in the augmenting path
            edge = self.predecessors[edge.dual.end_node.index]

        self.total_flow += added_flow
        # Start from the source node for the next iteration
        i = self.src

        return i





    def get_max_flow(self, src, snk):
        self.total_flow = 0
        self.src = src
        self.sink = snk

        self.initialize()
        return self.total_flow

if __name__ == '__main__':
    # Read the number of nodes and edges
    n, m = map(int, input().split())

    # Create a graph with n nodes
    g = Graph(n)

    # Read the edges
    for _ in range(m):
        u, v, c = map(int, input().split())
        g.link(g.nodes[u - 1], g.nodes[v - 1], c)

    # Compute the maximum flow from node 0 to node n - 1
    print(g.get_max_flow(g.nodes[0], g.nodes[n - 1]))
