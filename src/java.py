# https://github.com/Kawser-nerd/CLCDSA/blob/aee32551795763b54acb26856ab239370cac4e75/Source%20Codes/CodeJamData/17/04/4.java#L235

# /* A Max Flow solver base class. */
# 	public static abstract class MaxFlowSolver {
# 		/* List of nodes, indexed. */
# 		List<Node> nodes = new ArrayList<Node>();
#
# 		/* Create an edge between nodes n1 and n2 */
# 		public void link(Node n1, Node n2, long capacity) {
# 			/*
# 			 * Only the EdmondsKarp solver takes cost into account during
# 			 * getMaxFlow(). Setting it to 1 for problems that do not involve
# 			 * cost means it uses a shortest path search when finding augmenting
# 			 * paths. In practice, this does not seem to make a difference.
# 			 */
# 			link(n1, n2, capacity, 1);
# 		}
#
# 		/* Create an edge between nodes n1 and n2 and assign cost */
# 		public void link(Node n1, Node n2, long capacity, long cost) {
# 			Edge e12 = new Edge(n1, n2, capacity, true);
# 			Edge e21 = new Edge(n2, n1, 0, false);
# 			e12.dual = e21;
# 			e21.dual = e12;
# 			n1.edges.add(e12);
# 			n2.edges.add(e21);
# 			e12.cost = cost;
# 			e21.cost = -cost;
# 		}
#
# 		/* Create an edge between nodes n1 and n2 */
# 		void link(int n1, int n2, long capacity) {
# 			link(nodes.get(n1), nodes.get(n2), capacity);
# 		}
#
# 		/* Create a graph with n nodes. */
# 		protected MaxFlowSolver(int n) {
# 			for (int i = 0; i < n; i++)
# 				addNode();
# 		}
#
# 		protected MaxFlowSolver() {
# 			this(0);
# 		}
#
# 		public abstract long getMaxFlow(Node src, Node sink);
#
# 		/* Add a new node. */
# 		public Node addNode() {
# 			Node n = new Node();
# 			n.index = nodes.size();
# 			nodes.add(n);
# 			return n;
# 		}
# 	}
#
# 	/**
# 	 * Implements Ahuja/Orlin.
# 	 *
# 	 * Ahuja/Orlin describe this as an optimized variant of the original
# 	 * Edmonds-Karp shortest augmenting path algorithm.
# 	 *
# 	 * Ahuja, R. K. and Orlin, J. B. (1991), Distance-directed augmenting path
# 	 * algorithms for maximum flow and parametric maximum flow problems. Naval
# 	 * Research Logistics, 38: 413–430. doi:10.1002/1520-6750(199106)38:3
# 	 * <413::AID-NAV3220380310>3.0.CO;2-J and parametric maximum flow problems.
# 	 * This is algorithm DD1 in the paper.
# 	 */
# 	static class AhujaOrlin extends MaxFlowSolver {
# 		/* Create a graph with n nodes. */
# 		AhujaOrlin() {
# 			this(0);
# 		}
#
# 		AhujaOrlin(int n) {
# 			super(n);
# 		}
#
# 		@Override
# 		public long getMaxFlow(Node src, Node sink) {
# 			/**
# 			 * INITIALIZE. Perform a (reverse) breadth-first search of the
# 			 * residual network starting from the sink node to compute distance
# 			 * labels d(i).
# 			 */
# 			final int n = nodes.size();
# 			long totalFlow = 0;
# 			for (Node u : nodes) {
# 				u.dist = -1;
# 				u.mindj = n; // tracks min dist of children for relabeling
# 				u.currentarc = 0; // current active arc
# 			}
#
# 			int[] count = new int[n + 1]; // count[i] number of nodes u with
# 											// u.dist == i
# 			count[0]++; // count is used to bail as soon as maxflow is found
#
# 			Node[] Q = new Node[n]; // (Q, head, tail) are used as BFS queue
# 			int head = 0, tail = 0;
# 			sink.dist = 0;
# 			Q[tail++] = sink;
# 			while (head < tail) {
# 				Node x = Q[head++];
# 				for (Edge e : x.edges) {
# 					if (e.to.dist == -1) {
# 						e.to.dist = e.from.dist + 1;
# 						count[e.to.dist]++;
# 						Q[tail++] = e.to;
# 					}
# 				}
# 			}
#
# 			if (src.dist == -1) // no flow if source is not reachable
# 				return 0;
#
# 			src.minCapacity = Long.MAX_VALUE;
# 			Edge[] predecessors = new Edge[n]; // current augmenting path
# 			Node i = src; // i is the 'tip' of the augmenting path as in paper
#
# 			advance: while (src.dist < n) { // If d(s) >= n, then STOP.
# 				/*
# 				 * If the residual network contains an admissible arc (i, j),
# 				 * then set predecessors(j) := i If j = t then go to AUGMENT; otherwise,
# 				 * replace i by j and repeat ADVANCE(i).
# 				 */
# 				boolean augment = false;
#
# 				for (; i.currentarc < i.edges.size(); i.currentarc++) {
# 					Edge e = i.edges.get(i.currentarc);
# 					if (e.remaining() == 0) {
# 						continue;
# 					}
#
# 					Node j = e.to;
# 					i.mindj = min(i.mindj, j.dist + 1);
#
# 					/*
# 					 * an admissible edge (i, j) is one for which d(i) = d(j) +
# 					 * 1
# 					 */
# 					if (i.dist == j.dist + 1) {
# 						predecessors[j.index] = e;
# 						j.minCapacity = min(i.minCapacity, e.remaining());
# 						if (j == sink) {
# 							augment = true;
# 							break;
# 						} else {
# 							i = j;
# 							continue advance;
# 						}
# 					}
# 				}
# 				// either ran out of admissible edges, or reached sink and thus
# 				// are ready to augment
#
# 				if (!augment) {
# 					/*
# 					 * RETREAT: Update d(i): = min{d(j) + 1: rij > 0 and (i, j)
# 					 * \in A(i)}. If d(s) >= n, then STOP.
# 					 Otherwise, if i = s
# 					 * then go to ADVANCE(i); else replace i by predecessors(i) and go
# 					 * to ADVANCE(i).
# 					 */
#
# 					// check if maximum flow was already reached. This is
# 					// indicated
# 					// if a distance level is missing (a 'gap')
# 					if (--count[i.dist] == 0 && i.dist < src.dist)
# 						break;
# 					// TBD: i.dist describeds a min cut
#
# 					i.dist = i.mindj; // relabel
# 					count[i.dist]++;
#
# 					i.currentarc = 0; // reset current arc on node (i)
# 					i.mindj = n;
#
# 					if (i != src)
# 						i = predecessors[i.index].from;
# 				} else {
# 					/*
# 					 * AUGMENT. Let sigma: = min{ri: (i, j) \in P}. Augment
# 					 * sigma units of flow along P. Set i = s and go to
# 					 * ADVANCE(i).
# 					 */
# 					long addedFlow = sink.minCapacity;
# 					for (Edge edge = predecessors[sink.index]; edge != null; edge = predecessors[edge.dual.to.index]) {
# 						edge.addFlow(addedFlow);
# 					}
# 					totalFlow += addedFlow;
# 					i = src; // start over at source
# 				}
# 			}
# 			return totalFlow;
# 		}
# 	}
