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

def solve_max_flow_augmenting_paths(file_path):
    logger.info(f'Solving max flow problem with augmenting paths for {file_path}')

    # Read input file
    with open(file_path, 'r') as f:
        lines = f.readlines()

    # Parse input file
    nodes = int(lines[0].split()[1])
    source = int(lines[1].split()[1])
    sink = int(lines[2].split()[1])
    arcs = int(lines[3].split()[1])
    arcs_data = [tuple(map(int, line.split())) for line in lines[4:]]

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
    max_flow = max_flow_edmonds_karp(graph, capacities, source, sink)
    print(f'Max flow: {max_flow}')

    # Write output file
    write_output_file_to(".", file_path, nodes, capacities, max_flow)
    write_output_file_to("Path", file_path, nodes, capacities, max_flow)



def write_output_file_to(path, file_path, nodes, capacities, max_flow):
    if not os.path.exists(path):
        os.makedirs(path)
    with open(path + "/" + file_path.replace('Instances',
                                            'Path').replace(
            'inst', 'model').replace('.txt', '.path'), 'w') as f:
        if path == "Path":
            f.write(f'{max_flow}\n')
            for i in range(nodes):
                for j in range(nodes):
                    if capacities[i][j] > 0:
                        f.write(f'{i} {j} {capacities[i][j]}\n')

if __name__ == '__main__':
    import sys

    if 2 < len(sys.argv):
        if "debug" not in sys.argv:
            # logger disable
            logger.remove()
    solve_max_flow_augmenting_paths(os.path.join(sys.argv[1]))
