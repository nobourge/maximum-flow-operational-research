import sys

# Importe les fonctions nécessaires à la génération et à la résolution du modèle
from model_generation import read_graph_data, create_max_flow_model, solve_max_flow

# Obtient le nom du fichier de données du graphe à partir des arguments de la ligne de commande
file_path = sys.argv[1]

# Lit les données du graphe à partir du fichier
num_nodes, source_node, sink_node, num_arcs, arc_data = read_graph_data(
    file_path)

# Crée le modèle de flot maximum
model = create_max_flow_model(
    num_nodes, source_node, sink_node, num_arcs, arc_data)

# Résout le modèle et génère le fichier LP
output_file = file_path.replace(".txt", ".lp")
solve_max_flow(model, output_file)
