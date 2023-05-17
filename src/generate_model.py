import sys


def clean_file(file_path):
    # Read input file
    with open(file_path, 'r') as f:
        lines = f.readlines()

    # Supprimer les lignes avec le même noeud de départ et d'arrivée
    for index in range(4, len(lines)):
        lines[index] = lines[index].split()
        if lines[index][0] == lines[index][1]:
            lines[index] = ''
        else:
            lines[index] = ' '.join(lines[index]) + '\n'


    # Supprime les doublons à partir de la ligne 5
    unique_lines = lines[:4]  # Copie les 4 premières lignes
    unique_lines.extend(list(set(lines[4:])))  # Supprime les doublons et ajoute à la liste


    # Ecrit le nouveau fichier
    with open(file_path, 'w') as f:
        f.writelines(unique_lines)


def solve_max_flow_glpk(file_path):
    # Read input file
    with open(file_path, 'r') as f:
        lines = f.readlines()

    # Parse input file
    nodes = int(lines[0].split()[1])
    source = int(lines[1].split()[1])
    sink = int(lines[2].split()[1])
    arcs = int(lines[3].split()[1])
    arcs_data = [tuple(map(int, line.split())) for line in lines[4:]]

    # On liste les arcs sortants de la source
    arcs_source = [arc for arc in arcs_data if arc[0] == source]


    # On crée la condition de maximisation
    maximization = f'maximize \n v : f_{arcs_source[0][0]}_{arcs_source[0][1]}'
    for index in range(1, len(arcs_source)):
        maximization += f' + f_{arcs_source[index][0]}_{arcs_source[index][1]}'
    # Vérification condition maximization
    #print(maximization)

    # On crée les contraintes
    contraintes = f'subject to \n'
    contraintes_index = 1
    # On crée les contraintes liées aux capacités des arcs
    for arc in arcs_data:
        contraintes += f'c{contraintes_index} : f_{arc[0]}_{arc[1]} <= {arc[2]} \n'
        contraintes_index += 1
    # On crée les contraintes de conservation de flot
    for node in (node for node in range(1, nodes) if node != source and node != sink):
        contraintes += f'c{contraintes_index} : '
        # On construit la somme des flots entrants pour node
        for arc in arcs_data:
            if arc[0] == node:
                if contraintes[-2:] == ': ':
                    contraintes += f'f_{arc[0]}_{arc[1]} '
                else:
                    contraintes += f'+ f_{arc[0]}_{arc[1]} '
            if arc[1] == node:
                contraintes += f'- f_{arc[0]}_{arc[1]} '
        # On crée la partie droite de la contrainte
        contraintes += '= 0 \n'
        contraintes_index += 1
    contraintes += '\n'

    # On crée les integer
    integer = f'integer \n'
    for arc in arcs_source:
        integer += f'f_{arc[0]}_{arc[1]} \n'


    # Modèle à résoudre
    model = maximization + '\n' + contraintes + integer + '\n' + 'end'
    print(model)



file = sys.argv[1]
clean_file(file)
solve_max_flow_glpk(file)