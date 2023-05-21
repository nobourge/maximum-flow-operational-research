import os
import sys
import time
import subprocess
from loguru import logger



def clean_file(file_path):

    try :
        with open(file_path, 'r') as f :
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
            logger.info(f"Le fichier {file_path} a bien été nettoyé")
    except :
        logger.info(f'type {file_path} is not recognized')

def separate_arc_twins(arcs_data):
    arc_twins = []
    unique_arcs = []

    for i in range(len(arcs_data)):
        arc1 = arcs_data[i]
        is_twin = False

        for j in range(i + 1, len(arcs_data)):
            arc2 = arcs_data[j]

            if arc1[0] == arc2[0] and arc1[1] == arc2[1]:
                is_twin = True
                arc_twins.append(arc1)
                arc_twins.append(arc2)
                break

        if not is_twin:
            unique_arcs.append(arc1)

    return unique_arcs, arc_twins


def get_outgoing_arcs(node, arcs_data):
    outgoing_arcs = []
    for arc in arcs_data:
        if arc[0] == node or arc[1] == node:
            outgoing_arcs.append(arc)
    return outgoing_arcs


# ATTENTION TRES DANGEREUSE AVEC LA BOUCLE WHILE
# A N'UTILISER
def go_to_src():
    current_dir = os.getcwd()
    current_dir_name = os.path.basename(current_dir)

    while current_dir_name != 'src':
        parent_dir = os.path.join(current_dir, os.pardir)
        os.chdir(parent_dir)
        print(f"parent directory : {parent_dir}")
        current_dir_name = os.path.basename(os.getcwd())
        current_dir = os.getcwd()
        print(f"current dir : {os.getcwd()}")
        print(f"current dir name : {current_dir_name}")


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

    # Separate arc twins from arcs data
    arcs_data, arcs_twins = separate_arc_twins(arcs_data)

    # Get arcs outgoing from the source
    arcs_source = [arc for arc in arcs_data if arc[0] == source]
    arcs_source, arcs_twins_source = separate_arc_twins(arcs_source)
    print(arcs_twins_source)
    # Create the maximization objective

    maximization = f'maximize\nv : f_{arcs_source[0][0]}_{arcs_source[0][1]}'
    for arc in arcs_twins_source:
        maximization += f' + f_{arc[0]}_{arc[1]}_{arc[2]}'
    for index in range(1, len(arcs_source)):
        maximization += f' + f_{arcs_source[index][0]}_{arcs_source[index][1]}'
    # Create the constraints
    constraints = 'subject to\n'
    constraint_index = 1

    for arc in arcs_data:
        constraints += f'c{constraint_index}: f_{arc[0]}_{arc[1]} <= {arc[2]}\n'
        constraint_index += 1
    for arc in arcs_twins:
        constraints += f'c{constraint_index}: f_{arc[0]}_{arc[1]}_{arc[2]} <= {arc[2]}\n'
        constraint_index += 1

    # Create the flow conservation constraints
    for node in range(1, nodes):
        if node != source and node != sink:
            # Get arcs connected to the node
            arcs_node = [arc for arc in arcs_data if arc[0] == node or arc[1] == node]
            arcs_node, arcs_twins_node = separate_arc_twins(arcs_node)

            constraints += f'c{constraint_index}: '
            for arc in arcs_node:
                if arc[1] == node:
                    if constraints[-2:] == ': ':
                        constraints += f'f_{arc[0]}_{arc[1]} '
                    else:
                        constraints += f' + f_{arc[0]}_{arc[1]} '
                else:
                    constraints += f'- f_{arc[0]}_{arc[1]}'
            constraints += f' = 0\n'
            constraint_index += 1

            if len(arcs_twins_node) > 0:
                constraints += f'c{constraint_index}: '
                for arc in arcs_twins_node:
                    if arc[1] == node:
                        if constraints[-2:] == ': ':
                            constraints += f'f_{arc[1]}_{arc[0]}_{arc[2]} '
                        else:
                            constraints += f' + f_{arc[1]}_{arc[0]}_{arc[2]} '
                    else:
                        constraints += f'- f_{arc[1]}_{arc[0]}_{arc[2]}'
                    constraints += f' = 0\n'
                    constraint_index += 1

    # Create integer variables
    integer = 'integer \n'
    for arc in arcs_twins_source:
        integer += f'f_{arc[1]}_{arc[0]}_{arc[2]} \n'
    for arc in arcs_source:
        integer += f'f_{arc[0]}_{arc[1]} \n'

    # Write to a new file
    model = [maximization, "\n", constraints, "\n", integer]

    with open(file_path.replace('txt','lp'), 'w') as f:
        f.writelines(model)

    return model

if __name__ == '__main__':

    files = os.listdir('./Instances/Instances')
    files.sort()

    root = os.chdir('./')
    cumulated_time = 0

    last_file = ''

    for file in files:

        file_path = os.path.abspath(f"Instances/Instances/{file}")
        #clean_file(file_path)

        while cumulated_time < 300:

            start = time.time()
            try:
                logger.info(f"Run model generation on {file}")
                subprocess.run(["python3", "model_generation.py", file_path])

            except:
                logger.info("Linear model not produced")

            end = time.time()
            cumulated_time += end

            elapsed_time = end - start
            logger.info(f"Elapsed time : {elapsed_time} seconds")

            if not os.path.exists("Timing"):
                os.makedirs("Timing")
            with open('Timing/' + file.replace('inst', 'model').replace('.txt', '.lp'), 'w') as f:
                f.write(f'Elapsed time: {end - start} seconds')

            with open('Timing/all_in_seconds.txt', 'a') as f:
                f.write(f'{end - start} for {file} \n')

        last_file = file

    logger.info(f"Last lp file produced is {last_file}")

#    file_path = sys.argv[1]
#    file_path = os.path.abspath(file_path)
#    print(file_path)
#    print(type(file_path))
#    clean_file(file_path)
#    solve_max_flow_glpk(file_path)
