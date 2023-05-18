# script python3 nomm´e generate model.py
# prenant en param`etre en ligne de commande le nom
# d’une instance
# inst-n-p.txt
# dans le mˆeme dossier
# et qui g´en`ere un programme lin´eaire en nombre
# entiers de cette instance au format
# CPLEX LP vu en TP.
# Ce programme doit ˆetre sauv´e dans un fichier
# model-n-p.lp.
# Le script appel´e sur l’instance inst-300-0.3.txt via la commande
# python3 generate_model.py inst-300-0.3.txt
# doit g´en´erer un ficher model-300-0.3.lp.
# Comme utilis´e en TP, le fichier doit pouvoir ˆetre r´esolu et
# sauver les r´esultats avec la commande
# glpsol --lp model-300-0.3.lp -o model-300-0.3.sol


import pyomo.environ as pyo
from loguru import logger

def solve_max_flow_glpk(file_path):
    # Read input origin_file
    with open(file_path, 'r') as f:
        lines = f.readlines()

    # Parse input origin_file
    nodes = int(lines[0].split()[1])
    source = int(lines[1].split()[1])
    sink = int(lines[2].split()[1])
    arcs = int(lines[3].split()[1])
    arcs_data = [tuple(map(int, line.split())) for line in lines[4:]]
    logger.debug(f'nodes: {nodes}')
    logger.debug(f'source: {source}')
    logger.debug(f'sink: {sink}')
    logger.debug(f'arcs: {arcs}')
    logger.debug(f'arcs_data: {arcs_data}')
    logger.debug(f'arcs_data[0]: {arcs_data[0]}')


    # Create a concrete model fully defined and
    # ready to be solved
    model = pyo.ConcreteModel() # concrete is for

    # Define variables
    # model.f is a dictionary of variables indexed by
    # the set of arcs
    model.f = pyo.Var([(i, j) for i, j, c in arcs_data]
                      , within=pyo.NonNegativeReals)

    # Define objective
    # model.obj is an objective function
    # sense=pyo.maximize is for maximization
    model.obj = pyo.Objective(expr=pyo.summation(model.f)
                              , sense=pyo.maximize)

    # Define constraints
    model.node_balance = pyo.ConstraintList()
    for i in range(1, nodes + 1):
        if i == source:
            model.node_balance.add(pyo.summation(model.f[i, j] for i, j, c in arcs_data if i == source) == pyo.summation(model.f[j, i] for j, i, c in arcs_data if j == source))
        elif i == sink:
            model.node_balance.add(pyo.summation(model.f[i, j] for i, j, c in arcs_data if i == sink) == pyo.summation(model.f[j, i] for j, i, c in arcs_data if j == sink))
        else:
            model.node_balance.add(pyo.summation(model.f[i, j] for i, j, c in arcs_data if i == i) == pyo.summation(model.f[j, i] for j, i, c in arcs_data if j == i))

    model.arc_capacity = pyo.ConstraintList()
    for i, j, c in arcs_data:
        model.arc_capacity.add(model.f[i, j] <= c)

    # Solve model
    solver = pyo.SolverFactory('glpk')
    solver.solve(model)

    # Print solution
    print('Maximum flow: ', pyo.value(model.obj))
    print('Flow on arcs:')
    for i, j, c in arcs_data:
        print(f'f({i}, {j}) = {pyo.value(model.f[i, j])}')

    # Write solution to origin_file
    with open(file_path.replace('.txt', '.sol'), 'w') as f:
        f.write(f'{pyo.value(model.obj)}\n')
        for i, j, c in arcs_data:
            f.write(f'{i} {j} {pyo.value(model.f[i, j])}\n')

    # Write model to origin_file
    with open(file_path.replace('.txt', '.lp'), 'w') as f:
        model.pprint(ostream=f)

solve_max_flow_glpk('Instances/Instances/inst-300-0.3.txt')
