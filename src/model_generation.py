from pyomo.opt import SolverFactory
from pyomo.environ import *

def read_graph_data(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    num_nodes = int(lines[0].split()[1])
    source_node = int(lines[1].split()[1])
    sink_node = int(lines[2].split()[1])
    num_arcs = int(lines[3].split()[1])

    arc_data = []
    for line in lines[4:]:
        i, j, c = map(int, line.split())
        arc_data.append((i, j, c))

    return num_nodes, source_node, sink_node, num_arcs, arc_data


def create_max_flow_model(num_nodes, source_node, sink_node, num_arcs, arc_data):
    model = ConcreteModel()

    # Variables
    model.x = Var(range(num_arcs), domain=NonNegativeReals)

    # Objective function
    model.obj = Objective(expr=sum(model.x[i]
                          for i in range(num_arcs)), sense=maximize)

    # Constraints
    model.source_constraint = Constraint(expr=sum(model.x[i] for i in range(num_arcs) if arc_data[i][0] == source_node) -
                                         sum(model.x[i] for i in range(num_arcs) if arc_data[i][1] == source_node) == 1)

    model.sink_constraint = Constraint(expr=sum(model.x[i] for i in range(num_arcs) if arc_data[i][0] == sink_node) -
                                       sum(model.x[i] for i in range(num_arcs) if arc_data[i][1] == sink_node) == -1)


    model.flow_conservation = ConstraintList()
    for node in range(1, num_nodes + 1):
        if node != source_node and node != sink_node:
            inflow = sum(model.x[i]
                        for i in range(num_arcs) if arc_data[i][1] == node)
            outflow = sum(model.x[i]
                        for i in range(num_arcs) if arc_data[i][0] == node)
            model.flow_conservation.add(
                expr=inflow - outflow == 0)



    model.capacity_constraints = ConstraintList()
    for i in range(num_arcs):
        model.capacity_constraints.add(model.x[i] <= arc_data[i][2])

    return model

def solve_max_flow(model, output_file):
    # Création de l'instance du solveur GLPK
    solver = SolverFactory('glpk')

    # Résolution du modèle
    result = solver.solve(model)

    # Vérification du statut de la solution
    if result.solver.termination_condition == TerminationCondition.optimal:
        # Écriture du fichier LP
        model.write(output_file, format='lp')
    else:
        print("Le solveur n'a pas pu trouver de solution optimale.")


