# chat gpt generated
#
import pyomo.environ as pyo

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

    # Create model
    model = pyo.ConcreteModel()

    # Define variables
    model.f = pyo.Var([(i, j) for i, j, c in arcs_data], within=pyo.NonNegativeReals)

    # Define objective
    model.obj = pyo.Objective(expr=pyo.summation(model.f), sense=pyo.maximize)

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

    # Write solution to file
    with open(file_path.replace('.txt', '.sol'), 'w') as f:
        f.write(f'{pyo.value(model.obj)}\n')
        for i, j, c in arcs_data:
            f.write(f'{i} {j} {pyo.value(model.f[i, j])}\n')

    # Write model to file
    with open(file_path.replace('.txt', '.lp'), 'w') as f:
        model.pprint(ostream=f)
