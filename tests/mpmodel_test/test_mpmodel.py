from itertools import product

from src.ppopt.mpmodel import MPModeler, VariableType
from src.ppopt.mp_solvers.solve_mpqp import solve_mpqp, mpqp_algorithm
from src.ppopt.mp_solvers.solve_mpmiqp import solve_mpmiqp

import numpy


def test_market_problem_modeler_mplp():
    # make a mpLP for the market problem
    model = MPModeler()

    # define problem data
    factories = ['se', 'sd']
    markets = ['ch', 'to']
    capacities = {'se': 350, 'sd': 600}
    cost = {('se', 'ch'): 178, ('se', 'to'): 187, ('sd', 'ch'): 187, ('sd', 'to'): 151}

    # make a variable for each factory-market production pair
    x = {(f, m): model.add_var(name=f'x[{f},{m}]') for f, m in product(factories, markets)}

    # make a parameter for each market demand
    d = {m: model.add_param(name=f'd_[{m}]') for m in markets}

    # bounds on the production capacity of each factory
    model.add_constrs(sum(x[f, m] for m in markets) <= capacities[f] for f in factories)

    # demand satisfaction for each market
    model.add_constrs(sum(x[f, m] for f in factories) >= d[m] for m in markets)

    # bounds on the parametric demand
    model.add_constrs(d[m] <= 1000 for m in markets)
    model.add_constrs(d[m] >= 0 for m in markets)

    # non-negativity of the production variables
    model.add_constrs(x[f, m] >= 0 for f, m in product(factories, markets))

    # set the objective to minimize the total cost
    model.set_objective(sum(cost[f, m] * x[f, m] for f, m in product(factories, markets)))

    market_problem = model.formulate_problem()

    sol = solve_mpqp(market_problem)

    assert (len(sol.critical_regions) == 3)


def test_market_problem_modeler_mpqp():
    # make a mpLP for the market problem
    model = MPModeler()

    # define problem data
    factories = ['se', 'sd']
    markets = ['ch', 'to']
    capacities = {'se': 350, 'sd': 600}
    cost = {('se', 'ch'): 178, ('se', 'to'): 187, ('sd', 'ch'): 187, ('sd', 'to'): 151}

    # make a variable for each factory-market production pair
    x = {(f, m): model.add_var(name=f'x[{f},{m}]') for f, m in product(factories, markets)}

    # make a parameter for each market demand
    d = {m: model.add_param(name=f'd_[{m}]') for m in markets}

    # bounds on the production capacity of each factory
    model.add_constrs(sum(x[f, m] for m in markets) <= capacities[f] for f in factories)

    # demand satisfaction for each market
    model.add_constrs(sum(x[f, m] for f in factories) >= d[m] for m in markets)

    # bounds on the parametric demand
    model.add_constrs(d[m] <= 1000 for m in markets)
    model.add_constrs(d[m] >= 0 for m in markets)

    # non-negativity of the production variables
    model.add_constrs(x[f, m] >= 0 for f, m in product(factories, markets))

    # set the objective to minimize the total cost
    model.set_objective(sum(cost[f, m] * x[f, m] ** 2 + 25 * x[f, m] for f, m in product(factories, markets)))

    market_problem = model.formulate_problem()

    sol = solve_mpqp(market_problem)

    assert (len(sol.critical_regions) == 4)


def test_portfolio_modeler():
    # define problem data
    numpy.random.seed(123456789)
    num_assets = 10
    S = numpy.random.randn(num_assets, num_assets)
    S = S @ S.T / 10
    mu = numpy.random.rand(num_assets) / 100

    model = MPModeler()

    # make a variable for each asset
    assets = [model.add_var(name=f'w[{i}]') for i in range(num_assets)]

    # define the parametric return
    r = model.add_param(name='R')

    # investment must add to one
    model.add_constr(sum(assets) == 1)

    # the expeted return must be r
    model.add_constr(sum(mu[i] * assets[i] for i in range(num_assets)) == r)

    # all assets must be non-negative (no shorting)
    model.add_constrs(asset >= 0 for asset in assets)

    # parametric return must be constrained to be [min(mu), max(mu)]

    model.add_constr(r >= min(mu))
    model.add_constr(r <= max(mu))

    # set the objective to minimize the risk
    model.set_objective(sum(S[i, j] * assets[i] * assets[j] for i in range(num_assets) for j in range(num_assets)))

    portfolio = model.formulate_problem()

    sol = solve_mpqp(portfolio)


def test_simple_mpMILP_modeler():
    model = MPModeler()

    x_0 = model.add_var(vtype=VariableType.continuous)
    y_1 = model.add_var(vtype=VariableType.binary)
    y_2 = model.add_var(vtype=VariableType.binary)

    theta = model.add_param()

    model.add_constr(y_1 + y_2 <= 1)
    model.add_constr(x_0 <= theta)

    model.add_constr(-x_0 <= 0)
    model.add_constr(x_0 - y_1 <= 0)
    model.add_constr(x_0 - y_2 <= 0)

    model.add_constr(theta <= 2)
    model.add_constr(theta >= -2)

    model.set_objective(-3 * x_0)

    mpmilp = model.formulate_problem()

    mpmilp.solver.solvers['lp'] = 'glpk'

    sol = solve_mpmiqp(mpmilp)


def test_simple_mpMIQP_modeler():
    model = MPModeler()

    x_0 = model.add_var(vtype=VariableType.continuous)
    y_1 = model.add_var(vtype=VariableType.binary)
    y_2 = model.add_var(vtype=VariableType.binary)

    theta = model.add_param()

    model.add_constr(y_1 + y_2 <= 1)
    model.add_constr(x_0 <= theta)

    model.add_constr(-x_0 <= 0)
    model.add_constr(x_0 - y_1 <= 0)
    model.add_constr(x_0 - y_2 <= 0)

    model.add_constr(theta <= 2)
    model.add_constr(theta >= -2)

    model.set_objective(-3 * x_0 + x_0 ** 2 + y_1 ** 2 + y_2 ** 2)

    mpmilp = model.formulate_problem()

    mpmilp.solver.solvers['lp'] = 'glpk'

    sol = solve_mpmiqp(mpmilp)


def test_simple_mpMIQP_H_modeler():
    model = MPModeler()

    x_0 = model.add_var(vtype=VariableType.continuous)
    y_1 = model.add_var(vtype=VariableType.binary)
    y_2 = model.add_var(vtype=VariableType.binary)

    theta = model.add_param()

    model.add_constr(y_1 + y_2 == 1)
    model.add_constr(x_0 <= theta)

    model.add_constr(-x_0 <= 0)
    model.add_constr(x_0 - y_1 <= 0)
    model.add_constr(x_0 - y_2 <= 0)

    model.add_constr(theta <= 2)
    model.add_constr(theta >= -2)

    model.set_objective(-3 * x_0 + theta**2 + theta + x_0 ** 2 + y_1 ** 2 + y_2 ** 2 + theta * x_0 + theta * y_1 + theta * y_2 + x_0*theta)

    mpmilp = model.formulate_problem()

    mpmilp.solver.solvers['lp'] = 'glpk'
    mpmilp.solver.solvers['qp'] = 'daqp'

    sol = solve_mpmiqp(mpmilp)

def test_assigment_problem():
    model = MPModeler()

    # make problem data
    N = 2
    cost = numpy.random.rand(N, N)

    # make a variable for each possible worker task assignment
    x = {(i, j): model.add_var(name=f'x_[{i}][{j}]') for i in range(N) for j in range(N)}

    # add a parametric cost
    t = model.add_param()

    # set the objective
    model.set_objective(sum(cost[i, j] * x[i, j] for i in range(N) for j in range(N)) + t * x[0, 0])

    # assignment problem constraints
    model.add_constrs(sum(x[i, j] for i in range(N)) == 1 for j in range(N))
    model.add_constrs(sum(x[i, j] for j in range(N)) == 1 for i in range(N))

    model.add_constrs(x[i, j] >= 0 for i in range(N) for j in range(N))
    model.add_constrs(x[i, j] <= 1 for i in range(N) for j in range(N))

    # parametric constraints
    model.add_constr(t <= 5)
    model.add_constr(t >= -5)

    # formulate and solve
    assignment_problem = model.formulate_problem()

    sol = solve_mpqp(assignment_problem, mpqp_algorithm.combinatorial_graph)