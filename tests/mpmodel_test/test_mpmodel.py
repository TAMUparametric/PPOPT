from itertools import product

from src.ppopt.mpmodel import MPModel
from src.ppopt.mp_solvers.solve_mpqp import solve_mpqp

def test_market_problem_modeler_mplp():
    # make a mpLP for the market problem
    model = MPModel()

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

    assert(len(sol.critical_regions) == 3)

    print(sol)

def test_market_problem_modeler_mpqp():
    # make a mpLP for the market problem
    model = MPModel()

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
    model.set_objective(sum(cost[f, m] * x[f, m]**2 + 25 * x[f, m] for f, m in product(factories, markets)))

    market_problem = model.formulate_problem()

    sol = solve_mpqp(market_problem)

    assert(len(sol.critical_regions) == 4)