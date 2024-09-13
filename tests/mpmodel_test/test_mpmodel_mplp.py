from src.ppopt.mpmodel import MPModel
from itertools import product


def test_define_market_problem():
    model = MPModel()

    factories = ['se', 'sd']
    markets = ['ch', 'to']

    x = {(f, m): model.add_var(name=f'x_{f}_{m}') for f, m in product(factories, markets)}

    theta_1 = model.add_param()
    theta_2 = model.add_param()

    model.add_constr(x['se', 'ch'] + x['se', 'ch'] <= 350)
    model.add_constr(x['sd', 'ch'] + x['sd', 'to'] <= 600)

    model.add_constr(x['se', 'ch'] + x['sd', 'ch'] >= theta_1)
    model.add_constr(x['se', 'ch'] + x['sd', 'ch'] >= theta_2)

    model.add_constrs(x[f, m] >= 0 for f, m in product(factories, markets))

    model.set_objective(178*x['se', 'ch'] + 187*x['se', 'to'] + 187*x['sd', 'ch'] + 151*x['sd', 'to'])
