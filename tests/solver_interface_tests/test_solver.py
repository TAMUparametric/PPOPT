from src.ppopt.solver import Solver


def test_solver_constructor_1():
    _ = Solver()
    assert True


def test_solver_constructor_2():
    _ = Solver({'lp': 'gurobi', 'qp': 'gurobi'})
    assert True


def test_solver_wrong_solver_1():
    try:
        _ = Solver({'mlp': 'python'})
        assert False
    except RuntimeError:
        assert True
    except Exception:
        assert False


def test_solver_wrong_solver_2():
    try:
        _ = Solver({'lp': 'python'})
        assert False
    except RuntimeError:
        assert True
    except Exception:
        assert False


def test_solver_defined_problem_1():
    solver = Solver({'lp': 'gurobi', 'qp': 'gurobi'})
    try:
        solver.check_supported_problem('miqp')
    except RuntimeError:
        assert True
