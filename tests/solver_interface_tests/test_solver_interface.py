from src.ppopt.solver_interface.solver_interface import solve_lp, solve_qp, solve_milp, solve_miqp

def test_solver_not_supported_1():
    try:
        solve_lp(None, None, None, deterministic_solver='MyBigBrain')
        assert False
    except RuntimeError:
        assert True


def test_solver_not_supported_2():
    try:
        solve_qp(None, None, None, None, deterministic_solver='MyBigBrain')
        assert False
    except RuntimeError:
        assert True


def test_solver_not_supported_3():
    try:
        solve_miqp(None, None, None, None, deterministic_solver='MyBigBrain')
        assert False
    except RuntimeError:
        assert True


def test_solver_not_supported_4():
    try:
        solve_milp(None, None, None, deterministic_solver='MyBigBrain')
        assert False
    except RuntimeError:
        assert True
