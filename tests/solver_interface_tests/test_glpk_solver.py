import numpy
from src.ppopt.solver_interface.cvxopt_interface import solve_lp_cvxopt


def test_solve_lp_1():
    A = numpy.array([[1.0, 1.0], [1.0, -1.0]])
    b = numpy.array([[0.0], [0.0]])
    c = numpy.array([[0.0], [0.0]])

    soln = solve_lp_cvxopt(c, A, b, [0, 1])
    assert numpy.allclose(numpy.zeros(2), soln.sol)


def test_solve_lp_2():
    A = numpy.array([[-1, 0], [0, -1], [-1, 1]], dtype='float64')
    b = numpy.array([[0], [0], [1]], dtype='float64')
    c = numpy.array([[1], [1]], dtype='float64')

    soln = solve_lp_cvxopt(c, A, b)
    assert numpy.allclose(numpy.zeros(2), soln.sol)
    print(soln)

def test_infeasfible_lp():
    A = numpy.array([[1], [-1]], dtype='float64')
    b = numpy.array([[-1], [-1]], dtype='float64')
    c = None
    soln = solve_lp_cvxopt(c, A, b)
    assert soln is None


def test_indefinite_lp():
    assert solve_lp_cvxopt(None, None, None) is None


def test_indefinite_lp_2():
    assert solve_lp_cvxopt(None, numpy.zeros((0, 0)), None) is None
