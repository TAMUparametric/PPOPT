import numpy
from src.ppopt.utils.chebyshev_ball import chebyshev_ball


def test_chebyshev_ball_1():
    A = numpy.vstack((numpy.eye(5), -numpy.eye(5)))
    b = numpy.ones((10, 1))
    chebyshev_soln = chebyshev_ball(A, b, deterministic_solver='gurobi')
    # make sure it solved
    # solution is [0,0,0,0,0,1] -> y = [0,0,0,0,0], r = 1
    assert numpy.allclose(numpy.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0]), chebyshev_soln.sol)


def test_chebyshev_ball_2():
    A = numpy.vstack((numpy.eye(5), -numpy.eye(5)))
    b = numpy.ones((10, 1))
    chebyshev_soln = chebyshev_ball(A, b, bin_vars=None, deterministic_solver='gurobi')
    # make sure it solved
    # solution is [0,0,0,0,0,1] -> y = [0,0,0,0,0], r = 1
    assert numpy.allclose(numpy.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0]), chebyshev_soln.sol)
