import numpy
from src.ppopt.solver_interface.gurobi_solver_interface import solve_lp_gurobi, solve_qp_gurobi, solve_miqp_gurobi, solve_milp_gurobi


def test_solve_lp_1():
    A = numpy.array([[1.0, 1.0], [1.0, -1.0]])
    b = numpy.array([[0.0], [0.0]])
    c = numpy.array([[0.0], [0.0]])

    soln = solve_lp_gurobi(c, A, b, [0, 1])
    assert numpy.allclose(numpy.zeros(2), soln.sol)


def test_solve_lp_2():
    A = numpy.array([[-1, 0], [0, -1], [-1, 1]], dtype='float64')
    b = numpy.array([[0], [0], [1]], dtype='float64')
    c = numpy.array([[1], [1]], dtype='float64')

    soln = solve_lp_gurobi(c, A, b)
    assert numpy.allclose(numpy.zeros(2), soln.sol)


def test_solve_qp_1():
    rand_dim = numpy.random.randint(1, 10)
    Q = numpy.eye(rand_dim)
    soln = solve_qp_gurobi(Q, None, None, None)
    assert numpy.allclose(numpy.zeros(rand_dim), soln.sol)


def test_solve_qp_2():
    rand_dim = numpy.random.randint(1, 10)
    Q = numpy.eye(rand_dim)
    A = numpy.vstack((numpy.eye(rand_dim), -numpy.eye(rand_dim)))
    b = numpy.ones((rand_dim * 2, 1))

    soln = solve_qp_gurobi(Q, None, A, b)

    assert numpy.allclose(numpy.zeros(rand_dim), soln.sol)


def test_solve_qp_3():
    rand_dim = numpy.random.randint(1, 10)
    Q = None
    A = numpy.vstack((numpy.eye(rand_dim), -numpy.eye(rand_dim)))
    b = numpy.ones(rand_dim * 2)
    c = numpy.ones((rand_dim, 1))
    soln = solve_qp_gurobi(Q, c, A, b, [0])
    assert soln.sol[0] == 1
    assert numpy.allclose(soln.sol[1:], -numpy.ones(rand_dim - 1))


def test_solve_miqp_1():
    rand_dim = numpy.random.randint(1, 10)
    Q = None
    A = None
    b = None
    c = numpy.ones((rand_dim, 1))

    soln = solve_miqp_gurobi(Q, c, A, b, [], [0])
    print(soln)
    assert soln is None


def test_solve_miqp_2():
    rand_dim = numpy.random.randint(1, 10)
    Q = numpy.eye(rand_dim)
    A = numpy.vstack((numpy.eye(rand_dim), -numpy.eye(rand_dim)))
    b = numpy.block([[numpy.ones((rand_dim, 1))], [numpy.zeros((rand_dim, 1))]])
    c = numpy.zeros((rand_dim, 1))
    soln = solve_miqp_gurobi(Q, c, A, b, [], [0])
    assert numpy.allclose(numpy.zeros(rand_dim), soln.sol)


def test_solve_milp():
    rand_dim = numpy.random.randint(1, 10)
    A = numpy.vstack((numpy.eye(rand_dim), -numpy.eye(rand_dim)))
    b = numpy.ones(rand_dim * 2)
    c = None

    soln = solve_milp_gurobi(c, A, b, [], [0])
    assert numpy.allclose(numpy.zeros(rand_dim), soln.sol)


def test_infeasfible_lp():
    A = numpy.array([[1], [-1]], dtype='float64')
    b = numpy.array([[-1], [-1]], dtype='float64')
    c = None
    soln = solve_lp_gurobi(c, A, b)
    assert soln is None


def test_indefinite_lp_1():
    assert solve_lp_gurobi(None, None, None) is None


def test_indefinite_lp_2():
    assert solve_lp_gurobi(None, numpy.zeros((0, 0)), None) is None


def test_infeasfible_qp():
    A = numpy.array([[1], [-1]])
    b = numpy.array([[-1], [-1]])
    c = numpy.zeros((1, 1))
    soln = solve_qp_gurobi(None, c, A, b)
    assert soln is None
