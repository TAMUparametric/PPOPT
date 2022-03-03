import copy
# imports for suppressing output of display_tests
import os
import sys

import numpy
import pytest
from src.ppopt.mplp_program import MPLP_Program
from src.ppopt.mpqp_program import MPQP_Program
from src.ppopt.utils.general_utils import make_column
from src.ppopt.utils.constraint_utilities import constraint_norm


@pytest.fixture()
def linear_program() -> MPLP_Program:
    """a simple mplp to test the dimensional correctness of its functions"""
    A = numpy.eye(3)
    b = numpy.zeros((3, 1))
    F = numpy.ones((3, 10))
    A_t = numpy.block([[-numpy.eye(5)], [numpy.eye(5)]])
    b_t = numpy.ones((10, 1))
    c = numpy.ones((3, 1))
    H = numpy.zeros((A.shape[1], F.shape[1]))
    return MPLP_Program(A, b, c, H, A_t, b_t, F, equality_indices = [0])


@pytest.fixture()
def quadratic_program() -> MPQP_Program:
    """a simple mplp to test the dimensional correctness of its functions"""
    A = numpy.array(
        [[1, 1, 0, 0], [0, 0, 1, 1], [-1, 0, -1, 0], [0, -1, 0, -1], [-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0],
         [0, 0, 0, -1]])
    b = numpy.array([350, 600, 0, 0, 0, 0, 0, 0]).reshape(8, 1)
    c = 25 * make_column([1, 1, 1, 1])
    F = numpy.array([[0, 0], [0, 0], [-1, 0], [0, -1], [0, 0], [0, 0], [0, 0], [0, 0]])
    Q = 2.0 * numpy.diag([153, 162, 162, 126])

    CRa = numpy.vstack((numpy.eye(2), -numpy.eye(2)))
    CRb = numpy.array([1000, 1000, 0, 0]).reshape(4, 1)
    H = numpy.zeros((F.shape[1], Q.shape[0]))

    prog = MPQP_Program(A, b, c, H, Q, CRa, CRb, F)
    prog.scale_constraints()
    return prog


@pytest.fixture()
def simple_qp_program() -> MPQP_Program:
    Q = numpy.array([[1]])
    c = numpy.array([[0]])
    A = numpy.array([[1], [-1]])
    b = numpy.array([[5], [0]])
    F = numpy.array([[1], [0]])
    A_t = numpy.array([[-1], [1]])
    b_t = numpy.array([[0], [1]])
    H = numpy.zeros((F.shape[1], Q.shape[0]))
    return MPQP_Program(A, b, c, H, Q, A_t, b_t, F)


def test_active_set(linear_program):
    assert linear_program.equality_indices == [0]


def test_num_x(linear_program):
    assert linear_program.num_x() == 3


def test_num_t(linear_program):
    assert linear_program.num_t() == 10


def test_num_constraints(linear_program):
    assert linear_program.num_constraints() == 3


def test_warnings_1(linear_program):
    assert linear_program.warnings() == []


def test_warnings_2(linear_program):
    linear_program_2 = copy.deepcopy(linear_program)
    linear_program_2.A = numpy.eye(10)
    assert len(linear_program_2.warnings()) > 0


def test_warnings_3(linear_program):
    linear_program_2 = copy.deepcopy(linear_program)
    linear_program_2.b = numpy.zeros((4, 1))
    assert len(linear_program_2.warnings()) > 0


def test_warnings_4(linear_program):
    linear_program_2 = copy.deepcopy(linear_program)
    linear_program_2.b_t = numpy.zeros((12, 1))
    assert len(linear_program_2.warnings()) > 0


def test_warnings_5(linear_program):
    linear_program_2 = copy.deepcopy(linear_program)
    linear_program_2.c = numpy.zeros((12, 1))
    assert len(linear_program_2.warnings()) > 0


def test_warnings_6(linear_program):
    try:
        linear_program_2 = copy.deepcopy(linear_program)
        linear_program_2.c = numpy.zeros((12))
        assert False
    except Exception:
        assert True


def test_warnings_7(linear_program):
    linear_program_2 = copy.deepcopy(linear_program)
    linear_program_2.c = numpy.zeros((12))
    assert len(linear_program_2.warnings()) > 0


def test_warnings_8(linear_program):
    linear_program_2 = copy.deepcopy(linear_program)
    linear_program_2.b = numpy.zeros((12))
    assert len(linear_program_2.warnings()) > 0


def test_warnings_9(linear_program):
    linear_program_2 = copy.deepcopy(linear_program)
    linear_program_2.F = numpy.zeros((12, 10))
    assert len(linear_program_2.warnings()) > 0


def test_warnings_10(linear_program):
    linear_program_2 = copy.deepcopy(linear_program)
    linear_program_2.A = numpy.zeros((12, 10))
    assert len(linear_program_2.warnings()) > 0


def test_latex_lp(linear_program):
    _ = linear_program.latex()

    # supress output of print
    sys_stdout = sys.stdout
    sys.stdout = open(os.devnull, 'w')

    linear_program.display_latex()
    linear_program.display_warnings()

    # unsupress printing
    sys.stdout = sys_stdout

    assert True


def test_scale_constraints_1(linear_program):
    lp_2 = copy.deepcopy(linear_program)
    lp_2.scale_constraints()
    AF_block = numpy.block([lp_2.A, -lp_2.F])
    theta_block = lp_2.A_t
    assert numpy.allclose(numpy.ones((linear_program.num_x(), 1)), constraint_norm(AF_block))
    assert numpy.allclose(numpy.ones((linear_program.A_t.shape[0], 1)), constraint_norm(theta_block))


################
# QP programming
################

def test_warnings_qp_1(quadratic_program):
    qp_program_2 = copy.deepcopy(quadratic_program)
    qp_program_2.Q = numpy.zeros((3, 4))
    assert len(qp_program_2.warnings()) > 0


def test_warnings_qp_2(quadratic_program):
    qp_program_2 = copy.deepcopy(quadratic_program)
    qp_program_2.Q = numpy.zeros((5, 5))
    assert len(qp_program_2.warnings()) > 0


def test_warnings_qp_3(quadratic_program):
    qp_program_2 = copy.deepcopy(quadratic_program)
    qp_program_2.Q = numpy.diag([1, 0, -1])
    assert len(qp_program_2.warnings()) > 0


def test_warnings_qp_4(quadratic_program):
    qp_program_2 = copy.deepcopy(quadratic_program)
    qp_program_2.Q = numpy.diag([1, 0, 10 ** -5])
    assert len(qp_program_2.warnings()) > 0


def test_latex(quadratic_program):
    _ = quadratic_program.latex()

    # supress output of print
    sys_stdout = sys.stdout
    sys.stdout = open(os.devnull, 'w')

    quadratic_program.display_warnings()
    quadratic_program.display_latex()

    # unsupress printing
    sys.stdout = sys_stdout

    assert True


def test_solve_theta_mpqp_1(simple_qp_program):
    theta = numpy.array([[.5]])
    soln = simple_qp_program.solve_theta(theta)
    print(soln)
    assert numpy.allclose(soln.sol, numpy.array([0]))
    assert numpy.allclose(soln.dual, numpy.array([0, 0]))
    assert numpy.allclose(soln.slack, numpy.array([5.5, 0]))
    assert numpy.allclose(soln.active_set, numpy.array([1]))
