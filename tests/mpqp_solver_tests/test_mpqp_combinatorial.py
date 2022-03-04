import numpy

from src.ppopt.mp_solvers.mpqp_combinatorial import *
from src.ppopt.mp_solvers.solver_utils import generate_children_sets
from src.ppopt.mpqp_program import MPQP_Program
from ..test_fixtures import *


def test_check(filled_combo_tester):
    assert not filled_combo_tester.check([1])
    assert not filled_combo_tester.check([2])
    assert not filled_combo_tester.check([3])
    assert not filled_combo_tester.check([1, 5])
    assert not filled_combo_tester.check([1, 5, 2])

    assert filled_combo_tester.check([0, 4])
    assert filled_combo_tester.check([5])
    assert filled_combo_tester.check([5, 6])
    assert filled_combo_tester.check([5, 8])


# simple test, just for coverage
def test_add_1(blank_combo_tester):
    num_items = len(blank_combo_tester.combos)
    blank_combo_tester.add_combo([])
    assert len(blank_combo_tester.combos) == num_items + 1


# simple test, just for coverage
def test_add_2(filled_combo_tester):
    num_items = len(filled_combo_tester.combos)
    filled_combo_tester.add_combo([])
    assert len(filled_combo_tester.combos) == num_items + 1


def test_generate_children_1(blank_combo_tester):
    output = generate_children_sets([], 8, blank_combo_tester)
    assert output == [[0], [1], [2], [3], [4], [5], [6], [7]]


def test_generate_children_2(blank_combo_tester):
    output = generate_children_sets([1, 2, 3, 5], 8, blank_combo_tester, )
    assert output == [[1, 2, 3, 5, 6], [1, 2, 3, 5, 7]]


def test_generate_children_3(filled_combo_tester):
    output = generate_children_sets([], 8, filled_combo_tester)
    assert output == [[0], [4], [5], [6], [7]]


def test_generate_children_4(filled_combo_tester):
    output = generate_children_sets([0], 8, filled_combo_tester)
    assert output == [[0, 4], [0, 5], [0, 6], [0, 7]]


def test_generate_children_5(blank_combo_tester, linear_program):
    output = generate_children_sets(linear_program.equality_indices, linear_program.num_constraints(), blank_combo_tester)
    assert output == [[0, 1], [0, 2]]


def test_generate_children_6(blank_combo_tester, linear_program):
    output = generate_children_sets([0, 1], linear_program.num_constraints(), blank_combo_tester)
    assert output == [[0, 1, 2]]


def test_generate_children_7(blank_combo_tester, linear_program):
    A = numpy.array(
        [[1, 1, 0, 0], [0, 0, 1, 1], [-1, 0, -1, 0], [0, -1, 0, -1], [-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0],
         [0, 0, 0, -1]])
    b = numpy.array([350, 600, 0, 0, 0, 0, 0, 0]).reshape(8, 1)
    c = 25 * numpy.array([[1], [1], [1], [1]])
    F = numpy.array([[0, 0], [0, 0], [-1, 0], [0, -1], [0, 0], [0, 0], [0, 0], [0, 0]])
    Q = numpy.diag([153, 162, 162, 126])

    CRa = numpy.vstack((numpy.eye(2), -numpy.eye(2)))
    CRb = numpy.array([1000, 1000, 0, 0]).reshape(4, 1)
    H = numpy.zeros((F.shape[1], Q.shape[0]))
    program = MPQP_Program(A, b, c, H, Q, CRa, CRb, F, equality_indices = [0])

    output = generate_children_sets(program.equality_indices, program.num_constraints(), blank_combo_tester)
    assert output == [[0, 1], [0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 7]]


def test_check_feasibility_1(quadratic_program, blank_combo_tester):
    output = check_child_feasibility(quadratic_program, [[], [1], [2], [0, 1, 2, 3, 4]], blank_combo_tester)
    assert output == [[], [1], [2]]
