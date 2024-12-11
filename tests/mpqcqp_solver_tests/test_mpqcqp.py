# simple test, just for coverage
import numpy
import pytest

from src.ppopt.mp_solvers.solve_mpqcqp import mpqcqp_algorithm, solve_mpqcqp
from tests.test_fixtures import pappas_qcqp_1, pappas_qcqp_1_adapted, pappas_qcqp_2


def test_convex_mpqcqp(pappas_qcqp_1):
    sol = solve_mpqcqp(pappas_qcqp_1, mpqcqp_algorithm.combinatorial)

    assert len(sol) == 3

    theta_points = numpy.array([[0, 1.5], [1, 0.5], [0.5, 3]])
    expected_x_points = numpy.array([[-2], [-numpy.sqrt(0.5) - 1], [-2.5]])
    expected_objective_values = numpy.array([1, 1.08578643763, 1.25])
    for i, theta in enumerate(theta_points):
        x_point = sol.evaluate(theta)
        assert numpy.isclose(x_point, expected_x_points[i])
        objective_val = pappas_qcqp_1.evaluate_objective(x_point, theta)
        assert numpy.isclose(objective_val, expected_objective_values[i])


def test_convex_mpqcqp_implicit(pappas_qcqp_1):
    sol = solve_mpqcqp(pappas_qcqp_1, mpqcqp_algorithm.combinatorial_implicit)

    assert len(sol) == 3

    # theta_points = numpy.array([[0, 1.5], [1, 0.5], [0.5, 3]])
    # expected_x_points = numpy.array([[-2], [-numpy.sqrt(0.5) - 1], [-2.5]])
    # expected_objective_values = numpy.array([1, 1.08578643763, 1.25])
    # for i, theta in enumerate(theta_points):
    #     x_point = sol.evaluate(theta)
    #     assert numpy.isclose(x_point, expected_x_points[i])
    #     objective_val = pappas_qcqp_1.evaluate_objective(x_point, theta)
    #     assert numpy.isclose(objective_val, expected_objective_values[i])


def test_convex_mpqcqp_parallel(pappas_qcqp_1):
    sol = solve_mpqcqp(pappas_qcqp_1, mpqcqp_algorithm.combinatorial_parallel)

    assert len(sol) == 3

    theta_points = numpy.array([[0, 1.5], [1, 0.5], [0.5, 3]])
    expected_x_points = numpy.array([[-2], [-numpy.sqrt(0.5) - 1], [-2.5]])
    expected_objective_values = numpy.array([1, 1.08578643763, 1.25])
    for i, theta in enumerate(theta_points):
        x_point = sol.evaluate(theta)
        assert numpy.isclose(x_point, expected_x_points[i])
        objective_val = pappas_qcqp_1.evaluate_objective(x_point, theta)
        assert numpy.isclose(objective_val, expected_objective_values[i])

def test_convex_mpqcqp_2(pappas_qcqp_1_adapted):
    sol = solve_mpqcqp(pappas_qcqp_1_adapted, mpqcqp_algorithm.combinatorial)
    assert len(sol) == 1

def test_convex_mpqcqp_2_parallel(pappas_qcqp_1_adapted):
    sol = solve_mpqcqp(pappas_qcqp_1_adapted, mpqcqp_algorithm.combinatorial_parallel)
    assert len(sol) == 1

@pytest.mark.skip(reason="solving this nonconvex problem takes significantly longer than the other tests")
def test_nonconvex_qcqp(pappas_qcqp_2):
    sol = solve_mpqcqp(pappas_qcqp_2, mpqcqp_algorithm.combinatorial)

    assert len(sol) == 4
    theta_points = numpy.array([[0, 0], [1, 0], [-2, 2], [2, 3]])
    expected_x_points = numpy.array([[2.5], [3.09016994], [2], [-8.92261629]])
    expected_objective_values = numpy.array([-0.25, 0.09830056250525843, 0, 130.22616289332566])
    for i, theta in enumerate(theta_points):
        x_point = sol.evaluate(theta)
        assert numpy.isclose(x_point, expected_x_points[i])
        objective_val = pappas_qcqp_2.evaluate_objective(x_point, theta)
        assert numpy.isclose(objective_val, expected_objective_values[i])

@pytest.mark.skip(reason="solving this nonconvex problem takes significantly longer than the other tests")
def test_nonconvex_qcqp_parallel(pappas_qcqp_2):
    sol = solve_mpqcqp(pappas_qcqp_2, mpqcqp_algorithm.combinatorial_parallel)

    assert len(sol) == 4
    theta_points = numpy.array([[0, 0], [1, 0], [-2, 2], [2, 3]])
    expected_x_points = numpy.array([[2.5], [3.09016994], [2], [-8.92261629]])
    expected_objective_values = numpy.array([-0.25, 0.09830056250525843, 0, 130.22616289332566])
    for i, theta in enumerate(theta_points):
        x_point = sol.evaluate(theta)
        assert numpy.isclose(x_point, expected_x_points[i])
        objective_val = pappas_qcqp_2.evaluate_objective(x_point, theta)
        assert numpy.isclose(objective_val, expected_objective_values[i])