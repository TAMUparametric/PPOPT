# simple test, just for coverage
import numpy

from src.ppopt.mp_solvers.solve_mpqcqp import mpqcqp_algorithm, solve_mpqcqp
from tests.test_fixtures import pappas_qcqp_1


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