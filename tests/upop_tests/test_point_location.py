import numpy
import copy
from src.ppopt.upop.point_location import PointLocation

from tests.test_fixtures import factory_solution


def test_point_location(factory_solution):

    pl = PointLocation(factory_solution)
    pl.is_overlapping = False

    theta = numpy.array([[200.0], [200.0]])
    print(pl.evaluate(theta))
    print(factory_solution.evaluate(theta))
    print(factory_solution.program.solve_theta(theta))
    assert numpy.allclose(pl.evaluate(theta), factory_solution.evaluate(theta))


def test_point_location_overlap(factory_solution):
    new_sol = copy.deepcopy(factory_solution)
    new_sol.is_overlapping = True
    pl = PointLocation(factory_solution)

    theta = numpy.array([[200.0], [200.0]])
    print(pl.evaluate(theta))
    print(factory_solution.evaluate(theta))
    print(factory_solution.program.solve_theta(theta))
    assert numpy.allclose(pl.evaluate(theta), factory_solution.evaluate(theta))
