import numpy
from src.ppopt.upop.point_location import PointLocation

from tests.test_fixtures import factory_solution

def test_point_location(factory_solution):
    pl = PointLocation(factory_solution)

    theta = numpy.array([[200.0], [200.0]])

    assert numpy.allclose(pl.evaluate(theta), factory_solution.evaluate(theta))
