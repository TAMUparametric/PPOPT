import copy

import numpy
import pytest

from src.ppopt.critical_region import CriticalRegion
from src.ppopt.utils.general_utils import make_column

@pytest.fixture()
def region() -> CriticalRegion:
    """A square critical region with predictable properties."""
    A = numpy.eye(2)
    b = numpy.zeros((2, 1))
    C = numpy.eye(2)
    d = numpy.zeros((2, 1))
    E = numpy.block([[numpy.eye(2)], [-numpy.eye(2)]])
    f = make_column([1, 1, 0, 0])
    return CriticalRegion(A, b, C, d, E, f, [])

def test_docs(region):
    assert CriticalRegion.__doc__ is not None


def test_repr(region):
    assert len(region.__repr__()) > 0


def test_critical_region_construction(region):
    assert region is not None


def test_evaluate(region):
    theta = numpy.ones((2, 1))
    assert numpy.allclose(region.evaluate(theta), theta)


def test_lagrange_multipliers(region):
    theta_point = numpy.array([[1], [1]])
    assert numpy.allclose(theta_point, region.lagrange_multipliers(theta_point))


def test_is_inside_1(region):
    num_tests = 10
    for _ in range(num_tests):
        theta = numpy.random.random((2, 1))
        assert region.is_inside(theta)


def test_is_inside_2(region):
    num_tests = 10
    for _ in range(num_tests):
        theta = numpy.random.random((2, 1)) - numpy.array([[1], [1]])
        assert not region.is_inside(theta)


def test_is_full_dimension_1(region):
    assert region.is_full_dimension()


def test_is_full_dimension_2(region):
    region_2 = copy.deepcopy(region)
    region_2.f = make_column([0, 1, 0, 0])
    assert not region_2.is_full_dimension()


def test_get_constraints_1(region):
    region_2 = copy.deepcopy(region)
    assert len(region_2.get_constraints()) == 2
