import numpy
import pytest

from tests.test_fixtures import blank_solution, filled_solution, region

def test_add_region_1(blank_solution, region):
    assert len(blank_solution.critical_regions) == 0
    blank_solution.add_region(region)
    assert len(blank_solution.critical_regions) == 1


def test_add_region_2(blank_solution, region):
    num_items = len(blank_solution.critical_regions)
    blank_solution.add_region(region)
    assert len(blank_solution.critical_regions) == num_items + 1


def test_evaluate_1(blank_solution):
    theta = numpy.array([[.5], [.5]])
    assert blank_solution.evaluate(theta) is None


def test_evaluate_2(filled_solution):
    theta = numpy.array([[.5], [.5]])
    assert filled_solution.evaluate(theta) is not None
    assert numpy.allclose(filled_solution.evaluate(theta), theta)


def test_get_region_1(filled_solution):
    theta = numpy.array([[.5], [.5]])
    assert filled_solution.get_region(theta) is not None


def test_get_region_2(blank_solution):
    theta = numpy.array([[.5], [.5]])
    assert blank_solution.get_region(theta) is None

@pytest.mark.skip(reason="I am scaling the matrix array, expected output has changed")
def test_verify_1(factory_solution):
    theta = numpy.random.random((2, 1)) * 100

    region = factory_solution.get_region(theta)

    print('')
    print(region.evaluate(theta))
    print(region.lagrange_multipliers(theta))
    print(region.equality_indices)
    print(factory_solution.program.solve_theta(theta))

    assert factory_solution.verify_solution()
