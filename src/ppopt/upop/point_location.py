from typing import Optional

import numba
import numpy

from ..solution import Solution
from ..upop.upop_utils import find_unique_region_hyperplanes, get_outer_boundaries, get_chebychev_centers


class PointLocation:

    def __init__(self, solution: Solution):
        """
        Creates a compiled point location solving object for the specified solution.

        This is useful for real time applications on a server or desktop, as it solves the point location problem via direct enumeration in the order or single microseconds.

        For example: A 200 region solution can be evaluated in ~5 uSecs

        :param solution: A solution
        """

        self.solution = solution

        self.overall = numpy.block([[region.E, region.f] for region in self.solution.critical_regions])
        A = self.overall[:, :-1].copy()
        b = self.overall[:, -1].reshape((self.overall.shape[0], 1)).copy()
        self.overall_A = A
        self.overall_b = b

        [self.unique_indices, self.original_indices, self.original_parity] = find_unique_region_hyperplanes(solution)

        self.region_centers = get_chebychev_centers(solution)

        outer_indices = get_outer_boundaries(self.original_indices, self.original_parity)
        # outer_indices = verify_outer_boundary(solution, self.unique_indices, outer_indices, self.region_centers)

        self.outer_A = self.overall_A[self.original_indices][outer_indices].copy()
        self.outer_b = self.overall_b[self.original_indices][outer_indices].copy()

        # create region idx
        num_regions = len(self.solution.critical_regions)
        self.num_regions = num_regions

        region_constraints = numpy.array([0] * (num_regions + 1))
        for i, region in enumerate(self.solution.critical_regions):
            region_constraints[i + 1] = (region.E.shape[0] + region_constraints[i])

        self.region_constraints = region_constraints

        # this is the secret sauce, the core point location code is compiled to native instructions this reduces most overheads

        num_x = solution.program.num_x()

        @numba.njit
        def eval_(theta: numpy.ndarray) -> int:

            test = A @ theta <= b

            for j in range(num_regions):
                # if theta is in region
                if numpy.all(test[region_constraints[j]:region_constraints[j + 1]]):
                    return j

            return -1

        @numba.njit
        def eval__(j: int, theta: numpy.ndarray) -> numpy.ndarray:

            output = numpy.zeros((num_x, 1))
            theta_ = theta.flatten()

            for k in range(num_x):
                output[k] = numpy.dot(A[j * num_x + k], theta_) + b[j * num_x + k]

            return output

        self.eval_ = eval_
        self.eval__ = eval__

    def is_inside(self, theta: numpy.ndarray) -> bool:
        """
        Determines if the theta point in inside of the feasible space

        :param theta: A point in the theta space

        :return: True, if theta in region \n False, if theta not in region
        """
        return self.eval_(theta) != -1

    def locate(self, theta: numpy.ndarray) -> int:
        """
        Finds the index of the critical region that theta is inside

        :param theta:
        :return:
        """
        return self.eval_(theta)

    def evaluate(self, theta: numpy.ndarray) -> Optional[numpy.ndarray]:
        """
        Evaluates the value of x(theta), of the

        :param theta:
        :return:
        """

        idx = self.eval_(theta)
        if idx < 0:
            return None
        return self.solution.critical_regions[idx].evaluate(theta)
