from typing import Optional


import numpy

# Make this optional at some point, so we can run on more general platforms
import numba

from ..solution import Solution
from ..upop.upop_utils import find_unique_region_hyperplanes, get_outer_boundaries, get_chebychev_centers


class PointLocation:

    def __init__(self, solution: Solution):
        """
        Creates a compiled point location solving object for the specified solution.

        This is useful for real time applications on a server or desktop, as it solves the point location problem via
        direct enumeration.

        This is fast; for example a 200 region solution can be evaluated in single digit uSecs on modern computers

        :param solution: An explicit solution to a multiparametric program
        """

        # take in the solution
        self.solution = solution

        # build the overall matrix block - this is all of the region hyper plane constraints stacked ontop of each other

        A = numpy.block([[region.E] for region in self.solution.critical_regions])
        b = numpy.block([[region.f] for region in self.solution.critical_regions])

        self.region_centers = get_chebychev_centers(solution)

        # create region idx
        num_regions = len(self.solution.critical_regions)
        self.num_regions = num_regions

        region_constraints = numpy.zeros((num_regions+1,))

        for i, region in enumerate(self.solution.critical_regions):
            region_constraints[i + 1] = (region.E.shape[0] + region_constraints[i])

        self.region_constraints = region_constraints

        # this is the secret sauce, the core point location code is compiled to native instructions this reduces most overheads


        @numba.njit
        def get_region_overlap(theta: numpy.ndarray) -> numpy.ndarray:
            test = A @ theta <= b

            region_indicator = numpy.zeros((num_regions,))
            for j in range(num_regions):
                if numpy.all(test[region_constraints[j]:region_constraints[j + 1]]):
                    region_indicator[j] = 1

            return region_indicator

        @numba.njit
        def get_region_no_overlap(theta: numpy.ndarray) -> int:

            test = A @ theta <= b

            for j in range(num_regions):
                # if theta is in region
                if numpy.all(test[region_constraints[j]:region_constraints[j + 1]]):
                    return j

            return -1

        if solution.is_overlapping:
            self.get_region = get_region_overlap
        else:
            self.get_region = get_region_no_overlap

        def locate(theta:numpy.ndarray) -> int:
            if solution.is_overlapping:
                region_indicators = self.get_region(theta)
                obj_vals = numpy.array(map(self.solution.program.evaluate_objective(cr.evaluate(theta), theta) for index, cr in self.solution.critical_regions if region_indicators[index] ==1))
                return numpy.argmin(obj_vals)[0]
            else:
                return self.get_region(theta)

        self.eval_ = locate

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
