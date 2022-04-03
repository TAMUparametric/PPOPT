from dataclasses import dataclass
from typing import Union, List, Optional

import numpy

from .critical_region import CriticalRegion
from .geometry.polytope_operations import get_chebyshev_information
from .mplp_program import MPLP_Program
from .mpqp_program import MPQP_Program
from .mpmiqp_program import MPMIQP_Program
from .mpmilp_program import MPMILP_Program
from .utils.general_utils import make_column


@dataclass
class Solution:
    """The Solution object is the output of multiparametric solvers, it contains all the critical regions as well
    as holds a copy of the original problem that was solved. """
    program: Union[MPLP_Program, MPQP_Program, MPMIQP_Program, MPMILP_Program]
    critical_regions: List[CriticalRegion]

    def __init__(self, program: Union[MPLP_Program, MPQP_Program], critical_regions: List[CriticalRegion],
                 is_overlapping=False):
        """
        The explicit solution associated with

        :param program: The multiparametric program that is considered here
        :param critical_regions: The list of critical regions in the solution
        :param is_overlapping: A Flag that tells the point location routine that there are overlapping critical regions
        """
        self.program = program
        self.critical_regions = critical_regions
        self.is_overlapping = is_overlapping

    def add_region(self, region: CriticalRegion) -> None:
        """
        Adds a region to the solution

        :param region: region to add to the solution
        :return: None
        """
        self.critical_regions.append(region)

    def evaluate(self, theta_point: numpy.ndarray) -> Optional[numpy.ndarray]:
        """
        returns the optimal x* from the solution, if it exists

        :param theta_point: an uncertainty realization
        :return: the calculated x* from theta
        """

        cr = self.get_region(theta_point)

        if cr is None:
            return None

        return cr.evaluate(theta_point)

    def get_region(self, theta_point: numpy.ndarray) -> Optional[CriticalRegion]:
        """
        Find the critical region in the solution that corresponds to the theta provided

        The method finds all critical regions that the solution is inside and returns the solutions, x*, with the lowest
        objective function of all of these regions.

        In the case of no overlap we can make a shortcut

        :param theta_point: an uncertainty realization
        :return: the region that contains theta
        """
        if self.is_overlapping:
            return self.get_region_overlap(theta_point)
        else:
            return self.get_region_no_overlap(theta_point)

    def get_region_no_overlap(self, theta_point: numpy.ndarray) -> Optional[CriticalRegion]:
        """
        Find the critical region in the solution that corresponds to the provided theta, assumes that no critical regions overlap

        :param theta_point:
        :return:
        """
        for region in self.critical_regions:
            if region.is_inside(theta_point):
                return region
        return None

    def get_region_overlap(self, theta_point: numpy.ndarray) -> Optional[CriticalRegion]:
        """
        Find the critical region in the solution that corresponds to the provided theta

        :param theta_point: realization of uncertainty
        :return: the critical region that that theta is in with the lowest objective value or none
        """

        # start with the worst value possible for the best objective and without a selected cr
        best_objective = float("inf")
        best_cr = None

        for region in self.critical_regions:
            # check if theta is inside the critical region
            if region.is_inside(theta_point):
                # we are inside the critical region now evaluate x* and f*
                x_star = region.evaluate(theta_point)
                obj = self.program.evaluate_objective(x_star, theta_point)
                # if better then update
                if obj <= best_objective:
                    best_cr = region
                    best_objective = obj

        return best_cr

    def verify_solution(self) -> bool:
        """
        This can be called to verify that all the critical regions agree with the optimization problem. With problems
        with numerically small critical regions the deterministic optimizer value could fail. This does NOT necessarily
        mean that the critical region is at fault but that perhaps more analysis should be done. This is especially
        apparent with critical regions with chebychev radii on the order of sqrt(machine epsilon).

        In the case of overlapping critical regions this is not the proper analysis and a different method should be used.

        :return: True if all is verified, else False
        """

        # print(len(self.critical_regions))

        for region in self.critical_regions:
            sol = get_chebyshev_information(region)
            theta = make_column(sol.sol)[0:numpy.size(sol.sol) - 1]

            x_star = region.evaluate(theta)
            l_star = region.lagrange_multipliers(theta)
            active_set = region.active_set

            soln = self.program.solve_theta(theta)

            if not numpy.allclose(soln.sol, x_star.flatten()):
                return False
            if not numpy.allclose(soln.dual[soln.active_set], -l_star.flatten()):
                return False
            if numpy.allclose(soln.active_set, active_set):
                return False

        return True

    def verify_theta(self, theta_point: numpy.ndarray) -> bool:
        """
        Checks that the result of the solution is consistent with theta substituted multiparametric problem

        :param theta_point: an uncertainty realization
        :return: True if they are the same, False if they are different
        """
        region = self.get_region(theta_point)

        x_star = region.evaluate(theta_point)
        l_star = region.lagrange_multipliers(theta_point)
        r_active_set = region.active_set

        soln = self.program.solve_theta(theta_point)

        if numpy.allclose(soln.sol, x_star.flatten()):
            if numpy.allclose(soln.dual[soln.active_set], -l_star.flatten()):
                if numpy.allclose(soln.active_set, r_active_set):
                    return True

        return False

    def theta_dim(self) -> int:
        return self.program.num_t()

    def evaluate_objective(self, theta_point) -> Optional[numpy.ndarray]:
        """
        Given a realization of an uncertainty parameter, calculate the objective value

        :param theta_point:
        :return:
        """
        x_star = self.evaluate(theta_point)
        if x_star is not None:
            return self.program.evaluate_objective(x_star, theta_point)
        return None

    def is_mixed_integer_sol(self):
        return isinstance(self.program, MPMILP_Program) or isinstance(self.program, MPMIQP_Program)
