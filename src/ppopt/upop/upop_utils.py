import numpy
from typing import List, Tuple

from ..geometry.polytope_operations import get_chebyshev_information
from ..solution import Solution


def find_unique_hyperplanes(overall: numpy.ndarray) -> Tuple[List[int], List[int], List[int]]:
    """
    Generates the list of indices of the fundamental hyperplanes of the solution, as well as the indices of the associated hyperplanes from the original solution and the parity of the constraint

    This is linear w.r.t. number of hyper planes and is quite quick ~25 ns per constraint in the solution

    It first creates approximate(near exact) integer representations for each constraint for each region in the solution

    This approximation step is justified in that it will find equality between 2 constraints if the L2 norm of the difference is below 10E-12

    Then the positive and negative versions of these constraints [ -x < -1, x < 1 are on different sides of the same hyperplane] are made into a format that can be hashed (tuples of ints)

    With this is is relatively strait forward to check for uniqueness with the set

    The first loop scans thru all of the constraints and if the constraint contains a unique hyperplane

    1) if it is a unique hyper plane store the index, add the integer representation to the set, then index the integer representation to the index

    2) if it is not a unique hyperplane do nothing

    The second loop scans thru the constraints again and assigns them unique hyper plane indices and the parity(what side of the hyper plane that they are on)


    :param overall: The solution of a multiparametric programming problem
    :return: returns indices of fundamental hyperplanes, indices of constraints back to fundamental hyperplane, parity of constraint
    """

    #
    overall_p = (overall * 1000000000).astype(numpy.int64).tolist()
    overall_n = (overall * -1000000000).astype(numpy.int64).tolist()

    hyper_p = [tuple(x) for x in overall_p]
    hyper_n = [tuple(x) for x in overall_n]

    unique = set()
    indices = list()
    locator = dict()

    original_indices = list()
    original_parity = list()

    for i in range(overall.shape[0]):

        if not hyper_p[i] in unique and not hyper_n[i] in unique:
            unique.add(hyper_p[i])
            indices.append(i)
            locator[hyper_p[i]] = len(unique) - 1

    for i in range(overall.shape[0]):

        if hyper_p[i] in unique:
            original_indices.append(locator[hyper_p[i]])
            original_parity.append(1)
        else:
            original_indices.append(locator[hyper_n[i]])
            original_parity.append(-1)

    return indices, original_indices, original_parity


def find_unique_region_hyperplanes(solution: Solution) -> Tuple[List[int], List[int], List[int]]:
    """
    This is an overload of the find_unique_hyperplane function

    :param solution:
    :return:
    """
    overall = numpy.block([[region.E, region.f] for region in solution.critical_regions])
    return find_unique_hyperplanes(overall)


def find_unique_region_functions(solution: Solution) -> Tuple[List[int], List[int], List[int]]:
    overall = numpy.block([[region.A, region.b] for region in solution.critical_regions])
    return find_unique_hyperplanes(overall)


def get_outer_boundaries(indices: List[int], parity: List[int]):
    """
    Takes in the global constraint indices to the fundamental hyperplanes and their parity finds all planes with only one parity version aka only one verity of them appears in the original set.

    This method is linear w.r.t. number of indices, by the use of sets and hash maps

    :param indices: list of indices that maps the solution constraints into the fundamental hyperplanes
    :param parity: the side of the hyperplane that the constraint represents
    :return:
    """
    # forward iterate over all indices to place them in buckets

    visited = set()
    type_visited = dict()

    for i, index in enumerate(indices):
        # we have not visited this index
        if index not in visited:
            visited.add(index)
            type_visited[index] = parity[i]
        else:
            # we have visited this index before

            # check if we have scratched this index before
            if type_visited[index] == 5:
                continue

            # scratch this index
            elif type_visited[index] != parity[i]:
                type_visited[index] = 5

    outers = set()

    for i, index in enumerate(indices):
        if type_visited[index] != 5 and parity[i] == 1:
            outers.add(index)

    return list(outers)


def get_chebychev_centers(solution: Solution) -> List[numpy.ndarray]:
    """
    Calculates and returns a list of all of the theta chebychev centers for the critical regions in the solution

    :param solution: An mp programming Solution
    :return: A list of all of the chebychev centers of the regions in the solutions
    """

    return [get_chebyshev_information(region).sol[0:solution.program.num_t()].reshape((solution.program.num_t(), 1)) for
            region in solution.critical_regions]


def verify_outer_boundary(solution: Solution, hyper_indices: List[int], outer_indices: List[int],
                          chebychev_centers: List[numpy.ndarray] = None) -> List[int]:
    """
    This checks all of the possible outer boundary indices for errors, failures to solve for the minimal set of fundamental hyperplanes in the solution



    :param solution: An mp programming solution
    :param hyper_indices: The list of all fundamental hyperplane indices
    :param outer_indices: The list of identified exterior hyperplane indices
    :param chebychev_centers: the list of chebychev centers in the theta space for every critical region {Optional}
    :return: List of verified outer boundary constraints
    """
    if chebychev_centers is None:
        chebychev_centers = get_chebychev_centers(solution)

    A = numpy.block([[region.E] for region in solution.critical_regions])[hyper_indices]
    b = numpy.block([[region.f] for region in solution.critical_regions])[hyper_indices]

    output_indices = list()

    # check every chebychev center
    for boundary in outer_indices:

        # for every boundary
        is_valid_boundary = True

        for center in chebychev_centers:
            if not numpy.all(A[boundary] @ center < b[boundary]):
                is_valid_boundary = False

        if is_valid_boundary:
            output_indices.append(boundary)

    return output_indices


def get_descriptions(solution: Solution) -> dict:
    overall_constraints = numpy.block([[region.E, region.f] for region in solution.critical_regions])

    overall_functions = numpy.block([[region.A, region.b] for region in solution.critical_regions])

    desc = dict()

    desc['num_constraints'] = overall_constraints.shape[0]
    desc['theta_dim'] = solution.program.num_t()
    desc['x_dim'] = solution.program.num_x()
    desc['num_regions'] = len(solution.critical_regions)
    desc['num_functions'] = overall_functions.shape[0]

    return desc
