# noinspection PyProtectedMember
from pathos.multiprocessing import ProcessingPool as Pool
from settrie import SetTrie

from .solver_utils import generate_reduce, generate_extra
from ..mpqp_program import MPQP_Program
from ..solution import Solution
from ..utils.general_utils import num_cpu_cores
from ..utils.mpqp_utils import gen_cr_from_active_set


def full_process(program, candidate, murder_list):
    """

    :param program: A multiparametric program
    :param candidate: the active set combination that we are expanding on
    :param murder_list: the list containing all previously found
    :return: a list of the following form [List[Active Set combinations], active set to prune, Optional[CriticalRegion]]
    """
    to_attempt = []
    to_murder = None

    if not program.check_feasibility(list(candidate)):
        to_attempt.extend(generate_reduce(candidate, murder_list))
        to_murder = candidate
        return [to_attempt, to_murder, None]

    if not program.check_optimality(list(candidate)):
        to_attempt.extend(generate_reduce(candidate, murder_list))
        return [to_attempt, to_murder, None]

    region = gen_cr_from_active_set(program, list(candidate), check_full_dim=False)

    if region.is_full_dimension():
        to_attempt.extend(generate_reduce(candidate, murder_list))
        to_attempt.extend(generate_extra(candidate, region.regular_set[1], murder_list))

        return [to_attempt, to_murder, region]
    else:
        return [to_attempt, to_murder, None]


def solve(program: MPQP_Program, initial_active_sets=None, num_cores=-1) -> Solution:
    """
    Solves the MPQP program with a modified algorithm described in Oberdieck et al. 2016

    url: https://www.sciencedirect.com/science/article/pii/S0005109816303971

    :param program: MPQP to be solved
    :param initial_active_sets:An initial critical region to start this algorithm with, otherwise one will be found
    :param num_cores: number of cores to run this calculation on, default of -1 means use all available cores
    :return: the solution of the MPQP
    """
    if initial_active_sets is None:
        initial_active_sets = [program.gen_optimal_active_set()]

    # This will contain all the attempted active sets
    attempted = set()

    murder_list = SetTrie()

    to_attempt = [tuple(a_set) for a_set in initial_active_sets]
    to_attempt.append(tuple([]))
    to_attempt.extend([tuple([i]) for i in range(len(program.equality_indices), program.num_constraints())])

    # This will contain all the attempted active sets
    attempted = set()
    in_process = set()

    solution = Solution(program, [])

    murder_list = SetTrie()

    if num_cores == -1:
        num_cores = num_cpu_cores()

    pool = Pool(num_cores)

    tiered_to_attempt = [[]] * max(program.num_x() + 3, program.num_t() + 3)
    tiered_to_attempt[0].extend(to_attempt)

    while sum([len(tier) for tier in tiered_to_attempt]) > 0:

        check = lambda x: x not in attempted and not murder_list.hassubset(x)

        def f(x):
            return full_process(program, x, murder_list)

        to_attempt = []

        cursor = 0
        while len(to_attempt) == 0 and cursor < len(tiered_to_attempt):
            to_attempt.extend([x for x in tiered_to_attempt[cursor] if check(x)])
            tiered_to_attempt[cursor] = []
            cursor += 1

        print(f'Processing {len(to_attempt)} in this parallel swap')
        outputs = pool.map(f, to_attempt)

        for candidate in to_attempt:
            attempted.add(candidate)

        for output in outputs:
            for candidate in output[0]:
                if candidate not in in_process:
                    tiered_to_attempt[len(candidate)].append(candidate)
                    in_process.add(candidate)
            if output[1] is not None:
                murder_list.add(output[1])
            if output[2] is not None:
                solution.add_region(output[2])

        # print(f"Time to Serial Process {time.time() - start}")
        # print(f"Total Time at stage = {time.time() - start_run}")

    # pool.close()
    return solution


def solve_no_murder(program: MPQP_Program, initial_active_sets=None, num_cores=-1) -> Solution:
    """
    Solves the MPQP program with a modified algorithm described in Oberdieck et al. 2016 without excluding active
    sets based on an exclusion list.

    url: https://www.sciencedirect.com/science/article/pii/S0005109816303971

    :param program: MPQP to be solved
    :param initial_active_sets:An initial critical region to start this algorithm with, otherwise one will be found
    :param num_cores: number of cores to run this calculation on, default of -1 means use all available cores
    :return: the solution of the MPQP
    """
    if initial_active_sets is None:
        initial_active_sets = program.sample_theta_space()

    # This will contain all the attempted active sets
    attempted = set()

    to_attempt = [tuple(a_set) for a_set in initial_active_sets]
    to_attempt.append(tuple([]))
    to_attempt.extend([tuple([i]) for i in range(len(program.equality_indices), program.num_constraints())])

    # This will contain all the attempted active sets
    attempted = set()
    in_process = set()

    solution = Solution(program, [])

    murder_list = SetTrie()

    if num_cores == -1:
        num_cores = num_cpu_cores()

    pool = Pool(num_cores)

    tiered_to_attempt = [[]] * max(program.num_x() + 3, program.num_t() + 3)
    tiered_to_attempt[0].extend(to_attempt)

    while sum([len(tier) for tier in tiered_to_attempt]) > 0:

        check = lambda x: x not in attempted and not murder_list.hassubset(x)

        def f(x):
            return full_process(program, x, None)

        to_attempt = []

        cursor = 0
        while len(to_attempt) == 0 and cursor < len(tiered_to_attempt):
            to_attempt.extend([x for x in tiered_to_attempt[cursor] if check(x)])
            tiered_to_attempt[cursor] = []
            cursor += 1

        print(f'Processing {len(to_attempt)} in this parallel swap')
        outputs = pool.map(f, to_attempt)

        for candidate in to_attempt:
            attempted.add(candidate)

        for output in outputs:
            for candidate in output[0]:
                if candidate not in in_process:
                    tiered_to_attempt[len(candidate)].append(candidate)
                    in_process.add(candidate)
            if output[1] is not None:
                # pass
                murder_list.add(output[1])
            if output[2] is not None:
                solution.add_region(output[2])

    # pool.close()
    return solution
