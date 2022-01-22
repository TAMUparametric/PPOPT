import numpy
import pypoman

from ..critical_region import CriticalRegion
from ..mplp_program import MPLP_Program
from ..utils.chebyshev_ball import chebyshev_ball


def get_chebyshev_information(region: CriticalRegion, deterministic_solver='glpk'):
    region_constraints = region.get_constraints()
    return chebyshev_ball(*region_constraints, deterministic_solver=deterministic_solver)


def get_vertices(region: CriticalRegion, deterministic_solver='glpk'):
    """
    Generated the vertices of a Critical Region

    :param region: A critical region
    :param deterministic_solver: The optimization Solver to use
    :return: A numpy array of the vertices of a critical region with vertices stored in rows
    """
    return numpy.array(pypoman.compute_polytope_vertices(*region.get_constraints()))


def sample_region_convex_combination(region: CriticalRegion, dispersion=0, num_samples: int = 100,
                                     deterministic_solver='glpk'):
    """
    This is a class to sample a polytopic critical region
    Not SUPPORTED YET
    :param region:
    :param dispersion:
    :param num_samples:
    :param deterministic_solver:
    :return:
    """

    cheb_info = get_chebyshev_information(region, deterministic_solver)
    vertices = get_vertices(region, deterministic_solver)
    num_vertices = len(vertices)

    num_combs = min(max(num_vertices, 2), 4)

    if cheb_info is None:
        return None

    for _ in range(num_samples):
        sample_vertices = numpy.random.choice(num_vertices, num_combs, replace=False)
        print(type(sample_vertices))
        weight = numpy.random.random(num_combs)
        weight = weight / numpy.linalg.norm(weight)

        print(weight)
        print(vertices[sample_vertices])

    radius = cheb_info.sol[-1]
    point = cheb_info.sol[:-1]
    print(point)
    print(radius)
    print(vertices)


def find_extents(A, b, d, x):
    orth_vec = A @ d
    point_vec = A @ x

    dist = float('inf')

    for i in range(A.shape[0]):

        if orth_vec[i] <= 0:
            continue

        dist = min(dist, (b[i] - point_vec[i]) / orth_vec[i])

    return dist


def hit_and_run(p, x_0, n_steps: int = 10):
    # dimension size
    size = x_0.size

    def random_direction():
        vec = numpy.random.rand(size).reshape(size, -1)
        return vec / numpy.linalg.norm(vec, 2)

    for _ in range(n_steps):
        # generate a random direction
        random_direc = random_direction()

        # find the extend in the direction of the random direction and the opposite direction
        extent_forward = find_extents(p.A, p.b, random_direc, x_0)
        extend_backward = find_extents(p.A, p.b, -random_direc, x_0)

        # sample a delta x from the line
        pert = numpy.random.uniform(-extend_backward, extent_forward) * random_direc

        # check if still inside polytope
        if not numpy.all(p.A @ (x_0 + pert) <= p.b):
            continue

        # make step
        x_0 = pert + x_0

    return x_0


def sample_program_theta_space(program: MPLP_Program, num_samples: int = 10):
    pass
