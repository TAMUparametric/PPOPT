import numpy
import pypoman

from numba import jit


def make_subdomains(points):
    return numpy.array([0])
    # return scipy.spatial.Delaunay(points).simplices


@jit(nopython=True)
def make_simplex(n: int):
    a = numpy.zeros(shape=(n + 1, n))
    for i in range(n):
        a[i + 1][i] = 1
    return a


@jit(nopython=True)
def gen_tess_points_simplex(simplex):
    """

    :param simplex:
    :return:
    """
    width = simplex.shape[1]
    length = simplex.shape[0]
    new_length = length * (length + 1) // 2
    tess = numpy.zeros(shape=(new_length, width))

    for i in range(length):
        tess[i] = simplex[i]

    index = length
    extent = len(simplex)
    for i in range(extent):
        for j in range(i + 1, extent):
            tess[index] = .5 * (simplex[i] + simplex[j])
            index += 1
    return tess


# this adds only one subdivision point but in a pseudo optimal spot
@jit(nopython=True)
def revised_tess_simplex(simplex, half_split=False):
    # find the longest and shortest edges
    # ~O(n^2) where n is number of dimensions
    # this runs in the order of 10 usec

    shortest_so_far = float('inf')
    longest_so_far = -1
    longest_index_i = 0
    longest_index_j = 0

    for i in range(simplex.shape[0]):
        for j in range(i + 1, simplex.shape[0]):
            edge_ij = numpy.linalg.norm(simplex[i] - simplex[j])
            if edge_ij <= shortest_so_far:
                shortest_so_far = edge_ij
            elif edge_ij >= longest_so_far:
                longest_so_far = edge_ij
                longest_index_i = i
                longest_index_j = j

    # enforce that I do not want a simplex more bent than ~30 60 90
    # cuts the simplex into 2 based on the longest edge

    if longest_so_far >= 1.7 * shortest_so_far or half_split:
        split_point = .5 * (simplex[longest_index_i] + simplex[longest_index_j])
        left_split = [i for i in range(simplex.shape[0]) if i is not longest_index_j]
        right_split = [i for i in range(simplex.shape[0]) if i is not longest_index_i]

    # the simplex is sufficiently regular to take a piece out of the center
    else:
        split_point = numpy.sum(simplex, axis=0)
        combinations = [[0] * simplex.shape[0]] * simplex.shape[0]
        for i in range(simplex.shape[0]):
            combinations[i] = [j for j in range(simplex.shape[0]) if j is not i]


def make_domain_subdivision(A_t, b_t):
    print(A_t)
    print(b_t)

    x = pypoman.compute_polytope_vertices(A_t, b_t)
    print(x)

    x.append(numpy.sum(x, axis=0))

    simplices = numpy.array(make_subdomains(x))
    x_ = numpy.array(x)
    return [x_[i] for i in simplices]
