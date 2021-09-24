import numpy
import plotly.graph_objects as go
import pypoman
import time
from math import atan2
from matplotlib import pyplot
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon
from typing import List

from .solution import Solution
from .utils.general_utils import make_column


def sort_clockwise(vertices: List[numpy.ndarray]) -> List[numpy.ndarray]:
    """
    Sorts the vertices in clockwise order. This is important for rendering as if they were not sorted then you would see nonsense.

    :param vertices:
    :return:
    """

    # find the center
    x_center = 0
    y_center = 0
    for i in vertices:
        x_center += i[0]
        y_center += i[1]

    x_center = x_center / len(vertices)
    y_center = y_center / len(vertices)
    return sorted(vertices, key=lambda x: atan2((x[1] - y_center), (x[0] - x_center)))


# TODO: specify dimensions to fix
def gen_vertices(solution: Solution):
    """
    Generates the vertices associated with the critical regions in the solution.

    :param solution: a multiparametric region
    :return: a list of a collection of vertices sorted counterclockwise that correspond to the specific region

    """
    vertex_list = list()
    for region in solution.critical_regions:
        vertices = pypoman.compute_polytope_vertices(region.E, region.f)
        vertex_list.append(sort_clockwise(vertices))
    return vertex_list


def plotly_plot(solution: Solution, save_path: str = None, show=True) -> None:
    """
    Makes a plot via the plotly library, this is good for interactive figures that you can embed into webpages and handle interactively.

    :param solution:
    :param save_path: Keyword argument, if a directory path is specified it will save a html copy and a png to that directory
    :param show: Keyword argument, if True displays the plot otherwise does not display
    :return: no return, creates a graph of the solution
    """
    fig = go.Figure()
    vertex_list = gen_vertices(solution)

    for i, region_v in enumerate(vertex_list):
        x_ = [region_v[j][0] for j in range(len(region_v))]
        y_ = [region_v[j][1] for j in range(len(region_v))]

        fig.add_trace(go.Scatter(x=x_, y=y_, fill="toself", name=f'Critical Region {i}'))

    fig.update_layout(
        autosize=False,
        width=1000,
        height=1000
    )

    fig.update_layout(
        hoverlabel=dict(
            bgcolor='white'
        )
    )

    if save_path is not None:
        file_tag = str(time.time())
        fig.write_image(save_path + file_tag + ".png")
        fig.write_html(save_path + file_tag + ".html", include_plotyjs=False, full_html=False)

    if show:
        fig.show()


def parametric_plot(solution: Solution, save_path: str = None, show=True) -> None:
    """
    Makes a simple plot from a solution. This uses matplotlib to generate a plot, it is the general plotting backend.

    :param solution: a multiparametric solution
    :param save_path: if specified saves the plot in the directory
    :param show: Keyword argument, if True displays the plot otherwise does not display
    :return: no return, creates graph of solution
    """
    vertex_list = gen_vertices(solution)
    polygon_list = [Polygon(v) for v in vertex_list]

    fig, ax = pyplot.subplots()

    cm = pyplot.cm.get_cmap('Paired')
    colors = 100 * numpy.random.rand(len(solution.critical_regions))

    p = PatchCollection(polygon_list, cmap=cm, alpha=.8, edgecolors='black', linewidths=1)

    p.set_array(colors)
    ax.add_collection(p)
    pyplot.autoscale()

    if save_path is not None:
        pyplot.savefig(save_path + str(time.time()) + ".png", dpi=1000)

    if show:
        pyplot.show()


def parametric_plot_1D(solution: Solution, save_path: str = None, show=True) -> None:
    """
    Makes a simple plot of a 1D parametric solution

    :param solution:
    :param save_path:
    :param show:
    :return:
    """

    # check if the solution is actually 1 dimensional
    if solution.theta_dim() != 1:
        print(f"Solution is not 1D, the dimensionality of the solution is {solution.theta_dim()}")
        return None

    # see the dimensionality of the response variable x*

    # x_dim = solution.program.num_x()

    # set up the plotting object
    fig, ax = pyplot.subplots()

    # plot the critical regions w.r.t. x*
    for critical_region in solution.critical_regions:
        # get extents
        boundaries = critical_region.f / critical_region.E
        y = [critical_region.evaluate(theta=make_column(boundary)).flatten() for boundary in boundaries]
        ax.plot(boundaries, y, solid_capstyle='round')

    if save_path is not None:
        pyplot.savefig(save_path + str(time.time()) + ".png", dpi=1000)

    if show:
        pyplot.show()