from src.ppopt.plot import parametric_plot, plotly_plot, parametric_plot_1D
from src.ppopt.mp_solvers.solve_mpqp import solve_mpqp
from tests.test_fixtures import factory_solution, simple_mpqp_problem


def test_matplotlib_plot(factory_solution):
    parametric_plot(factory_solution, show=False)


def test_plotly_plot(factory_solution):
    plotly_plot(factory_solution, show=False)


def test_matplotlib_plot_1d(simple_mpqp_problem):
    parametric_plot_1D(solve_mpqp(simple_mpqp_problem), show=False)
