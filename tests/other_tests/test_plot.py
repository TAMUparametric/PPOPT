from src.ppopt.plot import parametric_plot, plotly_plot

from tests.test_fixtures import factory_solution

def test_matplotlib_plot(factory_solution):
    parametric_plot(factory_solution, show=False)


def test_plotly_plot(factory_solution):
    plotly_plot(factory_solution, show=False)
