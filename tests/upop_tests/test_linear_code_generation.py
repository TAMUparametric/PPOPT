from src.ppopt.upop.linear_code_gen import generate_code_js, generate_code_cpp, generate_code_matlab

from tests.test_fixtures import factory_solution


def test_generate_js_export(factory_solution):
    _ = generate_code_js(factory_solution)


def test_generate_cpp_export(factory_solution):
    _ = generate_code_cpp(factory_solution)


def test_generate_matlab_export(factory_solution):
    generate_code_matlab(factory_solution)
