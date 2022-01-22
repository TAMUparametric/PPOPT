from src.ppopt.upop.language_generation import gen_array, gen_variable


def test_array_cpp_1():
    value = gen_array([i for i in range(10)], 'my_data', 'double')
    print(value)


def test_array_js_1():
    value = gen_array([i for i in range(10)], 'my_data', 'double', lang='js')
    print(value)


def test_array_python_1():
    value = gen_array([i for i in range(10)], 'my_data', 'double', lang='python')
    print(value)


def test_variable_cpp_1():
    value = gen_variable(3.75, 'variable', 'double', lang='cpp')
    print(value)


def test_variable_cpp_2():
    value = gen_variable('3.75', 'variable', 'std::string', lang='cpp')
    print(value)


def test_variable_js_1():
    value = gen_variable(3.75, 'variable', 'double', lang='js')
    print(value)


def test_variable_js_2():
    value = gen_variable(3.75, 'variable', 'string', lang='js')
    print(value)


def test_variable_python_1():
    value = gen_variable(3.75, 'variable', 'double', lang='python')
    print(value)


def test_variable_python_2():
    value = gen_variable(3.75, 'variable', 'string', lang='python')
    print(value)
