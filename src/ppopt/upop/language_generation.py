# language specific generation code for C++
from typing import Tuple


def gen_cpp_array(data: list, name: str, vartype: str, options: Tuple[str] = ("const",)) -> str:
    """
    Generates a static C++ array from the provided data in the following format

    vartype name[size] = {data};

    :param data:
    :param name:
    :param vartype:
    :param options:
    :return:
    """
    if "const" in options:
        vartype = "const " + vartype

    return (f"{vartype} {name} [{len(data)}] = " + "{") + ','.join([str(i) for i in data]) + "};"


def gen_cpp_variable(data, name: str, vartype: str, options: Tuple[str] = ("const",)) -> str:
    if "const" in options:
        vartype = "const " + vartype

    # if string type
    if "string" in vartype.lower():
        return f"{vartype} {name} = \"{data}\";"

    return f"{vartype} {name} = {data!s};"


# language specific generation code for Python


def gen_python_array(data: list, name: str, vartype: str, options: Tuple[str] = ("const",)) -> str:
    return f"{name} = " + str(data)


def gen_python_variable(data, name: str, vartype: str, options: Tuple[str] = ("const",)) -> str:
    data_str = str(data)

    if "string" in vartype.lower():
        data_str = "\"" + data_str + "\""

    return f"{name} = {data_str}"


# language specific code for Javascript

def gen_js_array(data: list, name: str, vartype: str, options: Tuple[str] = ("const",)) -> str:
    if "const" in options:
        name = "const " + name

    data_payload = []

    if "string" in vartype:
        data_payload = ["\"" + str(i) + "\"" for i in data]
    else:
        data_payload = [str(i) for i in data]

    return f"{name} = [" + ",".join(data_payload) + "];"


def gen_js_variable(data, name: str, vartype: str, options: Tuple[str] = ("const",)) -> str:
    if "const" in options:
        name = "const " + name

    data_str = str(data)

    if "string" in vartype.lower():
        data_str = "\"" + data_str + "\""

    return f"{name} = {data_str};"


# general code generation interface

def gen_array(data: list, name: str, vartype: str, options: Tuple[str] = ("const",), lang='cpp') -> str:
    if lang == 'cpp':
        return gen_cpp_array(data, name, vartype, options=options)

    if lang == 'python':
        return gen_python_array(data, name, vartype, options=options)

    if lang == 'js':
        return gen_js_array(data, name, vartype, options=options)

    raise RuntimeError(f"Language {lang} is not a supported lang for array generation")


def gen_variable(data, name: str, vartype: str, options: Tuple[str] = ("const",), lang='cpp') -> str:
    if lang == 'cpp':
        return gen_cpp_variable(data, name, vartype, options=options)

    if lang == 'python':
        return gen_python_variable(data, name, vartype, options=options)

    if lang == 'js':
        return gen_js_variable(data, name, vartype, options=options)

    raise RuntimeError(f"Language {lang} is not a supported lang for variable generation")
