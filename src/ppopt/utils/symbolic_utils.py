import sympy
import gurobipy

from typing import List
import re

# Deprecated, used sympy to check if any constraints are redundant but was not able to catch them all
# def reduce_redundant_symbolic_constraints(constraints: List[sympy.core.relational.LessThan]) -> List[sympy.core.relational.LessThan]:
#     """
#     Reduces the constraints to a minimal set of constraints that are not redundant.

#     :param constraints: a list of symbolic constraints
#     :return: a list of constraints that are not redundant
#     """

#     constraint_list = []

#     for c in constraints:
#         equality = sympy.Eq(c.lhs, c.rhs)
#         constraint_set = [constraint for constraint in constraints if constraint != c]
#         constraint_set.append(equality)
#         if sympy.solve(constraint_set, c.free_symbols.pop()) != False:
#             constraint_list.append(c)

#     return constraint_list


# TODO this needs to be refactored later, I'm just going to build a gurobi model in here for now
def reduce_redundant_symbolic_constraints(constraints: List[sympy.core.relational.LessThan]) -> List[sympy.core.relational.LessThan]:
    """
    Reduces the constraints to a minimal set of constraints that are not redundant.

    :param constraints: a list of symbolic constraints
    :return: a list of symbolic constraints that are not redundant
    """

    # TODO this can probably be more efficient
    
    # Gurobi can't directly handle square roots, so we need to replace them with auxiliary variables
    # We assume that all square roots are of the form sqrt(expr(theta))
    # We replace sqrt(expr(theta)) with aux_i
    # We add the constraint expr(theta) = aux_(i+1)
    # We add the constraint aux_i = aux_(i+1)^0.5
    constraint_strings = [str(c) for c in constraints]
    counter = 0
    replacement_dict = {} # this stores the replacement for each square root term, so that we can re-use the same aux variables where appropriate instead of creating new ones each time
    for i_con, cs in enumerate(constraint_strings):
        for term in re.findall(r'sqrt\((.*?)\)', cs):
            if term not in replacement_dict:
                replacement_dict[term] = tuple((f'aux{counter}', f'aux{counter + 1}'))
                counter += 2
            constraint_strings[i_con] = cs.replace(f'sqrt({term})', replacement_dict[term][0])

    nonredundant_constraint_list = []
    syms = []

    # capture all the symbols in the constraints (this is basically all the thetas)
    for c in constraints:
        syms.extend(c.free_symbols)

    # ensure that we only have unique symbols
    # FIXME efficiency??
    syms = list(set(syms))
    syms.sort(key=str)

    for i_con, c in enumerate(constraint_strings):

        constraint_strings[i_con] = c.replace('<=', '==')

        model = gurobipy.Model()
        model.setParam('OutputFlag', 0)

        for s in syms:
            model.addVar(vtype=gurobipy.GRB.CONTINUOUS, lb=-gurobipy.GRB.INFINITY, ub=gurobipy.GRB.INFINITY, name=str(s))
        for i in range(counter):
            model.addVar(vtype=gurobipy.GRB.CONTINUOUS, lb=0, ub=gurobipy.GRB.INFINITY, name=f'aux{i}')

        # Build the model using a string, this way we can access all variables by name and use the constraint syntax from the strings
        exec_string = ''
        for v in syms:
            exec_string += str(v) + ' = model.getVarByName("' + str(v) + '")\n'
        for key, value in replacement_dict.items():
            exec_string += f"{value[0]} = model.getVarByName('{value[0]}')\n"
            exec_string += f"{value[1]} = model.getVarByName('{value[1]}')\n"
            exec_string += f"model.addConstr({key} == {value[1]})\n"
            exec_string += f"model.addGenConstrPow({value[1]}, {value[0]}, 0.5)\n"
        for constr in constraint_strings:
            exec_string += 'model.addConstr(' + constr + ')\n'

        model.update() # essential to get all information

        exec(exec_string)

        model.write("model.lp")
        model.optimize()
        status = model.status
        if status == gurobipy.GRB.OPTIMAL:
            nonredundant_constraint_list.append(constraints[i_con])

        constraint_strings[i_con] = constraint_strings[i_con].replace('==', '<=')

    return nonredundant_constraint_list