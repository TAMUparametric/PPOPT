import sympy
import gurobipy

from itertools import combinations

from typing import List, Dict, Tuple
import re

# # Deprecated, used sympy to check if any constraints are redundant but was not able to catch them all
# def reduce_redundant_symbolic_constraints(constraints: List[sympy.core.relational.LessThan]) -> List[sympy.core.relational.LessThan]:
#     """
#     Reduces the constraints to a minimal set of constraints that are not redundant.

#     :param constraints: a list of symbolic constraints
#     :return: a list of constraints that are not redundant
#     """

#     # first, we need to remove any duplicate constraints
#     constraints = remove_duplicate_symbolic_constraints(constraints)

#     constraint_list = []

#     for c in constraints:
#         equality = sympy.Eq(c.lhs, c.rhs)
#         constraint_set = [constraint for constraint in constraints if constraint != c]
#         constraint_set.append(equality)
#         if sympy.solve(constraint_set, c.free_symbols.pop()) != False:
#             constraint_list.append(c)

#     return constraint_list

def to_less_than_or_equal(constraint: sympy.core.relational) -> sympy.core.relational.LessThan:
    """
    Converts an inequality to a less than or equal constraint.

    :param constraint: an inequality
    :return: a less than or equal constraint
    """

    if '<' not in constraint.rel_op and '>' not in constraint.rel_op:
        raise ValueError('Constraint must be an inequality.')
    if constraint.rel_op == '>=' or constraint.rel_op == '>':
        return sympy.LessThan(-constraint.lhs, -constraint.rhs)
    if constraint.rel_op == '<':
        return sympy.LessThan(constraint.lhs, constraint.rhs)
    return constraint


def replace_square_roots_dictionary(constraint_strings: List[str]) -> Tuple[Dict[str, Tuple[str, str]], List[str], int]:
    """
    Replaces square root terms in a list of constraint strings with auxiliary variables.

    :param constraint_strings: a list of constraint strings
    :return: a dictionary of replacements and a list of constraint strings with square root terms replaced
    """

    counter = 0
    replacement_dict = {} # this stores the replacement for each square root term, so that we can re-use the same aux variables where appropriate instead of creating new ones each time
    for i_con, cs in enumerate(constraint_strings):
        for term in re.findall(r'sqrt\((.*?)\)', cs):
            if term not in replacement_dict:
                replacement_dict[term] = tuple((f'aux{counter}', f'aux{counter + 1}'))
                counter += 2
            constraint_strings[i_con] = cs.replace(f'sqrt({term})', replacement_dict[term][0])

    return replacement_dict, constraint_strings, counter


def remove_duplicate_symbolic_constraints(constraints: List[sympy.core.relational.LessThan], indices: List[int]) -> Tuple[List[sympy.core.relational.LessThan], List[int]]:
    """
    Removes duplicate constraints from a list of constraints.

    :param constraints: a list of symbolic constraints
    :return: a list of constraints with duplicates removed and their indices in the original list
    """
    unique_constraints = []
    new_indices = []
    for i, c in enumerate(constraints):
        unique = True
        for uc in unique_constraints:
            if sympy.simplify((c.lhs - c.rhs) - (uc.lhs - uc.rhs)) == 0:
                unique = False
                break
        if unique:
            unique_constraints.append(c)
            new_indices.append(indices[i])

    return unique_constraints, new_indices


def simplify_univariate_symbolic_constraints(constraints: List[sympy.core.relational.LessThan], indices: List[int]) -> Tuple[List[sympy.core.relational.LessThan], List[int]]:
    """
    Simplifies univariate constraints.

    :param constraints: a list of symbolic constraints
    :return: a list of simplified constraints and their indices in the original list
    """

    simplified_constraints = []
    new_indices = []
    for i, c in enumerate(constraints):
        constraint_index = indices[i]
        if len(c.free_symbols) == 1:
            simplified = sympy.solve(c, c.free_symbols.pop())
            if isinstance(simplified, sympy.And):
                for s in simplified.args:
                    simplified_constraints.append(s)
                    new_indices.append(constraint_index)
            else:
                simplified_constraints.append(simplified)
                new_indices.append(constraint_index)
        else:
            simplified_constraints.append(c)
            new_indices.append(constraint_index)

    return simplified_constraints, new_indices


def simplfiy_trivial_symbolic_constraints(constraints: List[sympy.core.relational.LessThan], indices: List[int]) -> Tuple[List[sympy.core.relational.LessThan], List[int]]:
    """
    Simplifies trivial constraints by trying to see if some sympy simplifications result in a True value.

    :param constraints: a list of symbolic constraints
    :return: a list of simplified constraints and their indices in the original list
    """

    simplified_constraints = []
    new_indices = []
    for i, c in enumerate(constraints):
        if sympy.factor(c) == True:
            continue
        if sympy.simplify(c) == True:
            continue
        if sympy.expand(c) == True:
            continue
        else:
            simplified_constraints.append(c)
            new_indices.append(indices[i])

    return simplified_constraints, new_indices


def build_gurobi_model_with_square_roots(constraint_strings: List[str], syms: List[sympy.Symbol], replacement_dict: Dict[str, Tuple[str, str]], num_aux: int) -> gurobipy.Model:
    """
    Builds a Gurobi model with square root terms replaced by auxiliary variables.

    :param constraint_strings: a list of constraint strings
    :param syms: a list of symbols
    :return: a Gurobi model
    """

    model = gurobipy.Model()
    model.setParam('OutputFlag', 0)

    for s in syms:
        model.addVar(vtype=gurobipy.GRB.CONTINUOUS, lb=-gurobipy.GRB.INFINITY, ub=gurobipy.GRB.INFINITY, name=str(s))
    for i in range(num_aux):
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

    return model


def reduce_redundant_symbolic_constraints(constraints: List[sympy.core.relational.LessThan], indices: List[int]) -> Tuple[List[sympy.core.relational.LessThan], List[int]]:
    """
    Reduces the constraints to a minimal set of constraints that are not redundant.

    :param constraints: a list of symbolic constraints
    :return: a list of symbolic constraints that are not redundant and their indices in the original list
    """

    # first, we need to simplify any univariate constraints
    constraints, indices = simplify_univariate_symbolic_constraints(constraints, indices)

    # next, we need to simplify any trivial constraints
    constraints, indices = simplfiy_trivial_symbolic_constraints(constraints, indices)

    # next, we need to remove any duplicate constraints
    constraints, indices = remove_duplicate_symbolic_constraints(constraints, indices)

    # finally, we make everything a less than or equal constraint (in particular, this can make some strict inequalities into non-strict inequalities)
    constraints = [to_less_than_or_equal(c) for c in constraints]

    # TODO this can probably be more efficient
    
    # Gurobi can't directly handle square roots, so we need to replace them with auxiliary variables
    # We assume that all square roots are of the form sqrt(expr(theta))
    # We replace sqrt(expr(theta)) with aux_i
    # We add the constraint expr(theta) = aux_(i+1)
    # We add the constraint aux_i = aux_(i+1)^0.5
    constraint_strings = [str(c) for c in constraints]
    replacement_dict, constraint_strings, num_aux = replace_square_roots_dictionary(constraint_strings)

    for i, c in enumerate(constraints):
        if isinstance(c, sympy.Equality):
            constraint_strings[i] = str(c.lhs - c.rhs) + ' == 0'

    nonredundant_constraint_list = []
    syms = []

    # capture all the symbols in the constraints (this is basically all the thetas)
    for c in constraints:
        syms.extend(c.free_symbols)

    # ensure that we only have unique symbols
    # FIXME efficiency??
    syms = list(set(syms))
    syms.sort(key=str)

    kept_indices = []

    for i_con, c in enumerate(constraint_strings):

        constraint_strings[i_con] = c.replace('<=', '==')

        model = build_gurobi_model_with_square_roots(constraint_strings, syms, replacement_dict, num_aux)

        model.optimize()
        status = model.status
        if status == gurobipy.GRB.OPTIMAL:
            nonredundant_constraint_list.append(constraints[i_con])
            kept_indices.append(indices[i_con])

        constraint_strings[i_con] = constraint_strings[i_con].replace('==', '<=')

    return nonredundant_constraint_list, kept_indices