from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union

import numpy

from .mplp_program import MPLP_Program
from .mpmilp_program import MPMILP_Program
from .mpmiqp_program import MPMIQP_Program
from .mpqp_program import MPQP_Program


class VariableType(Enum):
    """
    Defines the type of variable in the model

    continuous: a continuous variable
    binary: a binary variable
    parameter: a parameter variable
    """
    continuous = 1
    parameter = 2
    binary = 3


@dataclass
class ModelVariable:
    """
    Defines the Model Variable

    name: variable name
    var_type: the type of variable
    var_id: the index of the variable in the originating model
    """

    name: str
    var_type: VariableType
    var_id: int

    def make_expr(self) -> 'Expression':
        """
        Makes an expression from the variable

        :return: an expression the represents the variable
        """
        return Expression(0.0, {self: 1.0}, {})

    def __hash__(self):
        if self.var_type:
            return self.var_id
        else:
            return -self.var_id

    def __str__(self):
        return self.name

    def is_param(self):
        return self.var_type == VariableType.parameter

    def is_var(self):
        return self.var_type in {VariableType.continuous, VariableType.binary}


@dataclass
class Expression:
    """
    Expression is the base definition of a mathematical expression in the model that allows for programmatic
    construction of constraints and objectives that are used in the model.

    Some limitations, only expressions up to quadratic are supported. Orders higher than quadratic are not supported.
    """
    const: float
    linear_coeffs: Dict[ModelVariable, float]
    quad_coeffs: Dict[Tuple[ModelVariable, ModelVariable], float]

    def __add__(self, other) -> 'Expression':

        if isinstance(other, (int, float)):
            return self + Expression(other, {}, {})

        if isinstance(other, Expression):

            new_lc = self.linear_coeffs.copy()
            new_qc = self.quad_coeffs.copy()
            new_const = self.const

            for var, coeff in other.linear_coeffs.items():
                if var in new_lc:
                    new_lc[var] += coeff
                else:
                    new_lc[var] = coeff

            for (var1, var2), coeff in other.quad_coeffs.items():
                if (var1, var2) in new_qc:
                    new_qc[(var1, var2)] += coeff
                else:
                    new_qc[(var1, var2)] = coeff

            new_const += other.const

            return Expression(new_const, new_lc, new_qc).reduced_form()

        raise TypeError(
            f"Addition on expressions is only defined on numeric data or other Expressions not {type(other)}")

    def __radd__(self, other) -> 'Expression':
        return self + other

    def __neg__(self) -> 'Expression':
        return Expression(-self.const, {v: -c for v, c in self.linear_coeffs.items()},
                          {(v1, v2): -c for (v1, v2), c in self.quad_coeffs.items()})

    def __sub__(self, other) -> 'Expression':
        return self + (-other)

    def __rsub__(self, other) -> 'Expression':
        return (-self) + other

    def __mul__(self, other) -> 'Expression':

        if isinstance(other, (int, float)):
            # break into multiple lines
            return Expression(other * self.const, {v: other * c for v, c in self.linear_coeffs.items()},
                              {(v1, v2): other * c for (v1, v2), c in self.quad_coeffs.items()}).reduced_form()

        if isinstance(other, Expression):
            if len(other.quad_coeffs) > 0 or len(self.quad_coeffs) > 0:
                raise ValueError("Cannot multiply quadratic expressions, only linear expressions")

            # (a_i * x_i + b) * (c_j * x_j + d) -> (a_i*x_i)*(c_j * x_j) + b*(c_j * x_j) + d*(a_i*x_i) + b*d
            quad_terms = {}

            for v1, c1 in self.linear_coeffs.items():
                for v2, c2 in other.linear_coeffs.items():
                    new_coeff = c1 * c2
                    if new_coeff != 0.0:
                        quad_terms[(v1, v2)] = c1 * c2

            # break into multiple lines
            return (Expression(other.const * self.const, {}, quad_terms)
                    + self.const * Expression(0.0, other.linear_coeffs, {})
                    + other.const * Expression(0.0, self.linear_coeffs, {})
                    + other.const * self.const).reduced_form()

        raise TypeError(
            f"Multiplication on expressions is only defined on numeric types and (linear Expressions) not {type(other)}")

    def __pow__(self, power) -> 'Expression':

        if power == 0:
            return Expression(1, {}, {})

        if power == 1:
            return Expression(self.const, self.linear_coeffs, self.quad_coeffs).reduced_form()

        if power == 2:
            return self * self

        raise ValueError("Raising to a power on expressions is only defined on the integers 0, 1, 2")

    def __rmul__(self, other) -> 'Expression':
        return self * other

    def __truediv__(self, other) -> 'Expression':
        if isinstance(other, (int, float)):
            return self * (1.0 / other)

        raise TypeError(f"Division on expressions is only defined on numeric data not {type(other)}")

    def __eq__(self, other) -> 'Constraint':
        if isinstance(other, (int, float)):
            return self == Expression(other, {}, {})

        if isinstance(other, Expression):
            return Constraint(self - other, ConstraintType.equality)

        raise TypeError(
            f"constraint generation on expressions is only defined on numeric data or other Expressions not {type(other)}")

    def __le__(self, other) -> 'Constraint':
        if isinstance(other, (int, float)):
            return self <= Expression(other, {}, {})

        if isinstance(other, Expression):
            return Constraint(self - other, ConstraintType.inequality)

        raise TypeError(
            f"constraint generation on expressions is only defined on numeric data or other Expressions not {type(other)}")

    def __ge__(self, other) -> 'Constraint':
        if isinstance(other, (int, float)):
            return self >= Expression(other, {}, {})

        if isinstance(other, Expression):
            return Constraint(other - self, ConstraintType.inequality)

        raise TypeError(
            f"constraint generation on expressions is only defined on numeric data or other Expressions not {type(other)}")

    # check on __str__
    def __str__(self):

        output = ''

        output += str(self.const)

        for var, coeff in self.linear_coeffs.items():

            if coeff == 0.0:
                continue

            prefix = ' + ' if coeff > 0 else ' - '

            if numpy.isclose(abs(coeff),1):
                output += f'{prefix}{var}'
            else:
                output += f'{prefix}{abs(coeff)}{var}'

        for (v1, v2), coeff in self.quad_coeffs.items():

            if coeff == 0.0:
                continue

            prefix = ' + ' if coeff > 0 else ' - '

            if numpy.isclose(abs(coeff),1):
                output += f'{prefix}{v1}{v2}'
            else:
                output += f'{prefix}{abs(coeff)}{v1}{v2}'

        return output

    def is_quadratic(self):
        return len(self.quad_coeffs) > 0

    def is_linear(self):
        return len(self.quad_coeffs) == 0 and len(self.linear_coeffs) > 0

    def is_constant(self):
        return len(self.quad_coeffs) == 0 and len(self.linear_coeffs) == 0

    def reduced_form(self):
        """
        Returns the reduced form of the expression by removing any zero coefficients
        """
        return Expression(self.const, {v: c for v, c in self.linear_coeffs.items() if c != 0.0},
                          {(v1, v2): c for (v1, v2), c in self.quad_coeffs.items() if c != 0.0})

    def is_pure_parametric(self):
        """
        Returns True if the expression is a pure parametric expression
        """

        # get the most reduced form of the expression
        reduced_expr = self.reduced_form()

        for var in reduced_expr.linear_coeffs.keys():
            if var.var_type != VariableType.parameter:
                return False

        for (var1, var2) in reduced_expr.quad_coeffs.keys():
            if var1.var_type != VariableType.parameter or var2.var_type != VariableType.parameter:
                return False

        return True


class ConstraintType(Enum):
    """
    Defines the type of constraint

    equality: an equality constraint
    inequality: an inequality constraint
    """
    equality = 1
    inequality = 2


@dataclass
class Constraint:
    """
    The constraint class defines the constraints in the model, where the constraints are defined as expressions
    paired with a constraint type.

    All constraints are of the form expr <= 0 or expr == 0.
    """
    expr: Expression
    const_type: ConstraintType

    def __str__(self):

        if self.const_type == ConstraintType.equality:
            return str(self.expr) + ' == 0'
        else:
            return str(self.expr) + ' <= 0'

    def is_parametric_constraint(self):
        """
        Returns True if the constraint is a parametric constraint
        """

        return self.expr.is_pure_parametric()

    def is_mixed_constraint(self):
        """
        Returns True if the constraint is a mixed constraint
        """

        return not self.expr.is_pure_parametric()


@dataclass
class MPModeler:
    """
    The MPModel class is the base class for defining a multiparametric model using the new interface.
    """
    variables: List[ModelVariable]
    parameters: List[ModelVariable]

    constraints: List[Constraint]
    objective: Expression

    def __init__(self):
        self.variables = []
        self.parameters = []
        self.constraints = []
        self.objective = Expression(0, {}, {})

    def add_var(self, name: Optional[str] = None, vtype: VariableType = VariableType.continuous) -> Expression:
        """
        Adds a variable to the model, if no name is specified a default name is generated based on the vtype.

        :param name: the name of the variable
        :param vtype: the type of the variable
        :return: the expression that represents the variable
        """
        num_vars = len(self.variables)

        if name is None and vtype == VariableType.continuous:
            name = f"x_{num_vars}"

        if name is None and vtype == VariableType.binary:
            name = f"y_{num_vars}"

        self.variables.append(ModelVariable(name, vtype, num_vars))

        return self.variables[-1].make_expr()

    def add_param(self, name: Optional[str] = None) -> Expression:
        """
        Adds a parameter to the model, if no name is specified a default name is generated.

        :param name: the name of the variable
        :return: the expression that represents the variable
        """
        num_vars = len(self.parameters)

        if name is None:
            name = f"theta_{num_vars}"

        self.parameters.append(ModelVariable(name, VariableType.parameter, num_vars))

        return self.parameters[-1].make_expr()

    def add_constr(self, constr: Constraint):
        """
        Adds a constraint to the model

        Throws an error if the constraint is quadratic or if the constraint is not of type Constraint

        :param constr: the constraint to add
        """

        if isinstance(constr, Constraint):

            if constr.expr.is_quadratic():
                raise ValueError("Quadratic constraints are not supported")

            self.constraints.append(constr)
        else:
            raise TypeError(f"Constraints must be of type Constraint not {type(constr)}")

    def add_constrs(self, constrs):
        """
        Adds multiple constraints to the model

        Throws an error if any constraint is quadratic or if any constraint is not of type Constraint

        :param constrs: the constraints to add
        """
        for constr in constrs:
            self.add_constr(constr)

    def set_objective(self, obj):
        """
        Sets the objective of the model

        Throws an error if it is not of type Expression

        :param obj: the objective to set
        """
        if isinstance(obj, Expression):
            self.objective = obj
        else:
            raise TypeError(f"Objective must be of type Constraint not {type(obj)}")

    def __str__(self):

        output = 'Objective \n\n'

        output += str(self.objective) + '\n\n'

        output += 'Constraints \n\n'

        for con in self.constraints:
            output += str(con) + '\n'

        cont_vars = [var for var in self.variables if var.var_type == VariableType.continuous]
        bin_vars = [var for var in self.variables if var.var_type == VariableType.binary]
        params = self.parameters

        output += '\n\n'

        if len(cont_vars) > 0:
            output += 'Continuous Variables\n'
            output += '(' + ','.join([str(var) for var in cont_vars]) + f') in R^{len(cont_vars)}\n'
        if len(bin_vars) > 0:
            output += '(' + ','.join([str(var) for var in bin_vars]) + f') in B^{len(bin_vars)}\n'
        if len(params) > 0:
            output += 'Parameters\n'
            output += '(' + ','.join([str(var) for var in params]) + f') in R^{len(params)}\n'

        return output

    def formulate_problem(self, process: bool = True) -> Union[MPLP_Program, MPQP_Program, MPMILP_Program, MPMIQP_Program]:
        """
        Formulates the problem into the appropriate program type

        :param process: if the generated mpp should be processed or not
        :return: the formulated program of the appropriate type (mpLP, mpQP, mpMILP, mpMIQP)
        """

        # count number of variables of each type
        num_vars = len(
            [var for var in self.variables if var.var_type in [VariableType.continuous, VariableType.binary]])
        num_params = len(self.parameters)

        # partition the constraints into parametric (A@ theta -b <= 0) and mixed constraints(A@x + F@theta -b <= 0)
        mixed_constraints = [constr for constr in self.constraints if constr.is_mixed_constraint()]
        parametric_constraints = [constr for constr in self.constraints if constr.is_parametric_constraint()]

        # get the indices of the equality constraints
        equality_indices = [i for i, constr in enumerate(mixed_constraints) if
                            constr.const_type == ConstraintType.equality]

        # get the indices of the binary variables
        binary_indices = [var.var_id for var in self.variables if var.var_type == VariableType.binary]

        # Instantiate the mixed constraint matrices
        A = numpy.zeros((len(mixed_constraints), num_vars))
        F = numpy.zeros((len(mixed_constraints), num_params))
        b = numpy.zeros((len(mixed_constraints), 1))

        # instantiate the parametric constraint matrices
        A_t = numpy.zeros((len(parametric_constraints), num_params))
        b_t = numpy.zeros((len(parametric_constraints), 1))

        # fill in the mixed constraint matrices
        for constr_idx, constr in enumerate(mixed_constraints):

            # for each term in the
            for var, coeff in constr.expr.linear_coeffs.items():

                if var.is_var():
                    A[constr_idx, var.var_id] = coeff

                if var.is_param():
                    F[constr_idx, var.var_id] = -coeff

            # set the constant term
            b[constr_idx] = -constr.expr.const

        # fill in the parametric constraint matrices
        for constr_idx, constr in enumerate(parametric_constraints):

            # for each term in the
            for var, coeff in constr.expr.linear_coeffs.items():
                if var.is_param():
                    A_t[constr_idx, var.var_id] = coeff

            # set the constant term
            b_t[constr_idx] = -constr.expr.const

        # instantiate the objective matrices
        c = numpy.zeros((num_vars, 1))
        H = numpy.zeros((num_vars, num_params))
        c_c = numpy.array(self.objective.const)
        c_t = numpy.zeros((num_params, 1))
        Q = numpy.zeros((num_vars, num_vars))
        Q_t = numpy.zeros((num_params, num_params))

        # fill in the objective matrices
        for var, coeff in self.objective.linear_coeffs.items():
            if var.is_var():
                c[var.var_id] = coeff

            if var.is_param():
                c_t[var.var_id] = coeff

        for (v1, v2), coeff in self.objective.quad_coeffs.items():

            # if the quadratic term is between two variables add it to the Q matrix
            if v1.is_var() and v2.is_var():
                Q[v1.var_id, v2.var_id] += 0.5 * coeff
                Q[v2.var_id, v1.var_id] += 0.5 * coeff

            # if the quadratic term is between two parameters add it to the Q_t matrix
            if v1.is_param() and v2.is_param():
                Q_t[v1.var_id, v2.var_id] = coeff

            # if the quadratic term is between a variable and a parameter add it to the H matrix
            if v1.is_var() and v2.is_param():
                H[v1.var_id, v2.var_id] += coeff

            if v1.is_param() and v2.is_var():
                H[v2.var_id, v1.var_id] += coeff

        # if we don't have any quadratic terms then we either have a mpLP or a mpMILP
        if numpy.sum(numpy.abs(Q)) == 0:

            # if we don't have any binary variables then we have an mpLP
            if len(binary_indices) == 0:
                return MPLP_Program(A, b, c, H, A_t, b_t, F, c_c, c_t, Q_t, equality_indices=equality_indices,post_process=process)
            else:
                return MPMILP_Program(A, b, c, H, A_t, b_t, F, binary_indices, c_c, c_t, Q_t,
                                      equality_indices=equality_indices, post_process=process)

        # otherwise we have a mpQP or a mpMIQP
        if len(binary_indices) == 0:
            return MPQP_Program(A, b, c, H, 2 * Q, A_t, b_t, F, c_c, c_t, Q_t, equality_indices=equality_indices,post_process=process)
        else:
            return MPMIQP_Program(A, b, c, H, 2 * Q, A_t, b_t, F, binary_indices, c_c, c_t, Q_t,
                                  equality_indices=equality_indices,post_process=process)
