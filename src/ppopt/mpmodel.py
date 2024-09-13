from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Tuple, Optional, Union

from .mpmilp_program import MPMILP_Program
from .mpmiqp_program import MPMIQP_Program
from .mpqp_program import MPQP_Program
from .mplp_program import MPLP_Program

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

            return Expression(new_const, new_lc, new_qc)

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
                              {(v1, v2): other * c for (v1, v2), c in self.quad_coeffs.items()})

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
                    + other.const * self.const)

        raise TypeError(
            f"Multiplication on expressions is only defined on numeric types and (linear Expressions) not {type(other)}")

    def __pow__(self, power) -> 'Expression':

        if power == 0:
            return Expression(1, {}, {})

        if power == 1:
            return Expression(self.const, self.linear_coeffs, self.quad_coeffs)

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

            output += f'{prefix}{abs(coeff)}{var}'

        for (v1, v2), coeff in self.quad_coeffs.items():

            if coeff == 0.0:
                continue

            prefix = ' + ' if coeff > 0 else ' - '

            output += f'{prefix}{abs(coeff)}{v1}{v2}'

        return output

    def is_quadratic(self):
        return len(self.quad_coeffs) > 0

    def is_linear(self):
        return len(self.quad_coeffs) == 0 and len(self.linear_coeffs) > 0

    def is_constant(self):
        return len(self.quad_coeffs) == 0 and len(self.linear_coeffs) == 0

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
    expr: Expression
    const_type: ConstraintType

    def __str__(self):

        if self.const_type == ConstraintType.equality:
            return str(self.expr) + ' == 0'
        else:
            return str(self.expr) + ' <= 0'


@dataclass
class MPModel:
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

        num_vars = len(self.variables)

        if name is None and vtype == VariableType.continuous:
            name = f"x_{num_vars}"

        if name is None and vtype == VariableType.binary:
            name = f"y_{num_vars}"

        self.variables.append(ModelVariable(name, vtype, num_vars))

        return self.variables[-1].make_expr()

    def add_param(self, name: Optional[str] = None) -> Expression:
        num_vars = len(self.parameters)

        if name is None:
            name = f"theta_{num_vars}"

        self.parameters.append(ModelVariable(name, VariableType.parameter, num_vars))

        return self.parameters[-1].make_expr()

    def add_constr(self, constr: Constraint):

        if isinstance(constr, Constraint):

            if constr.expr.is_quadratic():
                raise ValueError("Quadratic constraints are not supported")

            self.constraints.append(constr)
        else:
            raise TypeError(f"Constraints must be of type Constraint not {type(constr)}")

    def add_constrs(self, others):

        for constr in others:
            self.add_constr(constr)

    def set_objective(self, obj):
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
        params = [param for param in self.parameters]

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

    def formulate_problem(self) -> Union[MPLP_Program,  MPQP_Program, MPMILP_Program, MPMIQP_Program]:
        #TODO: Implement conversion of expressions into the appropriate program
        pass
