import copy

import numpy
from typing import List, Union

from ..mpmilp_program import MPMILP_Program
from ..mpmiqp_program import MPMIQP_Program
from ..utils.constraint_utilities import remove_strongly_redundant_constraints
from ..critical_region import CriticalRegion


class MITree:

    def __init__(self, problem: Union[MPMILP_Program, MPMIQP_Program], fixed_bins: list = None, depth: int = 0):
        """
        This is the main data structure for the enumeration based algorithm were we are attempting to find feasible
        binary combinations in a more efficient manner

        This can have combinatorial blow ups, as the number of visited nodes can be up to is 2^(n+1) -1 where n is
        the number of binary combinations However this works fairly well if there are constraints on the binary
        variables in the mpMIXP that limit the number of feasible binary combinations.

        Generates  a node and all children nodes,
        :param problem: an MPMILP or MPMIQP program
        :param fixed_bins: the fixed binaries of this node
        :param depth: the depth of this node in the tree
        """
        self.b = None
        self.A = None
        self.problem = copy.copy(problem)
        self.depth = depth
        self.bin_indices = problem.binary_indices

        if fixed_bins is None:
            fixed_bins = []

        self.fixed_bins = fixed_bins

        if depth < len(self.bin_indices):
            self.is_leaf = False

            right_fix = [*self.fixed_bins, 0]
            left_fix = [*self.fixed_bins, 1]

            if self.problem.check_bin_feasibility(right_fix):
                self.right = MITree(self.problem, right_fix, depth + 1)
            else:
                self.right = None

            if self.problem.check_bin_feasibility(left_fix):
                self.left = MITree(self.problem, left_fix, depth + 1)
            else:
                self.left = None

        else:
            # we are a leaf node
            self.is_leaf = True
            self.left = None
            self.right = None

            if self.depth == len(self.bin_indices):
                self.problem.process_constraints()
                # pass

    def count_nodes(self) -> int:
        """
        Counts the nodes
        :return:
        """
        count = 1

        # if there are nodes in the left child then we need to add those as well
        if self.left is not None:
            count += self.left.count_nodes()

        # This is also true for the right child node
        if self.right is not None:
            count += self.right.count_nodes()
        return count

    def get_full_leafs(self):
        """
        Returns all nodes that have fully fixed binary combinations that are descended from this node.

        :return: list of all leaf nodes
        """
        total_leaves = []
        if self.is_leaf and self.depth == len(self.bin_indices):
            total_leaves.append(copy.deepcopy(self))
            return total_leaves

        if self.right is not None:
            total_leaves.extend(self.right.get_full_leafs())

        if self.left is not None:
            total_leaves.extend(self.left.get_full_leafs())

        return total_leaves

    def num_children(self):
        """
        Counts the number of direct children of this node

        :return: number of direct child nodes
        """
        count = 0

        # if there is a right child increment
        if self.right is not None:
            count += 1

        # if there is a left child increment
        if self.left is not None:
            count += 1

        return count

    # noinspection SpellCheckingInspection
    def generate_theta_feasible(self) -> None:
        """
        Given a feasible space of the following type, find an approximation of the theta feasible space, that while
        overestimates generates a fairly tight overestimation.

        .. math::
            \begin{align}
                A_{eq}x &= b_{eq} + F_{eq}\theta\\
                Ax &\leq b + F\theta\\
                A_\theta \theta &\leq b_\theta\\
                x_i &\in \mathbb{R} \text{ or } \mathbb{B}\\
            \end{align}

        :return: None
        """

        # build the problem constraint matrices
        A_block = numpy.block([[self.problem.A, -self.problem.F],
                               [numpy.zeros((self.problem.A_t.shape[0], self.problem.num_x())), self.problem.A_t]])
        b_block = numpy.block([[self.problem.b], [self.problem.b_t]])

        min_vals = []
        max_vals = []

        min_rows = []
        max_rows = []

        # for every F_i run the procedure
        for constraint_index in range(self.problem.num_constraints()):
            zed = numpy.zeros((self.problem.num_x()))
            opt_row = numpy.block([zed, self.problem.F[constraint_index]])

            # if F_i is just zeros then we can skip as it doesn't give any information
            if all(numpy.isclose(opt_row, numpy.zeros_like(opt_row))):
                continue

            # Solve the min F_i theta problem
            sol_obj_min = self.problem.solver.solve_milp(opt_row, A_block, b_block,
                                                         equality_constraints=self.problem.equality_indices,
                                                         bin_vars=self.bin_indices)

            # Solve the max F_i theta problem
            sol_obj_max = self.problem.solver.solve_milp(-opt_row, A_block, b_block,
                                                         equality_constraints=self.problem.equality_indices,
                                                         bin_vars=self.bin_indices)

            # In the case that something went wrong with the min step just skip
            if sol_obj_min is not None:
                min_vals.append(sol_obj_min.obj)
                min_rows.append(constraint_index)

            # in the case that something went wrong with the max step just skip
            if sol_obj_min is not None:
                max_vals.append(-sol_obj_max.obj)
                max_rows.append(constraint_index)

        # data marshel the min and max values into numpy arrays
        b_min = numpy.array(min_vals).reshape(-1, 1)
        b_max = numpy.array(max_vals).reshape(-1, 1)

        # more data marsheling to get the constraints L <= F theta <= U into A theta <= b format
        A, b = remove_strongly_redundant_constraints(
            numpy.block([[-self.problem.F[min_rows]], [self.problem.F[max_rows]], [self.problem.A_t]]),
            numpy.block([[-b_min], [b_max], [self.problem.b_t]]))

        # save this back to class variables
        self.A = A
        self.b = b

    def process_all(self) -> List[List[numpy.ndarray]]:
        """
        Calls generate_theta_feasible on all descendent nodes and returns the regions associated with this as a list
        of critical regions

        :return: A list of critical regions relating to the feasible space in the tree
        """
        self.generate_theta_feasible()
        total_leaves = [[self.A, self.b]]
        if self.is_leaf:
            return total_leaves

        if self.right is not None:
            total_leaves.extend(self.right.process_all())

        if self.left is not None:
            total_leaves.extend(self.left.process_all())

        return total_leaves

    def leaf_path(self) -> bool:
        """
        Checks if any descendant node is a leaf node, e.g. has full feasible binary combination

        :return: True or False
        """

        # if we are a leaf node then this is trivially true
        if self.is_leaf:
            return True
        # if we are not we need to look at the child nodes
        else:
            left_chain = False
            right_chain = False

            # check if the right node has a connection to a leaf node
            if self.right is not None:
                right_chain = self.right.leaf_path()
            # check if the left node has a connection to a leaf node
            if self.left is not None:
                left_chain = self.left.leaf_path()
            # if at least one direct child node is connected to a leaf node then so is this node
            return right_chain or left_chain

    def trim(self) -> None:
        """
        Trim is a method that cleans up the generated tree to make it minimal. Does the following reductions

        1) if a Node only has one child node, then takes the children of that node and hoists it to up

        e.g. N -> C -> ... ===> N -> ...

        2) if a Node is not connected to a leaf node, then it and all of its descends are removed
        :return: None
        """

        # if we are at a child node there is nothing to do
        if self.is_leaf:
            return

        # checks if the left child has a leaf path, if not then removes it
        if self.left is not None:
            if not self.left.leaf_path():
                self.left = None

        # checks if the right child has a leaf path, if not then removes it
        if self.right is not None:
            if not self.right.leaf_path():
                self.right = None

        # If there is only one child then can effectively 'hop' over this node
        if self.num_children() == 1:
            # if we are hopping over the left node
            if self.left is not None:
                self.problem = self.left.problem
                self.is_leaf = self.left.is_leaf
                self.depth = self.left.depth
                self.bin_indices = self.left.bin_indices

                new_left = copy.copy(self.left.left)
                new_right = copy.copy(self.left.right)

                self.left = new_left
                self.right = new_right

            # if we are hopping over the right node
            if self.right is not None:
                self.problem = self.right.problem
                self.is_leaf = self.right.is_leaf
                self.depth = self.right.depth
                self.bin_indices = self.right.bin_indices

                new_left = copy.copy(self.right.left)
                new_right = copy.copy(self.right.right)

                self.left = new_left
                self.right = new_right

        # if we have a left child call trim on that child
        if self.left is not None:
            self.left.trim()

        # if we have a right child call trim on that child
        if self.right is not None:
            self.right.trim()
