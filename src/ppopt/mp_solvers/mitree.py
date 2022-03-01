import copy

import numpy

from ..mplp_program import MPLP_Program
from ..utils.constraint_utilities import remove_strongly_redundant_constraints


class MITree:

    def __init__(self, problem: MPLP_Program, bin_indices: list, depth: int = 0):
        self.b = None
        self.A = None
        self.problem = copy.deepcopy(problem)
        self.depth = depth
        self.bin_indices = bin_indices

        def make_sub_problem(coeff: int):
            new_prob = copy.deepcopy(self.problem)
            new_row_A = numpy.array([[1 if i == self.bin_indices[depth] else 0 for i in range(new_prob.num_x())]])
            new_row_F = numpy.array([[0 for _ in range(self.problem.num_t())]])
            new_row_b = numpy.array([[coeff]])
            new_prob.A = numpy.block([[new_row_A], [new_prob.A]])
            new_prob.F = numpy.block([[new_row_F], [new_prob.F]])
            new_prob.b = numpy.block([[new_row_b], [new_prob.b]])
            new_prob.equality_indices = [0, *[i + 1 for i in new_prob.equality_indices]]
            return new_prob

        if depth < len(bin_indices):
            self.is_leaf = False

            right_prob = make_sub_problem(0)
            left_prob = make_sub_problem(1)

            if right_prob.check_feasibility(right_prob.equality_indices):
                # right_prob.process_constraints()
                self.right = BBTree(right_prob, bin_indices, depth + 1)
            else:
                self.right = None

            if left_prob.check_feasibility(left_prob.equality_indices):
                # left_prob.process_constraints()
                self.left = BBTree(left_prob, bin_indices, depth + 1)
            else:
                self.left = None

        else:
            # we are a leef node
            self.is_leaf = True
            self.left = None
            self.right = None

            if self.depth == len(self.bin_indices):
                self.problem.process_constraints()
                # pass

    def count_nodes(self) -> int:
        count = 1
        # print(self.problem.b[[self.bin_indices[i] for i in range(self.depth)]])
        if self.left is not None:
            count += self.left.count_nodes()
        if self.right is not None:
            count += self.right.count_nodes()
        return count

    def get_full_leafs(self):

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
        count = 0
        if self.right is not None:
            count += 1
        if self.left is not None:
            count += 1
        return count

    def generate_theta_feasible(self):

        # build the overestimating theta feasible space based on min maxing F_i theta
        self.problem.process_constraints()

        A_block = numpy.block([[self.problem.A, -self.problem.F],
                               [numpy.zeros((self.problem.A_t.shape[0], self.problem.num_x())), self.problem.A_t]])
        b_block = numpy.block([[self.problem.b], [self.problem.b_t]])

        min_vals = []
        max_vals = []

        min_rows = []
        max_rows = []

        print(
            f'PRocessing problem with {self.problem.num_constraints()} constraints and {self.problem.num_equality_constraints()}')
        for constraint_index in range(self.problem.num_constraints()):
            zed = numpy.zeros((self.problem.num_x()))
            opt_row = numpy.block([zed, self.problem.F[constraint_index]])

            if all(numpy.isclose(opt_row, numpy.zeros_like(opt_row))):
                continue
            # min problem
            sol_obj_min = self.problem.solver.solve_milp(opt_row, A_block, b_block,
                                                         equality_constraints=self.problem.equality_indices,
                                                         bin_vars=self.bin_indices)
            sol_obj_max = self.problem.solver.solve_milp(-opt_row, A_block, b_block,
                                                         equality_constraints=self.problem.equality_indices,
                                                         bin_vars=self.bin_indices)
            if sol_obj_min is not None:
                min_vals.append(sol_obj_min.obj)
                min_rows.append(constraint_index)

            if sol_obj_min is not None:
                max_vals.append(-sol_obj_max.obj)
                max_rows.append(constraint_index)

        b_min = numpy.array(min_vals).reshape(-1, 1)
        b_max = numpy.array(max_vals).reshape(-1, 1)

        A, b = remove_strongly_redundant_constraints(
            numpy.block([[-self.problem.F[min_rows]], [self.problem.F[max_rows]], [self.problem.A_t]]),
            numpy.block([[-b_min], [b_max], [self.problem.b_t]]))
        self.A = A
        self.b = b

    def process_all(self):
        from ppopt.critical_region import CriticalRegion
        self.generate_theta_feasible()
        total_leaves = [CriticalRegion(None, None, None, None, self.A, self.b, None, None, None, None)]
        if self.is_leaf and self.depth == len(self.bin_indices):
            return total_leaves

        if self.right is not None:
            total_leaves.extend(self.right.process_all())

        if self.left is not None:
            total_leaves.extend(self.left.process_all())

        return total_leaves

    def leaf_path(self):

        if self.is_leaf:
            return True
        else:
            left_chain = False
            right_chain = False

            if self.right is not None:
                right_chain = self.right.leaf_path()
            if self.left is not None:
                left_chain = self.left.leaf_path()
            return right_chain or left_chain

    def trim(self):
        # method that trims the exploration tree by removing dead ends and lines e.g. node chains that only have one child in a line

        if self.is_leaf:
            return

        if self.left is not None:
            if not self.left.leaf_path():
                self.left = None

        if self.right is not None:
            if not self.right.leaf_path():
                self.right = None

        # looking down not looking up to remove the reference to parent
        if self.num_children() == 1:
            if self.left is not None:
                self.problem = self.left.problem
                self.is_leaf = self.left.is_leaf
                self.depth = self.left.depth
                self.bin_indices = self.left.bin_indices

                new_left = copy.copy(self.left.left)
                new_right = copy.copy(self.left.right)

                self.left = new_left
                self.right = new_right

            if self.right is not None:
                self.problem = self.right.problem
                self.is_leaf = self.right.is_leaf
                self.depth = self.right.depth
                self.bin_indices = self.right.bin_indices

                new_left = copy.copy(self.right.left)
                new_right = copy.copy(self.right.right)

                self.left = new_left
                self.right = new_right

        if self.left is not None:
            self.left.trim()
        if self.right is not None:
            self.right.trim()
