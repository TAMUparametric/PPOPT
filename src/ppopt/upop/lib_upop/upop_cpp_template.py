# flake8: noqa
# noinspection SpellCheckingInspection,PyPep8
cpp_upop = """
#include<inttypes.h>
#include<string>
#include<array>
#include<bitset>

#ifndef UPOP_CPP_HEADER
#define UPOP_CPP_HEADER

namespace UPOP {
#define NOT_IN_FEASIBLE_SPACE -1
<==PayloadHere==>
template<int num_fundamental_planes>
	class Constraints {
	public:
		// Constructors Destructors
		Constraints() {};

		~Constraints() {};

		//computes constraint
		const bool get_value(const int i, const float_ t[theta_dim]) {

			int fundamental_plane_index = constraint_indices[i];
			int offset = fundamental_plane_index * theta_dim;

			float_ eval = 0.0;

			if (is_eval[fundamental_plane_index]) {
				// get the result 
				auto result = what_eval[fundamental_plane_index];
				return constraint_parity[i] == result;
			}

			for (int j = 0; j < theta_dim; j++) {
				eval += t[j] * constraint_matrix_data[j + offset];
			}

			//calculate the constraint
			bool value = eval < constraint_vector_data[fundamental_plane_index];
			value = constraint_parity[i] ? value : !value;

			// cache the results for next time
			is_eval[fundamental_plane_index] = true;

			// check parity and assing in positive parity refrence
			what_eval[fundamental_plane_index] = value == constraint_parity[i];

			return value;
		};

	private:
		std::bitset<num_fundamental_hyper_planes> is_eval;
		std::bitset<num_fundamental_hyper_planes> what_eval;
	};

	float_ evaluate_objective(float_* t, float_* x, bool include_theta_terms) {

		float_ obj = c_c;

		if (is_qp) {
			// is this is not a qp then we  can skip this term
			for (int i = 0; i < x_dim; i++) {
				for (int j = 0; j < x_dim; j++)
					obj += x[i] * x[j] * Q[i*x_dim + j];
			}
		}

		// rescale back to 0.5
		obj *= float_(.5);

		// calculate and add the cx term

		for (int i = 0; i < x_dim; i++) {
			obj += x[i] * c[i];
		}

		// calculate and add the t'H'x term
		for (int i = 0; i < x_dim; i++) {
			for (int j = 0; j < theta_dim; j++) {
				obj += t[j] * x[i] * H[i*x_dim + j];
			}
		}

		// this is a fairly common occurance
		if (!include_theta_terms)
			return obj;

		// calculate and add c_t t term
		for (int i = 0; i < theta_dim; i++) {
			obj += c_t[i] * t[i];
		}

		float_ tmp = float_(0);

		for (int i = 0; i < theta_dim; i++) {
			for (int j = 0; j < theta_dim; j++)
				tmp += t[i] * t[j] * Q_t[i*theta_dim + j];
		}

		return obj + float_(0.5)*tmp;
	}
	__inline bool is_inside_region(float_ * theta, int region_id, Constraints<num_fundamental_hyper_planes>& constraints) {

		for (int constraint_index = region_indicies[region_id]; constraint_index < region_indicies[region_id + 1]; constraint_index++)
			if (!constraints.get_value(constraint_index, theta))
				return false;

		return true;
	}

	void evaluate(int region_id, float_ theta[theta_dim], float_ output[x_dim]) {
		int offset = x_dim * region_id;

		if (region_id == NOT_IN_FEASIBLE_SPACE) {
			return;
		}

		for (int i = 0; i < x_dim; i++) {

			int function_vector_index = function_indices[i + offset];
			int function_index = function_vector_index * theta_dim;


			float_ value = 0;

			//std::cout << "Function Values ";
			//std::cout << function_index;
			//std::cout << " ";

			for (int j = 0; j < theta_dim; j++) {
				//std::cout << function_matrix_data[function_index + j] << " ";
				value += theta[j] * function_matrix_data[function_index + j];
			}

			value += function_vector_data[function_vector_index];
			//std::cout << function_vector_data[function_vector_index];
			if (function_parity[i + offset] == 0)
				value = -value;

			output[i] = value;

			//std::cout << std::endl;
		}
	}

	int locate_no_overlap(float_* theta) {
		// create constraints
		auto constraints = Constraints<num_fundamental_hyper_planes>();

		for (int region_id = 0; region_id < num_regions; region_id++) {
			// if no constraint is violated eager return the region_id
			if (is_inside_region(theta, region_id, constraints))
				return region_id;
		}

		// if not in any region, return special flag
		return NOT_IN_FEASIBLE_SPACE;
	}

	int locate_with_overlap(float_* theta) {
		auto constraints = Constraints<num_fundamental_hyper_planes>();

		//initialize the tracking variables with an empty selection and the worst possible obtainable objective
		auto best_region_id = NOT_IN_FEASIBLE_SPACE;
		auto best_obj = std::numeric_limits<float_>::max();

		// initialize the static output array
		float_ x_star[x_dim] = {};


		for (int region_id = 0; region_id < num_regions; region_id++) {

			// if we are not inside the current region then we can skip the next section
			if (!is_inside_region(theta, region_id, constraints))
				continue;

			// only compare the objective of regions that we are inside
			evaluate(region_id, theta, x_star);

			// evaluate the objective of the problem
			float_ curr_obj = evaluate_objective(theta, x_star, false);

			// swap region ID if better region was found
			if (curr_obj <= best_obj) {
				best_region_id = region_id;
				best_obj = curr_obj;
			}
		}

		return  best_region_id;
	}

	////////////////////////////
	// Public Interface
	////////////////////////////


	// Find the correct region, possibility of missing all regions
	int point_location(float_* theta) {
		// Calls the correct region location code
		if (solution_overlap)
			return locate_no_overlap(theta);
		else
			return locate_with_overlap(theta);
	}

	// Evaluate a region for x(theta) at a specific theta
	void evaluate_region(int region_id, float_* theta, float_* output) {

		// if we are not in the feasible space then we can return early
		if (region_id == NOT_IN_FEASIBLE_SPACE) {
			return;
		}

		// if we are in a region then we need to evaluate the critical region
		evaluate(region_id, theta, output);
	}

}

#endif // !UPOP_CPP_HEADER
"""
