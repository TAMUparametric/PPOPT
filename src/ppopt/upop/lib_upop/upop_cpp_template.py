# flake8: noqa
# noinspection SpellCheckingInspection
cpp_upop = """
#include<inttypes.h>
#include<string>
#include<bitset>

namespace UPOP{

#ifndef UPOP_CPP_HEADER
#define UPOP_CPP_HEADER

template<int N>
class BitArray {

public:

	BitArray() {}

	BitArray(std::string data_) {
		data = std::bitset<N>();

		for (int i = 0; i < N; i++)
			data[i] = data_[i] == '1';
	}

	~BitArray() {};

	bool get(int i) const{
		return data.test(i);
	};

	void set(int i, bool v) {
		data.set(i, v);
	};

public:
	int length = N;

private:
	std::bitset<N> data;
};

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
				// see if the 
				// std::cout << "Cached Constraint" << std::endl;
				return constraint_parity.get(i) == result;
			}


			//std::cout << "Variables for constraint ";
			//std::cout << fundamental_plane_index << " ";

			for (int j = 0; j < theta_dim; j++) {
				//std::cout << constraint_matrix_data[j + offset] << " ";
				eval += t[j] * constraint_matrix_data[j + offset];
			}
			
			//calculate the constraint
			//std::cout << constraint_vector_data[fundamental_plane_index];
			bool value = eval < constraint_vector_data[fundamental_plane_index];
			value = constraint_parity.get(i)? value : !value;
			//std::cout << std::endl;
			// cache the results for next time
			is_eval[fundamental_plane_index] = true;

			// check parity and assing in positive parity refrence
			what_eval[fundamental_plane_index] = value == constraint_parity.get(i);

			return value;
		};

	private:
		std::bitset<num_fundamental_hyper_planes> is_eval;
		std::bitset<num_fundamental_hyper_planes> what_eval;
	};


#define NOT_IN_FEASIBLE_SPACE -1

	class PointLocation
	{
	public:
		PointLocation() {};

		int point_location(float_* theta) {

			// create constraints
			auto constraints = Constraints<num_fundamental_hyper_planes>();

			for (int region_id = 0; region_id < num_regions; region_id++) {

				// create region is inside region flag 
				bool is_inside = true;

				// iterates over all of the constraints that are associated with current region
				for (int constraint = region_indicies[region_id]; constraint < region_indicies[region_id + 1]; constraint++)
					// if any of the constraints is violated, set flag to false and break out of loop
					if (!constraints.get_value(constraint, theta)) {
						is_inside = false;
						break;
					}

				// if no constraint is violated eager return the region_id
				if (is_inside)
					return region_id;
			}

			// if not in any region, return special flag
			return NOT_IN_FEASIBLE_SPACE;

		};


		void evaluate(int region_id, float_ theta[theta_dim], float_ output[x_dim]) {
			int offset = x_dim * region_id;

			//std::cout << "Evaluating Region ";
			//std::cout << region_id;
			//std::cout << " ";
			//std::cout << std::endl;

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
				if (function_parity.get(i + offset) == 0)
					value = -value;

				output[i] = value;

				//std::cout << std::endl;
			}
		}


		~PointLocation() {};

	private:


	};

	PointLocation point_location_helper = PointLocation();

	int point_location(float_* theta) {
		return point_location_helper.point_location(theta);
	}


	void evaluate(int region_id, float_* theta, float_* output) {
		point_location_helper.evaluate(region_id, theta, output);
	}


}

#endif
"""
