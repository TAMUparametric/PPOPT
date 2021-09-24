# flake8: noqa
# noinspection SpellCheckingInspection
js_upop = """

<==PayloadHere==>

class Constraints{
    constructor(){
        this.is_eval = Array(num_fundamental_hyper_planes);
        this.what_eval = Array(num_fundamental_hyper_planes);
    }

    get_value(i, theta){
        var fundamenral_plane_index = constraint_indices[i];
        var offset = fundamenral_plane_index * theta_dim;

        var accum = 0.0;
        
        for (var j = 0; j < theta_dim; j++){
            accum += theta[j] * constraint_matrix_data[j + offset];
        }

        var value = accum < constraint_vector_data[fundamenral_plane_index];
	
      	if (constraint_parity[i] == 0){
      			value = !value;
        }
      
        return value == constraint_parity[i];

    }
}

class UPOP{
    constructor(){}

    static locate(theta){

        var constraints = new Constraints();

        for (var region_id = 0; region_id < num_regions; region_id++){

            var is_inside = 1;

            for (var constraint = region_indicies[region_id]; constraint < region_indicies[region_id+1]; constraint++){

                if (constraints.get_value(constraint, theta) == 0){
                    is_inside = 0;
                    break;
                }

             }

             if (is_inside == 1){
                 return region_id;
             }
        }

        return -1;

    }

    static
    evaluate(region_id, theta){
        var offset = x_dim * region_id;
        var ouput = Array(x_dim);
        
        for (var i = 0; i < x_dim; i++){
            var function_vector_index = function_indices[i + offset];
          	var function_index = function_vector_index * theta_dim;
            var accum = 0;

            for (var j = 0; j < theta_dim; j++){
                accum += theta[j] * function_matrix_data[function_index+j];
            }
          
          	accum += function_vector_data[function_vector_index];

            if (function_parity[i+offset] == 0){
                accum = -accum;
            }

            ouput[i] = accum;
        }
        return ouput;
    }

}
"""
