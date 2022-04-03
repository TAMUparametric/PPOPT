# flake8: noqa
# noinspection SpellCheckingInspection
js_upop = """
/*
// Multiparametric solution Exported on <==DATESTAMP==>

<==PayloadHere==>

var Constraints = /** @class */ (function () {
    function Constraints() {
        this.is_value = new Array(num_fundamental_hyper_planes);
        this.what_value = new Array(num_fundamental_hyper_planes);
    }
    Constraints.prototype.get_value = function (i, t) {
        // get the real constraint index from the constraint index map
        var fundamental_plant_index = constraint_indices[i];
        var offset = fundamental_plant_index * theta_dim;
        // check if this is a previously visited constr
        if (this.is_value[fundamental_plant_index]) {
            // if visited then grab the cached value
            var result = this.what_value[fundamental_plant_index];
            return constraint_parity[i] == result;
        }
        // evaluate <A_i,t>
        var eval_ = 0;
        for (var j = 0; j < theta_dim; j++) {
            eval_ += t[j] * constraint_matrix_data[j + offset];
        }
        // see if we violated the constraint
        var value = eval_ < constraint_vector_data[fundamental_plant_index];
        value = constraint_parity[i] ? value : !value;
        // cache the value
        this.is_value[fundamental_plant_index] = true;
        this.what_value[fundamental_plant_index] = value == constraint_parity[i];
        // return the computed value
        return value;
    };
    return Constraints;
}());
function evaluate_objective(t, x, include_theta_therms) {
    // the
    var obj = c_c * 2.0;
    // checks to see if we need to include the Q term in the calulcation
    if (is_qp) {
        for (var i = 0; i < x_dim; i++) {
            for (var j = 0; j < x_dim; j++) {
                obj += x[i] * x[j] * Q[i * x_dim + j];
            }
        }
    }
    // rescale back to 0.5 due to the Q term being 0.5x'Qx
    obj *= 0.5;
    //calculate and add the c'x term
    for (var i = 0; i < x_dim; i++) {
        obj += x[i] * c[i];
    }
    // calculate and add the t'H'x
    for (var i = 0; i < x_dim; i++) {
        for (var j = 0; j < theta_dim; j++) {
            obj += x[i] * t[j] * H[i * x_dim + j];
        }
    }
    // if we don't need to calculate the f(t) terms then we can return early
    if (!include_theta_therms) {
        return obj;
    }
    // calcuate and add the c_t't term
    for (var i = 0; i < theta_dim; i++) {
        obj += t[i] * c_t[i];
    }
    // calculate and add 0.5t'Q_t t
    var tmp = 0;
    for (var i = 0; i < theta_dim; i++) {
        for (var j = 0; j < theta_dim; j++) {
            tmp += t[i] * t[j] * Q_t[i * theta_dim + j];
        }
    }
    return obj + tmp * 0.5;
}
function is_inside_region(t, region_id, constraints) {
    for (var constraint_index = region_indices[region_id]; constraint_index < region_indices[region_id + 1]; constraint_index++) {
        if (!constraints.get_value(constraint_index, t)) {
            return false;
        }
    }
    return true;
}
function evaluate_region(region_id, t) {
    var offset = x_dim * region_id;
    // check if we aren't in any region
    if (region_id == NOT_IN_FEASIBLE_SPACE) {
        return null;
    }
    // make sure we don't grab something over a side
    if (region_id >= num_regions) {
        return null;
    }
    // exstatiate the x(theta) array
    var x_star = Array(x_dim);
    for (var i = 0; i < x_dim; i++) {
        var function_vector_index = function_indices[i + offset];
        var function_index = function_vector_index * theta_dim;
        var value = 0.0;
        for (var j = 0; j < theta_dim; j++) {
            value += t[j] * function_matrix_data[function_index + j];
        }
        value += function_vector_data[function_vector_index];
        if (function_parity[i + offset] == false) {
            value = -value;
        }
        x_star[i] = value;
    }
    return x_star;
}
function locate_region(t) {
    if (solution_overlap) {
        return locate_no_overlap(t);
    }
    return locate_with_overlap(t);
}
function locate_no_overlap(t) {
    var constraints = new Constraints();
    for (var region_id = 0; region_id < num_regions; region_id++) {
        if (is_inside_region(t, region_id, constraints)) {
            return region_id;
        }
    }
    return NOT_IN_FEASIBLE_SPACE;
}
function locate_with_overlap(t) {
    var constraints = new Constraints();
    var best_region_id = NOT_IN_FEASIBLE_SPACE;
    var best_obj = Number.MAX_VALUE;
    for (var region_id = 0; region_id < num_regions; region_id++) {
        if (!is_inside_region(t, region_id, constraints)) {
            continue;
        }
        var x_curr = evaluate_region(region_id, t);
        var curr_obj = evaluate_objective(t, x_curr, false);
        if (curr_obj <= best_obj) {
            best_obj = curr_obj;
            best_region_id = region_id;
        }
    }
    return best_region_id;
}
"""