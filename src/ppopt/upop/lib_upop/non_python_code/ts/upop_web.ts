// Multiparametric solution Exported on

const region_indices:number[] = [0,2,4,6 ];
const NOT_IN_FEASIBLE_SPACE:number = -1;

const constraint_indices:number[] = [ 0,1,0,1,0,1 ];
const constraint_parity:boolean[] = [true,true,true,true,true,true];

const function_indices:number[] = [0,0,0,0,0,1,0,1,0 ];
const function_parity:boolean[] = [true,true,true,true,true,true,true,true,true];
const solution_overlap:boolean = true;
const is_qp:boolean = false;
const theta_dim:number = 1;
const x_dim:number = 3;
const num_hyperplanes:number = 6;
const num_functions:number = 9;
const num_regions:number = 3;
const num_fundamental_hyper_planes:number = 2;
const constraint_matrix_data:number[] = [-1.0,1.0 ];
const constraint_vector_data:number[] = [ 0.0,2.0 ];
const function_matrix_data:number[] = [ 0.0,0.0 ];
const function_vector_data:number[] = [ 0.0,1.0 ];
const Q:number[] = [ 1 ];
const c:number[] = [ -3,0,0 ];
const H:number[] = [ 0.0,0.0,0.0 ];
const c_c:number = 0.0;
const c_t:number[] = [ 0.0 ];
const Q_t:number[] = [ 0.0 ];

class Constraints {
    is_value: boolean[] = new Array(num_fundamental_hyper_planes);
    what_value: boolean[] = new Array(num_fundamental_hyper_planes);

    public get_value(i:number, t:number[]):boolean{

        // get the real constraint index from the constraint index map
        const fundamental_plant_index:number = constraint_indices[i];
        const offset: number = fundamental_plant_index * theta_dim;

        // check if this is a previously visited constr
        if (this.is_value[fundamental_plant_index]){
            // if visited then grab the cached value
            const result = this.what_value[fundamental_plant_index];
            return constraint_parity[i] == result;
        }

        // evaluate <A_i,t>
        let eval_ : number = 0;
        for (let j = 0; j < theta_dim; j++) {
            eval_ += t[j]*constraint_matrix_data[j+offset];
        }

        // see if we violated the constraint
        let value:boolean = eval_ <  constraint_vector_data[fundamental_plant_index];
        value = constraint_parity[i] ? value : !value;

        // cache the value
        this.is_value[fundamental_plant_index] = true;
        this.what_value[fundamental_plant_index] = value == constraint_parity[i];

        // return the computed value
        return value;
    }

}

function evaluate_objective(t:number[], x:number[], include_theta_therms):number{

    // the
    let obj:number = c_c*2.0;

    // checks to see if we need to include the Q term in the calculation
    if (is_qp){
        for (let i = 0; i <x_dim; i++) {
            for (let j = 0; j <x_dim; j++) {
                obj += x[i]*x[j]*Q[i*x_dim + j];
            }
        }
    }

    // rescale back to 0.5 due to the Q term being 0.5x'Qx

    obj *= 0.5;

    //calculate and add the c'x term

    for (let i = 0; i < x_dim; i++) {
        obj += x[i]*c[i];
    }

    // calculate and add the t'H'x
    for (let i = 0; i < x_dim; i++) {
        for (let j = 0; j < theta_dim; j++) {
            obj += x[i]*t[j]*H[i*x_dim + j];
        }
    }

    // if we don't need to calculate the f(t) terms then we can return early
    if (!include_theta_therms){
        return obj;
    }

    // calculate and add the c_t't term
    for (let i = 0; i <theta_dim; i++) {
        obj += t[i]*c_t[i];
    }

    // calculate and add 0.5t'Q_t t
    let tmp:number = 0;

    for (let i = 0; i < theta_dim; i++) {
        for (let j = 0; j < theta_dim; j++) {
            tmp += t[i]*t[j]*Q_t[i*theta_dim + j];
        }
    }

    return obj + tmp*0.5;
}

function is_inside_region(t:number[],region_id:number, constraints:Constraints):boolean{
    for (let constraint_index = region_indices[region_id]; constraint_index < region_indices[region_id+1]; constraint_index++) {
        if (!constraints.get_value(constraint_index, t)) {
            return false
        }
    }
    return true;
}

function evaluate_region(region_id:number, t:number[]):number[]{

    const offset = x_dim*region_id;

    // check if we aren't in any region
    if (region_id == NOT_IN_FEASIBLE_SPACE){
        return null;
    }

    // make sure we don't grab something over a side
    if (region_id >= num_regions){
        return null;
    }

    // create the x(theta) array
    let x_star:number[] = Array(x_dim);

    for (let i = 0; i < x_dim; i++) {

        let function_vector_index:number = function_indices[i + offset];
        let function_index:number = function_vector_index * theta_dim;

        let value:number = 0.0;

        for (let j = 0; j < theta_dim; j++) {
            value += t[j] * function_matrix_data[function_index + j];
        }

        value += function_vector_data[function_vector_index];

        if (function_parity[i + offset] == false){
            value = -value;
        }
        x_star[i] = value;
    }

    return x_star;
}

function locate_region(t:number[]):number{
    if (solution_overlap){
        return locate_no_overlap(t);
    }
    return locate_with_overlap(t);
}

function locate_no_overlap(t:number[]):number {
    let constraints = new Constraints();

    for (let region_id = 0; region_id < num_regions; region_id++) {
        if (is_inside_region(t,region_id,constraints)){
            return region_id;
        }
    }

    return NOT_IN_FEASIBLE_SPACE;
}

function locate_with_overlap(t:number[]):number {
    let constraints = new Constraints();

    let best_region_id = NOT_IN_FEASIBLE_SPACE;
    let best_obj = Number.MAX_VALUE;

    for (let region_id = 0; region_id < num_regions; region_id++) {

        if (!is_inside_region(t, region_id, constraints)){
            continue
        }
        const x_curr = evaluate_region(region_id, t);
        const curr_obj = evaluate_objective(t, x_curr, false);

        if (curr_obj <= best_obj){
            best_obj = curr_obj;
            best_region_id = region_id;
        }

    }

    return best_region_id;
}