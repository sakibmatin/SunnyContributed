#=
    Wrapping the Bayesian Optimization in PyCall for easy use case in Julia. 
=#
using Sunny
using CairoMakie # Good for non-interactive plots. 
using DelimitedFiles
using ProgressMeter 
using StaticArrays, LinearAlgebra
using Statistics
using Measurements # Useful for Error Propagation (when relevant)

# Here, we set up PyCall.jl 
# Review the readme.md for instructions on setting up python packages.
using Pkg
# (Optional) Use pre-defined conda environment. 
# ENV["PYTHON"] = "/vast/home/smatin/.conda/envs/bopt/bin/python3"
Pkg.build("PyCall") # TODO automate checks of re-building if needed. 
using PyCall

# Include Bayesian Optimization functions (python). 
@pyinclude("./Bopt.py")
# include("./forward.jl")
# include("./loss.jl")


### Parameter for Bayesian Optimization. 
# The bounds for different parameters in the Model. 
py"""
bounds={
    "R1" : [0.7500, 1.2500], 
    "R2" : [0.000, 0.200], 
    "R3" : [0.000, 0.200], 
    "R4" : [0.000, 0.100],
}
# Maximum iterations for Bayesian Optimization. 
#   Good rule of thumb is : no. of parameter * 20
max_iter = 80
"""
###


function Objective(params)
    # Convert Bopt parameter dictionary to Sunny model parameters. 
    R1  = get(params, "R1", 0.0)
    R2  = get(params, "R2", 0.0)
    R3a = get(params, "R3", 0.0)
    R3b = get(params, "R4", 0.0)
    ID  = get(params, "ID", 0)
    display(params)

    # Global Parameters for System. 
    L = 24 
    num_samples = 20 # No. of samples to average over. 
    val_J1 = 3.27/(3/2)*5/2  # Scale of parameters. 
    steps_equlb = 5000
    steps_intensity = 1000
    
    J1 = val_J1 * R1
    J2 = val_J1 * R2
    J3a = val_J1 * R3a
    J3b = val_J1 * R3b

    # Compute the Loss function. 
    Loss = SqObjective(
        J1, J2, J3a, J3b, 
        L, num_samples, 
        steps_equlb, steps_intensity, 
        ID
    )

    # Returns objective (negative of loss function.)
    return -Loss
end



function FIT(bounds, max_iter)
    py"""

    $SafetyChecks(bounds)

    # Define GP process. 
    print("Setting up Kernel")
    kernel = Matern(
        length_scale=1.0, 
        length_scale_bounds=(1e-03, 1e3), 
        nu=2.5
    )
    GP_model = GaussianProcessRegressor(
        kernel=kernel, 
        normalize_y=False, 
        optimizer='fmin_l_bfgs_b',  
        n_restarts_optimizer=30  
    )
    
    X, y, idx_list = Restart(bounds)
    start = int(idx_list[-1][0]) + 1

    if start == 1:
        print("Initial Run")
        idx_list = np.array([start])
        X = np.vstack([np.mean(bounds[p]) for p in bounds]).T
        params ={"ID":1}
        for p,i in zip(bounds, range(len(bounds))):
            params[p] = X[0,i]
        y = np.array([$Objective(params)])
        start = start + 1

    for idx in range(start, max_iter):
        print("Bayesian Opt Step :: %i"%idx)
        # Fit GP
        GP_model.fit(X, y)
    
        # Enforce alteration of high exploration and exploitation
        if idx % 4 == 0 : 
            exploreRate = 0.25 
        else : 
            exploreRate = 0.0
        
        # Evaluate Objective
        x_next = Opt_Acquisition(X, GP_model, bounds=bounds, explore=exploreRate)
        x_next = x_next.round(decimals=4, out=None)
        params ={"ID":idx}
        for p,i in zip(bounds, range(len(bounds))):
            params[p] = x_next[0,i]
        X = np.vstack([X, x_next])
        y = np.vstack([y, $Objective(params) ])
        idx_list = np.vstack([idx_list, idx])
        
        best_so_far = np.argmax(y)
        print("Best Loss ", np.max(y), '\n', "Params = ", X[best_so_far])
        
        # Update Log
        data = np.hstack((idx_list, X, y))
        header = [p for p in bounds]
        header.append("Obj")
        header.insert(0, "ID")
        df = pd.DataFrame(data, columns=header,)
        df["ID"] = df["ID"].astype(int)
        df.to_csv("Bopt_Log.csv", sep='\t')
    """ 
end


function SafetyChecks(bounds)
    if length(bounds) > 10
        @warn "Using Bayesian Optimization for more than 10 variables can be slow. "
    end 

    for b in bounds
        @assert b[2][1] < b[2][2] "Set ordered bounds for the parameters."
    end 
end


@time FIT(py"bounds", py"max_iter")
