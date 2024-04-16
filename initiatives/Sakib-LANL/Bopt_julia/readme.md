# Documentation
Instructions to set up the Bayesian Optimization to work with Sunny scripts. 


## Bayesian Optimization
Here, the Bayesian Optimization is hard-coded for arbitrary number of variables without noise. The file name is `bopt.jl`. In the future, the error-bars associated with the Loss can be incorporated. 

The relevant part of the code that should be updated
```julia
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
```

The user has to define the `Objective(params)` function for the their specific problem. The `params` corresponds to a dictionary of parameters to simulate. `params` has all the keys of `bounds` + `ID`, which is the "Index" for the Bayesian optimization.

```julia
function Objective(params)
    # Convert Bopt parameter dictionary to Sunny model parameters. 
    R1  = get(params, "R1", 0.0)
    R2  = get(params, "R2", 0.0)
    R3a = get(params, "R3", 0.0)
    R3b = get(params, "R4", 0.0)
    ID  = get(params, "ID", 0)

    # ... All other Forward Simulation happens here... #

    # Returns objective (negative of loss function.)
    return -Loss
```




## Setting up Python environment
The Bayesian Optimization (in `Bopt.py`) uses a Python backend. We use the `PyCall` to communicate between Julia and Python. 


### Method 1 : Using `CondaPkg.jl` (Recommended)
We let Julia set up the python packages. We can use `CondaPkj.jl`. 

We install `CondaPkg` for Julia
```julia
pkg> add CondaPkg
```

Installing the Python packages needed for Bayesian Optimization. This is done through the Julia REPL.
```julia
julia> using CondaPkg
julia> # now press ] to enter the Pkg REPL
pkg> conda status                # see what has already been installed (if any)
pkg> conda add python numpy scipy matplotlib scikit-learn
```
The installation should take a few min. 

We can use PyCall at the start of the Julia script.
```julia
using Pkg
Pkg.build("PyCall")
using PyCall
```


### Method 2: Setting up a conda environment
Here, we can set up a Python environment separate from Julia. There are several ways to set up a Python Environment. In general, using `conda` is recommended to avoid conflicts with any existing installations or projects. 

Assuming `conda` is already installed, then we can create a new environment named `Bopt` using the command. 
```bash
conda create -n Bopt python numpy scipy matplotlib scikit-learn 
```
This automatically installs the latest version of the different packages that are consistent with each other. These are the minimum packages utilized for the Bayesian Optimization scripts. During the installation process you may be prompted to verify the installation with a `y/n` question. 

After a successful installation, you can initialize the conda environment to ensure there are no errors. 
```bash
conda activate Bopt
```

Using Pycall in a Julia script
```julia
using Pkg
# We have to use the location of the conda environment (example.)
ENV["PYTHON"] = "/vast/home/smatin/.conda/envs/Bopt/bin/python3"
Pkg.build("PyCall") 
using PyCall
```
