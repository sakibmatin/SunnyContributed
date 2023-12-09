# Documentation
This will help set up the appropriate python environment for the Bayesian Optimization script to work with Sunny scripts. 



## Bayesian Optimization Python script
Here, we are using Bayesian Optimization for fitting to experimental data using Sunny. 

Here, the Bayesian Optimization is hard-coded for arbitrary number of variables without noise. The file name is `Bopt.py`. In the future, the error-bars associated with the Loss can be incorporated. 

The only relevant part of the code that needs to be updated: 
```python
### GLOBAL PARAMS
# This sets the bound for the Bayesian Optimization Search space. 
# The number for the bounds should be consistent with the units used in Sunny. 
interval = {
    "J1" : [0.5, 2.25], 
    "J2" : [0.0, 0.5], 
    "J3" : [0.00, 0.5], 
    "J4" : [0.0, 0.0],
} 
JULIA_FILE = "./mgCrO.jl" 
###
```



## Setting up Python Environment
There are several ways to set up a Python Environment. In general, using `conda` is recommended to avoid conflicts with any existing installations or projects. 

Assuming `conda` is already installed, then we can create a new environment named `Bopt` using the command. 
```bash
conda create -n Bopt python numpy scipy matplotlib scikit-learn 
```
This automatically installs the latest version of the different packages that are consistent with each other. These are the minimum packages utilized for the Bayesian Optimization scripts. During the installation process you may be prompted to verify the installation with a `y/n` question. 

After a successful installation, you can initialize the conda environment. 
```bash
conda activate Bopt
```

Then you can run the Bayesian Optimization using one of the following commands in the command line (linux OS)
```bash
# Run as a foreground process
python3  Bopt_4var.py

# Run as a back-ground process
python3  Bopt_4var.py & 

# Back-ground process where all the necessary output is sent to a file 
# not recommended on local machines such as desktops
stdbuf -oL python3 Bopt_4var.py > log_bopt.txt & 
```



## Changes to the Sunny script
We have to add a few lines to the current Sunny script to ensure it produces the necessary output files for the Bayesian Optimization script. 
```julia
import JSON  # JSON package can be installed using Julia's package manager. 

# Loading Parameters from file. 
params = JSON.parsefile("BO.json") 
J1 = get(params, "J1", 0.0)
J2 = get(params, "J2", 0.0)
J3 = get(params, "J3", 0.0)
J4 = get(params, "J4", 0.0)
idx = get(params, "ID", 0)

# Execute the lines that actually performs simulations
SQ_Calculate(J1, J2, J3, J4, idx; kwargs)

# Creating this file signals to the Bayesian Optimization script 
# that the parameter evaluation executed correctly. 
open("idx_$idx.json", "w") do f 
    JSON.print(f, params)
end 
```
Finally, we have to add a few extra lines in the to the function `SQ_Calculate`, that computes the actual loss function:
```julia
open("loss_$idx.csv", "w") do file
        write(file, string(-loss))
    end
# Loss is actual loss/cost we want to minimize. 
# The Bayesian Optimization reads the loss from this particular file. 
```
