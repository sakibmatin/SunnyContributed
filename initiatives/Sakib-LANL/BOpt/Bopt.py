""" Basic Bayesian Optimizaiton in Python using Scikit Learn. 

    Hard-coded for arbitrary no. of parameters and noise-free scalar objective. 
    
    Using Expected Improvement Acquisition Function
    5/2-Matern Kernel for Gaussian Process. 
    
    We are maximizing the negative-Loss function. 
        This is a consequence of the fact the Bayesian Optimization sign convention. 
"""
import os, sys
import math
import json
import numpy as np
import scipy

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern

from warnings import catch_warnings
from warnings import simplefilter
import warnings
warnings.filterwarnings(action='once')

# Optional Plotting stuff. 
from mpl_toolkits import mplot3d
from matplotlib import pyplot as plt
import matplotlib.style
import matplotlib as mpl
mpl.style.use('classic')



### GLOBAL PARAMS
# This sets the bound for the Bayesian Optimization Search space. 
interval = {
    "R1" : [0.5, 1.0], 
    "R2" : [0.0, 0.5], 
    "R3" : [0.00, 0.5], 
    "R4" : [0.0, 0.1],
} 
JULIA_FILE = "./mgCrO.jl" 
###


def Restart(interval):
    """ Restart Bayesian Optimization from the same directory if possible. 

    Args:
        interval (dictionary): Set of intervals or bounds for all parameters. 

    Returns:
        x_set : Set of free parameters corresponding to `interval`.
        y_set : Set of evaluated Objective function (negatie of loss fucntion).
        idx   : Index to resume from. 
    """
    FILE_LOC = os.path.dirname(os.path.realpath(__file__))
    x_set = np.array([]) ; y_set = np.array([])
    
    for idx in range(1, 1000):    
        # Check if the loss function exists. 
        if os.path.isfile(FILE_LOC+"/loss_%i.csv"%idx):
            
            # Load the actual param file. 
            with open("idx_%i.json"%idx, 'r') as f:
                data = json.load(f)
                loss = np.genfromtxt(f"loss_{idx}.csv")
                Params = [data[p] for p in interval]
                
            # Stack all variables together
            if idx == 1:
                x_set = np.vstack(Params).T
                y_set = np.array([loss])
            else:
                x_set = np.vstack((x_set, Params))
                y_set = np.vstack((y_set, [loss]))
                
        else:
            break    
          
    return x_set, y_set, idx



def Objective(idx, X):
    """Evaluate Objective by using julia subprocess. 

    Args:
        idx (integer): Index of BOpt to be evaluated. 
        X (np.array): Parameters to be evaluated. 
        
    Returns:
        objective (float) : Objective (negative of loss/chi-squared) to be maximized
    """
    
    params ={"ID":idx}
    for p,i in zip(interval, range(len(interval))):
        params[p] = X[0,i]
    with open('BO.json', 'w') as f:
        json.dump(params, f)
    
    # Use os.subprocess for runing the Julia process . 
    # os.system(f"/vast/home/smatin/Julia/julia-1.9.1/bin/julia {JULIA_FILE}") # Specific installation of julia. 
    os.system(f"julia {JULIA_FILE}") 
    
    objective = np.genfromtxt(f"loss_{idx}.csv")
    
    return objective



def surrogate(model, X) : 
    """Helper Function to supress warning when evaluating surrogate models. 

    Args:
        model (GP model): Trained surrogoate GP model 
        X (array): Parameters to evaluate. 

    Returns:
        model predictions (float) : Model predictions of the GP. 
    """
    warnings.filterwarnings("ignore")
    return model.predict(X, return_std=True)



def Expected_Improvement(X, XS, model, explore) :
    """Expected Improvement Surrogate function

    Args:
        X (array): All evaluated parameters. 
        XS (array): Trial parameter for EI calculation. 
        model (GP): Gaussian Process model fit to data
        explore (float): Degree of exploration.  

    Returns:
        EI (float): Expected Improvmeent score. 
    """
    # Find best Score of Data-points. 
    yhat, _ = surrogate(model, X) 
    best = max(yhat) # Current best of the surrogate model.
    
    # Calculate Mean and StdDev via Surrogate Model. 
    mu, std =  surrogate(model, XS) 
    
    # Expected Improvment Calculation.
    I = mu - best - explore 
    Z = np.divide(I, std+1e-8)
    EI = I*scipy.stats.norm.cdf(Z) + std*scipy.stats.norm.pdf(Z)
    
    return EI 



def Opt_Acquisition(X, model, bounds, explore=0.0): 
    """Optimize the Acquisition function to find the next set of parameters to evaluate. 

    Args:
        X (array): Parameters that have been already evaluated. 
        model (GP): Gaussian Process as the surrogate model.
        bounds (dictionary): Set of bounds for all parameters. 
        explore (float, optional): Exploration rate. Defaults to 0.0.

    Returns:
        (array): Set of parameters to evaluate for next B.O. iteration. 
    """
    to_search = np.array([np.random.uniform(low=bounds[b][0], high=bounds[b][1], size=50-len(bounds)) for b in bounds])
    XR = to_search.T
    
    # Calculates optimum EI score
    EIScores = Expected_Improvement(X, XR, model, explore=explore)
    return np.array([XR[np.argmax(EIScores)]])



def SafetyChecks(interval):
    """ Ensures that the intervals for the Bayesian Optimization are in the correct format. 

    Args:
        interval (dictionary): Dictionary for the differt parameters and the appropriate bounds. 
    """
    if len(interval) > 10:
        warnings.warn("Using Bayesian Optimization for more than 10 variables can be slow. ")

    # Verify that the bounds for each parameter is ordered.
    for b in interval:
        assert interval[b][0] < interval[b][1]
        
        

def main():
    SafetyChecks(interval)
    
    # Restart or Random start based on information available. 
    X, y, start = Restart(interval)
    
    # Set up the Gaussian Process as Surrogate Model. 
    kernel = Matern(
        length_scale=1.0, 
        length_scale_bounds=(1e-03, 1e3), 
        nu=2.5
    )
    GP_model = GaussianProcessRegressor(
        kernel=kernel, 
        normalize_y=False, 
        optimizer='fmin_l_bfgs_b',  
        n_restarts_optimizer=25  # 
    )
    
    # Start with single intial sample instead of random sample. 
    if start == 1 :
        print("Single Initial Sample")
        X = np.vstack([np.mean(interval[p]) for p in interval]).T
        y = np.array([Objective(start, X)])
        start = start + 1

    # Bayesian Optimization Loop 
    for idx in range(start, 200):
        print("Bayesian Opt Step :: %i"%idx)
    
        # Fit GP
        GP_model.fit(X, y)
        
        # Enforce alteration of high exploration and exploitation
        if idx % 5 == 0 : 
            exploreRate = 0.25 
        else : 
            exploreRate = 0.0

        x_next = Opt_Acquisition(X, GP_model, bounds=interval, explore=exploreRate)
        
        y_actual = Objective(idx, x_next)
        X = np.vstack([X, x_next])
        y = np.vstack([y, y_actual])
                    

        # Table for tracking progress and plotting.
        best_so_far = np.argmax(y)
        print("Best Loss ", np.max(y), '\n', "Params = ", X[best_so_far])
    
        # Write to file
        with open("Bopt_table.txt", "w") as fOut:
            np.savetxt(
                fOut,
                np.hstack((X,y)),
                fmt='%10.4f', 
                delimiter=',', 
            )
        
        # Plot as a sanity check
        plt.plot(y, '.', marker='o')    
        plt.xlabel("Index", fontsize=15, fontweight="bold")
        plt.ylabel("-Loss", fontsize=12, fontweight="bold")
        plt.tight_layout()
        plt.savefig("Loss_idx.png", dpi=300)
        plt.close('all')


if __name__ == "__main__":
    main() 