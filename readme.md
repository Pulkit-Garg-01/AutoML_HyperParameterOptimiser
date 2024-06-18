# Project Title: Bayesian vs Hyperopt Optimization for RandomForest Classifier

## Introduction
This project aims to compare two optimization techniques, Bayesian Optimization and Hyperopt, for tuning the hyperparameters of a RandomForestClassifier. The objective is to identify the best set of hyperparameters that maximize the ROC AUC score for a given dataset, in this case, the Wine dataset.

## Project Structure

### Main Files and Directories

- `main.py`: The main script to run the optimization and comparison.
- `src/`: Directory containing the source code.
  - `Bayesian_Optimizer.py`: Implements the Bayesian Optimization class.
  - `Objective_Bayesian.py`: Defines the objective function for Bayesian Optimization.
  - `Objective_Hyperopt.py`: Defines the objective function for Hyperopt.
  - `Cross_validation.py`: Contains the function for cross-validation.
  - `Learning_Rate_Distribution.py`: Generates the learning rate distribution plot.
  - `Objective_Value_Distribution.py`: Generates the objective value distribution plot.

## Installation

### Requirements
- Python 3.7+
- NumPy
- Pandas
- SciPy
- Scikit-learn
- Hyperopt
- Matplotlib

### Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/Pulkit-Garg-01/AutoML_HyperParameterOptimiser
   cd AutoML_HyperParameterOptimiser
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Running the Optimization
To run the Bayesian and Hyperopt optimizations and compare their results, execute the main script:
```bash
python main.py
```

### Output
The script will print the best parameters found by each optimizer and plot the following:

1. **Convergence Plot**: Shows the negative mean ROC AUC over iterations for Bayesian Optimization.
2. **Learning Rate Distribution**: Compares the distribution of ROC AUC scores for Bayesian Optimization and Hyperopt.
3. **Objective Value Distribution**: Compares the distribution of objective values for Bayesian Optimization and Hyperopt.

Additionally, the script will print the cross-validation ROC AUC scores and mean ROC AUC for the default RandomForestClassifier and the best parameters found by both optimizers.

## Code Overview

### Bayesian Optimization
The `BayesianOptimizer` class is used to perform Bayesian Optimization. It requires:
- `func`: The objective function to optimize.
- `float_param_ranges`: A dictionary specifying the ranges for continuous parameters.
- `int_param_candidates`: A dictionary specifying the candidates for integer parameters.
- `n_init_points`: The number of initial points to sample.
- `max_iter`: The maximum number of iterations.
- `acq_type`: The acquisition function type ('EI' in this case).

### Hyperopt
Hyperopt's `fmin` function is used to perform the optimization. It requires:
- `fn`: The objective function to optimize.
- `space`: The search space for the parameters.
- `algo`: The algorithm to use (TPE in this case).
- `max_evals`: The maximum number of evaluations.
- `trials`: A Trials object to store the results.
- `rstate`: A random state for reproducibility.

### Cross-validation
The `cross_validate_with_params` function performs cross-validation using the specified parameters and returns the ROC AUC scores.

### Plotting
The `learning_rate_distribution` and `objective_value_distribution` functions generate and display the respective plots.

## Conclusion
This project demonstrates how to use Bayesian Optimization and Hyperopt to tune hyperparameters for a RandomForestClassifier. By comparing the performance of both optimizers, you can gain insights into the efficiency and effectiveness of these techniques in finding optimal hyperparameters. The results include convergence plots, learning rate distributions, and cross-validation scores, providing a comprehensive analysis of the optimization process.
