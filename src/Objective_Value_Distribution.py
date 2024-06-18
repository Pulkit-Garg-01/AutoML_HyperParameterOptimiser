import matplotlib.pyplot as plt 

def objective_value_distribution(bayesian_obj_values, hyperopt_obj_values):
    plt.figure(figsize=(14, 7))

    plt.subplot(1, 2, 1)
    plt.hist(bayesian_obj_values, bins=30, alpha=0.7, label='Bayesian Optimizer', color='red')
    plt.xlabel('Objective Function Value')
    plt.ylabel('Frequency')
    plt.title('Bayesian Optimizer Objective Function Value Distribution')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.hist(hyperopt_obj_values, bins=30, alpha=0.7, label='Hyperopt', color='black')
    plt.xlabel('Objective Function Value')
    plt.ylabel('Frequency')
    plt.title('Hyperopt Objective Function Value Distribution')
    plt.legend()

    # Bayesian optimizer versus Hyperopt Objective Function Value Distribution 
    plt.tight_layout()
    plt.show()