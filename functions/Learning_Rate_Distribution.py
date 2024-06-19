import matplotlib.pyplot as plt

def learning_rate_distribution(bayesian_roc_auc_scores, hyperopt_roc_auc_scores, default_scores):

    plt.figure(figsize=(14, 7))

    # Bayesian Optimizer results
    plt.subplot(1, 3, 1)
    plt.hist(bayesian_roc_auc_scores, bins=10, alpha=0.7, label='Bayesian Optimizer', color='red')
    plt.xlabel('ROC AUC Score')
    plt.ylabel('Frequency')
    plt.title('Bayesian Optimizer Learning Rate Distribution')
    plt.legend()

    # Hyperopt results
    plt.subplot(1, 3, 2)
    plt.hist(hyperopt_roc_auc_scores, bins=10, alpha=0.7, label='Hyperopt', color='black')
    plt.xlabel('ROC AUC Score')
    plt.ylabel('Frequency')
    plt.title('Hyperopt Learning Rate Distribution')
    plt.legend()

    # Default Model results
    plt.subplot(1, 3, 3)
    plt.hist(default_scores, bins=10, alpha=0.7, label='Default Model')
    plt.xlabel('ROC AUC Score')
    plt.ylabel('Frequency')
    plt.title('Default Model Learning Rate Distribution')
    plt.legend()

    #Bayesian versus Hyperopt versus Default model learning rate distribution
    plt.tight_layout()
    plt.show()
    return
