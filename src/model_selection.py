"""
Continual Hyperparameter Selection
"""
from src.datasets.seq_mnist import SequentialMNIST
from src.datasets.cifar10 import SequentialCIFAR10
from src.datasets.perm_mnist import PermutedMNIST
from src.datasets.rotated_mnist import RotatedMNIST
from src.replay import DarkExperienceReplay
from src import metric

import utils

to_class = {
    'SequentialMNIST': SequentialMNIST,
    'PermutedMNIST': PermutedMNIST,
    'RotatedMNIST': RotatedMNIST,
    'SequentialCIFAR10': SequentialCIFAR10
}


def continual_hyperparameter_selection(DATASET, buffer_size=500, accuracy_drop=0.2, hyperparamater_drop=0.5, plus_plus=False):
    """
    Performs model selection according to the continual hyperparameter selection algorithm. 
    For each task, maximize plasticity by optimizing learning rate and buffer size, then maximize stability by
    optimizing alpha and beta with a decay factor. The latter part is only performed if the accuracy on the validation
    set drops below a certain threshold.

    Args:
        DATASET (str): The dataset to perform model selection on.
        buffer_size (int): The length of the buffer to use in the algorithm.
        accuracy_drop (float): The threshold for the accuracy drop on the validation set.
        hyperparamater_drop (float): The decay factor for alpha and beta.
        plus_plus (bool): Whether to use the ++ variant of the algorithm.

    Returns:
        dict: A dictionary containing the best hyperparameters found by the algorithm
    """

    dataset = to_class.get(DATASET)()

    # load hyperparameters
    hyperparameters = utils.load_hparams(dataset.NAME)

    # learning rates coarse grid
    learning_rates = hyperparameters['lr']

    # necessary metrics
    cl_metrics = metric.Metric(dataset)

    # loaders
    loader_id_pairs = dataset.get_train_loader()

    # DER settings
    alpha = 1.0
    beta = None
    if plus_plus:
        beta = 1.0

    der = DarkExperienceReplay(dataset=dataset, buffer_size=buffer_size)

    best_weights = None
    best_lr = None
    best_model = None

    for train_loader, task_id in loader_id_pairs:

        # plasticity
        best_accuracy = 0
        for lr in learning_rates:
            model, current_accuracy = utils.fine_tune(dataset, lr, task_id, weights=best_weights)
            if current_accuracy > best_accuracy:
                best_accuracy = current_accuracy
                best_lr = lr
                best_weights = model.state_dict()
                best_model = model

        # stability
        while True:
            der.set_optimizer(best_lr, best_model)
            der.train(
                train_loader=train_loader,
                task_id=task_id,
                replay_batch_size=32,
                alpha=alpha,
                beta=beta,
                verbose=False
            )
            # evaluate the model on the validation set
            current_accuracy = der.eval(dataset.get_val_loader(task_id))
            # the current accuracy is too low for our threshold, we need to decrease stability to allow more forgetting
            if current_accuracy < (1 - accuracy_drop) * best_accuracy:
                alpha = hyperparamater_drop * alpha
                beta = hyperparamater_drop * beta if beta is not None else None
            else:
                break

        # optimal weights checkpointing for next task
        best_weights = der.model.state_dict()
        print(end='\r')
        print(
            f"Task {task_id} - Best LR: {best_lr} "
            f"- Best Accuracy on Validation set: {best_accuracy:.2f}", end='\n'
        )

        cl_metrics(der.model, task_id, mode='test')

    setting = dataset.setting
    if setting == 'domain-il':
        print("\n ===  Accuracies on test sets - DIL === \n")
    else:
        print("\n ===  Accuracies on test sets - CIL === \n")
    print(cl_metrics.accuracies_CIL)
    if setting in ['task-il', 'class-il']:
        print("\n ===  Accuracies on test sets - TIL === \n")
        print(cl_metrics.accuracies_TIL, end='\n')

    cl_metrics.print_metrics()

    return {
        'best_lr': best_lr,
        'best_alpha': alpha,
        'best_beta': beta,
    }
