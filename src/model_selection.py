"""
Continual Hyperparameter Selection
"""
import config
from src.datasets.seq_mnist import SequentialMNIST
from src.datasets.cifar10 import SequentialCIFAR10
from src.datasets.perm_mnist import PermutedMNIST
from src.datasets.rotated_mnist import RotatedMNIST
from src.replay import DarkExperienceReplay
from itertools import product
from src import metric

import utils

to_class = {
    'SequentialMNIST': SequentialMNIST,
    'PermutedMNIST': PermutedMNIST,
    'RotatedMNIST': RotatedMNIST,
    'SequentialCIFAR10': SequentialCIFAR10
}


def continual_hyperparameter_selection(DATASET, accuracy_drop=0.2, hyperparamater_drop=0.5, plus_plus=False):
    """
    Performs model selection according to the continual hyperparameter selection algorithm. 
    For each task, maximize plasticity by optimizing learning rate and buffer size, then maximize stability by
    optimizing alpha and beta with a decay factor. The latter part is only performed if the accuracy on the validation
    set drops below a certain threshold.

    Args:
        DATASET (str): The dataset to perform model selection on.
        accuracy_drop (float): The threshold for the accuracy drop on the validation set.
        hyperparamater_drop (float): The decay factor for alpha and beta.
        plus_plus (bool): Whether to use the ++ variant of the algorithm.

    Returns:
        dict: A dictionary containing the best hyperparameters found by the algorithm
    """

    dataset = to_class.get(DATASET)()

    hyperparameters = utils.load_hparams(dataset.NAME)
    learning_rates = hyperparameters['lr']
    buffer_sizes = hyperparameters['buffer_size']

    cl_metrics = metric.Metric(dataset)

    loader_id_pairs = dataset.get_train_loader()

    alpha = 1.0
    beta = None
    if plus_plus:
        beta = 1.0

    weights = None
    for train_loader, task_id in loader_id_pairs:

        keys = ('lr', 'buffer_size')
        values = (learning_rates, buffer_sizes)
        parameters_combination = [
            dict(zip(keys, combination)) for combination in product(*values)
        ]

        print("Plasticity", end='\r')
        # plasticity
        best_accuracy = 0
        for combination in parameters_combination:
            lr = combination['lr']
            buffer_size = combination['buffer_size']
            # we maintain the weights of the previous task
            der = DarkExperienceReplay(dataset=dataset, lr=lr, buffer_size=buffer_size, weights=weights)
            der.train(train_loader=train_loader, task_id=task_id, replay_batch_size=32, alpha=alpha, beta=beta)
            cl_metrics(der.model, task_id, mode='val')
            current_accuracy = cl_metrics.get_task_accuracy(task_id)
            if current_accuracy > best_accuracy:
                best_accuracy = current_accuracy
                best_lr = lr
                best_buffer_size = buffer_size

        i = 1
        # stability
        while True:
            print(f"Stability attempt {i}", end='\r')
            i = i + 1
            der = DarkExperienceReplay(dataset=dataset, lr=best_lr, buffer_size=best_buffer_size, weights=weights)
            der.train(train_loader=train_loader, task_id=task_id, replay_batch_size=32, alpha=alpha, beta=beta)
            cl_metrics(der.model, task_id, mode='val')
            current_accuracy = cl_metrics.get_task_accuracy(task_id)
            if current_accuracy < (1 - accuracy_drop) * best_accuracy:
                alpha = hyperparamater_drop * alpha
                beta = hyperparamater_drop * beta if beta is not None else None
            else:
                break
        
        # save optimal weights for the next task
        weights = der.model.state_dict()
        print()
        print(
            f"Task {task_id} - Best LR: {best_lr} - Best Buffer Size: {best_buffer_size} - Best Accuracy on Validation set: {best_accuracy}\n")

        cl_metrics(der.model, task_id, mode='test')

    print(cl_metrics.accuracies_CIL)
    cl_metrics.print_metrics()

    return {
        'best_lr': best_lr,
        'best_buffer_size': best_buffer_size,
        'best_alpha': alpha,
        'best_beta': beta,
    }
