import torch
from src.replay import DarkExperienceReplay
from src import metric
from src.datasets.seq_mnist import SequentialMNIST
from src.datasets.cifar10 import SequentialCIFAR10
from src.datasets.perm_mnist import PermutedMNIST
from src.datasets.rotated_mnist import RotatedMNIST


def run_experiment(DATASET='SequentialMNIST', alpha=0.1, beta=None, lr=0.03, buffer_size=500) -> str:
    """
    Runs an experiment, training a model on a dataset using Dark Experience Replay (DER, even in the ++ version).
    """

    dataset = to_class.get(DATASET)()
    setting = dataset.setting

    der = DarkExperienceReplay(dataset=dataset, lr=lr, buffer_size=buffer_size)
    cl_metrics = metric.Metric(dataset)

    loader_id_pairs = dataset.get_train_loader()

    for train_loader, task_id in loader_id_pairs:
        print(f"Experience ({task_id}) - Training Samples: {len(train_loader.dataset)}")
        der.train(train_loader=train_loader, task_id=task_id, replay_batch_size=32, alpha=alpha, beta=beta)
        cl_metrics(der.model, task_id)

    if setting == 'domain-il':
        print("\n ===  Accuracies - DIL === \n")
    else :
        print("\n ===  Accuracies - CIL === \n")
    print(cl_metrics.accuracies_CIL)
    if setting in ['task-il', 'class-il']:
        print("\n ===  Accuracies - TIL === \n")
        print(cl_metrics.accuracies_TIL)

    cl_metrics.print_metrics()
    return cl_metrics


# dictionary to map the dataset name to the class
to_class = {
    'SequentialMNIST': SequentialMNIST,
    'PermutedMNIST': PermutedMNIST,
    'RotatedMNIST': RotatedMNIST,
    'SequentialCIFAR10': SequentialCIFAR10,
}