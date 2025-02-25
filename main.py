import torch
from src.replay import DarkExperienceReplay
from src import metric
from src.datasets.seq_mnist import SequentialMNIST
from src.datasets.cifar10 import SequentialCIFAR10
from src.datasets.tiny_imagenet import SequentialTinyImageNet
from src.datasets.perm_mnist import PermutedMNIST

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_experiment(setting='task-il', DATASET='SequentialMNIST', alpha=0.1, beta=None, lr=0.03, buffer_size=500) -> str:

    DATASET = to_class.get(DATASET)
    dataset = DATASET(setting=setting)

    der = DarkExperienceReplay(dataset=dataset, lr=lr, buffer_size=buffer_size)
    cl_metrics = metric.Metric(dataset.N_TASKS)
    random_model_results = []

    loader_id_pairs = dataset.get_train_loader()

    for train_loader, task_id in loader_id_pairs:
        print(f"Experience ({task_id}) - Training Samples: {len(train_loader.dataset)}")

        der.train(train_loader=train_loader, task_id=task_id, replay_batch_size=32, alpha=alpha, beta=beta)
        random_model = dataset.get_model()
        test_loader = dataset.get_test_loader(task_id)
        random_model_results.append(random_model.eval(test_loader=test_loader, task_id=task_id))

        for experience_id in range(len(loader_id_pairs)):
            test_loader = dataset.get_test_loader(experience_id)
            acc = der.eval(test_loader=test_loader, metric=cl_metrics, task_id=task_id, experience_id=experience_id)
            if experience_id == task_id:
                print(f"Accuracy on task {task_id} is {acc}")
                print()

    print(cl_metrics.accuracy_table)

    return cl_metrics.get_metrics(random_model_results)



to_class = {
    'SequentialMNIST': SequentialMNIST,
    'PermutedMNIST': PermutedMNIST,
    'SequentialCIFAR10': SequentialCIFAR10,
    'SequentialTinyImageNet': SequentialTinyImageNet
}