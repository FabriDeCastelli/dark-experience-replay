import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import torch.nn.functional as F
import config

from src import models
from PIL import Image


class SequentialMNIST:
    NAME = 'seq-mnist'
    N_CLASSES_PER_TASK = 2
    N_TASKS = 5
    N_CLASSES = N_CLASSES_PER_TASK * N_TASKS
    SIZE = (28, 28)
    N_CHANNELS = 1

    def __init__(self, setting='task-il'):
        self.setting = setting

    class CustomMNIST(datasets.MNIST):
        def __init__(self, *args, **kwargs):

            super(SequentialMNIST.CustomMNIST, self).__init__(*args, **kwargs)

        def __getitem__(self, index):
            """ returns the augmented image, target and the not augmented image"""
            img, target = self.data[index], self.targets[index]
            img = Image.fromarray(img.numpy(), mode='L').copy()
            if self.transform is not None:
                img = self.transform(img)

            return img, target, img
        
    def get_train_loader(self):
        tasks = []

        transform = transforms.ToTensor()
        train_dataset = self.CustomMNIST(root=config.DATASET_PATH, train=True, download=True, transform=transform)

        for task_id in range(self.N_TASKS):

            label_start = task_id * self.N_CLASSES_PER_TASK
            label_end = (task_id + 1) * self.N_CLASSES_PER_TASK - 1

            train_indices = torch.where((train_dataset.targets >= label_start) & (train_dataset.targets <= label_end))[0]
            train_subset = Subset(train_dataset, train_indices)
            if self.setting == 'task-il':
                # down to 0, 1
                train_subset.dataset.targets[train_indices] = train_subset.dataset.targets[train_indices] - label_start
            train_dataloader = DataLoader(train_subset, batch_size=self.get_batch_size(), shuffle=True)

            tasks.append((train_dataloader, task_id))
        
        return tasks
    

    def get_test_loader(self, task_id: int):
        # get correct labels
        label_start = task_id * self.N_CLASSES_PER_TASK
        label_end = (task_id + 1) * self.N_CLASSES_PER_TASK - 1

        test_dataset = datasets.MNIST(root=config.DATASET_PATH, train=False, download=True, transform=transforms.ToTensor())
        test_indices = torch.where((test_dataset.targets >= label_start) & (test_dataset.targets <= label_end))[0]
        test_subset = Subset(test_dataset, test_indices)
        if self.setting == 'task-il':
            # down to 0, 1
            test_subset.dataset.targets[test_indices] = test_subset.dataset.targets[test_indices] - label_start
        return DataLoader(test_subset, batch_size=self.get_batch_size())
    
    def get_batch_size(self):
        return 64
    
    @staticmethod
    def get_transform():
        return None

    def get_epochs(self):
        return 1
    
    def get_loss(self):
        return F.cross_entropy
    
    def get_model(self):
        if self.setting == 'task-il':
            return models.MultiHeadMLP(input_size=784, hidden_size=100, num_heads=5, head_output_size=2)
        return models.SingleHeadMLP(input_size=784, hidden_size=100, output_size=10)


