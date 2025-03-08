import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset, random_split
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

    def __init__(self):
        self.setting = 'task-il'
        self.tasks = []
        self.test_loaders = []
        self.val_loaders = [] 
        self.setup_loaders()

    class CustomMNIST(datasets.MNIST):
        def __init__(self, *args, **kwargs):
            super(SequentialMNIST.CustomMNIST, self).__init__(*args, **kwargs)

        def __getitem__(self, index):
            img, target = self.data[index], self.targets[index]
            img = Image.fromarray(img.numpy(), mode='L').copy()
            if self.transform is not None:
                img = self.transform(img)
            return img, target, img 

    def setup_loaders(self):
        transform = transforms.ToTensor()
        train_dataset = self.CustomMNIST(root=config.DATASET_PATH, train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST(root=config.DATASET_PATH, train=False, download=True, transform=transform)

        for task_id in range(self.N_TASKS):
            label_start = task_id * self.N_CLASSES_PER_TASK
            label_end = (task_id + 1) * self.N_CLASSES_PER_TASK - 1

            train_indices = torch.where((train_dataset.targets >= label_start) & (train_dataset.targets <= label_end))[0]
            test_indices = torch.where((test_dataset.targets >= label_start) & (test_dataset.targets <= label_end))[0]

            train_subset = Subset(train_dataset, train_indices)
            test_subset = Subset(test_dataset, test_indices)

            # Split training data into training (90%) and validation (10%)
            val_size = int(0.1 * len(train_subset))
            train_size = len(train_subset) - val_size
            train_subset, val_subset = random_split(train_subset, [train_size, val_size])

            train_dataloader = DataLoader(train_subset, batch_size=self.get_batch_size(), shuffle=True)
            val_dataloader = DataLoader(val_subset, batch_size=self.get_batch_size(), shuffle=False)
            test_dataloader = DataLoader(test_subset, batch_size=self.get_batch_size(), shuffle=False)

            self.tasks.append((train_dataloader, task_id))
            self.val_loaders.append(val_dataloader)  
            self.test_loaders.append(test_dataloader)

    def get_train_loader(self):
        return self.tasks

    def get_test_loader(self, task_id: int):
        return self.test_loaders[task_id]
    
    def get_val_loader(self, task_id: int):
        return self.val_loaders[task_id]
    
    def get_batch_size(self):
        return 64
    
    def get_transform(self):
        return None

    def get_epochs(self):
        return 1
    
    def get_loss(self):
        return F.cross_entropy
    
    def get_model(self, weights=None):
        return models.SingleHeadMLP(input_size=784, hidden_size=100, output_size=self.N_CLASSES, weights=weights)
