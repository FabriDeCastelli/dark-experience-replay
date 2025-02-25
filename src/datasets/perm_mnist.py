from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
import config
from src.augmentations import random_permutation
import torch

from src import models
from PIL import Image


class PermutedMNIST:
    NAME = 'perm-mnist'
    SETTING = 'domain-il'
    N_CLASSES_PER_TASK = 10
    N_TASKS = 20
    SIZE = (28, 28)

    def __init__(self, setting='domain-il'):
        self.setting = setting
        self.tasks = []
        self.test_loaders = []
        self.setup_loaders() 

    class CustomMNIST(datasets.MNIST):
        def __init__(self, *args, **kwargs):
            super(PermutedMNIST.CustomMNIST, self).__init__(*args, **kwargs)

        def __getitem__(self, index):
            """ returns the augmented image, target and the not augmented image"""
            img, target = self.data[index], self.targets[index]
            img = Image.fromarray(img.numpy(), mode='L').copy()
            if self.transform is not None:
                img = self.transform(img)

            return img, target, img
        
    def setup_loaders(self):
        # gets both train and test loaders for each task and stores them in a state variable
        # this is to avoid recomputing the loaders for each task
        
        self.permutations = [torch.randperm(28 * 28) for _ in range(self.N_TASKS)]

        for task_id in range(self.N_TASKS):
            transform = transforms.Compose([
                transforms.ToTensor(),
                lambda x: x.view(-1)[self.permutations[task_id]].view(28, 28)  # Apply the same permutation
            ])

            train_dataset = self.CustomMNIST(root=config.DATASET_PATH, train=True, download=True, transform=transform)
            train_dataloader = DataLoader(train_dataset, batch_size=self.get_batch_size(), shuffle=True)
            self.tasks.append((train_dataloader, task_id))
            test_dataset = datasets.MNIST(root=config.DATASET_PATH, train=False, download=True, transform=transform)
            self.test_loaders.append(DataLoader(test_dataset, batch_size=self.get_batch_size()))
        
    def get_train_loader(self):
        return self.tasks
    
    def get_test_loader(self, task_id: int):
        return self.test_loaders[task_id]

    @staticmethod
    def get_transform():
        return None

    def get_loss(self):
        return F.cross_entropy

    def get_batch_size(self) -> int:
        return 128

    def get_epochs(self):
        return 1
    
    def get_model(self):
        return models.SingleHeadMLP(input_size=28*28, hidden_size=100, output_size=10)
