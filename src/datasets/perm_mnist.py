from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import torch.nn.functional as F
import config
import torch

from src import models
from PIL import Image


class PermutedMNIST:
    NAME = 'perm-mnist'
    SETTING = 'domain-il'
    N_CLASSES_PER_TASK = 10
    N_TASKS = 20
    SIZE = (28, 28)

    def __init__(self):
        self.setting = self.SETTING
        self.tasks = []
        self.test_loaders = []
        self.val_loaders = []
        self.permutations = []
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

        def get_transform(perm):
            """ Ensure each dataset gets its own independent transformation """
            return transforms.Compose([
                transforms.ToTensor(),
                lambda x: x.view(-1)[perm].view(28, 28)
            ])

        for task_id in range(self.N_TASKS):
            permutation = torch.randperm(28 * 28)
            self.permutations.append(permutation)

            train_transform = get_transform(permutation)
            train_dataset = self.CustomMNIST(root=config.DATASET_PATH, train=True, download=True,
                                             transform=train_transform)

            # split training data into training (90%) and validation (10%)
            val_size = int(0.1 * len(train_dataset))
            train_size = len(train_dataset) - val_size
            train_subset, val_subset = random_split(train_dataset, [train_size, val_size])

            test_transform = get_transform(permutation)
            test_dataset = datasets.MNIST(root=config.DATASET_PATH, train=False, download=True,
                                          transform=test_transform)

            train_dataloader = DataLoader(train_subset, batch_size=self.get_batch_size(), shuffle=True)
            val_dataloader = DataLoader(val_subset, batch_size=self.get_batch_size())
            test_dataloader = DataLoader(test_dataset, batch_size=self.get_batch_size())

            self.tasks.append((train_dataloader, task_id))
            self.val_loaders.append(val_dataloader)
            self.test_loaders.append(test_dataloader)

    def get_train_loader(self):
        return self.tasks

    def get_test_loader(self, task_id: int):
        return self.test_loaders[task_id]

    def get_val_loader(self, task_id: int):
        return self.val_loaders[task_id]

    def get_transform(self):
        return None

    def get_loss(self):
        return F.cross_entropy

    def get_batch_size(self) -> int:
        return 128

    def get_epochs(self):
        return 1

    def get_model(self, weights=None):
        return models.SingleHeadMLP(input_size=28 * 28, hidden_size=100, output_size=10, weights=weights)
