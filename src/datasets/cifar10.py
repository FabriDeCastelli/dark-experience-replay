import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset, random_split
import torch.nn.functional as F
import config

from src import models
from PIL import Image


class SequentialCIFAR10:
    NAME = 'seq-cifar10'
    N_CLASSES_PER_TASK = 2
    N_TASKS = 5
    N_CLASSES = N_CLASSES_PER_TASK * N_TASKS
    SIZE = (32, 32)
    N_CHANNELS = 3
    MEAN, STD = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2615)
    TRANSFORM = transforms.Compose(
        [transforms.RandomCrop(32, padding=4),
         transforms.RandomHorizontalFlip(),
         transforms.ToTensor(),
         transforms.Normalize(MEAN, STD)])
    
    TEST_TRANSFORM = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])


    def __init__(self):
        self.setting = 'task-il'
        self.tasks = []
        self.test_loaders = []
        self.val_loaders = []
        self.setup_loaders() 

    class CustomCIFAR10(datasets.CIFAR10):
        def __init__(self, *args, **kwargs):
            super(SequentialCIFAR10.CustomCIFAR10, self).__init__(*args, **kwargs)
            self.not_aug_transform = transforms.Compose([transforms.ToTensor()])

        def __getitem__(self, index):
            """ returns the augmented image, target and the not augmented image"""
            img, target = self.data[index], self.targets[index]
            img = Image.fromarray(img, mode='RGB').copy()
            original_img = img.copy()
            not_aug_img = self.not_aug_transform(original_img)
            if self.transform is not None:
                img = self.transform(img)

            return img, target, not_aug_img
        
    def setup_loaders(self):
        train_dataset = self.CustomCIFAR10(root=config.DATASET_PATH, train=True, download=True, transform=self.TRANSFORM)
        test_dataset = datasets.CIFAR10(root=config.DATASET_PATH, train=False, download=True, transform=self.TEST_TRANSFORM)

        for task_id in range(self.N_TASKS):
            label_start = task_id * self.N_CLASSES_PER_TASK
            label_end = (task_id + 1) * self.N_CLASSES_PER_TASK - 1

            train_dataset_targets = torch.tensor(train_dataset.targets)
            test_dataset_targets = torch.tensor(test_dataset.targets)

            train_indices = torch.where((train_dataset_targets >= label_start) & (train_dataset_targets <= label_end))[0]
            test_indices = torch.where((test_dataset_targets >= label_start) & (test_dataset_targets <= label_end))[0]

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
        return transforms.Compose([
            transforms.ToPILImage(),
            self.TRANSFORM
        ])

    def get_epochs(self):
        return 1
    
    def get_loss(self):
        return F.cross_entropy
    
    def get_model(self, weights=None):
        return models.SingleHeadResNet18(num_classes=self.N_CLASSES, weights=weights)
