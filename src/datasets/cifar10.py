import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import torch.nn.functional as F
import config

from src import models
from PIL import Image
from src.datasets.sampler import TestSamplerByID


class SequentialCIFAR10:
    NAME = 'seq-cifar10'
    SETTING = 'class-il'
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

    class CustomCIFAR10(datasets.CIFAR10):
        def __init__(self, *args, **kwargs):
            super(SequentialCIFAR10.CustomCIFAR10, self).__init__(*args, **kwargs)
        
        def __getitem__(self, index):
            """ returns the augmented image, target and the not augmented image"""

            img, target = self.data[index], self.targets[index]
            img = Image.fromarray(img, mode='RGB').copy()
            
            img_aug = SequentialCIFAR10.TRANSFORM(img)
            return img_aug, target, transforms.ToTensor()(img)
        

    def get_data_loaders(self):


        train_dataset = self.CustomCIFAR10(root=config.DATASET_PATH, train=True, download=True, transform=self.TRANSFORM)
        test_dataset = self.CustomCIFAR10(root=config.DATASET_PATH, train=False, download=True, transform=self.TEST_TRANSFORM)

        tasks = []
        cumulative_test_indices = torch.tensor([], dtype=torch.int16)
        
        for task_id in range(self.N_TASKS):
            label_start = task_id * self.N_CLASSES_PER_TASK
            label_end = (task_id + 1) * self.N_CLASSES_PER_TASK - 1

            train_dataset_targets = torch.tensor(train_dataset.targets)
            test_dataset_targets = torch.tensor(test_dataset.targets)

            train_indices = torch.where((train_dataset_targets >= label_start) & (train_dataset_targets <= label_end))[0]
            test_indices = torch.where((test_dataset_targets >= label_start) & (test_dataset_targets <= label_end))[0]
            
            train_subset = Subset(train_dataset, train_indices)
            cumulative_test_indices = torch.cat((cumulative_test_indices, test_indices))
            test_subset = Subset(test_dataset, cumulative_test_indices)
        
            train_dataloader = DataLoader(train_subset, batch_size=self.get_batch_size(), shuffle=True)
            test_dataloader = DataLoader(test_subset, batch_size=self.get_batch_size(), shuffle=False)

            tasks.append((train_dataloader, test_dataloader, task_id))
        
        return tasks
    
    def get_test_from_experience(self, experience_id: int):
        test_dataset = datasets.CIFAR10(root=config.DATASET_PATH, train=False, download=True, transform=self.TEST_TRANSFORM)
        sampler = TestSamplerByID(test_dataset, range(experience_id * self.N_CLASSES_PER_TASK, (experience_id + 1) * self.N_CLASSES_PER_TASK))
        return DataLoader(test_dataset, batch_size=self.get_batch_size(), sampler=sampler)

    
    @staticmethod
    def get_transform():
        return transforms.Compose([transforms.ToPILImage(), SequentialCIFAR10.TRANSFORM])
    
    def get_batch_size(self):
        return 32

    def get_epochs(self):
        return 50
    
    def get_loss(self):
        return F.cross_entropy
    
    def get_model(self):
        return models.ResNet18(num_classes=10)
    
