import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Subset, Sampler
import torch.nn.functional as F
import config
import numpy as np
from src import models
import numpy as np
from src.datasets.sampler import TestSamplerByID


class SequentialTinyImageNet:
    NAME = 'seq-tinyimg'
    SETTING = 'class-il'
    N_CLASSES_PER_TASK = 20
    N_TASKS = 10
    N_CLASSES = N_CLASSES_PER_TASK * N_TASKS
    MEAN, STD = (0.4802, 0.4480, 0.3975), (0.2770, 0.2691, 0.2821)
    SIZE = (64, 64)
    N_CHANNELS = 3
    TRANSFORM = transforms.Compose(
        [transforms.RandomCrop(64, padding=4),
         transforms.RandomHorizontalFlip(),
         transforms.ToTensor(),
         transforms.Normalize(MEAN, STD)])
    TEST_TRANSFORM = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])

    def __init__(self):
        self.training_data = []
        self.training_targets = []
        self.test_data = []
        self.test_targets = []

    def _load_data(self):
        for i in range(20):
            self.training_data.append(np.load(config.DATASET_PATH + '/tiny_imagenet/x_train_%02d.npy'.format(i+1)))
            self.training_targets.append(np.load(config.DATASET_PATH + '/tiny_imagenet/y_train_%02d.npy'.format(i+1)))
            self.test_data.append(np.load(config.DATASET_PATH + '/tiny_imagenet/x_val_%02d.npy'.format(i+1)))
            self.test_targets.append(np.load(config.DATASET_PATH + '/tiny_imagenet/y_val_%02d.npy'.format(i+1)))
        self.training_data = np.concatenate(np.array(self.training_data))
        self.training_targets = np.concatenate(np.array(self.training_targets))
        self.test_data = np.concatenate(np.array(self.test_data))
        self.test_targets = np.concatenate(np.array(self.test_targets))

    def get_data_loaders(self):
        self._load_data()
        tasks = []
        cumulative_test_indices = torch.tensor([], dtype=torch.int16)

        for task_id in range(self.N_TASKS):
            label_start = task_id * self.N_CLASSES_PER_TASK
            label_end = (task_id + 1) * self.N_CLASSES_PER_TASK - 1

            train_indices = torch.where((self.training_targets >= label_start) & (self.training_targets <= label_end))[0]
            test_indices = torch.where((self.test_targets >= label_start) & (self.test_targets <= label_end))[0]
            
            train_subset = Subset(self.training_data, train_indices)
            cumulative_test_indices = torch.cat((cumulative_test_indices, test_indices))
            test_subset = Subset(self.test_data, cumulative_test_indices)
        
            train_dataloader = DataLoader(train_subset, batch_size=self.get_batch_size(), shuffle=True)
            test_dataloader = DataLoader(test_subset, batch_size=self.get_batch_size(), shuffle=False)

            tasks.append((train_dataloader, test_dataloader, task_id))
        
        return tasks
    
    def get_test_from_experience(self, experience_id: int):
        sampler = TestSamplerByID(self.test_data, range(experience_id * self.N_CLASSES_PER_TASK, (experience_id + 1) * self.N_CLASSES_PER_TASK))
        return DataLoader(self.test_data, batch_size=self.get_batch_size(), sampler=sampler)
    
    def get_batch_size(self):
        return 32
    
    def get_epochs(self):
        return 50
    
    def get_loss(self):
        return F.cross_entropy
    
    def get_model(self):
        return models.ResNet18(num_classes=200)


