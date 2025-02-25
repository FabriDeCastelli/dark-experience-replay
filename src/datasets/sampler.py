
from torch.utils.data import Sampler

class TestSamplerByID(Sampler):
    def __init__(self, dataset, target_classes):
        """
        Args:
            dataset (torch.utils.data.Dataset): The dataset to sample from (e.g., CIFAR10).
            target_classes (list): List of class IDs to sample from (e.g., [0, 1]).
        """
        self.dataset = dataset
        self.target_classes = target_classes
        self.indices = self._get_class_indices()

    def _get_class_indices(self):
        """ Get the indices of samples that belong to the target classes. """
        indices = []
        for idx, (_, target) in enumerate(self.dataset):
            if target in self.target_classes:
                indices.append(idx)
        return indices

    def __iter__(self):
        """ Return an iterator of indices for the target classes. """
        return iter(self.indices)

    def __len__(self):
        """ Return the length of the sampler (i.e., how many indices we are returning). """
        return len(self.indices)
