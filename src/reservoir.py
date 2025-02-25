import random
from torch.utils.data import DataLoader, TensorDataset
import torch
from utils import apply_transform

class ReservoirBuffer:
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.n_seen_examples = 0
        self.x_buffer = None
        self.z_buffer = None
        self.y_buffer = None
        self.task_id_buffer = None
        self.is_filled = None

    def add(self, x, z, y=None, task_id=None):
        """Adds data (x, z, y) to the buffer using reservoir sampling."""

        if self.x_buffer is None:
            data_shape_x = x.shape[1:] if isinstance(x, torch.Tensor) else x[0].shape
            data_shape_z = z.shape[1:] if isinstance(z, torch.Tensor) else z[0].shape
            if y is not None:
                self.y_buffer = torch.empty((self.buffer_size, 1), dtype=torch.float32)
            
            self.x_buffer = torch.empty((self.buffer_size, *data_shape_x), dtype=torch.float32)
            self.z_buffer = torch.empty((self.buffer_size, *data_shape_z), dtype=torch.float32)
            self.task_id_buffer = torch.empty((self.buffer_size, 1), dtype=torch.int64)
            self.is_filled = torch.zeros(self.buffer_size, dtype=torch.bool)

        for i in range(x.shape[0]):
            self._add_single(x[i], z[i], y[i] if y is not None else None, task_id=task_id)


    def _add_single(self, x, z, y=None, task_id=None):
        """Helper function to add a single data point."""

        self.n_seen_examples += 1
        if not self.is_filled.all():
            index = self.is_filled.logical_not().nonzero(as_tuple=True)[0][0]
            self.x_buffer[index] = x.clone().detach()
            self.z_buffer[index] = z.clone().detach()
            if y is not None:
                self.y_buffer[index] = y.clone().detach()
            if task_id is not None:
                self.task_id_buffer[index] = task_id
            self.is_filled[index] = True
        else:
            replace_index = random.randint(0, self.n_seen_examples - 1)
            if replace_index < self.buffer_size:
                self.x_buffer[replace_index] = x.clone().detach()
                self.z_buffer[replace_index] = z.clone().detach()
                if y is not None:
                    self.y_buffer[replace_index] = y.clone().detach()
                if task_id is not None:
                    self.task_id_buffer[replace_index] = task_id



    def get_data_loader(self, batch_size):
        """Returns a DataLoader for the buffer."""
        if self.x_buffer is None or self.is_filled.sum() == 0:
            return None
        valid_x = self.x_buffer[self.is_filled]
        valid_z = self.z_buffer[self.is_filled]
        if self.y_buffer is None:
            dataset = TensorDataset(valid_x, valid_z)
            return DataLoader(dataset, batch_size=batch_size, shuffle=True)
        valid_y = self.y_buffer[self.is_filled]
        dataset = TensorDataset(valid_x, valid_z, valid_y)
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)

    def sample(self, batch_size=1, transform=None):
        """Returns batch_size random samples (x, z, y) from the buffer."""
        if self.x_buffer is None:
            return None
    
        valid_x = self.x_buffer[self.is_filled]
        valid_z = self.z_buffer[self.is_filled]
        batch_size = min(batch_size, valid_x.shape[0])
        indices = torch.randint(0, valid_x.shape[0], (batch_size,))
        
        task_ids = None
        if self.task_id_buffer is not None:
            task_ids = self.task_id_buffer[indices]
        
        if self.y_buffer is None:
            return apply_transform(valid_x[indices], transform), valid_z[indices], None, task_ids

        valid_y = self.y_buffer[self.is_filled]
        return apply_transform(valid_x[indices]), valid_z[indices], valid_y, task_ids
        
    def __len__(self):
        return self.n_seen_examples
    
    def is_empty(self):
        return self.n_seen_examples == 0

