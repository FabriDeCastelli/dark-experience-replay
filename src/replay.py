
import torch.nn.functional as F
from src.reservoir import ReservoirBuffer
import torch


class DarkExperienceReplay:
    """
    Dark Experience Replay (DER).
    """
    def __init__(self, dataset, lr=None, buffer_size=500, weights=None):
        self.dataset = dataset
        self.buffer = ReservoirBuffer(buffer_size)
        self.model = dataset.get_model(weights)
        if lr is not None:
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr)
        else:
            self.optimizer = None
        self.criterion = dataset.get_loss()
        self.transform = dataset.get_transform()

    def train(self, train_loader, task_id, replay_batch_size, alpha=0.5, beta=None, verbose=True):
        """
        Train the model using Dark Experience Replay (DER).

        Args:
            train_loader (torch.utils.data.DataLoader): Training data loader.
            task_id (int): the current task id.
            replay_batch_size (int): Replay batch size.
            alpha (float): Alpha regularization term.
            beta (float): Beta regularization term. If specified it the algorithm is DER++.
            verbose (bool): If True, print the loss at each epoch.
        """

        # get predefined number of epochs
        epochs = self.dataset.get_epochs()

        if train_loader.batch_size < replay_batch_size:
            raise ValueError("Replay batch size must be smaller or equal to training batch size.")
        
        for epoch in range(epochs):
            for x_aug, y, x in train_loader:

                self.optimizer.zero_grad()

                # get predictions
                task_ids = torch.full((x_aug.size(0),), task_id, dtype=torch.long)
                outputs = self.model.forward(x_aug, task_ids)
                loss = self.criterion(outputs, y)

                if not self.buffer.is_empty():

                    # compute alpha term
                    x_prime, z_prime, _, _task_ids = self.buffer.sample(batch_size=replay_batch_size, transform=self.transform)
                    buffer_outputs = self.model.forward(x_prime, _task_ids)
                    loss += alpha * F.mse_loss(buffer_outputs, z_prime)

                    # compute beta term
                    if beta is not None:
                        x_second, _, y_second, _task_ids = self.buffer.sample(batch_size=replay_batch_size, transform=self.transform)
                        output_class = self.model.forward(x_second, _task_ids)
                        loss += beta * self.criterion(output_class, y_second)

                loss.backward()
                self.optimizer.step()
                self.buffer.add(x=x, z=outputs, y=y if beta is not None else None, task_id=task_id)

            if verbose:
                print(f"Epoch {epoch+1}/{epochs} - Loss: {loss.item()}", end="\r")

    def eval(self, val_loader):
        return self.model.eval(val_loader)

    def get_buffer(self):
        return self.buffer

    def set_optimizer(self, lr, model):
        self.optimizer = torch.optim.SGD(model.parameters(), lr=lr)
        self.model = model
