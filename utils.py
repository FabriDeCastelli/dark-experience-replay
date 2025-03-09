import PIL
import torch
from typing import Any
import yaml
import config


def apply_transform(x: torch.Tensor, transform, autosqueeze=False) -> torch.Tensor:
    """Applies a transform to a batch of images.

    Args:
        x: a batch of images.
        transform: the transform to apply.
        autosqueeze: whether to automatically squeeze the output tensor.

    Returns:
        The transformed batch of images.
    """
    if transform is None:
        return x

    if isinstance(x, PIL.Image.Image):
        return transform(x)

    out = torch.stack([transform(xi) for xi in x.cpu()], dim=0).to(x.device)
    if autosqueeze and out.shape[0] == 1:
        out = out.squeeze(0)
    return out


def fine_tune(dataset, lr, task_id, weights=None):
    """
    Fine-tunes the model on the current task (does not use the buffer).

    Args:
        dataset: the dataset to fine-tune on
        lr: the learning rate
        task_id: the current task id
        weights: the weights to load into the model

    Returns:
        the model's weights and the accuracy on the validation set
    """
    epochs = dataset.get_epochs()
    model = dataset.get_model(weights=weights)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    criterion = dataset.get_loss()
    task_ids = torch.full((dataset.get_batch_size(),), task_id, dtype=torch.long)
    train_loader = dataset.get_train_loader()[task_id][0]
    for epoch in range(epochs):
        for x, y, _ in train_loader:
            optimizer.zero_grad()
            outputs = model.forward(x, task_ids)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

    val_loader = dataset.get_val_loader(task_id)
    return model, model.eval(val_loader, task_id)


def read_yaml(path: str) -> dict[str, Any]:
    """
    Reads a file in .yaml format.

    :param path: the path of the file to read
    :return: the dictionary contained in the file
    """
    with open(path, "r") as file:
        dictionary = yaml.load(file, Loader=yaml.FullLoader)
    return dictionary


def load_hparams(model: str) -> dict[str, Any]:
    """
    Loads the hyperparameters for a certain model.

    :param model: the name of the model
    :return: the hyperparameters for the given model
    """
    return read_yaml(config.HPARAMS_ROOT.format(model))
