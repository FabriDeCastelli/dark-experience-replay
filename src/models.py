import torch
import torchvision as tv


class SingleHeadMLP(torch.nn.Module):
    """
    Single Head MLP ideally used for CIL settings.
    """

    def __init__(self, input_size=784, hidden_size=100, output_size=10, weights=None):
        super(SingleHeadMLP, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, hidden_size)
        self.fc2 = torch.nn.Linear(hidden_size, hidden_size)
        self._features = torch.nn.Sequential(
            self.fc1,
            torch.nn.ReLU(),
            self.fc2,
            torch.nn.ReLU()
        )
        self.classifier = torch.nn.Linear(hidden_size, output_size)
        self.net = torch.nn.Sequential(
            self._features,
            self.classifier
        )

        if weights is not None:
            self.load_state_dict(weights)

    def forward(self, x, *args):
        x = x.view(x.size(0), -1)
        return self.net(x)

    def eval(self, test_loader, *args):
        correct = 0
        total = 0
        with torch.no_grad():
            for data in test_loader:
                x, y = data[:2]
                outputs = self(x)
                _, predicted = torch.max(outputs.data, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()

        return correct / total * 100


class SingleHeadResNet18(torch.nn.Module):
    """
    Single Head ResNet18 ideally used for CIL settings (not pretrained).
    """
    def __init__(self, num_classes=10, weights=None):
        super(SingleHeadResNet18, self).__init__()
        self.net = tv.models.resnet18(pretrained=False)
        self.net.fc = torch.nn.Linear(self.net.fc.in_features, num_classes)

        if weights is not None:
            self.load_state_dict(weights)

    def forward(self, x, *args):
        return self.net(x)

    def eval(self, test_loader, *args):
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in test_loader:
                outputs = self(x)
                _, predicted = torch.max(outputs.data, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()

        return correct / total


# region: multi-head models (not used)

class MultiHeadMLP(torch.nn.Module):
    """
    Multi Head MLP ideally used for TIL settings.
    """

    def __init__(self, input_size=784, hidden_size=100, num_heads=5, head_output_size=2, weights=None):
        super(MultiHeadMLP, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, hidden_size)
        self.fc2 = torch.nn.Linear(hidden_size, hidden_size)
        self._features = torch.nn.Sequential(
            self.fc1,
            torch.nn.ReLU(),
            self.fc2,
            torch.nn.ReLU()
        )
        self.heads = torch.nn.ModuleList([torch.nn.Linear(hidden_size, head_output_size) for _ in range(num_heads)])

        if weights is not None:
            self.load_state_dict(weights)

    def forward(self, x, head_ids, *args):
        x = x.view(x.size(0), -1)
        features = self._features(x)
        all_head_outputs = torch.stack([head(features) for head in self.heads], dim=1)

        batch_indices = torch.arange(x.size(0), dtype=torch.long)
        selected_outputs = all_head_outputs[batch_indices, head_ids]
        return selected_outputs

    def eval(self, test_loader, task_id, *args):
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in test_loader:
                outputs = self(x, task_id)
                _, predicted = torch.max(outputs.data, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()

        return correct / total


class MultiHeadResNet18(torch.nn.Module):
    """
    Multi Head ResNet18 ideally used for TIL settings (not pretrained).
    """

    def __init__(self, num_heads=5, head_output_size=2, weights=None):
        super(MultiHeadResNet18, self).__init__()
        self.resnet18 = tv.models.resnet18(pretrained=False)
        self.heads = torch.nn.ModuleList(
            [torch.nn.Linear(self.resnet18.fc.in_features, head_output_size) for _ in range(num_heads)])

        if weights is not None:
            self.load_state_dict(weights)

    def forward(self, x, head_ids, *args):
        features = self.resnet18(x)
        all_head_outputs = torch.stack([head(features) for head in self.heads], dim=1)

        batch_indices = torch.arange(x.size(0), dtype=torch.long)
        selected_outputs = all_head_outputs[batch_indices, head_ids]

        return selected_outputs

    def eval(self, test_loader, task_id, *args):
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in test_loader:
                outputs = self(x, task_id)
                _, predicted = torch.max(outputs.data, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()

        return correct / total

# endregion
