
import torch
import torchvision as tv

class SingleHeadMLP(torch.nn.Module):
    def __init__(self, input_size=784, hidden_size=100, output_size=10):
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
    
    def forward(self, x, task_id=None):
        x = x.view(x.size(0), -1)
        return self.net(x)
    
    def eval(self, test_loader, task_id=None):
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in test_loader:
                outputs = self(x)
                _, predicted = torch.max(outputs.data, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()
        
        return correct / total
    

class MultiHeadMLP(torch.nn.Module):
    def __init__(self, input_size=784, hidden_size=100, num_heads=5, head_output_size=2):
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
    

    def forward(self, x, head_ids):
        x = x.view(x.size(0), -1)
        features = self._features(x)  
        all_head_outputs = torch.stack([head(features) for head in self.heads], dim=1)
        
        batch_indices = torch.arange(x.size(0), dtype=torch.long) 
        selected_outputs = all_head_outputs[batch_indices, head_ids] 
        
        return selected_outputs
    
    def eval(self, test_loader, task_id):
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in test_loader:
                outputs = self(x, task_id)
                _, predicted = torch.max(outputs.data, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()
        
        return correct / total




class ResNet18(torch.nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet18, self).__init__()
        self.resnet18 = tv.models.resnet18(pretrained=False)
        self.resnet18.fc = torch.nn.Linear(self.resnet18.fc.in_features, num_classes)
    
    def forward(self, x):
        return self.resnet18(x)
    
    def eval(self, test_loader):
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in test_loader:
                outputs = self(x)
                _, predicted = torch.max(outputs.data, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()
        
        return correct / total


