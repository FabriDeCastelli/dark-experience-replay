import torch
import numpy as np

class Metric:

    def __init__(self, num_tasks):
        self.accuracy_table = np.zeros((num_tasks, num_tasks))
        self.num_tasks = num_tasks

    def __call__(self, data_loader, model, task_id, experience_id):
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in data_loader:
                outputs = model.forward(x, experience_id)
                _, predicted = torch.max(outputs.data, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()
        accuracy = correct / total
        self.accuracy_table[task_id, experience_id] = accuracy
        return accuracy
    
    def get_metrics(self, random_results):

        last_model = self.accuracy_table[self.num_tasks-1, self.num_tasks-1]
        average_over_time = np.mean(self.accuracy_table)
        full_stream = np.mean(self.accuracy_table[-1, :])
        forgetting = self._forgetting()
        backward_transfer = self._backward_transfer()
        forward_transfer = self._forward_transfer(random_results=random_results)

        return f"\nAccuracy: \t {last_model:.2f} \n" + \
                f"Average over time: {average_over_time:.2f} \n" + \
                f"Full stream: \t{full_stream:.2f}\n" + \
                f"Forgetting: \t{forgetting:.2f} \n" + \
                f"Backward transfer: {backward_transfer:.2f} \n" + \
                f"Forward transfer: {forward_transfer:.2f} \n"
    
    def _backward_transfer(self):
        l = []
        for i in range(self.num_tasks - 1):
            l.append(self.accuracy_table[-1, i] - self.accuracy_table[i, i])
        return np.mean(l)
    
    def _forward_transfer(self, random_results):
        l = []
        for i in range(1, self.num_tasks):
            l.append(self.accuracy_table[i-1, i] - random_results[i])
        return np.mean(l)
    
    def _forgetting(self):
        return -self._backward_transfer()

    