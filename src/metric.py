import torch
import numpy as np


class Metric:

    def __init__(self, dataset):
        self.num_tasks = dataset.N_TASKS
        self.accuracies_CIL = np.zeros((self.num_tasks, self.num_tasks))
        self.accuracies_TIL = np.zeros((self.num_tasks, self.num_tasks))
        self.random_accuracies_TIL = []
        self.random_accuracies_CIL = []
        self.dataset = dataset
        self.setting = dataset.setting
        self.random_baseline_accuracy()

    @torch.no_grad()
    def __call__(self, model, task_id, mode='test'):
        """
        Sets the model to evaluation mode and computes all accuracies on the validation or test set.
        In case of test set we compute the accuracy for each task, in case of validation only for the current task.

        Args:
            model: model to evaluate
            task_id: current task id
            mode: 'val' or 'test'
        
        Returns:
            None
        """

        status = model.net.training
        model.net.eval()

        if mode == 'val':
            val_loader = self.dataset.get_val_loader(task_id)
            correct, correct_mask_classes, total = 0.0, 0.0, 0.0
            to = (task_id + 1) * self.dataset.N_CLASSES_PER_TASK

            for data in val_loader:

                x, y = data[:2]

                outputs = model(x)
                if self.setting in ['task-il', 'class-il']:
                    _, predicted = torch.max(outputs[:, :to].data, 1)
                else:
                    _, predicted = torch.max(outputs.data, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()

                # mask classes to provide both TIL and CIL evaluation
                if self.setting in ['class-il', 'task-il']:
                    self.mask_classes(outputs, task_id)

                    _, predicted = torch.max(outputs.data, 1)
                    correct_mask_classes += (predicted == y).sum().item()

            if correct > correct_mask_classes and self.setting in ['task-il', 'class-il']:
                print("WARNING: TIL performs worse than CIL")

            self.accuracies_CIL[task_id, task_id] = correct / total * 100
            self.accuracies_TIL[task_id, task_id] = correct_mask_classes / total * 100
            return

        for experience_id in range(self.num_tasks):

            test_loader = self.dataset.get_test_loader(experience_id)
            correct, correct_mask_classes, total = 0.0, 0.0, 0.0
            to = (experience_id + 1) * self.dataset.N_CLASSES_PER_TASK

            # test loop
            for data in test_loader:

                x, y = data[:2]

                outputs = model(x)
                if self.setting in ['task-il', 'class-il']:
                    _, predicted = torch.max(outputs[:, :to].data, 1)
                else:
                    _, predicted = torch.max(outputs.data, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()

                # mask classes to provide both TIL and CIL evaluation
                if self.setting in ['class-il', 'task-il']:
                    self.mask_classes(outputs, experience_id)

                    _, predicted = torch.max(outputs.data, 1)
                    correct_mask_classes += (predicted == y).sum().item()

            if correct > correct_mask_classes and self.setting in ['task-il', 'class-il']:
                print("WARNING: TIL performs worse than CIL")

            self.accuracies_CIL[task_id, experience_id] = correct / total * 100
            self.accuracies_TIL[task_id, experience_id] = correct_mask_classes / total * 100

        model.net.train(status)

    def mask_classes(self, outputs, task_id):
        """
        Masks classes in case of task-il setting.

        Args:
            outputs: model outputs
            task_id: current task id
        
        Returns:
            None
        """
        start = task_id * self.dataset.N_CLASSES_PER_TASK
        end = (task_id + 1) * self.dataset.N_CLASSES_PER_TASK
        outputs[:, :start] = -float('inf')
        outputs[:, end:] = -float('inf')

    def random_baseline_accuracy(self):
        """ Compute the random baseline accuracy for each task. """
        random_baseline = self.dataset.get_model()

        for i in range(self.num_tasks):
            test_loader = self.dataset.get_test_loader(i)
            correct = 0
            total = 0
            to = (i + 1) * self.dataset.N_CLASSES_PER_TASK
            with torch.no_grad():
                for x, y in test_loader:
                    outputs = random_baseline(x)
                    _, predicted = torch.max(outputs[:, :to].data, 1)
                    total += y.size(0)
                    correct += (predicted == y).sum().item()

                    if self.setting in ['class-il', 'task-il']:
                        self.mask_classes(outputs, i)
                        _, predicted = torch.max(outputs.data, 1)
                        correct += (predicted == y).sum().item()

            self.random_accuracies_CIL.append(correct / total * 100)
            self.random_accuracies_TIL.append(correct / total * 100)

    def print_metrics(self):
        """ Print evaluation metrics for both CIL and TIL settings. """

        if self.setting == 'domain-il':
            self.print_DIL()
            return  # Exit early if domain-il is selected

        last_model_CIL = self.accuracies_CIL[self.num_tasks - 1, self.num_tasks - 1]
        last_model_TIL = self.accuracies_TIL[self.num_tasks - 1, self.num_tasks - 1]
        avg_CIL = np.tril(self.accuracies_CIL).sum() / (self.num_tasks * (self.num_tasks + 1) / 2)
        avg_TIL = np.tril(self.accuracies_TIL).sum() / (self.num_tasks * (self.num_tasks + 1) / 2)
        full_stream_CIL = np.mean(self.accuracies_CIL[-1, :])
        full_stream_TIL = np.mean(self.accuracies_TIL[-1, :])
        forgetting_CIL = self._forgetting(self.accuracies_CIL)
        forgetting_TIL = self._forgetting(self.accuracies_TIL)
        backward_transfer_CIL = self._backward_transfer(self.accuracies_CIL)
        backward_transfer_TIL = self._backward_transfer(self.accuracies_TIL)
        forward_transfer_CIL = self._forward_transfer(self.accuracies_CIL, self.random_accuracies_CIL)
        forward_transfer_TIL = self._forward_transfer(self.accuracies_TIL, self.random_accuracies_TIL)

        print(f"\n=== Task-IL (TIL) vs Class-IL (CIL) Metrics ===\n")
        print(f"Accuracy - Last Model (CIL): \t {last_model_CIL:.2f}")
        print(f"Accuracy - Last Model (TIL): \t {last_model_TIL:.2f}\n")
        print(f"Accuracy - Average (CIL): \t {avg_CIL:.2f}")
        print(f"Accuracy - Average (TIL): \t {avg_TIL:.2f}\n")
        print(f"Accuracy - Full Stream (CIL): \t {full_stream_CIL:.2f}")
        print(f"Accuracy - Full Stream (TIL): \t {full_stream_TIL:.2f}\n")
        print(f"Forgetting (CIL): \t {forgetting_CIL:.2f}")
        print(f"Forgetting (TIL): \t {forgetting_TIL:.2f}\n")
        print(f"Backward Transfer (CIL): \t {backward_transfer_CIL:.2f}")
        print(f"Backward Transfer (TIL): \t {backward_transfer_TIL:.2f}\n")
        print(f"Forward Transfer (CIL): \t {forward_transfer_CIL:.2f}")
        print(f"Forward Transfer (TIL): \t {forward_transfer_TIL:.2f}\n")

    def print_DIL(self):
        """ Print evaluation metrics only for Domain-IL (CIL metrics only). """
        last_model_CIL = self.accuracies_CIL[self.num_tasks - 1, self.num_tasks - 1]
        avg_CIL = np.tril(self.accuracies_CIL).sum() / (self.num_tasks * (self.num_tasks + 1) / 2)
        full_stream_CIL = np.mean(self.accuracies_CIL[-1, :])
        forgetting_CIL = self._forgetting(self.accuracies_CIL)
        backward_transfer_CIL = self._backward_transfer(self.accuracies_CIL)
        forward_transfer_CIL = self._forward_transfer(self.accuracies_CIL, self.random_accuracies_CIL)

        print(f"\n=== Domain-IL (DIL Metrics Only) ===\n")
        print(f"Accuracy - Last Model (DIL): \t {last_model_CIL:.2f}")
        print(f"Accuracy - Average (DIL): \t {avg_CIL:.2f}")
        print(f"Accuracy - Full Stream (DIL): \t {full_stream_CIL:.2f}")
        print(f"Forgetting (DIL): \t {forgetting_CIL:.2f}")
        print(f"Backward Transfer (DIL): \t {backward_transfer_CIL:.2f}")
        print(f"Forward Transfer (DIL): \t {forward_transfer_CIL:.2f}")

    def get_task_accuracy(self, task_id):
        """
        Gets the accuracy of a model at a given task. 
        In case of not domain-il setting we return an average of CIL and TIL accuracies (done for continual hyperparameter selection).

        Args:
            task_id: task id

        Returns:
            task accuracy
        """
        if self.setting != 'domain-il':
            CIL = self.accuracies_CIL[task_id, task_id]
            TIL = self.accuracies_TIL[task_id, task_id]
            return (CIL + TIL) / 2
        return self.accuracies_CIL[task_id, task_id]

    def _backward_transfer(self, accuracies):
        """ Compute backward transfer for a given accuracy matrix. """
        l = []
        for i in range(self.num_tasks - 1):
            l.append(accuracies[-1, i] - accuracies[i, i])
        return np.mean(l) if l else 0

    def _forward_transfer(self, accuracies, random_accuracies):
        """ Compute forward transfer for a given accuracy matrix. """
        l = []
        for i in range(1, self.num_tasks):
            l.append(accuracies[i - 1, i] - random_accuracies[i])
        return np.mean(l) if l else 0

    def _forgetting(self, accuracies):
        """ Compute forgetting as negative backward transfer. """
        return -self._backward_transfer(accuracies)
