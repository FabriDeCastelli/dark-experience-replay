# Dark Experience Replay

This repository contains an implementation of Dark Experience Replay ([original paper](https://arxiv.org/pdf/2004.07211)) in PyTorch. Dark Experience Replay is a method that relies on _dark knowledge_ for distilling past experiences, sampled of the entire training trajectory. It is a combination of Experience Replay and Knowledge Distillation. The idea is to store the past non-normalized logits of the network in a replay buffer and use them to train the network to then use them to push the model response for next tasks to take into account previous tasks output distribution.

In this repository you will find the following files
```
📂 Project Root
├── 📂 src
│   ├── 📂 datasets
│   │   ├── ✈️ cifar10.py
│   │   ├── ➡️ seq_mnist.py
│   │   ├── 🔃 perm_mnist.py
│   │   └── 🔄 rotated_mnist.py
│   ├── 📏 metric.py
│   ├── 🗳️ model_selection.py
│   ├── ⊞ models.py
│   ├── 🎬 replay.py
│   └── ⛁ reservoir.py
├── 📂 hyperparameters
│   ├── 📜 perm-mnist.yaml
│   ├── 📜 seq-mnist.yaml
│   └── 📜 rot-mnist.yaml
├── 📂 notebooks
│   ├── 🌍 showcase.iynb
│   ├── 🌍 showcase_cifar10.iynb
│   └── 🌍 validation.iynb
├── ⚙️ config.py
├── ✅ utils.py
├── ▶️ main.py
├── 📜 LICENSE
├── 📖 README.md
├── 🛠 pyproject.toml
└── 🛠 poetry.lock
```

## Contents
In order, the relevant contents are:
- `src/`
    - `datasets/**.py`: contains the used benchmarks
    - `metric.py`: contains the metric used for the experiments
    - `model_selection.py`: implementation of the continual hyperparameter selection framework
    - `models.py`: contains the models used for the experiments (MLP and ResNet18)
    - `replay.py`: contains the implementation of Dark Experience Replay
    - `reservoir.py`: contains the implementation of the reservoir sampling
- `notebooks/**.ipynb`: contains the showcase to demonstrate the usage of the code
- `hyperparameters/**.yaml`: contains the hyperparameters used for the experiments
- `config.py`: contains the configuration for the experiments
- `utils.py`: contains utility functions
- `main.py`: contains the main script to run the experiments