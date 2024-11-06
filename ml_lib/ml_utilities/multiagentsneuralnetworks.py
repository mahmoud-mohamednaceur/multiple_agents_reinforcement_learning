import torch
import torch.nn as nn
import torch.optim as optim


# This module defines various neural network architectures for reinforcement learning.
# It includes simple networks and more complex architectures, allowing for different optimizers
# and loss functions. These networks are designed to model the Q-value function in
# deep reinforcement learning tasks, with flexibility in structure and training parameters.

# SimpleNetwork has only one hidden layer and uses the Adam optimizer
class SimpleNetwork(nn.Module):
    """
    A simple feedforward neural network with one hidden layer.
    Uses ReLU activation and the Adam optimizer for training.
    """

    def __init__(self, input_dim, fc1_dim, fc2_dim, fc3_dim, fc4_dim, n_action, lr, loss):
        super(SimpleNetwork, self).__init__()
        self.lr = lr
        self.network = nn.Sequential(
            nn.Linear(input_dim, fc1_dim),
            nn.ReLU(),
            nn.Linear(fc1_dim, n_action),
        )

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = loss

    def forward(self, state):
        """Forward pass to compute actions from the input state."""
        actions = self.network(state)
        return actions


class MoreLayersNetwork(nn.Module):
    """
    A more complex feedforward neural network with multiple hidden layers.
    Uses ReLU activation and the Adam optimizer for training.
    """

    def __init__(self, input_dim, fc1_dim, fc2_dim, fc3_dim, fc4_dim, n_action, lr, loss):
        super(MoreLayersNetwork, self).__init__()
        self.lr = lr
        self.network = nn.Sequential(
            nn.Linear(input_dim, fc1_dim),
            nn.ReLU(),
            nn.Linear(fc1_dim, fc2_dim),
            nn.ReLU(),
            nn.Linear(fc2_dim, fc3_dim),
            nn.ReLU(),
            nn.Linear(fc3_dim, n_action),
        )

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = loss

    def forward(self, state):
        """Forward pass to compute actions from the input state."""
        actions = self.network(state)
        return actions


class SimpleNetworkWithDiffrentOptimizer(nn.Module):
    """
    A simple feedforward neural network with one hidden layer using Adagrad optimizer.
    """

    def __init__(self, input_dim, fc1_dim, fc2_dim, fc3_dim, fc4_dim, n_action, lr, loss):
        super(SimpleNetworkWithDiffrentOptimizer, self).__init__()
        self.lr = lr
        self.network = nn.Sequential(
            nn.Linear(input_dim, fc1_dim),
            nn.ReLU(),
            nn.Linear(fc1_dim, n_action),
        )

        # Optimizer with Adagrad
        self.optimizer = optim.Adagrad(self.parameters(), lr=lr)
        self.loss = loss

    def forward(self, state):
        """Forward pass to compute actions from the input state."""
        actions = self.network(state)
        return actions


class MoreLayersNetworkDiffrentOptimizer(nn.Module):
    """
    A more complex feedforward neural network with multiple hidden layers using Adagrad optimizer.
    """

    def __init__(self, input_dim, fc1_dim, fc2_dim, fc3_dim, fc4_dim, n_action, lr, loss):
        super(MoreLayersNetworkDiffrentOptimizer, self).__init__()
        self.lr = lr
        self.network = nn.Sequential(
            nn.Linear(input_dim, fc1_dim),
            nn.ReLU(),
            nn.Linear(fc1_dim, fc2_dim),
            nn.ReLU(),
            nn.Linear(fc2_dim, fc3_dim),
            nn.ReLU(),
            nn.Linear(fc3_dim, n_action),
        )

        self.optimizer = optim.Adagrad(self.parameters(), lr=lr)
        self.loss = loss

    def forward(self, state):
        """Forward pass to compute actions from the input state."""
        actions = self.network(state)
        return actions


class SimpleDiffrentLossFunction(nn.Module):
    """
    A simple feedforward neural network with one hidden layer that allows for
    different loss functions during training.
    """

    def __init__(self, input_dim, fc1_dim, fc2_dim, fc3_dim, fc4_dim, n_action, lr, loss):
        super(SimpleDiffrentLossFunction, self).__init__()
        self.lr = lr
        self.network = nn.Sequential(
            nn.Linear(input_dim, fc1_dim),
            nn.ReLU(),
            nn.Linear(fc1_dim, n_action),
        )

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = loss

    def forward(self, state):
        """Forward pass to compute actions from the input state."""
        actions = self.network(state)
        return actions


class MoreLayerDiffrentLossFunction(nn.Module):
    """
    A more complex feedforward neural network with multiple hidden layers that
    allows for different loss functions during training.
    """

    def __init__(self, input_dim, fc1_dim, fc2_dim, fc3_dim, fc4_dim, n_action, lr, loss):
        super(MoreLayerDiffrentLossFunction, self).__init__()
        self.lr = lr

        self.network = nn.Sequential(
            nn.Linear(input_dim, fc1_dim),
            nn.ReLU(),
            nn.Linear(fc1_dim, fc2_dim),
            nn.ReLU(),
            nn.Linear(fc2_dim, fc3_dim),
            nn.ReLU(),
            nn.Linear(fc3_dim, n_action),
        )

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = loss

    def forward(self, state):
        """Forward pass to compute actions from the input state."""
        actions = self.network(state)
        return actions


class Qnetwork(nn.Module):
    """
    A feedforward neural network specifically designed for Q-learning, with one hidden layer.
    Uses ReLU activation and the Adam optimizer for training.
    """

    def __init__(self, input_dim, fc1_dim, fc2_dim, fc3_dim, fc4_dim, n_action, lr, loss):
        super(Qnetwork, self).__init__()
        self.lr = lr
        self.network = nn.Sequential(
            nn.Linear(input_dim, fc1_dim),
            nn.ReLU(),
            nn.Linear(fc1_dim, n_action),
        )
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = loss

    def forward(self, state):
        """Forward pass to compute actions from the input state."""
        actions = self.network(state)
        return actions
