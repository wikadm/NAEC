import numpy as np
import pandas as pd
from typing import Tuple, List, Optional

class NeuralNet:

    def __init__(self,
                 layers: List[int],
                 epochs: int = 100,
                 learning_rate: float = 0.01,
                 momentum: float = 0.9,
                 activation: str = 'sigmoid',
                 validation_split: float = 0.2):


        """
        Initialize the Neural Network.

        layers : list of int
            Number of units in each layer 
        epochs : int
            Number of training epochs
        learning_rate : float
            Learning rate for weight updates
        momentum : float
            Momentum parameter for weight updates
        activation : str
            Activation function ('sigmoid', 'relu', 'linear', 'tanh')
        validation_split : float
            Percentage of data to use for validation 
            0 means no validation
        """

        self.L = len(layers) 
        self.n = layers  
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.fact = activation  
        self.validation_split = validation_split

        # Initialize arrays for network structure
        self.h = [None] * self.L  # Fields
        self.xi = [None] * self.L  # Activations
        self.w = [None] * self.L  # Weights
        self.theta = [None] * self.L  # Thresholds
        self.delta = [None] * self.L  # Error propagation
        self.d_w = [None] * self.L  # Weight changes
        self.d_theta = [None] * self.L  # Threshold changes
        self.d_w_prev = [None] * self.L  # Previous weight changes (momentum)
        self.d_theta_prev = [None] * self.L  # Previous threshold changes (momentum)

        # Initialize weights and thresholds
        self._initialize_weights()

        # Store loss history
        self.train_loss_history = []
        self.val_loss_history = []
