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

        def _initialize_weights(self):
        """Initialize weights and thresholds with random small values."""
        for l in range(1, self.L):
            # Xavier/He initialization
            if self.fact == 'relu':
                # He initialization for ReLU
                limit = np.sqrt(2.0 / self.n[l-1])
            else:
                # Xavier initialization for sigmoid/tanh
                limit = np.sqrt(6.0 / (self.n[l-1] + self.n[l]))

            # Initialize weights: w[l] has shape (n[l], n[l-1])
            self.w[l] = np.random.uniform(-limit, limit, (self.n[l], self.n[l-1]))

            # Initialize thresholds
            self.theta[l] = np.zeros(self.n[l])

            # Initialize change matrices for momentum
            self.d_w_prev[l] = np.zeros_like(self.w[l])
            self.d_theta_prev[l] = np.zeros_like(self.theta[l])

        def _activation_function(self, x: np.ndarray) -> np.ndarray:
        """Apply activation function."""
        if self.fact == 'sigmoid':
            return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))
        elif self.fact == 'relu':
            return np.maximum(0, x)
        elif self.fact == 'tanh':
            return np.tanh(x)
        elif self.fact == 'linear':
            return x
        else:
            raise ValueError(f"Unknown activation function: {self.fact}")
            
        def _activation_derivative(self, x: np.ndarray) -> np.ndarray:
        """Compute derivative of activation function."""
        if self.fact == 'sigmoid':
            sig = self._activation_function(x)
            return sig * (1 - sig)
        elif self.fact == 'relu':
            return (x > 0).astype(float)
        elif self.fact == 'tanh':
            return 1 - np.tanh(x) ** 2
        elif self.fact == 'linear':
            return np.ones_like(x)
        else:
            raise ValueError(f"Unknown activation function: {self.fact}")
            
    def _forward_propagation(self, X: np.ndarray) -> np.ndarray:
        """
        Perform forward propagation.
        
        Parameters:
        -----------
        X : np.ndarray
            Input data of shape (n_features,)
            
        Returns:
        --------
        np.ndarray
            Output of the network
        """
        # Set input layer activations
        self.xi[0] = X
        
        # Forward propagation through layers
        for l in range(1, self.L):
            # Calculate fields: h[l] = w[l] @ xi[l-1] - theta[l]
            self.h[l] = np.dot(self.w[l], self.xi[l-1]) - self.theta[l]
            
            # Apply activation function (linear for output layer in regression)
            if l == self.L - 1:  # Output layer for regression
                self.xi[l] = self.h[l]  # Linear activation for regression
            else:
                self.xi[l] = self._activation_function(self.h[l])
                
        return self.xi[self.L - 1]
    
    def _backward_propagation(self, y_true: float):
        """
        Perform backward propagation.
        
        Parameters:
        -----------
        y_true : float
            True target value
        """
        # Calculate error at output layer
        output = self.xi[self.L - 1]
        
        # For regression with linear output
        self.delta[self.L - 1] = np.array([output[0] - y_true])
        
        # Backpropagate errors
        for l in range(self.L - 2, 0, -1):
            # delta[l] = activation_derivative(h[l]) * (w[l+1].T @ delta[l+1])
            self.delta[l] = self._activation_derivative(self.h[l]) * np.dot(self.w[l+1].T, self.delta[l+1])
            
    def _update_weights(self):
        """Update weights and thresholds using gradient descent with momentum."""
        for l in range(1, self.L):
            # Calculate weight changes
            self.d_w[l] = -self.learning_rate * np.outer(self.delta[l], self.xi[l-1])
            self.d_theta[l] = self.learning_rate * self.delta[l]
            
            # Add momentum
            self.d_w[l] += self.momentum * self.d_w_prev[l]
            self.d_theta[l] += self.momentum * self.d_theta_prev[l]
            
            # Update weights and thresholds
            self.w[l] += self.d_w[l]
            self.theta[l] += self.d_theta[l]
            
            # Store for next iteration
            self.d_w_prev[l] = self.d_w[l].copy()
            self.d_theta_prev[l] = self.d_theta[l].copy()
    
