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
            
            # Clip to prevent overflow
            self.h[l] = np.clip(self.h[l], -500, 500)
            
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
        
        # Clip output delta to prevent exploding gradients
        self.delta[self.L - 1] = np.clip(self.delta[self.L - 1], -10, 10)
        
        # Backpropagate errors
        for l in range(self.L - 2, 0, -1):
            # delta[l] = activation_derivative(h[l]) * (w[l+1].T @ delta[l+1])
            self.delta[l] = self._activation_derivative(self.h[l]) * np.dot(self.w[l+1].T, self.delta[l+1])
            
            # Clip deltas to prevent exploding gradients
            self.delta[l] = np.clip(self.delta[l], -10, 10)
            
    def _update_weights(self):
        """Update weights and thresholds using gradient descent with momentum."""
        for l in range(1, self.L):
            # Calculate weight changes
            self.d_w[l] = -self.learning_rate * np.outer(self.delta[l], self.xi[l-1])
            self.d_theta[l] = self.learning_rate * self.delta[l]
            
            # Add momentum
            self.d_w[l] += self.momentum * self.d_w_prev[l]
            self.d_theta[l] += self.momentum * self.d_theta_prev[l]
            
            # Gradient clipping to prevent exploding gradients
            max_grad = 5.0
            self.d_w[l] = np.clip(self.d_w[l], -max_grad, max_grad)
            self.d_theta[l] = np.clip(self.d_theta[l], -max_grad, max_grad)
            
            # Update weights and thresholds
            self.w[l] += self.d_w[l]
            self.theta[l] += self.d_theta[l]
            
            # Check for NaN or Inf
            if np.any(np.isnan(self.w[l])) or np.any(np.isinf(self.w[l])):
                # Reset to small random values
                limit = 0.01
                self.w[l] = np.random.uniform(-limit, limit, self.w[l].shape)
                self.theta[l] = np.zeros(self.n[l])
            
            # Store for next iteration
            self.d_w_prev[l] = self.d_w[l].copy()
            self.d_theta_prev[l] = self.d_theta[l].copy()
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Train the neural network.
        
        Parameters:
        -----------
        X : np.ndarray
            Training data of shape (n_samples, n_features)
        y : np.ndarray
            Target values of shape (n_samples,)
        """
        # Split data into training and validation if needed
        n_samples = X.shape[0]
        
        if self.validation_split > 0:
            n_val = int(n_samples * self.validation_split)
            indices = np.random.permutation(n_samples)
            
            val_idx = indices[:n_val]
            train_idx = indices[n_val:]
            
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
        else:
            X_train, y_train = X, y
            X_val, y_val = None, None
            
        # Initialize arrays for changes
        for l in range(1, self.L):
            self.d_w[l] = np.zeros_like(self.w[l])
            self.d_theta[l] = np.zeros_like(self.theta[l])
            
        # Training loop
        for epoch in range(self.epochs):
            # Shuffle training data
            indices = np.random.permutation(len(X_train))
            X_train_shuffled = X_train[indices]
            y_train_shuffled = y_train[indices]
            
            # Train on each sample
            train_errors = []
            nan_detected = False
            for i in range(len(X_train)):
                # Forward propagation
                output = self._forward_propagation(X_train_shuffled[i])
                
                # Check for NaN
                if np.isnan(output[0]) or np.isinf(output[0]):
                    nan_detected = True
                    break
                
                # Backward propagation
                self._backward_propagation(y_train_shuffled[i])
                
                # Update weights
                self._update_weights()
                
                # Store error
                train_errors.append((output[0] - y_train_shuffled[i]) ** 2)
            
            # If NaN detected, stop training
            if nan_detected:
                print(f"  Warning: NaN detected at epoch {epoch}. Stopping training early.")
                # Fill remaining epochs with current loss
                remaining = self.epochs - epoch
                for _ in range(remaining):
                    self.train_loss_history.append(self.train_loss_history[-1] if self.train_loss_history else float('inf'))
                    self.val_loss_history.append(self.val_loss_history[-1] if self.val_loss_history else float('inf'))
                break
                
            # Calculate epoch losses
            train_loss = np.mean(train_errors) if train_errors else float('inf')
            self.train_loss_history.append(train_loss)
            
            # Validation loss
            if X_val is not None:
                val_predictions = self.predict(X_val)
                val_loss = np.mean((val_predictions - y_val) ** 2)
                self.val_loss_history.append(val_loss)
            else:
                self.val_loss_history.append(0)
                
            # Print progress every 10 epochs
            if epoch % 10 == 0:
                if X_val is not None:
                    print(f"Epoch {epoch}/{self.epochs} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
                else:
                    print(f"Epoch {epoch}/{self.epochs} - Train Loss: {train_loss:.6f}")
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions.
        
        Parameters:
        -----------
        X : np.ndarray
            Input data of shape (n_samples, n_features)
            
        Returns:
        --------
        np.ndarray
            Predictions of shape (n_samples,)
        """
        predictions = []
        for i in range(X.shape[0]):
            output = self._forward_propagation(X[i])
            # Handle NaN predictions
            if np.isnan(output[0]) or np.isinf(output[0]):
                predictions.append(0.0)
            else:
                predictions.append(output[0])
        return np.array(predictions)
    
    def loss_epochs(self) -> np.ndarray:
        """
        Return the loss history.
        
        Returns:
        --------
        np.ndarray
            Array of shape (n_epochs, 2) containing training and validation losses
        """
        return np.column_stack((self.train_loss_history, self.val_loss_history))
