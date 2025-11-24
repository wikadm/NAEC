"""
Neural and Evolutionary Computation - Activity 1
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import warnings
warnings.filterwarnings('ignore')

from nn import NeuralNet

np.random.seed(42)
print("="*80)
print("NEURAL AND EVOLUTIONARY COMPUTATION - ACTIVITY 1")
print("="*80)

##############################################################################
##### PART 1: DATA PREPROCESSING
##############################################################################
print("\nPART 1: DATA PREPROCESSING")
print("-"*40)

# Load dataset
print("Loading California Housing Dataset...")
housing = fetch_california_housing()
X_raw = pd.DataFrame(housing.data, columns=housing.feature_names)
y_raw = housing.target

# Add categorical features
X_raw['HouseAge_Category'] = pd.cut(X_raw['HouseAge'], bins=5, labels=False)
X_raw['Income_Category'] = pd.qcut(X_raw['MedInc'], q=4, labels=False)

print(f"Dataset: {X_raw.shape[0]} samples, {X_raw.shape[1]} features")
print(f"Features: {list(X_raw.columns)}")

# Normalize data
print("\nNormalizing data...")
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_normalized = scaler_X.fit_transform(X_raw)
y_normalized = scaler_y.fit_transform(y_raw.reshape(-1, 1)).ravel()

# Split data
print("Splitting data (80% train, 20% test)...")
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X_normalized, y_normalized, test_size=0.2, random_state=42, shuffle=True
)

print(f"Training set: {X_train_val.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")


#########################################################################
######PART 2: DHYPERPARAMETER COMPARISON
#########################################################################

def calculate_metrics(y_true, y_pred, scaler_y):
    y_true_orig = scaler_y.inverse_transform(y_true.reshape(-1, 1)).ravel()
    y_pred_orig = scaler_y.inverse_transform(y_pred.reshape(-1, 1)).ravel()

    mse = mean_squared_error(y_true_orig, y_pred_orig)
    mae = mean_absolute_error(y_true_orig, y_pred_orig)
    mape = np.mean(np.abs((y_true_orig - y_pred_orig) / (y_true_orig + 1e-10))) * 100

    return mse, mae, mape

print("\nTesting 10 hyperparameter configurations...")
print("(This will take a few minutes)\n")

configurations = [
    # layers, epochs, learning_rate, momentum, activation
    [[10, 20, 1], 100, 0.01, 0.9, 'sigmoid'],
    [[10, 30, 1], 100, 0.01, 0.9, 'sigmoid'],
    [[10, 20, 10, 1], 100, 0.01, 0.9, 'sigmoid'],
    [[10, 50, 1], 100, 0.005, 0.9, 'tanh'],
    [[10, 30, 1], 150, 0.01, 0.8, 'tanh'],
    [[10, 25, 1], 100, 0.02, 0.85, 'relu'],
    [[10, 50, 25, 1], 150, 0.001, 0.95, 'relu'],
    [[10, 100, 50, 1], 150, 0.001, 0.9, 'sigmoid'],
    [[10, 64, 32, 1], 200, 0.005, 0.9, 'tanh'],
    [[10, 40, 20, 1], 150, 0.005, 0.9, 'tanh'],
]
results = []
best_mse = float('inf')
best_model = None
best_pred = None

for i, (layers, epochs, lr, momentum, activation) in enumerate(configurations):
    print(f"Config {i+1}/10: Layers={layers}, Epochs={epochs}, LR={lr}, Act={activation}")


# Train model
nn = NeuralNet(
        layers=layers,
        epochs=epochs,
        learning_rate=lr,
        momentum=momentum,
        activation=activation,
        validation_split=0.2
    )
    
    nn.fit(X_train_val, y_train_val)
    y_pred = nn.predict(X_test)
    
    # Calculate metrics
    mse, mae, mape = calculate_metrics(y_test, y_pred, scaler_y)
    
    results.append({
        'Config': i+1,
        'Layers': '-'.join(map(str, layers)),
        'Epochs': epochs,
        'LR': lr,
        'Momentum': momentum,
        'Activation': activation,
        'MSE': mse,
        'MAE': mae,
        'MAPE': mape
    })
    
    print(f"  MSE={mse:.4f}, MAE={mae:.4f}, MAPE={mape:.2f}%")
    
    # Keep best model
    if mse < best_mse:
        best_mse = mse
        best_model = nn
        best_pred = y_pred
        best_config = i+1

# Display results table
print("\n" + "="*80)
print("HYPERPARAMETER COMPARISON TABLE")
print("="*80)
results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))
print(f"\nBest Configuration: #{best_config} with MSE={best_mse:.4f}")



