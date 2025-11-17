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
