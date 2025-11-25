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


#######################################################################
#Model Comparison
#######################################################################
print("\n" + "="*80)
print("PART 3.2: MODEL COMPARISON")
print("-"*40)

# 1. Best Custom Neural Network 
print("\n1. Custom Neural Network (BP):")
best_bp_mse, best_bp_mae, best_bp_mape = calculate_metrics(y_test, best_pred, scaler_y)
print(f"   MSE={best_bp_mse:.4f}, MAE={best_bp_mae:.4f}, MAPE={best_bp_mape:.2f}%")

# 2. Linear Regression
print("\n2. Linear Regression (MLR-F):")
mlr = LinearRegression()
mlr.fit(X_train_val, y_train_val)
y_pred_mlr = mlr.predict(X_test)
mlr_mse, mlr_mae, mlr_mape = calculate_metrics(y_test, y_pred_mlr, scaler_y)
print(f"   MSE={mlr_mse:.4f}, MAE={mlr_mae:.4f}, MAPE={mlr_mape:.2f}%")

# 3. PyTorch Neural Network 
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim

    class SimpleNN(nn.Module):
        def __init__(self, input_size):
            super(SimpleNN, self).__init__()
            self.fc1 = nn.Linear(input_size, 50)
            self.fc2 = nn.Linear(50, 25)
            self.fc3 = nn.Linear(25, 1)
            self.activation = nn.Tanh()

        def forward(self, x):
            x = self.activation(self.fc1(x))
            x = self.activation(self.fc2(x))
            x = self.fc3(x)
            return x

    print("\n3. PyTorch Neural Network (BP-F):")

    # Prepare data
    X_train_torch = torch.FloatTensor(X_train_val)
    y_train_torch = torch.FloatTensor(y_train_val.reshape(-1, 1))
    X_test_torch = torch.FloatTensor(X_test)

    # Train model
    pytorch_nn = SimpleNN(X_train_val.shape[1])
    criterion = nn.MSELoss()
    optimizer = optim.Adam(pytorch_nn.parameters(), lr=0.001)

    print("   Training PyTorch model...")
    pytorch_nn.train()
    for epoch in range(150):
        optimizer.zero_grad()
        outputs = pytorch_nn(X_train_torch)
        loss = criterion(outputs, y_train_torch)
        loss.backward()
        optimizer.step()
        if epoch % 50 == 0:
            print(f"   Epoch {epoch}, Loss: {loss.item():.6f}")

    # Predict
    pytorch_nn.eval()
    with torch.no_grad():
        y_pred_pytorch = pytorch_nn(X_test_torch).numpy().ravel()

    pytorch_mse, pytorch_mae, pytorch_mape = calculate_metrics(y_test, y_pred_pytorch, scaler_y)
    print(f"   MSE={pytorch_mse:.4f}, MAE={pytorch_mae:.4f}, MAPE={pytorch_mape:.2f}%")

    has_pytorch = True
except ImportError:
    print("\n3. PyTorch not installed - skipping BP-F comparison")
    has_pytorch = False

# Final comparison table
print("\n" + "="*80)
print("FINAL MODEL COMPARISON")
print("="*80)

if has_pytorch:
    comparison = pd.DataFrame({
        'Model': ['BP (Custom)', 'BP-F (PyTorch)', 'MLR-F (Sklearn)'],
        'MSE': [best_bp_mse, pytorch_mse, mlr_mse],
        'MAE': [best_bp_mae, pytorch_mae, mlr_mae],
        'MAPE (%)': [best_bp_mape, pytorch_mape, mlr_mape]
    })
else:
    comparison = pd.DataFrame({
        'Model': ['BP (Custom)', 'MLR-F (Sklearn)'],
        'MSE': [best_bp_mse, mlr_mse],
        'MAE': [best_bp_mae, mlr_mae],
        'MAPE (%)': [best_bp_mape, mlr_mape]
    })

print(comparison.to_string(index=False))

# Create simple scatter plots
print("\nGenerating scatter plots...")

plt.figure(figsize=(12, 4))

# Plot 1: Best Neural Network
plt.subplot(1, 3, 1)
y_true_orig = scaler_y.inverse_transform(y_test.reshape(-1, 1)).ravel()
y_pred_orig = scaler_y.inverse_transform(best_pred.reshape(-1, 1)).ravel()
plt.scatter(y_true_orig, y_pred_orig, alpha=0.5, s=10)
plt.plot([y_true_orig.min(), y_true_orig.max()], [y_true_orig.min(), y_true_orig.max()], 'r--')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title(f'BP (Custom)\nMSE={best_bp_mse:.3f}')
plt.grid(True, alpha=0.3)

# Plot 2: Linear Regression
plt.subplot(1, 3, 2)
y_pred_mlr_orig = scaler_y.inverse_transform(y_pred_mlr.reshape(-1, 1)).ravel()
plt.scatter(y_true_orig, y_pred_mlr_orig, alpha=0.5, s=10)
plt.plot([y_true_orig.min(), y_true_orig.max()], [y_true_orig.min(), y_true_orig.max()], 'r--')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title(f'MLR-F\nMSE={mlr_mse:.3f}')
plt.grid(True, alpha=0.3)

# Plot 3: PyTorch
if has_pytorch:
    plt.subplot(1, 3, 3)
    y_pred_pytorch_orig = scaler_y.inverse_transform(y_pred_pytorch.reshape(-1, 1)).ravel()
    plt.scatter(y_true_orig, y_pred_pytorch_orig, alpha=0.5, s=10)
    plt.plot([y_true_orig.min(), y_true_orig.max()], [y_true_orig.min(), y_true_orig.max()], 'r--')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title(f'BP-F (PyTorch)\nMSE={pytorch_mse:.3f}')
    plt.grid(True, alpha=0.3)

plt.suptitle('Model Predictions vs Actual Values', y=1.05)
plt.tight_layout()
plt.savefig('model_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

# Plot loss evolution for best model
print("\nGenerating loss evolution plot...")
loss_history = best_model.loss_epochs()
plt.figure(figsize=(8, 5))
plt.plot(loss_history[:, 0], label='Training Loss', linewidth=2)
plt.plot(loss_history[:, 1], label='Validation Loss', linewidth=2)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Best Model - Loss Evolution')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('loss_evolution.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n" + "="*80)
print("ACTIVITY COMPLETED!")
print("="*80)
print("\nGenerated files:")
print("- model_comparison.png: Scatter plots comparing all models")
print("- loss_evolution.png: Training/validation loss for best model")
print("\nKey Results:")
print(f"- Best Custom NN: MSE={best_bp_mse:.4f}")
print(f"- Linear Regression: MSE={mlr_mse:.4f}")
if has_pytorch:
    print(f"- PyTorch NN: MSE={pytorch_mse:.4f}")

