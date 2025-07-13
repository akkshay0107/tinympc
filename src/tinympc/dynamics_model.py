import sys
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

# Add project root to PATH
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.load_toml_config import load_toml_config

INPUT_DIMS = 8 # 6 state variables + 2 controls
OUTPUT_DIMS = 6 # 6 state variables

class RocketDynamicsDataset(Dataset):
    def __init__(self, X: torch.Tensor, y: torch.Tensor):
        self.X = X
        self.y = y

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


class DynamicsModel(nn.Module):
    def __init__(self, input_dim: int = 8, output_dim: int = 6, hidden_dims: List[int] = [64, 64, 64],
                 dropout_rate: float = 0.1):
        super(DynamicsModel, self).__init__()
        layers = []
        prev_dim = input_dim

        # Simple relu feed forward
        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
            if dropout_rate > 0 and i < len(hidden_dims) - 1:
                layers.append(nn.Dropout(p=dropout_rate))

        layers.append(nn.Linear(prev_dim, output_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def load_data(data_path: str, test_size: float = 0.2, val_size: float = 0.1,
             batch_size: int = 64, random_state: int = 1,
             normalize: bool = False) -> Tuple[DataLoader, DataLoader, DataLoader]:
    data = pd.read_csv(data_path)

    # Input features ([S_t, A_t])
    input_cols = [
        'current_pos_x', 'current_pos_y', 'current_angle',
        'current_vel_x', 'current_vel_y', 'current_angular_vel',
        'left_thruster', 'right_thruster'
    ]

    # Output features ([S_(t+1) - S_t])
    output_cols = [
        'delta_pos_x', 'delta_pos_y', 'delta_angle',
        'delta_vel_x', 'delta_vel_y', 'delta_angular_vel'
    ]

    # Train test validation split
    X = data[input_cols].values
    y = data[output_cols].values

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    val_size_adjusted = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_size_adjusted, random_state=random_state
    )

    # Converting to the convention for Pytorch
    y_train = torch.FloatTensor(y_train)
    y_val = torch.FloatTensor(y_val)
    y_test = torch.FloatTensor(y_test)
    if normalize:
        scaler = StandardScaler()
        X_train = torch.FloatTensor(scaler.fit_transform(X_train))
        X_val = torch.FloatTensor(scaler.transform(X_val))
        X_test = torch.FloatTensor(scaler.transform(X_test))
    else:
        X_train = torch.FloatTensor(X_train)
        X_val = torch.FloatTensor(X_val)
        X_test = torch.FloatTensor(X_test)

    train_dataset = RocketDynamicsDataset(X_train, y_train)
    val_dataset = RocketDynamicsDataset(X_val, y_val)
    test_dataset = RocketDynamicsDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


def train_model(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
               num_epochs: int = 100, learning_rate: float = 1e-3,
               patience: int = 10, model_save_path: Optional[str] = None,
               log_dir: Optional[str] = None) -> Dict:
    # Only pytorch-cpu installed in project
    device = torch.device('cpu')
    model = model.to(device)

    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    # TensorBoard setup
    writer = None
    if log_dir:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_dir = os.path.join(log_dir, f'run_{timestamp}')
        writer = SummaryWriter(log_dir=log_dir)
        print(f'TensorBoard logs saved to: {log_dir}')

    # Training variables
    best_val_loss = float('inf')
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': []}
    best_model_state = model.state_dict()

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            outputs = model(X_batch)
            loss = loss_fn(outputs, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * X_batch.size(0)

        # Forcibly suppressing pyright type checker since
        # RocketDynamicsDataset implements __len__
        train_loss /= len(train_loader.dataset) # type: ignore[arg]

        val_loss = 0.0
        model.eval()
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                val_loss += loss_fn(outputs, y_batch).item() * X_batch.size(0)

        val_loss /= len(val_loader.dataset) # type: ignore[arg]

        scheduler.step(val_loss)

        # Save history + output progress
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        print(f'Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')

        # Log to TensorBoard
        if writer:
            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('Loss/validation', val_loss, epoch)
            writer.add_scalar('Learning Rate', optimizer.param_groups[0]['lr'], epoch)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict()

            if model_save_path:
                torch.save({
                    'model_state_dict': best_model_state,
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'epoch': epoch
                }, model_save_path)
                print(f'Model saved to {model_save_path}')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'Early stopping after {patience} epochs')
                if writer:
                    writer.add_text('Training', f'Early stopping after {patience} epochs')
                break

    if writer:
        writer.flush()
        writer.close()
        print(f'Training metrics saved, run tensorboard --logdir={log_dir}')

    return {
        'history': history,
        'best_model_state': best_model_state,
        'best_val_loss': best_val_loss,
        'log_dir': log_dir if log_dir else None,
    }

def evaluate_model(model: nn.Module, test_loader: DataLoader) -> Dict[str, float]:
    device = torch.device('cpu')
    model = model.to(device)
    model.eval()
    loss_fn = nn.MSELoss()
    test_loss = 0.0
    all_predictions = []
    all_targets = []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            test_loss += loss_fn(outputs, y_batch).item() * X_batch.size(0)
            all_predictions.append(outputs)
            all_targets.append(y_batch)
    all_predictions = torch.cat(all_predictions, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    test_loss /= len(test_loader.dataset) # type: ignore[arg]
    rmse = torch.sqrt(torch.tensor(test_loss))

    # sMAPE
    epsilon = 1e-8 # avoid division by zero
    smape = torch.mean(torch.abs(all_targets - all_predictions) /
                      (torch.abs(all_targets) + torch.abs(all_predictions) + epsilon)) * 200

    # R^2
    ss_res = torch.sum((all_targets - all_predictions) ** 2)
    ss_tot = torch.sum((all_targets - torch.mean(all_targets)) ** 2)
    r2 = 1 - (ss_res / ss_tot)

    # Calculate per-variable metrics
    per_variable_metrics = {}
    variable_names = ['delta_pos_x', 'delta_pos_y', 'delta_angle', 'delta_vel_x', 'delta_vel_y', 'delta_angular_vel']
    for i, var_name in enumerate(variable_names):
        # Per-variable MSE
        var_mse = torch.mean((all_targets[:, i] - all_predictions[:, i]) ** 2)

        # Per-variable sMAPE
        var_smape = torch.mean(torch.abs(all_targets[:, i] - all_predictions[:, i]) /
                              (torch.abs(all_targets[:, i]) + torch.abs(all_predictions[:, i]) + epsilon)) * 200

        # Per-variable R^2
        var_ss_res = torch.sum((all_targets[:, i] - all_predictions[:, i]) ** 2)
        var_ss_tot = torch.sum((all_targets[:, i] - torch.mean(all_targets[:, i])) ** 2)
        var_r2 = 1 - (var_ss_res / var_ss_tot)

        per_variable_metrics[f'{var_name}_mse'] = var_mse.item()
        per_variable_metrics[f'{var_name}_smape'] = var_smape.item()
        per_variable_metrics[f'{var_name}_r2'] = var_r2.item()

    # mean absolute error
    mae = torch.mean(torch.abs(all_targets - all_predictions))
    # Maximum absolute error
    max_error = torch.max(torch.abs(all_targets - all_predictions))
    # Median absolute error (more robust to outliers)
    median_ae = torch.median(torch.abs(all_targets - all_predictions))

    # compiling all metrics
    metrics = {
        'mse': test_loss,
        'rmse': rmse.item(),
        'mae': mae.item(),
        'smape': smape.item(),
        'r2': r2.item(),
        'max_error': max_error.item(),
        'median_ae': median_ae.item(),
        **per_variable_metrics
    }
    return metrics

def print_evaluation_results(metrics: Dict[str, float]):
    # Overall metrics
    print("Test Metrics:")
    print(f"  MSE:           {metrics['mse']:.6f}")
    print(f"  RMSE:          {metrics['rmse']:.6f}")
    print(f"  MAE:           {metrics['mae']:.6f}")
    print(f"  sMAPE:         {metrics['smape']:.2f}%")
    print(f"  R^2:           {metrics['r2']:.4f}")
    print(f"  Max Error:     {metrics['max_error']:.6f}")
    print(f"  Median AE:     {metrics['median_ae']:.6f}")

    # Per-variable metrics
    variable_names = ['delta_pos_x', 'delta_pos_y', 'delta_angle', 'delta_vel_x', 'delta_vel_y', 'delta_angular_vel']
    has_per_var_metrics = any(f'{var}_mse' in metrics for var in variable_names)
    if has_per_var_metrics:
        print("\nPer-Variable Metrics:")
        print(f"{'Variable':<15} {'MSE':<10} {'sMAPE':<8} {'R^2':<8}")
        print("-" * 45)
        for var in variable_names:
            mse_key = f'{var}_mse'
            smape_key = f'{var}_smape'
            r2_key = f'{var}_r2'
            if mse_key in metrics:
                print(f"{var:<15} {metrics[mse_key]:<10.6f} {metrics[smape_key]:<8.2f} {metrics[r2_key]:<8.4f}")


def main():
    config = load_toml_config(str(PROJECT_ROOT / 'dynamics_model_training_config.toml'))
    (PROJECT_ROOT / 'models').mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    train_loader, val_loader, test_loader = load_data(
        data_path=config['data_path'],
        test_size=config['test_size'],
        val_size=config['val_size'],
        batch_size=config['batch_size'],
        random_state=config['random_state'],
        normalize=config['normalize']
    )

    print("Initializing model...")
    # TODO: move input and output dims to constants
    model = DynamicsModel(
        input_dim=INPUT_DIMS,
        output_dim=OUTPUT_DIMS,
        hidden_dims=config['hidden_dims'],
        dropout_rate=config['dropout_rate']
    )

    print("Training model...")
    training_result = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=config['num_epochs'],
        learning_rate=config['learning_rate'],
        patience=config['patience'],
        model_save_path=config['model_save_path'],
        log_dir=config['log_dir']
    )

    # Load best model and evaluate on test set
    checkpoint = torch.load(config['model_save_path'])
    model.load_state_dict(checkpoint['model_state_dict'])
    print("\nEvaluating on test set...")
    metrics = evaluate_model(model, test_loader)
    print_evaluation_results(metrics)

    print("\nTraining completed successfully!")
    print(f"Best validation loss: {training_result['best_val_loss']:.6f}")
    print(f"Model saved to {config['model_save_path']}")

if __name__ == "__main__":
    main()
