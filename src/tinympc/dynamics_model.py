import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter


class RocketDynamicsDataset(Dataset):
    def __init__(self, X: torch.Tensor, y: torch.Tensor):
        self.X = X
        self.y = y

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


class DynamicsModel(nn.Module):
    def __init__(self, input_dim: int = 8, output_dim: int = 6, hidden_dims: List[int] = [64, 64, 64]):
        super(DynamicsModel, self).__init__()
        layers = []
        prev_dim = input_dim

        # Simple relu feed forward
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, output_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def load_data(data_path: str, test_size: float = 0.2, val_size: float = 0.1,
             batch_size: int = 64, random_state: int = 1) -> Tuple[DataLoader, DataLoader, DataLoader]:
    data = pd.read_csv(data_path)

    # Input features ([S_t, A_t])
    input_cols = [
        'current_pos_x', 'current_pos_y', 'current_angle',
        'current_vel_x', 'current_vel_y', 'current_angular_vel',
        'left_thruster', 'right_thruster'
    ]

    # Output features ([S_(t+1)])
    output_cols = [
        'resulting_pos_x', 'resulting_pos_y', 'resulting_angle',
        'resulting_vel_x', 'resulting_vel_y', 'resulting_angular_vel'
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
    X_train = torch.FloatTensor(X_train)
    y_train = torch.FloatTensor(y_train)
    X_val = torch.FloatTensor(X_val)
    y_val = torch.FloatTensor(y_val)
    X_test = torch.FloatTensor(X_test)
    y_test = torch.FloatTensor(y_test)

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
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    loss_fn = nn.MSELoss()
    test_loss = 0.0

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            test_loss += loss_fn(outputs, y_batch).item() * X_batch.size(0)

    test_loss /= len(test_loader.dataset) # type: ignore[arg]

    return {
        'mse': test_loss,
        'rmse': np.sqrt(test_loss)
    }


def main():
    # Project root directory (one level up from src/tinympc)
    PROJECT_ROOT = Path(__file__).parent.parent.parent

    # Hyperparameters + layer architecture
    config = {
        'data_path': str(PROJECT_ROOT / 'data' / 'physics_data.csv'),
        'model_save_path': str(PROJECT_ROOT / 'models' / 'dynamics_model.pth'),
        'log_dir': str(PROJECT_ROOT / 'runs'),
        'batch_size': 128,
        'hidden_dims': [256, 256, 256],
        'learning_rate': 1e-3,
        'num_epochs': 100,
        'patience': 15,
        'test_size': 0.15,
        'val_size': 0.15,
        'random_state': 1
    }

    (PROJECT_ROOT / 'models').mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    train_loader, val_loader, test_loader = load_data(
        data_path=config['data_path'],
        test_size=config['test_size'],
        val_size=config['val_size'],
        batch_size=config['batch_size'],
        random_state=config['random_state']
    )

    print("Initializing model...")
    model = DynamicsModel(
        input_dim=8,  # 6 state + 2 action dimensions
        output_dim=6,  # 6 state dimensions
        hidden_dims=config['hidden_dims']
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
    print(f"Test MSE: {metrics['mse']:.6f}")
    print(f"Test RMSE: {metrics['rmse']:.6f}")

    # Save config
    with open('training_config.json', 'w') as f:
        json.dump(config, f, indent=4)

    print("\nTraining completed successfully!")
    print(f"Best validation loss: {training_result['best_val_loss']:.6f}")
    print(f"Model saved to {config['model_save_path']}")

if __name__ == "__main__":
    main()
