import sys
import os
from pathlib import Path

# Add project root to PATH
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from src.tinympc.dynamics_model import DynamicsModel, load_data
from src.utils.load_toml_config import load_toml_config

def test_dynamics_model_inference():
    data_path = PROJECT_ROOT / 'data' / 'physics_data.csv'

    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found at {data_path}")

    _, _, test_loader = load_data(
        data_path=str(data_path),
        test_size=0.15,
        val_size=0.15,
        batch_size=1,
        random_state=1
    )

    training_config_path = str(PROJECT_ROOT / "dynamics_model_training_config.toml")
    config = load_toml_config(training_config_path)
    
    model = DynamicsModel(
        input_dim=8,
        output_dim=6,
        hidden_dims=config["hidden_dims"],
        dropout_rate=config["dropout_rate"]
    )
    
    # Load pre-trained weights
    model_path = config["model_save_path"]
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model file not found at {model_path}")

    checkpoint = torch.load(str(model_path), map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded pre-trained model from {model_path}")
    print(f"Model  trained for {checkpoint.get('epoch', 'unknown')} epochs")
    print(f"Training loss: {checkpoint.get('train_loss', 'unknown'):.6f}")
    print(f"Validation loss: {checkpoint.get('val_loss', 'unknown'):.6f}")

    model.eval()
    test_sample, target = next(iter(test_loader))
    with torch.no_grad():
        prediction = model(test_sample)

    # checking for correct shape
    # and no invalid values (inf / nan)
    assert prediction.shape == (1, 6), \
        f"Expected prediction shape (1, 6), got {prediction.shape}"

    assert torch.all(torch.isfinite(prediction)), "Prediction contains NaN or infinite values"

    print("\nDynamics Model Inference Test")
    print("-" * 50)
    print(f"Input shape: {test_sample.shape}")
    print(f"Output shape: {prediction.shape}")
    print("\nInput sample:", test_sample.numpy().flatten())
    print("Predicted output:", prediction.numpy().flatten())
    print("Expected output:", target.numpy().flatten())
    print("-" * 50)

if __name__ == "__main__":
    test_dynamics_model_inference()
