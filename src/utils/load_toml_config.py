import tomllib
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent

def load_toml_config(config_path: str) -> dict:
    with open(config_path, "rb") as f:
        nested_config = tomllib.load(f)
    
    for key, value in nested_config["paths"].items():
        nested_config["paths"][key] = str(PROJECT_ROOT / value)
    
    flattened_config = {}
    # Only handling one level of denesting for now
    for key, value in nested_config.items():
        if isinstance(value, dict):
            flattened_config.update(value)
        else:
            flattened_config[key] = value
    return flattened_config

if __name__ == "__main__":
    # Testing on the dynamics model config
    config_path = PROJECT_ROOT / "dynamics_model_training_config.toml"
    config = load_toml_config(config_path)
    print(config)
    