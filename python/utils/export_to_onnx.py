import torch
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))
from src.ppo import PolicyNet

def export_to_onnx(model, input_path, output_path, obs_dim):
    model.load_state_dict(torch.load(input_path))
    model.eval()

    # ONNX needs a dummy input to trace through the model
    dummy_input = torch.randn(1, obs_dim)

    print(f"Exporting model to {output_path}...")
    torch.onnx.export(
        model,
        (dummy_input,),
        output_path,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    print("Export complete.")

if __name__ == "__main__":
    # Exporting only the policy net for generating actions
    OBS_DIM = 6
    ACT_DIM = 2
    model = PolicyNet(OBS_DIM, ACT_DIM)
    export_to_onnx(model, "./models/policy_net.pth", "./models/policy_net.onnx", OBS_DIM)
