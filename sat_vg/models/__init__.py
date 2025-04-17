from .sat_vg import SatVG
import torch

def build_model(args):
    """
    Build the SatVG model based on the provided arguments.
    
    Args:
        args: Arguments for model configuration
    
    Returns:
        SatVG model instance
    """
    model = SatVG(args)
    
    if args.device.startswith('cuda'):
        # Get the specific CUDA device
        device = torch.device(args.device)
        print(f"Moving model to {device}")
        model.to_cuda(device)
    
    return model 