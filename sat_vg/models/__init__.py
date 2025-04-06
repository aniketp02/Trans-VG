from .sat_vg import SatVG

def build_model(args):
    """
    Build the SatVG model based on the provided arguments.
    
    Args:
        args: Arguments for model configuration
    
    Returns:
        SatVG model instance
    """
    return SatVG(args) 