import os
import json
import torch
import numpy as np
from pathlib import Path


class Logger:
    """
    Simple logger for training metrics and parameters.
    
    Saves metrics as JSON files and provides methods for logging
    parameters, scalars, and histograms.
    """
    def __init__(self, log_dir):
        """
        Initialize the logger.
        
        Args:
            log_dir: Directory to save logs
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        self.scalar_dir = self.log_dir / 'scalars'
        self.scalar_dir.mkdir(exist_ok=True)
        
        self.histogram_dir = self.log_dir / 'histograms'
        self.histogram_dir.mkdir(exist_ok=True)
        
        self.params_file = self.log_dir / 'params.json'
        
        # Initialize metrics dictionaries
        self.scalar_metrics = {}
    
    def log_params(self, params):
        """
        Log parameters to a JSON file.
        
        Args:
            params: Dictionary of parameters
        """
        # Convert any non-serializable values to strings
        serializable_params = {}
        for k, v in params.items():
            if isinstance(v, (int, float, str, bool, list, dict)) or v is None:
                serializable_params[k] = v
            else:
                serializable_params[k] = str(v)
        
        with open(self.params_file, 'w') as f:
            json.dump(serializable_params, f, indent=4)
    
    def log_scalar(self, name, value, step):
        """
        Log a scalar value.
        
        Args:
            name: Metric name
            value: Scalar value
            step: Step (e.g., epoch) number
        """
        # Create the scalar directory if it doesn't exist
        os.makedirs(self.scalar_dir, exist_ok=True)
        
        # Create subdirectory if name contains a slash
        if '/' in name:
            subdir = os.path.dirname(name)
            os.makedirs(self.scalar_dir / subdir, exist_ok=True)
        
        if name not in self.scalar_metrics:
            self.scalar_metrics[name] = []
        
        self.scalar_metrics[name].append((step, value))
        
        # Save to file
        with open(self.scalar_dir / f'{name}.json', 'w') as f:
            json.dump(self.scalar_metrics[name], f)
    
    def log_histogram(self, values, step, name):
        """
        Log a histogram of values.
        
        Args:
            values: Array of values
            step: Step (e.g., epoch) number
            name: Histogram name
        """
        if isinstance(values, torch.Tensor):
            values = values.cpu().numpy()
        
        # Compute histogram bins
        hist, bin_edges = np.histogram(values, bins=50)
        
        # Save histogram data
        histogram_data = {
            'step': step,
            'counts': hist.tolist(),
            'bin_edges': bin_edges.tolist()
        }
        
        # Append to existing file or create new file
        histogram_file = self.histogram_dir / f'{name}.json'
        
        if histogram_file.exists():
            with open(histogram_file, 'r') as f:
                all_data = json.load(f)
            all_data.append(histogram_data)
        else:
            all_data = [histogram_data]
        
        with open(histogram_file, 'w') as f:
            json.dump(all_data, f)
    
    def log_images(self, images, step, name):
        """
        Log images (not implemented, placeholder for future).
        
        Args:
            images: Images to log
            step: Step number
            name: Image name
        """
        # This could be implemented to save images if needed
        pass
    
    def get_scalar(self, name):
        """
        Get scalar values for a metric.
        
        Args:
            name: Metric name
            
        Returns:
            List of (step, value) tuples
        """
        return self.scalar_metrics.get(name, []) 