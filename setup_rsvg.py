#!/usr/bin/env python3
import os
import shutil
from pathlib import Path
import torch

def setup_rsvg_dataset():
    """
    Prepare the RSVG dataset structure for TransVG
    """
    # Ensure the necessary directories exist
    rsvg_dir = Path("./rsvg")
    processed_dir = rsvg_dir / "processed"
    os.makedirs(processed_dir, exist_ok=True)
    
    # Create the dataset directory structure expected by TransVG
    rsvg_data_dir = Path("./rsvg/rsvg")  # This is the directory TransVG will look for
    os.makedirs(rsvg_data_dir, exist_ok=True)
    os.makedirs(rsvg_dir / "images", exist_ok=True)
    
    # Process the dataset using our processing script
    print("Processing RSVG dataset...")
    from datasets.process_rsvg import process_rsvg_dataset
    process_rsvg_dataset(str(rsvg_dir), str(processed_dir))
    
    # Copy the processed files to the expected locations
    for split in ['train', 'val', 'test']:
        src_file = processed_dir / f"rsvg_{split}.pth"
        if src_file.exists():
            # Copy to root folder
            root_dst_file = rsvg_dir / f"rsvg_{split}.pth"
            shutil.copy(src_file, root_dst_file)
            print(f"Copied {src_file} to {root_dst_file}")
            
            # Copy to the expected data directory structure
            data_dst_file = rsvg_data_dir / f"rsvg_{split}.pth"
            shutil.copy(src_file, data_dst_file)
            print(f"Copied {src_file} to {data_dst_file}")
            
            # Verify file contents
            data = torch.load(str(root_dst_file))
            print(f"File {root_dst_file} contains {len(data)} examples")
    
    # Create a simple .exists file to mark the dataset as ready
    with open(rsvg_data_dir / ".exists", "w") as f:
        f.write("RSVG dataset exists")
    
    # Create a datasets.txt file to list the available datasets
    with open(rsvg_dir / "datasets.txt", "w") as f:
        f.write("rsvg\n")
    
    print("RSVG dataset setup complete")

if __name__ == "__main__":
    setup_rsvg_dataset() 