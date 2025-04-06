#!/usr/bin/env python3
import os
import torch
import random
from pathlib import Path

def create_synthetic_rsvg_data():
    """
    Create synthetic RSVG data for testing purposes
    """
    rsvg_dir = Path("./rsvg")
    processed_dir = rsvg_dir / "processed"
    data_dir = rsvg_dir / "rsvg"
    
    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)
    
    # Create synthetic train, val, test datasets
    train_data = create_dataset(5500, "train")
    val_data = create_dataset(1200, "val")
    test_data = create_dataset(1200, "test")
    
    # Save datasets to all possible locations
    for dataset, name in [(train_data, "train"), (val_data, "val"), (test_data, "test")]:
        filename = f"rsvg_{name}.pth"
        
        # Save to root rsvg directory
        torch.save(dataset, rsvg_dir / filename)
        print(f"Saved {len(dataset)} items to {rsvg_dir / filename}")
        
        # Save to processed directory
        torch.save(dataset, processed_dir / filename)
        print(f"Saved {len(dataset)} items to {processed_dir / filename}")
        
        # Save to rsvg/rsvg directory
        torch.save(dataset, data_dir / filename)
        print(f"Saved {len(dataset)} items to {data_dir / filename}")
    
    # Create a marker file
    with open(data_dir / ".exists", "w") as f:
        f.write("RSVG dataset exists")
    
    print("Synthetic RSVG dataset created successfully")

def create_dataset(size, split_name):
    """Create a synthetic dataset with the given size"""
    dataset = []
    
    for i in range(size):
        # Generate a synthetic image name
        image_name = f"{i:06d}_{i+1000:06d}_1024_32615_synthetic_{split_name}.jpg"
        
        # Generate a random bounding box [x1, y1, x2, y2]
        x1 = random.randint(10, 300)
        y1 = random.randint(10, 300)
        width = random.randint(50, 200)
        height = random.randint(50, 200)
        bbox = [float(x1), float(y1), float(x1 + width), float(y1 + height)]
        
        # Generate a synthetic referring expression
        expression = f"find the {random.choice(['red', 'blue', 'green'])} {random.choice(['house', 'building', 'structure'])} in the {random.choice(['top', 'bottom'])} {random.choice(['left', 'right'])} area"
        
        # Create a synthetic item [img_file, None, bbox, phrase, None] format
        item = [image_name, None, bbox, expression, None]
        dataset.append(item)
    
    return dataset

if __name__ == "__main__":
    create_synthetic_rsvg_data() 