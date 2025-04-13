import os
import shutil
import torch
from typing import Dict, List, Tuple
from pathlib import Path
import zipfile
from tqdm import tqdm


def extract_classes_from_filename(filename: str) -> str:
    """Extract class name from RSVG image filename."""
    # Format: X_Y_dist_EPSG_class.jpg
    parts = filename.split('_')
    if len(parts) >= 5:
        return parts[-1].split('.')[0]
    return "unknown"


def process_rsvg_dataset(
    rsvg_root: str,
    output_root: str,
    split_files: Dict[str, str]
) -> None:
    """Process RSVG dataset into classification format."""
    # Create output directories
    for split in ['train', 'val']:
        os.makedirs(os.path.join(output_root, split), exist_ok=True)
    
    # Process each split
    for split_name, split_file in split_files.items():
        # Load split data
        data = torch.load(os.path.join(rsvg_root, split_file))
        
        # Process each entry
        for entry in tqdm(data, desc=f"Processing {split_name}"):
            img_name = entry[0]  # First element is image name
            class_name = extract_classes_from_filename(img_name)
            
            # Create class directory
            class_dir = os.path.join(output_root, split_name, class_name)
            os.makedirs(class_dir, exist_ok=True)
            
            # Copy image
            src_path = os.path.join(rsvg_root, 'images', img_name)
            dst_path = os.path.join(class_dir, img_name)
            
            if os.path.exists(src_path):
                shutil.copy2(src_path, dst_path)


def process_opt_rsvg_dataset(
    opt_rsvg_root: str,
    output_root: str,
    rsvg_classes: List[str]
) -> None:
    """Process OPT-RSVG dataset and add to existing classes."""
    # Unzip annotations
    annotations_zip = os.path.join(opt_rsvg_root, 'OPT-RSVG', 'Annotations.zip')
    with zipfile.ZipFile(annotations_zip, 'r') as zip_ref:
        zip_ref.extractall(opt_rsvg_root)
    
    # Process images (implementation depends on OPT-RSVG format)
    # This is a placeholder - you'll need to adapt based on actual OPT-RSVG format
    pass


def process_dior_rsvg_dataset(
    dior_rsvg_root: str,
    output_root: str,
    rsvg_classes: List[str]
) -> None:
    """Process DIOR-RSVG dataset and add to existing classes."""
    # Process images (implementation depends on DIOR-RSVG format)
    # This is a placeholder - you'll need to adapt based on actual DIOR-RSVG format
    pass


def main():
    # Define paths
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    rsvg_root = os.path.join(base_dir, 'rsvg')
    opt_rsvg_root = os.path.join(base_dir, 'opt-rsvg')
    dior_rsvg_root = os.path.join(base_dir, 'dior-rsvg')
    output_root = os.path.join(base_dir, 'backbone_finetuning', 'data')
    
    # Define split files
    split_files = {
        'train': 'rsvg_train.pth',
        'val': 'rsvg_val.pth'
    }
    
    # Process RSVG dataset
    print("Processing RSVG dataset...")
    process_rsvg_dataset(rsvg_root, output_root, split_files)
    
    # Get unique classes from RSVG
    rsvg_classes = set()
    for split in ['train', 'val']:
        split_dir = os.path.join(output_root, split)
        if os.path.exists(split_dir):
            rsvg_classes.update(os.listdir(split_dir))
    
    # Process additional datasets if needed
    if os.path.exists(opt_rsvg_root):
        print("Processing OPT-RSVG dataset...")
        process_opt_rsvg_dataset(opt_rsvg_root, output_root, list(rsvg_classes))
    
    if os.path.exists(dior_rsvg_root):
        print("Processing DIOR-RSVG dataset...")
        process_dior_rsvg_dataset(dior_rsvg_root, output_root, list(rsvg_classes))
    
    print("Dataset processing completed!")


if __name__ == "__main__":
    main() 