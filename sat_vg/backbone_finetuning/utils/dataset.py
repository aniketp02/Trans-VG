import os
import json
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms.functional as TF
from typing import Dict, List, Tuple, Optional, Union, Callable

from .transforms import get_transform
from .mask_transforms import prepare_mask, apply_mask_transforms, resize_binary_mask

class SatelliteDataset(Dataset):
    """Dataset class for satellite imagery with utility methods for processing."""
    
    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        annotation_file: Optional[str] = None,
        image_size: int = 224,
    ):
        """
        Initialize the satellite dataset with utility methods.
        
        Args:
            root_dir: Root directory containing the dataset
            split: Dataset split ('train', 'val', or 'test')
            transform: Optional transform for input images
            target_transform: Optional transform for targets
            annotation_file: Optional path to annotation file
            image_size: Target image size for resizing
        """
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        self.image_size = image_size
        
        # Initialize empty data lists
        self.image_paths = []
        self.labels = []
        self.annotations = {}
        
        # Counter for missing images statistics
        self.missing_images_count = 0
        self.total_attempts = 0
        
        # Load annotations if provided
        if annotation_file and os.path.exists(annotation_file):
            self.load_annotations(annotation_file)
            
        # Load image paths based on the split
        self._load_image_paths()
    
    def _load_image_paths(self):
        """Load image paths from the root directory based on the split."""
        # Check if a split file exists (common format: split.txt with image IDs)
        split_file = os.path.join(self.root_dir, f"{self.split}.txt")
        
        if os.path.exists(split_file):
            # Load image IDs from split file
            with open(split_file, 'r') as f:
                image_ids = [line.strip() for line in f.readlines()]
                
            # Look for image directory (common names)
            for img_dir_name in ['images', 'JPEGImages', 'Images', 'imgs']:
                img_dir = os.path.join(self.root_dir, img_dir_name)
                if os.path.exists(img_dir) and os.path.isdir(img_dir):
                    # For each image ID, look for corresponding image file
                    for img_id in image_ids:
                        # Try different extensions
                        for ext in ['.jpg', '.jpeg', '.png', '.tif', '.tiff']:
                            img_path = os.path.join(img_dir, f"{img_id}{ext}")
                            if os.path.exists(img_path):
                                self.image_paths.append(img_path)
                                # Add a default label of 0 (can be updated later)
                                self.labels.append(0)
                                break
                    
                    # If we found images, don't check other directories
                    if self.image_paths:
                        break
                        
            # Check for a labels file associated with the split
            labels_file = os.path.join(self.root_dir, f"{self.split}_labels.txt")
            if os.path.exists(labels_file) and len(self.image_paths) > 0:
                # Format expected: one label per line, corresponding to image order
                with open(labels_file, 'r') as f:
                    self.labels = [int(line.strip()) for line in f.readlines()]
                    
                # Make sure we have as many labels as images
                if len(self.labels) != len(self.image_paths):
                    print(f"WARNING: Number of labels ({len(self.labels)}) doesn't match number of images ({len(self.image_paths)})")
                    # Truncate labels or add default labels as needed
                    if len(self.labels) > len(self.image_paths):
                        self.labels = self.labels[:len(self.image_paths)]
                    else:
                        self.labels.extend([0] * (len(self.image_paths) - len(self.labels)))
        else:
            # If no split file exists, look for images directly
            for img_dir_name in ['images', 'JPEGImages', 'Images', 'imgs']:
                img_dir = os.path.join(self.root_dir, img_dir_name)
                if os.path.exists(img_dir) and os.path.isdir(img_dir):
                    # Look for images with common extensions
                    for ext in ['.jpg', '.jpeg', '.png', '.tif', '.tiff']:
                        img_paths = [os.path.join(img_dir, f) for f in os.listdir(img_dir) 
                                    if f.lower().endswith(ext)]
                        self.image_paths.extend(img_paths)
                        # Add default labels
                        self.labels.extend([0] * len(img_paths))
                    
                    # If we found images, don't check other directories
                    if self.image_paths:
                        break
            
            # If still no images, look for split-specific directory
            if not self.image_paths:
                split_dir = os.path.join(self.root_dir, self.split)
                if os.path.exists(split_dir) and os.path.isdir(split_dir):
                    for ext in ['.jpg', '.jpeg', '.png', '.tif', '.tiff']:
                        img_paths = [os.path.join(split_dir, f) for f in os.listdir(split_dir) 
                                    if f.lower().endswith(ext)]
                        self.image_paths.extend(img_paths)
                        # Add default labels
                        self.labels.extend([0] * len(img_paths))
                        
                    # Try to infer labels from directory structure
                    # Common format: class_name/image.jpg
                    try:
                        self._infer_labels_from_directory_structure()
                    except Exception as e:
                        print(f"Failed to infer labels from directory structure: {e}")
        
        # Print warning if no images found
        if not self.image_paths:
            print(f"WARNING: No images found for split '{self.split}' in directory '{self.root_dir}'")
            print(f"Tried looking for split file: {split_file}")
            print(f"Make sure your dataset structure is correct.")
            
        else:
            print(f"Loaded {len(self.image_paths)} images for split '{self.split}'")
            
        # Try to load class names
        self._load_class_names()
    
    def _infer_labels_from_directory_structure(self):
        """
        Try to infer labels from directory structure.
        Assumes format: root_dir/class_name/image.jpg
        """
        # Get unique parent directories
        parent_dirs = set()
        for img_path in self.image_paths:
            parent_dir = os.path.basename(os.path.dirname(img_path))
            parent_dirs.add(parent_dir)
            
        # Create class mapping (directory name -> class index)
        self.class_to_idx = {dir_name: idx for idx, dir_name in enumerate(sorted(parent_dirs))}
        
        # Update labels based on parent directory
        new_labels = []
        for img_path in self.image_paths:
            parent_dir = os.path.basename(os.path.dirname(img_path))
            class_idx = self.class_to_idx.get(parent_dir, 0)
            new_labels.append(class_idx)
            
        self.labels = new_labels
        
    def _load_class_names(self):
        """Load class names from a class mapping file if available."""
        # Check for class names file
        for class_file in ['classes.txt', 'class_names.txt', 'labels.txt']:
            class_path = os.path.join(self.root_dir, class_file)
            if os.path.exists(class_path):
                with open(class_path, 'r') as f:
                    self.classes = [line.strip() for line in f.readlines()]
                print(f"Loaded {len(self.classes)} class names from {class_path}")
                # Create class to index mapping
                self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
                return
                
        # If no class file found but we have annotations with class names
        if hasattr(self, 'annotations') and self.annotations:
            # Try to extract unique class names from annotations
            class_names = set()
            for ann in self.annotations.values():
                if 'category' in ann:
                    class_names.add(ann['category'])
                elif 'class' in ann:
                    class_names.add(ann['class'])
                elif 'label' in ann:
                    class_names.add(ann['label'])
            
            if class_names:
                self.classes = sorted(class_names)
                self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
                print(f"Extracted {len(self.classes)} class names from annotations")
                return
                
        # Default: create numeric class names based on unique labels
        unique_labels = sorted(set(self.labels))
        self.classes = [str(label) for label in unique_labels]
        self.class_to_idx = {str(label): label for label in unique_labels}
        print(f"Created {len(self.classes)} default class names from labels")
    
    def load_annotations(self, annotation_file: str) -> None:
        """
        Load annotations from a JSON file.
        
        Args:
            annotation_file: Path to annotation file
        """
        with open(annotation_file, 'r') as f:
            self.annotations = json.load(f)
    
    def get_image_info(self, idx: int) -> Dict:
        """
        Get information about an image.
        
        Args:
            idx: Index of the image
            
        Returns:
            Dict containing image information
        """
        if idx >= len(self):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self)}")
            
        image_path = self.image_paths[idx]
        image_id = os.path.basename(image_path).split('.')[0]
        
        info = {
            'image_id': image_id,
            'image_path': image_path,
            'height': None,
            'width': None
        }
        
        # Try to get image dimensions
        try:
            with Image.open(image_path) as img:
                info['height'], info['width'] = img.height, img.width
        except Exception as e:
            print(f"Warning: Could not open image {image_path}: {e}")
        
        # Add annotation info if available
        if image_id in self.annotations:
            info.update(self.annotations[image_id])
            
        return info
    
    def get_class_counts(self) -> Dict[int, int]:
        """
        Get counts of each class in the dataset.
        
        Returns:
            Dictionary mapping class indices to their counts
        """
        class_counts = {}
        for label in self.labels:
            if isinstance(label, (int, np.integer)):
                class_counts[label] = class_counts.get(label, 0) + 1
            elif isinstance(label, torch.Tensor) and label.dim() == 0:
                label_int = label.item()
                class_counts[label_int] = class_counts.get(label_int, 0) + 1
            elif isinstance(label, (list, np.ndarray, torch.Tensor)) and len(label) > 0:
                # For multi-label cases
                for l in label:
                    if isinstance(l, (int, np.integer)) or (isinstance(l, torch.Tensor) and l.dim() == 0):
                        l_int = l if isinstance(l, (int, np.integer)) else l.item()
                        class_counts[l_int] = class_counts.get(l_int, 0) + 1
                        
        return class_counts
    
    def get_num_classes(self) -> int:
        """
        Get the number of unique classes in the dataset.
        
        Returns:
            Number of unique classes
        """
        class_counts = self.get_class_counts()
        return len(class_counts) if class_counts else 0
    
    def get_stats(self) -> Dict:
        """
        Get dataset statistics.
        
        Returns:
            Dictionary with dataset statistics
        """
        stats = {
            'num_samples': len(self),
            'class_counts': self.get_class_counts(),
            'split': self.split
        }
        return stats
        
    def __len__(self) -> int:
        """Return the length of the dataset."""
        return len(self.image_paths)
        
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Union[torch.Tensor, Dict]]:
        """
        Get an item from the dataset.
        
        Args:
            idx: Index of the item
            
        Returns:
            Tuple of (image, target)
        """
        # This is a base implementation, should be overridden in subclasses
        image_path = self.image_paths[idx]
        
        # Increment total attempts counter
        self.total_attempts += 1
        
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            # Don't print individual errors, just count them
            self.missing_images_count += 1
            
            # Print summary periodically
            if self.missing_images_count == 1 or self.missing_images_count % 100 == 0:
                print(f"Missing images: {self.missing_images_count}/{self.total_attempts} "
                      f"({self.missing_images_count/self.total_attempts:.1%})")
                
            # Return a blank image as fallback
            image = Image.new('RGB', (self.image_size, self.image_size), color=0)
            
        target = self.labels[idx] if idx < len(self.labels) else 0
        
        if self.transform:
            image = self.transform(image)
            
        if self.target_transform and not isinstance(target, dict):
            target = self.target_transform(target)
            
        return image, target

# Helper function to create a dataloader
def create_satellite_dataloader(
    root_dir: str,
    split: str = 'train',
    batch_size: int = 16,
    img_size: int = 224,
    num_workers: int = 4,
    use_augmentation: bool = True,
    aug_intensity: str = 'medium',
):
    """Create a dataloader for satellite dataset"""
    from torch.utils.data import DataLoader
    
    # Get the appropriate transform
    transform = get_transform(
        img_size=img_size,
        is_training=(split == 'train'),
        use_augmentation=use_augmentation,
        aug_intensity=aug_intensity
    )
    
    # Create dataset
    dataset = SatelliteDataset(
        root_dir=root_dir,
        split=split,
        transform=transform,
        image_size=img_size
    )
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == 'train'),
        num_workers=num_workers,
        pin_memory=True,
        drop_last=(split == 'train')
    )
    
    return dataloader 