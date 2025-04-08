import random
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F
from PIL import Image
import numpy as np

class RandomGaussianBlur:
    def __init__(self, p=0.5, kernel_size=3, sigma=(0.1, 2.0)):
        self.p = p
        self.kernel_size = kernel_size
        self.sigma = sigma

    def __call__(self, img):
        if random.random() < self.p:
            sigma = random.uniform(self.sigma[0], self.sigma[1])
            return F.gaussian_blur(img, self.kernel_size, [sigma, sigma])
        return img

class RandomColorJitter:
    def __init__(self, p=0.5, brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1):
        self.p = p
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    def __call__(self, img):
        if random.random() < self.p:
            return F.adjust_brightness(img, random.uniform(1-self.brightness, 1+self.brightness))
            img = F.adjust_contrast(img, random.uniform(1-self.contrast, 1+self.contrast))
            img = F.adjust_saturation(img, random.uniform(1-self.saturation, 1+self.saturation))
            img = F.adjust_hue(img, random.uniform(-self.hue, self.hue))
        return img

class RandomErasing:
    def __init__(self, p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0):
        self.p = p
        self.scale = scale
        self.ratio = ratio
        self.value = value

    def __call__(self, img):
        if random.random() < self.p:
            # Convert to tensor if not already
            if not isinstance(img, torch.Tensor):
                img = F.to_tensor(img)
            
            # Get image dimensions
            _, h, w = img.shape
            
            # Calculate area
            area = h * w
            
            # Calculate target area
            target_area = random.uniform(self.scale[0], self.scale[1]) * area
            aspect_ratio = random.uniform(self.ratio[0], self.ratio[1])
            
            # Calculate dimensions
            h_er = int(round(np.sqrt(target_area * aspect_ratio)))
            w_er = int(round(np.sqrt(target_area / aspect_ratio)))
            
            # Ensure dimensions are within bounds
            if h_er < h and w_er < w:
                # Calculate position
                top = random.randint(0, h - h_er)
                left = random.randint(0, w - w_er)
                
                # Apply erasing
                img[:, top:top+h_er, left:left+w_er] = self.value
            
            # Convert back to PIL if input was PIL
            if not isinstance(img, torch.Tensor):
                img = F.to_pil_image(img)
        
        return img

def get_transforms(args, is_train=True):
    """Get data transforms for training and validation."""
    transforms = []
    
    # Basic transforms
    transforms.extend([
        T.Resize((args.img_size, args.img_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    if is_train:
        # Add training-specific transforms
        if args.aug_crop:
            transforms.insert(0, T.RandomResizedCrop(args.img_size, scale=(0.8, 1.0)))
        
        if args.aug_scale:
            transforms.insert(0, T.RandomAffine(degrees=0, scale=(0.8, 1.2)))
        
        if args.aug_translate:
            transforms.insert(0, T.RandomAffine(degrees=0, translate=(0.1, 0.1)))
        
        if args.aug_blur:
            transforms.insert(0, RandomGaussianBlur(p=0.5))
        
        if args.aug_color:
            transforms.insert(0, RandomColorJitter(p=0.5))
        
        if args.aug_erase:
            transforms.append(RandomErasing(p=0.5))
    
    return T.Compose(transforms) 