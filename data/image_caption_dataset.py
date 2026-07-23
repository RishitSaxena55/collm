import json
import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class ImageCaptionDataset(Dataset):
    def __init__(self, data_list, target_transform=None, aug_transform=None, image_dir=""):
        """
        Args:
            data_list (list or str): List of dicts [{"image_path": "...", "caption": "..."}] 
                                     or path to a JSONL/JSON file containing such dicts.
            target_transform: Transform for the standard target image v_i.
            aug_transform: Heavy augmentation transform for v'_i.
            image_dir: Base directory for resolving relative paths.
        """
        # Load data from file path or use the list directly
        if isinstance(data_list, str):
            with open(data_list, 'r', encoding='utf-8') as f:
                self.data = [json.loads(line) for line in f]
        else:
            self.data = data_list
            
        self.image_dir = image_dir
            
        # Default standard transform for target image
        self.target_transform = target_transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], 
                                 std=[0.26862954, 0.26130258, 0.27577711]), # CLIP normalizations
        ])
        
        # Default SimCLR-style contrastive augmentations for the augmented image
        self.aug_transform = aug_transform or transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], 
                                 std=[0.26862954, 0.26130258, 0.27577711]),
        ])

    def __len__(self):
        return len(self.data)

    def _get_image_path(self, item):
        return item.get('image_path')

    def _get_caption(self, item):
        return item.get('caption')

    def __getitem__(self, idx):
        item = self.data[idx]
        image_path = self._get_image_path(item)
        if self.image_dir and image_path:
            image_path = os.path.join(self.image_dir, image_path)
            
        caption = self._get_caption(item)
        
        # Load image (convert to RGB in case of grayscale/RGBA)
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            # Handle corrupted or missing images simply by creating a blank image
            # In production you might want to skip or return a random other index
            image = Image.new('RGB', (224, 224), (0, 0, 0))
            print(f"Warning: Failed to load image {image_path}: {e}")
        
        # Apply transformations
        v_i = self.target_transform(image)
        v_prime_i = self.aug_transform(image)
        
        return v_i, v_prime_i, caption

class LLaVADataset(ImageCaptionDataset):
    """
    Subclass for the LLaVA 558k dataset format.
    Expects JSON keys: 'image' for image paths and 'blip_caption' for captions.
    """
    def _get_image_path(self, item):
        return item.get('image')

    def _get_caption(self, item):
        return item.get('blip_caption')
