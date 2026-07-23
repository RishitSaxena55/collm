import json
import random
import os
import torch
from torch.utils.data import Dataset
from PIL import Image

class TripletDataset(Dataset):
    """
    Dataset for supervised Composed Image Retrieval (CIR) fine-tuning.
    Loads explicit triplets: Reference Image, Target Image, and Modification Text.
    """
    def __init__(self, jsonl_path, transform=None, image_dir=""):
        """
        Args:
            jsonl_path (str): Path to the JSONL dataset file.
            transform (callable, optional): Transform to be applied on the PIL images.
            image_dir (str): Base directory for relative image paths.
        """
        self.data = []
        with open(jsonl_path, 'r') as f:
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line.strip())
                self.data.append(item)
                
        self.transform = transform
        self.image_dir = image_dir

    def __len__(self):
        return len(self.data)

    def _process_text(self, item):
        """Extracts and processes the modification text from the raw JSON item."""
        return item.get("mod_text", "")

    def _get_image_paths(self, item):
        """Extracts relative image paths from the raw JSON item."""
        return item.get("reference_img", ""), item.get("target_img", "")

    def __getitem__(self, idx):
        item = self.data[idx]
        ref_rel, tgt_rel = self._get_image_paths(item)
        
        ref_img_path = os.path.join(self.image_dir, ref_rel) if self.image_dir else ref_rel
        target_img_path = os.path.join(self.image_dir, tgt_rel) if self.image_dir else tgt_rel
        
        mod_text = self._process_text(item)
        
        try:
            # Load images
            ref_image = Image.open(ref_img_path).convert('RGB')
            target_image = Image.open(target_img_path).convert('RGB')
        except (FileNotFoundError, OSError) as e:
            # Gracefully handle missing or corrupted images by falling back to the next item
            print(f"Warning: Failed to load images for idx {idx} ({ref_img_path} or {target_img_path}). Error: {e}. Falling back to next sample.")
            next_idx = (idx + 1) % len(self)
            # Prevent infinite recursion if all images are broken
            if next_idx == idx:
                raise RuntimeError("All images in the dataset appear to be missing or corrupted.")
            return self.__getitem__(next_idx)

        # Apply visual transforms
        if self.transform:
            ref_image = self.transform(ref_image)
            target_image = self.transform(target_image)

        return ref_image, target_image, mod_text

class MTCIRDataset(TripletDataset):
    """
    Subclass for the MTCIR dataset which contains multiple modification texts
    per image pair to increase diversity. Randomly samples one text per epoch.
    """
    def _get_image_paths(self, item):
        return item.get("ref_relpath", ""), item.get("tgt_relpath", "")

    def _process_text(self, item):
        texts = item.get("texts", [])
        if isinstance(texts, list) and len(texts) > 0:
            return random.choice(texts)
        elif isinstance(texts, str):
            return texts
        return ""

class FIQDataset(TripletDataset):
    """
    Subclass for the Fashion-IQ dataset which contains two modification texts
    per image pair. Concatenates them using " and ".
    """
    def _get_image_paths(self, item):
        # FIQ uses 'candidate' as ref and 'target' as target, and needs .jpg appended
        return item.get("candidate", "") + ".jpg", item.get("target", "") + ".jpg"

    def _process_text(self, item):
        texts = item.get("captions", [])
        if isinstance(texts, list) and len(texts) > 0:
            return " and ".join(texts)
        elif isinstance(texts, str):
            return texts
        return ""
