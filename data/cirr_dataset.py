"""
data/cirr_dataset.py  —  CIRR dataset loader for composed image retrieval.
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import torch
from torch.utils.data import Dataset
from PIL import Image

def resolve_cirr_root(data_root: str) -> Path:
    """Resolve data_root to the exact directory containing 'captions' and 'image_splits'."""
    root = Path(data_root)
    if (root / "captions").exists():
        return root
    elif (root / "cirr" / "captions").exists():
        return root / "cirr"
    else:
        # Fallback to root for clear error reporting downstream
        return root

def _load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"CIRR required file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

class CIRRTrainDataset(Dataset):
    """
    CIRR Training Dataset.
    Adapted for CoLLM: Returns (target_img, ref_img, caption)
    """
    def __init__(self, data_root: str, split: str = "train", transform=None):
        self.root = resolve_cirr_root(data_root)
        self.transform = transform
        self.split = split

        cap_path = self.root / "captions" / f"cap.rc2.{split}.json"
        split_path = self.root / "image_splits" / f"split.rc2.{split}.json"

        self.captions = _load_json(cap_path)
        self.path_map = _load_json(split_path)
        self.img_raw = self.root / "img_raw"

    def _resolve_img_path(self, img_id: str) -> Path:
        if img_id not in self.path_map:
            raise KeyError(f"Image ID '{img_id}' not found in split mapping!")
        rel_path = self.path_map[img_id].lstrip("./")
        full_path = self.img_raw / rel_path
        if not full_path.exists():
            raise FileNotFoundError(f"CIRR image '{img_id}' missing at path: {full_path}")
        return full_path

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx: int):
        item = self.captions[idx]
        ref_id = item["reference"]
        target_id = item["target_hard"]
        caption = item["caption"]

        ref_path = self._resolve_img_path(ref_id)
        target_path = self._resolve_img_path(target_id)

        try:
            ref_img = Image.open(ref_path).convert("RGB")
        except (FileNotFoundError, OSError):
            ref_img = Image.new('RGB', (224, 224))
            
        try:
            target_img = Image.open(target_path).convert("RGB")
        except (FileNotFoundError, OSError):
            target_img = Image.new('RGB', (224, 224))

        if self.transform is not None:
            ref_img = self.transform(ref_img)
            target_img = self.transform(target_img)

        # Output format explicitly tailored for CoLLM train.py unpacking
        # (target_images, ref_images, caption) -> maps to (v_i, v_prime_i, captions)
        return target_img, ref_img, caption

class CIRRQueryDataset(Dataset):
    """
    CIRR Query Evaluation Dataset for val and test1.
    """
    def __init__(self, data_root: str, split: str = "val", transform=None):
        self.root = resolve_cirr_root(data_root)
        self.split = split
        self.transform = transform

        cap_path = self.root / "captions" / f"cap.rc2.{split}.json"
        split_path = self.root / "image_splits" / f"split.rc2.{split}.json"

        self.captions = _load_json(cap_path)
        self.path_map = _load_json(split_path)
        self.img_raw = self.root / "img_raw"

    def _resolve_img_path(self, img_id: str) -> Path:
        if img_id not in self.path_map:
            raise KeyError(f"Image ID '{img_id}' not found in split mapping!")
        rel_path = self.path_map[img_id].lstrip("./")
        full_path = self.img_raw / rel_path
        if not full_path.exists():
            raise FileNotFoundError(f"CIRR image '{img_id}' missing at path: {full_path}")
        return full_path

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx: int) -> dict:
        item = self.captions[idx]
        ref_id = item["reference"]
        caption = item["caption"]
        pairid = item["pairid"]
        subset_ids = item.get("img_set", {}).get("members", [])

        # target_hard is only present in train and val, absent in test1
        target_id = item.get("target_hard", "")

        ref_path = self._resolve_img_path(ref_id)
        
        try:
            ref_img = Image.open(ref_path).convert("RGB")
        except (FileNotFoundError, OSError):
            ref_img = Image.new('RGB', (224, 224))

        if self.transform is not None:
            ref_img = self.transform(ref_img)

        return {
            "ref_images": ref_img,
            "texts": caption,
            "target_id": target_id,
            "ref_id": ref_id,
            "subset_ids": subset_ids,
            "pairid": pairid
        }

class CIRRPoolDataset(Dataset):
    """
    CIRR Gallery Pool Dataset for val and test1.
    """
    def __init__(self, data_root: str, split: str = "val", transform=None):
        self.root = resolve_cirr_root(data_root)
        self.split = split
        self.transform = transform

        split_path = self.root / "image_splits" / f"split.rc2.{split}.json"
        self.path_map = _load_json(split_path)
        self.img_raw = self.root / "img_raw"

        # Unique gallery items as list of (img_id, full_path)
        self.data = []
        for img_id, rel_path in self.path_map.items():
            clean_rel = rel_path.lstrip("./")
            full_path = self.img_raw / clean_rel
            self.data.append((img_id, str(full_path)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        img_id, img_path = self.data[idx]
        
        try:
            img = Image.open(img_path).convert("RGB")
        except (FileNotFoundError, OSError):
            img = Image.new('RGB', (224, 224))

        if self.transform is not None:
            img = self.transform(img)

        return {
            "target_images": img,
            "target_id": img_id
        }
