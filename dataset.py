"""
Dataset and DataLoader Module
"""
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Tuple
from config import Config


def pad_sequence(seq: List[int], max_len: int) -> Tuple[List[int], List[int]]:
    """Pad sequence and return mask"""
    seq = seq[-max_len:]
    pad_len = max_len - len(seq)
    return seq + [0]*pad_len, [1]*len(seq) + [0]*pad_len


class DINDataset(Dataset):
    """PyTorch Dataset for DIN with user profiles"""
    
    def __init__(self, samples: List[Dict], item_map: Dict, profile_map: Dict, 
                 age_map: Dict, occ_map: Dict, max_len: int = 50):
        self.item_map = item_map
        self.profile_map = profile_map
        self.age_map = age_map
        self.occ_map = occ_map
        self.max_len = max_len
        
        # Filter valid samples
        self.samples = []
        for s in samples:
            if s["candidate_post_id"] not in item_map:
                continue
            if s["user_id"] not in profile_map:
                continue
            hist = [pid for pid in s["history_post_ids"] if pid in item_map]
            if len(hist) == 0:
                continue
            self.samples.append({**s, "history_post_ids": hist})
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        s = self.samples[idx]
        
        # User profile
        profile = self.profile_map[s["user_id"]]
        age_id = self.age_map[profile["age_band"]]
        occ_id = self.occ_map[profile["occupation"]]
        loc_id = profile["user_location_id"]
        
        # Candidate
        cand = self.item_map[s["candidate_post_id"]]
        
        # History
        hist_dist = [self.item_map[p]["district_id"] for p in s["history_post_ids"]]
        hist_price = [self.item_map[p]["price_id"] for p in s["history_post_ids"]]
        hist_area = [self.item_map[p]["area_id"] for p in s["history_post_ids"]]
        
        # Pad
        hist_dist, mask = pad_sequence(hist_dist, self.max_len)
        hist_price, _ = pad_sequence(hist_price, self.max_len)
        hist_area, _ = pad_sequence(hist_area, self.max_len)
        
        return (
            torch.tensor(hist_dist, dtype=torch.long),
            torch.tensor(hist_price, dtype=torch.long),
            torch.tensor(hist_area, dtype=torch.long),
            torch.tensor(mask, dtype=torch.float32),
            torch.tensor(cand["district_id"], dtype=torch.long),
            torch.tensor(cand["price_id"], dtype=torch.long),
            torch.tensor(cand["area_id"], dtype=torch.long),
            torch.tensor(age_id, dtype=torch.long),
            torch.tensor(occ_id, dtype=torch.long),
            torch.tensor(loc_id, dtype=torch.long),
            torch.tensor(s["label"], dtype=torch.float32)
        )


def create_dataloaders(config: Config):
    """Create train/val/test dataloaders"""
    # Load data
    with open(config.ITEM_FEATURES, "r") as f:
        items = json.load(f)
    item_map = {it["list_id"]: it for it in items}
    
    with open(config.TRAINING_DATA, "r") as f:
        samples = json.load(f)
    
    with open(config.USER_PROFILES, "r") as f:
        profiles = json.load(f)
    profile_map = {p["user_id"]: p for p in profiles}
    
    with open(config.FEATURE_MAPPING, "r") as f:
        mappings = json.load(f)
    age_map = mappings["age_map"]
    occ_map = mappings["occupation_map"]
    
    # Compute max_len
    hist_lens = [len(s["history_post_ids"]) for s in samples]
    max_len = int(np.percentile(hist_lens, 95))
    print(f"📏 Using max_hist_len = {max_len} (95th percentile)")
    
    # Split data
    np.random.seed(config.RANDOM_SEED)
    np.random.shuffle(samples)
    
    n_val = int(len(samples) * config.VAL_RATIO)
    n_test = int(len(samples) * config.TEST_RATIO)
    
    train_samples = samples[:(len(samples)-n_val-n_test)]
    val_samples = samples[(len(samples)-n_val-n_test):(len(samples)-n_test)]
    test_samples = samples[(len(samples)-n_test):]
    
    print(f"📊 Data split: Train={len(train_samples)}, Val={len(val_samples)}, Test={len(test_samples)}")
    
    # Create datasets
    train_ds = DINDataset(train_samples, item_map, profile_map, age_map, occ_map, max_len)
    val_ds = DINDataset(val_samples, item_map, profile_map, age_map, occ_map, max_len)
    test_ds = DINDataset(test_samples, item_map, profile_map, age_map, occ_map, max_len)
    
    # Create loaders
    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS)
    val_loader = DataLoader(val_ds, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS)
    test_loader = DataLoader(test_ds, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS)
    
    return train_loader, val_loader, test_loader