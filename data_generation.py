"""
Synthetic Data Generation Module
"""
import json
import numpy as np
from tqdm import tqdm
from typing import List, Dict, Tuple
from config import Config


class DataGenerator:
    """Generate realistic synthetic user-item interactions with user profiles"""
    
    def __init__(self, config: Config):
        self.config = config
        self.items = []
        self.item_map = {}
        self.districts = []
        self.price_bands = {}
        
    def load_items(self):
        """Load processed items"""
        print(f"📁 Loading items from {self.config.ITEM_FEATURES}...")
        with open(self.config.ITEM_FEATURES, "r", encoding="utf-8") as f:
            self.items = json.load(f)
        self.item_map = {item["list_id"]: item for item in self.items}
        
        # Extract unique districts
        self.districts = list(set(item["district_id"] for item in self.items))
        
        # Group items by price band
        self.price_bands = {}
        for item in self.items:
            pid = item["price_id"]
            if pid not in self.price_bands:
                self.price_bands[pid] = []
            self.price_bands[pid].append(item)
        
        print(f"   Loaded {len(self.items)} items")
        print(f"   Districts: {len(self.districts)}")
        print(f"   Price bands: {len(self.price_bands)}")
    
    def generate_user_profile(self, user_id: int) -> Dict:
        """Generate a user profile"""
        age_band = np.random.choice(self.config.AGE_BANDS, p=self.config.AGE_PROBS)
        occupation = np.random.choice(self.config.OCCUPATIONS, p=self.config.OCCUPATION_PROBS)
        
        # User location from random item district
        random_item = np.random.choice(self.items)
        user_location_id = random_item["district_id"]
        
        return {
            "user_id": user_id,
            "age_band": age_band,
            "occupation": occupation,
            "user_location_id": user_location_id
        }
    
    def select_anchor_item(self, profile: Dict) -> Dict:
        """Select anchor item based on user profile"""
        occupation = profile["occupation"]
        
        # Price preferences based on occupation
        if occupation == "student":
            if np.random.random() < 0.7:
                preferred_prices = [1, 2, 3, 4, 5]  # Low prices
            else:
                preferred_prices = [6, 7]  # High prices
        else:  # working
            if np.random.random() < 0.7:
                preferred_prices = [5, 6, 7]  # High prices
            else:
                preferred_prices = [1, 2, 3, 4]  # Low prices
        
        # Get items in preferred price bands
        candidates = []
        for pid in preferred_prices:
            if pid in self.price_bands:
                candidates.extend(self.price_bands[pid])
        
        if not candidates:
            return np.random.choice(self.items)
        
        return np.random.choice(candidates)
    
    def compute_similarity(self, item_a: Dict, item_b: Dict, profile: Dict) -> float:
        """Compute similarity score between two items considering user profile"""
        score = 0.0
        user_loc = profile["user_location_id"]
        
        # District matching
        if item_a["district_id"] == item_b["district_id"]:
            score += 0.5
        
        # User location bonus
        if item_b["district_id"] == user_loc:
            score += 0.3
        
        # Price matching
        if item_a["price_id"] == item_b["price_id"]:
            score += 0.3
        
        # Area matching
        if item_a["area_id"] == item_b["area_id"]:
            score += 0.2
        
        # Add noise
        score += np.random.normal(0, 0.05)
        
        return np.clip(score, 0, 1)
    
    def generate_user_history(self, anchor_item: Dict, profile: Dict) -> List[int]:
        """Generate browsing history for a user"""
        hist_len = np.random.randint(self.config.MIN_HIST_LEN, self.config.MAX_HIST_LEN)
        history = []
        
        # 70% similar items (sim >= 0.6)
        n_similar = int(hist_len * 0.7)
        similar = [
            item for item in self.items
            if self.compute_similarity(item, anchor_item, profile) >= 0.6
        ]
        if similar:
            selected = np.random.choice(
                similar, 
                min(n_similar, len(similar)), 
                replace=False
            )
            history.extend(selected)
        
        # 20% medium similarity (0.3 <= sim < 0.6)
        n_medium = int(hist_len * 0.2)
        medium = [
            item for item in self.items
            if 0.3 <= self.compute_similarity(item, anchor_item, profile) < 0.6
        ]
        if medium:
            selected = np.random.choice(
                medium, 
                min(n_medium, len(medium)), 
                replace=False
            )
            history.extend(selected)
        
        # 10% random
        n_random = hist_len - len(history)
        if n_random > 0:
            selected = np.random.choice(
                self.items, 
                min(n_random, len(self.items)), 
                replace=False
            )
            history.extend(selected)
        
        np.random.shuffle(history)
        return [item["list_id"] for item in history[:hist_len]]
    
    def generate_samples(self) -> Tuple[List[Dict], List[Dict]]:
        """Generate all training samples and user profiles"""
        print(f"🎲 Generating samples for {self.config.NUM_USERS} users...")
        samples = []
        profiles = []
        
        for user_id in tqdm(range(self.config.NUM_USERS), desc="Generating"):
            # Generate user profile
            profile = self.generate_user_profile(user_id)
            profiles.append(profile)
            
            # Select anchor item
            anchor = self.select_anchor_item(profile)
            
            # Generate history
            history_ids = self.generate_user_history(anchor, profile)
            if len(history_ids) < self.config.MIN_HIST_LEN:
                continue
            
            # Positive sample
            pos_candidates = [
                item for item in self.items
                if self.compute_similarity(item, anchor, profile) >= 0.6 and 
                   item["list_id"] not in history_ids
            ]
            if not pos_candidates:
                continue
            
            pos_item = np.random.choice(pos_candidates)
            samples.append({
                "user_id": profile["user_id"],
                "history_post_ids": history_ids,
                "candidate_post_id": pos_item["list_id"],
                "label": 1
            })
            
            # Negative samples
            num_negs = int(self.config.NEG_POS_RATIO)
            num_hard = int(num_negs * self.config.HARD_NEG_RATIO)
            
            # Hard negatives
            hard_negs = [
                item for item in self.items
                if 0.3 <= self.compute_similarity(item, anchor, profile) < 0.5 and
                   item["list_id"] not in history_ids
            ]
            for _ in range(min(num_hard, len(hard_negs))):
                neg = np.random.choice(hard_negs)
                samples.append({
                    "user_id": profile["user_id"],
                    "history_post_ids": history_ids,
                    "candidate_post_id": neg["list_id"],
                    "label": 0
                })
            
            # Easy negatives
            easy_negs = [
                item for item in self.items
                if self.compute_similarity(item, anchor, profile) < 0.3 and
                   item["list_id"] not in history_ids
            ]
            num_easy = num_negs - num_hard
            for _ in range(min(num_easy, len(easy_negs))):
                neg = np.random.choice(easy_negs)
                samples.append({
                    "user_id": profile["user_id"],
                    "history_post_ids": history_ids,
                    "candidate_post_id": neg["list_id"],
                    "label": 0
                })
        
        return samples, profiles
    
    def analyze_data(self, samples: List[Dict], profiles: List[Dict]):
        """Print statistics"""
        print(f"\n📊 Data Statistics:")
        print(f"   Total samples: {len(samples):,}")
        print(f"   Total users: {len(profiles):,}")
        
        # Label distribution
        n_pos = sum(s["label"] for s in samples)
        print(f"   Positive: {n_pos:,} ({n_pos/len(samples)*100:.1f}%)")
        print(f"   Negative: {len(samples)-n_pos:,} ({(1-n_pos/len(samples))*100:.1f}%)")
        
        # History length
        hist_lens = [len(s["history_post_ids"]) for s in samples]
        print(f"   History length: min={min(hist_lens)}, max={max(hist_lens)}, "
              f"mean={np.mean(hist_lens):.1f}, median={np.median(hist_lens):.0f}")
        
        # Profile distribution
        ages = [p["age_band"] for p in profiles]
        occs = [p["occupation"] for p in profiles]
        print(f"\n   Age distribution:")
        for age in self.config.AGE_BANDS:
            count = ages.count(age)
            print(f"      {age}: {count} ({count/len(ages)*100:.1f}%)")
        print(f"   Occupation distribution:")
        for occ in self.config.OCCUPATIONS:
            count = occs.count(occ)
            print(f"      {occ}: {count} ({count/len(occs)*100:.1f}%)")
    
    def save_data(self, samples: List[Dict], profiles: List[Dict]):
        """Save training data and user profiles"""
        # Save samples
        with open(self.config.TRAINING_DATA, "w", encoding="utf-8") as f:
            json.dump(samples, f, indent=2, ensure_ascii=False)
        print(f"💾 Saved {len(samples)} samples → {self.config.TRAINING_DATA}")
        
        # Save profiles
        profiles_to_save = []
        for p in profiles:
            profiles_to_save.append({
                "user_id": int(p["user_id"]),
                "age_band": str(p["age_band"]),
                "occupation": str(p["occupation"]),
                "user_location_id": int(p["user_location_id"])
            })
        
        with open(self.config.USER_PROFILES, "w", encoding="utf-8") as f:
            json.dump(profiles_to_save, f, indent=2, ensure_ascii=False)
        print(f"💾 Saved {len(profiles_to_save)} user profiles → {self.config.USER_PROFILES}")
    
    def run(self):
        """Run generation pipeline"""
        print("\n" + "="*70)
        print("🏗️  STEP 2: SYNTHETIC DATA GENERATION WITH USER PROFILES")
        print("="*70 + "\n")
        
        self.load_items()
        samples, profiles = self.generate_samples()
        self.analyze_data(samples, profiles)
        self.save_data(samples, profiles)
        
        print("\n✅ Data generation completed!\n")
        return samples, profiles