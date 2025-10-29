"""
Data Preprocessing Module
"""
import os
import json
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from config import Config


class DataPreprocessor:
    """Preprocess raw data into features"""
    
    def __init__(self, config: Config):
        self.config = config
        self.le_district = LabelEncoder()
        self.le_price = LabelEncoder()
        self.le_area = LabelEncoder()
        
        # Create data directory if not exists
        os.makedirs(config.DATA_DIR, exist_ok=True)
    
    def load_raw_data(self) -> pd.DataFrame:
        """Load raw JSON data"""
        print(f"📁 Loading raw data from {self.config.RAW_DATA}...")
        with open(self.config.RAW_DATA, "r", encoding="utf-8") as f:
            data = json.load(f)
        df = pd.DataFrame(data)
        print(f"   Loaded {len(df)} items")
        return df
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean data"""
        print("🧹 Cleaning data...")
        initial = len(df)
        
        # Drop missing
        df = df.dropna(subset=["price", "size", "area_name", "list_id"])
        
        # Filter outliers
        df = df[
            (df["price"] >= self.config.PRICE_MIN) & 
            (df["price"] <= self.config.PRICE_MAX) &
            (df["size"] >= self.config.AREA_MIN) & 
            (df["size"] <= self.config.AREA_MAX)
        ]
        
        # Remove duplicates
        df = df.drop_duplicates(subset=["list_id"])
        
        removed = initial - len(df)
        print(f"   Kept {len(df)} items (removed {removed}, {removed/initial*100:.1f}%)")
        return df
    
    def create_bands(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create price and area bands"""
        print("🏷️  Creating feature bands...")
        
        # Price bands
        df["price_band"] = pd.cut(
            df["price"],
            bins=[0, 1e6, 2e6, 3e6, 5e6, 7e6, 10e6, float('inf')],
            labels=["<1M", "1-2M", "2-3M", "3-5M", "5-7M", "7-10M", ">10M"]
        )
        
        # Area bands
        df["area_band"] = pd.cut(
            df["size"],
            bins=[0, 15, 25, 35, 50, float('inf')],
            labels=["<15", "15-25", "25-35", "35-50", ">50"]
        )
        
        print(f"   Created {df['price_band'].nunique()} price bands")
        print(f"   Created {df['area_band'].nunique()} area bands")
        return df
    
    def encode_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical features"""
        print("🔢 Encoding features...")
        
        df["district_id"] = self.le_district.fit_transform(df["area_name"]) + 1
        df["price_id"] = self.le_price.fit_transform(df["price_band"]) + 1
        df["area_id"] = self.le_area.fit_transform(df["area_band"]) + 1
        
        print(f"   Districts: {df['district_id'].nunique()}")
        print(f"   Price bands: {df['price_id'].nunique()}")
        print(f"   Area bands: {df['area_id'].nunique()}")
        return df
    
    def save_outputs(self, df: pd.DataFrame):
        """Save processed data"""
        print("💾 Saving outputs...")
        
        # Save item features
        features = df[["list_id", "district_id", "price_id", "area_id"]].to_dict(orient="records")
        with open(self.config.ITEM_FEATURES, "w", encoding="utf-8") as f:
            json.dump(features, f, indent=2)
        print(f"   Saved {len(features)} items → {self.config.ITEM_FEATURES}")
        
        # Save mappings
        mappings = {
            "vocab_sizes": {
                "n_districts": int(df["district_id"].max() + 1),
                "n_prices": int(df["price_id"].max() + 1),
                "n_areas": int(df["area_id"].max() + 1),
                "n_ages": len(self.config.AGE_BANDS) + 1,
                "n_occupations": len(self.config.OCCUPATIONS) + 1
            },
            "district_map": {str(k): int(v) for k, v in zip(
                self.le_district.classes_, 
                self.le_district.transform(self.le_district.classes_) + 1
            )},
            "price_map": {str(k): int(v) for k, v in zip(
                self.le_price.classes_, 
                self.le_price.transform(self.le_price.classes_) + 1
            )},
            "area_map": {str(k): int(v) for k, v in zip(
                self.le_area.classes_, 
                self.le_area.transform(self.le_area.classes_) + 1
            )},
            "age_map": {age: i+1 for i, age in enumerate(self.config.AGE_BANDS)},
            "occupation_map": {occ: i+1 for i, occ in enumerate(self.config.OCCUPATIONS)}
        }
        with open(self.config.FEATURE_MAPPING, "w", encoding="utf-8") as f:
            json.dump(mappings, f, indent=2)
        print(f"   Saved mappings → {self.config.FEATURE_MAPPING}")
    
    def run(self):
        """Run preprocessing pipeline"""
        print("\n" + "="*70)
        print("🔧 STEP 1: DATA PREPROCESSING")
        print("="*70 + "\n")
        
        df = self.load_raw_data()
        df = self.clean_data(df)
        df = self.create_bands(df)
        df = self.encode_features(df)
        self.save_outputs(df)
        
        print("\n✅ Preprocessing completed!\n")
        return df