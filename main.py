"""
Main Pipeline for DIN Room Rental Recommendation

Usage:
    python main.py --mode all
    python main.py --mode preprocess
    python main.py --mode generate
    python main.py --mode train
"""

import os
import json
import argparse
import numpy as np
import torch
import warnings
warnings.filterwarnings('ignore')

from config import Config
from preprocessing import DataPreprocessor
from data_generation import DataGenerator
from dataset import create_dataloaders
from model import DINModel
from trainer import Trainer
from utils import plot_training_curves


def main(mode: str = "all"):
    """
    Main pipeline
    
    Args:
        mode: 'preprocess' | 'generate' | 'train' | 'all'
    """
    config = Config()
    
    # Set random seeds
    np.random.seed(config.RANDOM_SEED)
    torch.manual_seed(config.RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.RANDOM_SEED)
    
    print("\n" + "🎯 " + "="*66)
    print("🎯 DIN MODEL WITH USER PROFILE FOR ROOM RENTAL RECOMMENDATION")
    print("🎯 " + "="*66 + "\n")
    
    # Step 1: Preprocessing
    if mode in ["preprocess", "all"]:
        preprocessor = DataPreprocessor(config)
        preprocessor.run()
    
    # Step 2: Data Generation
    if mode in ["generate", "all"]:
        generator = DataGenerator(config)
        generator.run()
    
    # Step 3: Training
    if mode in ["train", "all"]:
        # Load vocab sizes
        with open(config.FEATURE_MAPPING, "r") as f:
            mappings = json.load(f)
        vocab_sizes = mappings["vocab_sizes"]
        
        # Create dataloaders
        print("\n" + "="*70)
        print("📦 PREPARING DATALOADERS")
        print("="*70 + "\n")
        train_loader, val_loader, test_loader = create_dataloaders(config)
        
        # Create model
        print("\n" + "="*70)
        print("🏗️  BUILDING MODEL WITH USER PROFILE")
        print("="*70 + "\n")
        model = DINModel(
            n_districts=vocab_sizes['n_districts'],
            n_prices=vocab_sizes['n_prices'],
            n_areas=vocab_sizes['n_areas'],
            n_ages=vocab_sizes['n_ages'],
            n_occupations=vocab_sizes['n_occupations'],
            config=config
        )
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"✅ Model created with {n_params:,} parameters")
        print(f"   Item embeddings: districts={vocab_sizes['n_districts']}, "
              f"prices={vocab_sizes['n_prices']}, areas={vocab_sizes['n_areas']}")
        print(f"   Profile embeddings: ages={vocab_sizes['n_ages']}, "
              f"occupations={vocab_sizes['n_occupations']}")
        print(f"   Embed dim: {config.EMBED_DIM}")
        print(f"   Attention hidden: {config.ATTN_HIDDEN}")
        print(f"   MLP hidden: {config.MLP_HIDDEN}")
        print(f"   Device: {config.DEVICE}")
        
        # Train
        trainer = Trainer(model, config, train_loader, val_loader, test_loader)
        history = trainer.train()
        
        # Plot curves
        plot_training_curves(history)
        
        # Save history
        with open(f"{config.SAVE_DIR}/history.json", "w") as f:
            json.dump(history, f, indent=2)
        print(f"💾 Saved training history → {config.SAVE_DIR}/history.json")
    
    print("\n" + "🎉 " + "="*66)
    print("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
    print("🎉 " + "="*66 + "\n")
    
    # Print summary
    if mode in ["train", "all"]:
        print("📁 Generated Files:")
        print(f"   - {config.ITEM_FEATURES}")
        print(f"   - {config.FEATURE_MAPPING}")
        print(f"   - {config.TRAINING_DATA}")
        print(f"   - {config.USER_PROFILES}")
        print(f"   - {config.SAVE_DIR}/{config.BEST_MODEL}")
        print(f"   - {config.SAVE_DIR}/history.json")
        print(f"   - training_curves.png")
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DIN Pipeline with User Profile")
    parser.add_argument(
        "--mode",
        type=str,
        default="all",
        choices=["preprocess", "generate", "train", "all"],
        help="Pipeline mode"
    )
    args = parser.parse_args()
    
    main(mode=args.mode)