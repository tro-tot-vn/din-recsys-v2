"""
Training Module
"""
import os
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, log_loss
from typing import Dict
from config import Config


class Trainer:
    """DIN Trainer with User Profile"""
    
    def __init__(self, model, config: Config, train_loader, val_loader, test_loader):
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        
        self.device = torch.device(config.DEVICE)
        self.model.to(self.device)
        
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY
        )
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=config.LR_DECAY_STEP,
            gamma=config.LR_DECAY_FACTOR
        )
        
        self.best_auc = -1
        self.patience_counter = 0
        self.history = {"train_loss": [], "train_auc": [], "val_loss": [], "val_auc": []}
        
        os.makedirs(config.SAVE_DIR, exist_ok=True)
    
    def train_epoch(self, epoch):
        """Train one epoch"""
        self.model.train()
        losses, y_true, y_pred = [], [], []
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch:02d} [Train]")
        for batch in pbar:
            batch = [t.to(self.device) for t in batch]
            hdist, hprice, harea, mask, cdist, cprice, carea, age, occ, loc, labels = batch
            
            logits = self.model(hdist, hprice, harea, mask, cdist, cprice, carea, age, occ, loc)
            loss = self.criterion(logits, labels)
            
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.MAX_GRAD_NORM)
            self.optimizer.step()
            
            losses.append(loss.item())
            y_true.append(labels.cpu().numpy())
            y_pred.append(torch.sigmoid(logits).detach().cpu().numpy())
            
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        avg_loss = np.mean(losses)
        y_true = np.concatenate(y_true)
        y_pred = np.concatenate(y_pred)
        auc = roc_auc_score(y_true, y_pred)
        
        return avg_loss, auc
    
    @torch.no_grad()
    def evaluate(self, loader, desc="Val"):
        """Evaluate"""
        self.model.eval()
        losses, y_true, y_pred = [], [], []
        
        for batch in tqdm(loader, desc=desc):
            batch = [t.to(self.device) for t in batch]
            hdist, hprice, harea, mask, cdist, cprice, carea, age, occ, loc, labels = batch
            
            logits = self.model(hdist, hprice, harea, mask, cdist, cprice, carea, age, occ, loc)
            loss = self.criterion(logits, labels)
            
            losses.append(loss.item())
            y_true.append(labels.cpu().numpy())
            y_pred.append(torch.sigmoid(logits).cpu().numpy())
        
        avg_loss = np.mean(losses)
        y_true = np.concatenate(y_true)
        y_pred = np.concatenate(y_pred)
        auc = roc_auc_score(y_true, y_pred)
        logloss = log_loss(y_true, np.clip(y_pred, 1e-7, 1-1e-7))
        
        return avg_loss, auc, logloss
    
    def train(self):
        """Main training loop"""
        print("\n" + "="*70)
        print("🚀 STEP 3: MODEL TRAINING")
        print("="*70 + "\n")
        
        for epoch in range(1, self.config.EPOCHS + 1):
            # Train
            train_loss, train_auc = self.train_epoch(epoch)
            
            # Validate
            val_loss, val_auc, val_ll = self.evaluate(self.val_loader, f"Epoch {epoch:02d} [Val]  ")
            
            # Update LR
            self.scheduler.step()
            lr = self.optimizer.param_groups[0]['lr']
            
            # Save history
            self.history["train_loss"].append(train_loss)
            self.history["train_auc"].append(train_auc)
            self.history["val_loss"].append(val_loss)
            self.history["val_auc"].append(val_auc)
            
            # Print summary
            print(f"\nEpoch {epoch:02d} | LR: {lr:.6f}")
            print(f"  Train: Loss={train_loss:.4f}, AUC={train_auc:.4f}")
            print(f"  Val:   Loss={val_loss:.4f}, AUC={val_auc:.4f}, LogLoss={val_ll:.4f}")
            
            # Save best
            if val_auc > self.best_auc:
                self.best_auc = val_auc
                self.patience_counter = 0
                torch.save({
                    'epoch': epoch,
                    'model': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'auc': val_auc
                }, f"{self.config.SAVE_DIR}/{self.config.BEST_MODEL}")
                print(f"  💾 Saved best model (AUC: {val_auc:.4f})")
            else:
                self.patience_counter += 1
            
            # Early stopping
            if self.patience_counter >= self.config.EARLY_STOP_PATIENCE:
                print(f"\n⚠️  Early stopping at epoch {epoch}")
                break
        
        # Test
        print(f"\n{'='*70}")
        print("📊 TESTING ON TEST SET")
        print("="*70 + "\n")
        
        # Load best model
        checkpoint = torch.load(f"{self.config.SAVE_DIR}/{self.config.BEST_MODEL}")
        self.model.load_state_dict(checkpoint['model'])
        
        test_loss, test_auc, test_ll = self.evaluate(self.test_loader, "Testing")
        
        print(f"\n🎯 Test Results:")
        print(f"   AUC: {test_auc:.4f}")
        print(f"   LogLoss: {test_ll:.4f}")
        print(f"   Loss: {test_loss:.4f}")
        
        print(f"\n✅ Training completed! Best Val AUC: {self.best_auc:.4f}\n")
        
        return self.history