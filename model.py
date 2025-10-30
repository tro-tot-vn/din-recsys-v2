"""
DIN Model Architecture
"""
import torch
import torch.nn as nn
from config import Config


class DINModel(nn.Module):
    """Deep Interest Network with User Profile Features"""
    
    def __init__(self, n_districts, n_prices, n_areas, n_ages, n_occupations, config: Config):
        super().__init__()
        
        # Item embeddings
        self.emb_dist = nn.Embedding(n_districts, config.EMBED_DIM, padding_idx=0)
        self.emb_price = nn.Embedding(n_prices, config.EMBED_DIM, padding_idx=0)
        self.emb_area = nn.Embedding(n_areas, config.EMBED_DIM, padding_idx=0)
        
        # User profile embeddings (smaller dimensions)
        self.emb_age = nn.Embedding(n_ages, config.USER_AGE_DIM, padding_idx=0)
        self.emb_occ = nn.Embedding(n_occupations, config.USER_OCC_DIM, padding_idx=0)
        self.emb_loc = nn.Embedding(n_districts, config.USER_LOC_DIM, padding_idx=0)
        
        # User profile compression MLP
        self.user_mlp = nn.Sequential(
            nn.Linear(config.USER_AGE_DIM + config.USER_OCC_DIM + config.USER_LOC_DIM, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, config.USER_COMPRESSED_DIM)
        )
        
        # Attention network
        attn_layers = []
        in_dim = config.EMBED_DIM * 3
        for h in config.ATTN_HIDDEN:
            attn_layers += [nn.Linear(in_dim, h), nn.PReLU(), nn.Dropout(config.DROPOUT)]
            in_dim = h
        attn_layers += [nn.Linear(in_dim, 1)]
        self.attention = nn.Sequential(*attn_layers)
        
        # MLP for final prediction
        mlp_layers = []
        in_dim = config.EMBED_DIM * 2 + config.USER_COMPRESSED_DIM  # user_interest(64) + candidate(64) + profile(16) = 144
        for h in config.MLP_HIDDEN:
            mlp_layers += [nn.Linear(in_dim, h), nn.PReLU(), nn.Dropout(config.DROPOUT)]
            in_dim = h
        mlp_layers += [nn.Linear(in_dim, 1)]
        self.mlp = nn.Sequential(*mlp_layers)
    
    def embed_item(self, dist, price, area):
        """Embed an item"""
        return self.emb_dist(dist) + self.emb_price(price) + self.emb_area(area)
    
    def embed_profile(self, age, occ, loc):
        """Embed user profile with concat + MLP compression"""
        age_emb = self.emb_age(age)      # [B, 16]
        occ_emb = self.emb_occ(occ)      # [B, 8]
        loc_emb = self.emb_loc(loc)      # [B, 32]
        
        # Concat and compress
        profile_concat = torch.cat([age_emb, occ_emb, loc_emb], dim=-1)  # [B, 56]
        profile_compressed = self.user_mlp(profile_concat)  # [B, 16]
        
        return profile_compressed
    
    def get_user_reg_loss(self):
        """L2 regularization on user embeddings"""
        return (
            torch.sum(self.emb_age.weight ** 2) +
            torch.sum(self.emb_occ.weight ** 2) +
            torch.sum(self.emb_loc.weight ** 2)
        )
    
    def forward(self, hist_dist, hist_price, hist_area, mask, 
                cand_dist, cand_price, cand_area, age, occ, loc):
        """
        Args:
            hist_*: [B, T]
            mask: [B, T]
            cand_*: [B]
            age, occ, loc: [B]
        Returns:
            logits: [B]
        """
        B, T = hist_dist.shape
        
        # Embed history & candidate
        hist_emb = self.embed_item(hist_dist, hist_price, hist_area)  # [B,T,D]
        cand_emb = self.embed_item(cand_dist, cand_price, cand_area)  # [B,D]
        
        # Embed user profile
        profile_emb = self.embed_profile(age, occ, loc)  # [B,D]
        
        # Attention
        cand_expand = cand_emb.unsqueeze(1).expand(-1, T, -1)  # [B,T,D]
        attn_input = torch.cat([hist_emb, cand_expand, hist_emb * cand_expand], dim=-1)  # [B,T,3D]
        attn_scores = self.attention(attn_input).squeeze(-1)  # [B,T]
        attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        attn_weights = torch.relu(attn_scores).unsqueeze(-1)  # [B,T,1]
        
        # Weighted sum -> user interest
        user_interest = (attn_weights * hist_emb).sum(dim=1)  # [B,D]
        
        # Prediction: concat [user_interest, candidate, profile]
        mlp_input = torch.cat([user_interest, cand_emb, profile_emb], dim=-1)  # [B,3D]
        logits = self.mlp(mlp_input).squeeze(-1)  # [B]
        
        # Regularization loss
        reg_loss = self.get_user_reg_loss()
        
        return logits, reg_loss