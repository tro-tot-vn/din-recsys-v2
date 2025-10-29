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
        
        # User profile embeddings
        self.emb_age = nn.Embedding(n_ages, config.EMBED_DIM, padding_idx=0)
        self.emb_occ = nn.Embedding(n_occupations, config.EMBED_DIM, padding_idx=0)
        self.emb_loc = nn.Embedding(n_districts, config.EMBED_DIM, padding_idx=0)
        
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
        in_dim = config.EMBED_DIM * 3  # user_interest + candidate + profile
        for h in config.MLP_HIDDEN:
            mlp_layers += [nn.Linear(in_dim, h), nn.PReLU(), nn.Dropout(config.DROPOUT)]
            in_dim = h
        mlp_layers += [nn.Linear(in_dim, 1)]
        self.mlp = nn.Sequential(*mlp_layers)
    
    def embed_item(self, dist, price, area):
        """Embed an item"""
        return self.emb_dist(dist) + self.emb_price(price) + self.emb_area(area)
    
    def embed_profile(self, age, occ, loc):
        """Embed user profile"""
        return self.emb_age(age) + self.emb_occ(occ) + self.emb_loc(loc)
    
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
        attn_scores = attn_scores.masked_fill(mask == 0, 0)
        attn_weights = torch.relu(attn_scores).unsqueeze(-1)  # [B,T,1]
        
        # Weighted sum -> user interest
        user_interest = (attn_weights * hist_emb).sum(dim=1)  # [B,D]
        
        # Prediction: concat [user_interest, candidate, profile]
        mlp_input = torch.cat([user_interest, cand_emb, profile_emb], dim=-1)  # [B,3D]
        logits = self.mlp(mlp_input).squeeze(-1)  # [B]
        
        return logits