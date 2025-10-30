"""
Configuration for DIN Pipeline
"""
import torch


class Config:
    """Configuration for entire pipeline"""
    
    # ============= PATHS =============
    DATA_DIR = "data"
    RAW_DATA = f"{DATA_DIR}/posts_data.json"
    ITEM_FEATURES = f"{DATA_DIR}/item_features.json"
    FEATURE_MAPPING = f"{DATA_DIR}/feature_mappings.json"
    TRAINING_DATA = f"{DATA_DIR}/training_data.json"
    USER_PROFILES = f"{DATA_DIR}/user_profiles.json"
    
    SAVE_DIR = "checkpoints"
    BEST_MODEL = "din_best.pt"
    
    # ============= DATA PREPROCESSING =============
    PRICE_MIN = 1e5
    PRICE_MAX = 2e7
    AREA_MIN = 5.0
    AREA_MAX = 100.0
    
    # ============= USER PROFILE =============
    AGE_BANDS = ["18-22", "23-30", "31-45", ">45"]
    AGE_PROBS = [0.35, 0.4, 0.2, 0.05]
    OCCUPATIONS = ["student", "working"]
    OCCUPATION_PROBS = [0.6, 0.4]
    
    # ============= DATA GENERATION =============
    NUM_USERS = 5000
    MIN_HIST_LEN = 3
    MAX_HIST_LEN = 50
    NEG_POS_RATIO = 1.0
    HARD_NEG_RATIO = 0.3
    
    # ============= DATA SPLIT =============
    VAL_RATIO = 0.15
    TEST_RATIO = 0.15
    RANDOM_SEED = 42
    
    # ============= MODEL ARCHITECTURE =============
    EMBED_DIM = 64
    ATTN_HIDDEN = [80, 40]
    MLP_HIDDEN = [128, 64]
    DROPOUT = 0.2
    
    # User profile embedding dimensions
    USER_AGE_DIM = 16        # 4 age groups -> 16 dims
    USER_OCC_DIM = 8         # 2 occupations -> 8 dims
    USER_LOC_DIM = 32        # ~50 districts -> 32 dims
    USER_COMPRESSED_DIM = 16 # Final compressed user vector
    USER_REG_WEIGHT = 0.01   # L2 regularization weight
    
    # ============= TRAINING =============
    BATCH_SIZE = 256
    EPOCHS = 20
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-5
    LR_DECAY_FACTOR = 0.9
    LR_DECAY_STEP = 3
    MAX_GRAD_NORM = 5.0
    EARLY_STOP_PATIENCE = 5
    
    # ============= DEVICE =============
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4