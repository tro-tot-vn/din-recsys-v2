import torch, json, numpy as np
from model import DINModel
from config import Config

# ===============================
# 1️⃣ Load data & model
# ===============================
config = Config()

# Load item features
with open(config.ITEM_FEATURES, "r", encoding="utf-8") as f:
    items = json.load(f)
item_map = {it["list_id"]: it for it in items}

# Load user profiles
with open(config.USER_PROFILES, "r", encoding="utf-8") as f:
    profiles = json.load(f)
profile_map = {p["user_id"]: p for p in profiles}

# Load vocab sizes and mappings
with open(config.FEATURE_MAPPING, "r", encoding="utf-8") as f:
    mappings = json.load(f)
vocab_sizes = mappings["vocab_sizes"]
age_map = mappings["age_map"]
occ_map = mappings["occupation_map"]


model = DINModel(
    n_districts=vocab_sizes["n_districts"],
    n_prices=vocab_sizes["n_prices"],
    n_areas=vocab_sizes["n_areas"],
    n_ages=vocab_sizes["n_ages"],
    n_occupations=vocab_sizes["n_occupations"],
    config=config
)


checkpoint = torch.load(f"{config.SAVE_DIR}/{config.BEST_MODEL}", map_location="cpu")
model.load_state_dict(checkpoint["model"])
model.eval()

print("✅ Model loaded successfully!\n")

# ===============================
# 2️⃣ Load mappings to display real names
# ===============================
rev_district = {v: k for k, v in mappings["district_map"].items()}
rev_price    = {v: k for k, v in mappings["price_map"].items()}
rev_area     = {v: k for k, v in mappings["area_map"].items()}

def describe_post(pid):
    info = item_map.get(pid)
    if not info:
        return f"⚠️ Post {pid} not found."
    return (
        f"Post {pid:<9} | "
        f"District: {rev_district[info['district_id']]:<15} | "
        f"Price: {rev_price[info['price_id']]:<6} | "
        f"Area: {rev_area[info['area_id']]:<6}"
    )

# ===============================
# 3️⃣ Test data (example)
# ===============================
# User profile (select first user or define your own)
user_id = 0
user_profile = profile_map[user_id]

history_post_ids = [      128010560,
      128038844,
      127089178,
      127849962,
      128117946]
candidate_post_id = 128171995

for pid in history_post_ids + [candidate_post_id]:
    if pid not in item_map:
        raise KeyError(f"❌ Post ID {pid} not found in item_features.json!")

print("===================================")
print("🧩  DIN PREDICTION SAMPLE (Manual IDs)")
print("===================================")
print("=== User Profile ===")
print(f"User ID: {user_profile['user_id']}")
print(f"Age: {user_profile['age_band']}, Occupation: {user_profile['occupation']}")
print(f"Location: {rev_district[user_profile['user_location_id']]}")

print("\n=== User history ===")
for pid in history_post_ids:
    print(describe_post(pid))
print("\n=== Candidate ===")
print(describe_post(candidate_post_id))
print("===================================")

# ===============================
# 4️⃣ Prepare input tensors
# ===============================
def pad_seq(seq, max_len):
    seq = seq[-max_len:]
    pad_len = max_len - len(seq)
    return seq + [0]*pad_len, [1]*len(seq) + [0]*pad_len

MAX_T = 50

hist = [pid for pid in history_post_ids if pid in item_map]
hdist  = [item_map[p]["district_id"] for p in hist]
hprice = [item_map[p]["price_id"] for p in hist]
harea  = [item_map[p]["area_id"] for p in hist]
hdist, mask = pad_seq(hdist, MAX_T)
hprice,_ = pad_seq(hprice, MAX_T)
harea,_  = pad_seq(harea, MAX_T)

hdist  = torch.tensor([hdist])
hprice = torch.tensor([hprice])
harea  = torch.tensor([harea])
mask   = torch.tensor([mask], dtype=torch.float32)

c = item_map[candidate_post_id]
cdist  = torch.tensor([c["district_id"]])
cprice = torch.tensor([c["price_id"]])
carea  = torch.tensor([c["area_id"]])

# User profile features
age_id = age_map[user_profile["age_band"]]
occ_id = occ_map[user_profile["occupation"]]
loc_id = user_profile["user_location_id"]

age = torch.tensor([age_id])
occ = torch.tensor([occ_id])
loc = torch.tensor([loc_id])

# ===============================
# 5️⃣ Predict CTR
# ===============================
with torch.no_grad():
    logit, _ = model(hdist, hprice, harea, mask, cdist, cprice, carea, age, occ, loc)
    prob = torch.sigmoid(logit).item()

print(f"\nPredicted click probability: {prob:.3f}")

# ===============================
# 6️⃣ Print attention weights
# ===============================
with torch.no_grad():
    # Get embeddings of history and candidate
    hist_e = model.embed_item(hdist, hprice, harea)
    cand_e = model.embed_item(cdist, cprice, carea)
    B, T, D = hist_e.shape
    cand_expand = cand_e.unsqueeze(1).expand(-1, T, -1)
    att_in = torch.cat([hist_e, cand_expand, hist_e * cand_expand], dim=-1)
    w = model.attention(att_in).squeeze(-1)
    w = w.masked_fill(mask == 0, -1e9)
    w = torch.relu(w).squeeze(0).cpu().numpy()


print("\n=== Attention Weights ===")
for weight, pid in zip(w[:len(history_post_ids)], history_post_ids):
    print(f"[{weight:.2f}]  {describe_post(pid)}")
print("===================================")