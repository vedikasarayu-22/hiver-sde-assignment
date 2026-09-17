import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
CACHE_DIR = os.path.join(BASE_DIR, ".cache")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

# Datasets
APPLE_RAW_TWEETS_PATH = os.path.join(DATA_DIR, "applesupport_tweets.csv")
RESOLUTION_PAIRS_PATH = os.path.join(DATA_DIR, "apple_resolution_pairs.csv")
RETRIEVAL_INDEX_PATH = os.path.join(DATA_DIR, "apple_retrieval_index.csv")
GOLDEN_SET_PATH = os.path.join(DATA_DIR, "human_verified_golden_set.csv")
CANDIDATE_GOLDEN_SET_PATH = os.path.join(DATA_DIR, "candidate_golden_set.csv")
HUMAN_REVIEW_TEMPLATE_PATH = os.path.join(DATA_DIR, "human_review.csv")

# Caching
LLM_CACHE_PATH = os.path.join(CACHE_DIR, "llm_cache.json")
EMBEDDING_CACHE_PATH = os.path.join(CACHE_DIR, "embedding_cache.pkl")

# Taxonomy Categories
FINAL_TAXONOMY = [
    "device_hardware_battery",
    "software_update_bugs",
    "account_appleid_security",
    "billing_subscriptions_refunds",
    "connectivity_network_bluetooth",
    "app_thirdparty_behavior",
    "audio_accessory_hardware",
    "general_inquiry_other"
]
