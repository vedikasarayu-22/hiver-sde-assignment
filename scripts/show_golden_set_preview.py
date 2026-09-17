import os
import sys
import pandas as pd

# Ensure root directory is on python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

from src.config import CANDIDATE_GOLDEN_SET_PATH, GOLDEN_SET_PATH

def show_preview():
    cand_df = pd.read_csv(CANDIDATE_GOLDEN_SET_PATH)
    review_df = pd.read_csv(GOLDEN_SET_PATH)

    print("=" * 80)
    print("STEP 5: GOLDEN SET WORKFLOW PREVIEW & MANUAL REVIEW INSTRUCTIONS")
    print("=" * 80)

    print(f"\n1. CANDIDATE GOLDEN SET PREVIEW (First 10 Rows of candidate_golden_set.csv):")
    print(cand_df[['tweet_id', 'candidate_intent', 'candidate_escalation', 'difficulty']].head(10).to_string(index=False))

    print(f"\n\n2. HUMAN REVIEW TEMPLATE PREVIEW (First 10 Rows of human_review.csv):")
    print(review_df[['tweet_id', 'candidate_intent', 'human_final_intent', 'candidate_escalation', 'human_final_escalation', 'human_verified']].head(10).to_string(index=False))

    print("\n" + "=" * 80)
    print("EXACT INSTRUCTIONS FOR MANUAL HUMAN REVIEW:")
    print("=" * 80)
    print("File to edit: data/human_review.csv")
    print("Columns to review and update for each of the 200 rows:")
    print("  1. 'human_final_intent': Replace 'UNVERIFIED' with the true intent category (e.g., 'device_hardware_battery', 'software_update_bugs', 'account_appleid_security', 'billing_subscriptions_refunds', 'connectivity_network_bluetooth', 'app_thirdparty_behavior', 'audio_accessory_hardware', 'general_inquiry_other').")
    print("  2. 'human_final_escalation': Replace 'UNVERIFIED' with True or False.")
    print("  3. 'human_verified': Change from False to True once reviewed.")
    print("  4. 'reviewer_notes': (Optional) Add any clarifying notes.")
    print("=" * 80)

if __name__ == "__main__":
    show_preview()
