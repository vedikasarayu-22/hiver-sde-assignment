import os
import sys
import pandas as pd

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
GOLDEN_PATH = os.path.join(DATA_DIR, "human_review.csv")

def review_cli():
    if not os.path.exists(GOLDEN_PATH):
        print(f"Error: {GOLDEN_PATH} not found.")
        return

    df = pd.read_csv(GOLDEN_PATH)
    unverified = df[df['human_verified'] == False]
    
    print("=" * 70)
    print("HUMAN GOLDEN SET REVIEW HELPER CLI")
    print("=" * 70)
    print(f"Total Rows: {len(df)} | Unverified Rows: {len(unverified)}")
    print("\nInstructions:")
    print("1. Open 'data/human_review.csv' in Excel/VS Code or use this script.")
    print("2. For each row, set 'human_final_intent' to the true intent.")
    print("3. Set 'human_final_escalation' to True or False.")
    print("4. Set 'human_verified' to True after inspecting.")
    print("=" * 70)

    print("\nFirst 3 Unverified Rows Preview:")
    for idx, row in unverified.head(3).iterrows():
        print(f"\n[ID: {row['tweet_id']}] Customer: {row['customer_text'][:100]}...")
        print(f"  • Candidate Intent: {row['candidate_intent']} | Candidate Escalation: {row['candidate_escalation']}")
        print(f"  • Expected Human Action: Fill 'human_final_intent' and 'human_final_escalation'")

if __name__ == "__main__":
    review_cli()
