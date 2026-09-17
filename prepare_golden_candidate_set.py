import os
import sys
import pandas as pd
import numpy as np

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

from src.config import RESOLUTION_PAIRS_PATH, RETRIEVAL_INDEX_PATH, CANDIDATE_GOLDEN_SET_PATH, GOLDEN_SET_PATH

def prepare_golden_and_index():
    if not os.path.exists(RESOLUTION_PAIRS_PATH):
        print(f"Error: {RESOLUTION_PAIRS_PATH} not found. Run explore_apple_data.py first.")
        return

    df = pd.read_csv(RESOLUTION_PAIRS_PATH)
    print(f"Total available paired resolution threads: {len(df):,}")
    
    def get_candidate_escalation(text, response):
        t_lower = str(text).lower()
        r_lower = str(response).lower()
        
        reasons = []
        if 'dm' in r_lower or 'direct message' in r_lower:
            reasons.append("Brand requested DM to collect private info")
        if any(w in t_lower for w in ['refund', 'money', 'card', 'charge', 'billing', 'bought']):
            reasons.append("Financial/Billing dispute requiring private account lookup")
        if any(w in t_lower for w in ['lock', 'locked', 'hacked', 'recover', 'security', 'password', 'apple id']):
            reasons.append("Account security/Apple ID lockout requiring verification")
        if any(w in t_lower for w in ['crack', 'damaged', 'broken', 'repair', 'hardware']):
            reasons.append("Physical hardware repair required")
            
        if reasons:
            return True, "; ".join(reasons)
        else:
            return False, "Standard technical inquiry capable of auto-handling"

    df['candidate_escalation_tuple'] = df.apply(lambda r: get_candidate_escalation(r['customer_text'], r['brand_response']), axis=1)
    df['candidate_escalation'] = df['candidate_escalation_tuple'].apply(lambda x: x[0])
    df['candidate_escalation_reason'] = df['candidate_escalation_tuple'].apply(lambda x: x[1])
    df.drop(columns=['candidate_escalation_tuple'], inplace=True)
    df.rename(columns={'primary_intent': 'candidate_intent'}, inplace=True)

    # Stratified sample of exactly 200 candidates for Golden Set
    np.random.seed(42)
    sample_size = 200
    
    intents = df['candidate_intent'].unique()
    samples = []
    
    per_intent_target = max(15, sample_size // len(intents))
    grouped = df.groupby('candidate_intent')
    for intent, group in grouped:
        n_sample = min(len(group), per_intent_target)
        samples.append(group.sample(n=n_sample, random_state=42))
        
    candidate_df = pd.concat(samples)
    
    if len(candidate_df) < sample_size:
        remaining_needed = sample_size - len(candidate_df)
        remaining_pool = df[~df['tweet_id'].isin(candidate_df['tweet_id'])]
        extra_sample = remaining_pool.sample(n=remaining_needed, random_state=42)
        candidate_df = pd.concat([candidate_df, extra_sample])
    elif len(candidate_df) > sample_size:
        candidate_df = candidate_df.sample(n=sample_size, random_state=42)
        
    print(f"\nExtracted Candidate Golden Evaluation set: {len(candidate_df)} examples.")

    # Assign difficulty levels heuristically for review assistance
    def assign_difficulty(row):
        txt = str(row['customer_text']).lower()
        if len(txt) > 140 or any(w in txt for w in ['and', 'also', 'after', 'but']):
            return "Complex"
        elif any(w in txt for w in ['how', 'why', 'error', 'locked']):
            return "Moderate"
        return "Simple"

    candidate_df['difficulty'] = candidate_df.apply(assign_difficulty, axis=1)

    # Prevent evaluation leakage: remove candidate evaluation tweets from historical retrieval index
    index_df = df[~df['tweet_id'].isin(candidate_df['tweet_id'])].copy()
    print(f"Historical Retrieval Index Size (strictly leak-free): {len(index_df):,} examples.")
    
    index_df.to_csv(RETRIEVAL_INDEX_PATH, index=False, encoding='utf-8')
    print(f"Saved leak-free retrieval index to: {RETRIEVAL_INDEX_PATH}")

    # Candidate Golden Set (preserving candidate labels)
    candidate_df.to_csv(CANDIDATE_GOLDEN_SET_PATH, index=False, encoding='utf-8')
    print(f"Saved candidate golden set to: {CANDIDATE_GOLDEN_SET_PATH}")

    # Human Review CSV (Explicitly marking unverified human ground truth fields)
    human_review_df = candidate_df[['tweet_id', 'customer_text', 'brand_response', 'candidate_intent', 'candidate_escalation', 'difficulty']].copy()
    human_review_df['human_final_intent'] = "UNVERIFIED"
    human_review_df['human_final_escalation'] = "UNVERIFIED"
    human_review_df['reviewer_notes'] = ""
    human_verified_series = pd.Series([False] * len(human_review_df), name='human_verified')
    human_review_df['human_verified'] = human_verified_series
    
    # Reorder columns as specified
    col_order = ['tweet_id', 'customer_text', 'brand_response', 'candidate_intent', 'human_final_intent', 'candidate_escalation', 'human_final_escalation', 'difficulty', 'reviewer_notes', 'human_verified']
    human_review_df = human_review_df[col_order]
    
    human_review_df.to_csv(GOLDEN_SET_PATH, index=False, encoding='utf-8')
    print(f"Saved human review template to: {GOLDEN_SET_PATH}")

if __name__ == "__main__":
    prepare_golden_and_index()
