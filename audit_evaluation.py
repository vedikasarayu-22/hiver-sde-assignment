import os
import sys
import pandas as pd
import numpy as np

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

from src.config import GOLDEN_SET_PATH, RETRIEVAL_INDEX_PATH, CANDIDATE_GOLDEN_SET_PATH
from src.agent import TrivialBaselineAgent, SimpleBaselineAgent, GroundedAIAgent
from src.llm_judge import LLMJudgeEvaluator

def run_audit():
    print("=" * 80)
    print("EVALUATION AUDIT SCRIPT")
    print("=" * 80)

    golden_df = pd.read_csv(GOLDEN_SET_PATH)
    retrieval_df = pd.read_csv(RETRIEVAL_INDEX_PATH)
    candidate_df = pd.read_csv(CANDIDATE_GOLDEN_SET_PATH)

    # 1. Check Data Leakage
    golden_ids = set(golden_df['tweet_id'].astype(str))
    retrieval_ids = set(retrieval_df['tweet_id'].astype(str))
    leakage = golden_ids.intersection(retrieval_ids)
    print(f"[CHECK 1] Data Leakage Check:")
    print(f"  • Golden Set count: {len(golden_ids)}")
    print(f"  • Retrieval Index count: {len(retrieval_ids):,}")
    print(f"  • Overlapping tweet IDs (Leakage): {len(leakage)} -> {'VALID (0 Leakage)' if len(leakage)==0 else 'INVALID (LEAK DETECTED)'}")

    # 2. Check Golden Set Human Verification Status
    human_verified_count = golden_df.get('human_verified', pd.Series([False]*len(golden_df))).sum()
    print(f"\n[CHECK 2] Golden Set Hand-Labelling Status:")
    print(f"  • Total examples in human_review.csv: {len(golden_df)}")
    print(f"  • Human verified flags set: {human_verified_count}")
    print(f"  • Are labels rule-generated candidate labels? YES")
    print(f"  • Status: Candidate Golden Set (Requires Human Labeling/Verification before claiming hand-labelled status)")

    # 3. Inspect Baseline 1 vs Main Agent Judge Scoring on 20 examples
    judge = LLMJudgeEvaluator()
    b1_agent = TrivialBaselineAgent()
    main_agent = GroundedAIAgent()

    print("\n" + "=" * 80)
    print("MANUAL INSPECTION OF 20 EXAMPLES — BASELINE 1 vs MAIN AGENT JUDGE SCORES")
    print("=" * 80)

    sample_20 = golden_df.head(20)
    
    b1_inspection = []
    main_inspection = []

    for idx, row in sample_20.iterrows():
        query = row['customer_text']
        ref = row['brand_response']

        # Baseline 1
        b1_out = b1_agent.process(query)
        b1_j = judge.evaluate_reply(query, b1_out['draft_reply'], ref, b1_out['retrieved_contexts'])
        b1_inspection.append({
            'query': query,
            'b1_reply': b1_out['draft_reply'],
            'groundedness': b1_j['groundedness'],
            'correctness': b1_j['correctness'],
            'tone': b1_j['tone'],
            'safety': b1_j['safety'],
            'overall': b1_j['overall_score']
        })

        # Main Agent
        main_out = main_agent.process(query)
        main_j = judge.evaluate_reply(query, main_out['draft_reply'], ref, main_out['retrieved_contexts'])
        main_inspection.append({
            'query': query,
            'main_reply': main_out['draft_reply'],
            'retrieved_top1': main_out['retrieved_contexts'][0]['brand_response'] if main_out['retrieved_contexts'] else "None",
            'groundedness': main_j['groundedness'],
            'correctness': main_j['correctness'],
            'tone': main_j['tone'],
            'safety': main_j['safety'],
            'overall': main_j['overall_score']
        })

    print("\n--- SAMPLE INSPECTION: BASELINE 1 (First 5 of 20) ---")
    for i, ex in enumerate(b1_inspection[:5]):
        print(f"\n[Ex {i+1}] Query: {ex['query'][:100]}...")
        print(f"      Baseline 1 Reply: {ex['b1_reply']}")
        print(f"      Scores -> Groundedness: {ex['groundedness']}, Correctness: {ex['correctness']}, Tone: {ex['tone']}, Safety: {ex['safety']} | OVERALL: {ex['overall']}")

    print("\n--- SAMPLE INSPECTION: MAIN AGENT (First 5 of 20) ---")
    for i, ex in enumerate(main_inspection[:5]):
        print(f"\n[Ex {i+1}] Query: {ex['query'][:100]}...")
        print(f"      Main Agent Reply: {ex['main_reply']}")
        print(f"      Retrieved Evidence: {ex['retrieved_top1'][:80]}...")
        print(f"      Scores -> Groundedness: {ex['groundedness']}, Correctness: {ex['correctness']}, Tone: {ex['tone']}, Safety: {ex['safety']} | OVERALL: {ex['overall']}")

if __name__ == "__main__":
    run_audit()
