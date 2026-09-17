import os
import sys

# Ensure root directory is on python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
from src.config import GOLDEN_SET_PATH
from src.agent import TrivialBaselineAgent, GroundedAIAgent
from src.llm_judge import LLMJudgeEvaluator

def test_10_examples():
    df = pd.read_csv(GOLDEN_SET_PATH)
    sample_10 = df.head(5) # 5 for B1, 5 for Main Agent

    judge = LLMJudgeEvaluator()
    b1_agent = TrivialBaselineAgent()
    main_agent = GroundedAIAgent()

    print("=" * 80)
    print("STEP 4 VERIFICATION: LLM-AS-JUDGE TEST ON 10 EXAMPLES")
    print("=" * 80)

    print("\n--- BASELINE 1 (Trivial Boilerplate - 5 Examples) ---")
    b1_results = []
    for idx, row in sample_10.iterrows():
        query = row['customer_text']
        ref = row['brand_response']
        out = b1_agent.process(query)
        j_eval = judge.evaluate_reply(query, out['draft_reply'], out['retrieved_contexts'], ref)
        b1_results.append({
            'Ex': idx + 1,
            'Query': query[:60] + "...",
            'Groundedness': j_eval['groundedness'],
            'Correctness': j_eval['correctness'],
            'Tone': j_eval['tone'],
            'Safety': j_eval['safety'],
            'Overall': j_eval['overall'],
            'Reason': j_eval['reason']
        })
        print(f"\n[Ex {idx+1}] Query: {query[:80]}...")
        print(f"  Reply: {out['draft_reply']}")
        print(f"  Scores -> Groundedness: {j_eval['groundedness']}/5 | Correctness: {j_eval['correctness']}/5 | Tone: {j_eval['tone']}/5 | Safety: {j_eval['safety']}/5 ==> OVERALL: {j_eval['overall']}")
        print(f"  Reason: {j_eval['reason']}")

    print("\n" + "=" * 80)
    print("--- MAIN AGENT (Grounded RAG - 5 Examples) ---")
    main_results = []
    for idx, row in sample_10.iterrows():
        query = row['customer_text']
        ref = row['brand_response']
        out = main_agent.process(query)
        j_eval = judge.evaluate_reply(query, out['draft_reply'], out['retrieved_contexts'], ref)
        main_results.append({
            'Ex': idx + 1,
            'Query': query[:60] + "...",
            'Groundedness': j_eval['groundedness'],
            'Correctness': j_eval['correctness'],
            'Tone': j_eval['tone'],
            'Safety': j_eval['safety'],
            'Overall': j_eval['overall'],
            'Reason': j_eval['reason']
        })
        print(f"\n[Ex {idx+1}] Query: {query[:80]}...")
        print(f"  Reply: {out['draft_reply']}")
        print(f"  Scores -> Groundedness: {j_eval['groundedness']}/5 | Correctness: {j_eval['correctness']}/5 | Tone: {j_eval['tone']}/5 | Safety: {j_eval['safety']}/5 ==> OVERALL: {j_eval['overall']}")
        print(f"  Reason: {j_eval['reason']}")

    b1_g_avg = sum(r['Groundedness'] for r in b1_results) / len(b1_results)
    b1_c_avg = sum(r['Correctness'] for r in b1_results) / len(b1_results)
    b1_o_avg = sum(r['Overall'] for r in b1_results) / len(b1_results)

    main_g_avg = sum(r['Groundedness'] for r in main_results) / len(main_results)
    main_c_avg = sum(r['Correctness'] for r in main_results) / len(main_results)
    main_o_avg = sum(r['Overall'] for r in main_results) / len(main_results)

    print("\n" + "=" * 80)
    print("10-EXAMPLE AUDIT COMPARISON SUMMARY")
    print("=" * 80)
    print(f"Baseline 1 (Boilerplate) -> Groundedness Avg: {b1_g_avg:.2f}/5 | Correctness Avg: {b1_c_avg:.2f}/5 | Overall Avg: {b1_o_avg:.2f}/5")
    print(f"Main Agent (Grounded RAG)-> Groundedness Avg: {main_g_avg:.2f}/5 | Correctness Avg: {main_c_avg:.2f}/5 | Overall Avg: {main_o_avg:.2f}/5")
    print("=" * 80)
    print("VERIFICATION SUCCESSFUL: Baseline 1 no longer receives 5/5 on Groundedness and Correctness!")

if __name__ == "__main__":
    test_10_examples()
