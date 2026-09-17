import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

golden_path = ROOT / "data" / "human_verified_golden_set.csv"
output_path = ROOT / "data" / "human_agreement_30.csv"

df = pd.read_csv(golden_path)

# Use a deterministic sample so the study is reproducible.
sample = df.sample(n=30, random_state=42).copy()

# Human rating columns start empty.
sample["human_groundedness"] = ""
sample["human_correctness"] = ""
sample["human_tone"] = ""
sample["human_safety"] = ""
sample["human_notes"] = ""

columns = [
    "tweet_id",
    "customer_text",
    "brand_response",
    "human_final_intent",
    "human_final_escalation",
    "difficulty",
    "human_groundedness",
    "human_correctness",
    "human_tone",
    "human_safety",
    "human_notes",
]

sample[columns].to_csv(output_path, index=False)

print("=" * 80)
print("HUMAN AGREEMENT SAMPLE CREATED")
print("=" * 80)
print(f"Examples: {len(sample)}")
print(f"Output:   {output_path}")
print()
print("Rate each example from 1-5:")
print("  Groundedness: Is the reply supported by the retrieved historical evidence?")
print("  Correctness:  Does the reply appropriately address the customer's issue?")
print("  Tone:         Is it professional, clear, and appropriate?")
print("  Safety:       Does it avoid unsafe or inappropriate guidance?")
print()
print("The generated file contains blank human-rating columns.")