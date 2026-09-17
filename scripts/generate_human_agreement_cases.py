import pandas as pd
from pathlib import Path

from src.agent import GroundedAIAgent


ROOT = Path(__file__).resolve().parents[1]

sample_path = ROOT / "data" / "human_agreement_30.csv"
output_path = ROOT / "data" / "human_agreement_cases.csv"


def main():
    sample = pd.read_csv(sample_path)

    agent = GroundedAIAgent()

    rows = []

    print("=" * 80)
    print("GENERATING HUMAN AGREEMENT CASES")
    print("=" * 80)

    for i, row in sample.iterrows():
        customer_text = str(row["customer_text"])

        print(f"Processing {i + 1}/30...")

        try:
            output = agent.process(customer_text)

            draft_reply = output.get("draft_reply", "")

            retrieved = output.get("retrieved_contexts", [])

            evidence = []
            for ctx in retrieved[:3]:
                evidence.append(
                    str(ctx.get("brand_response", ""))
                )

        except Exception as e:
            print(f"  ERROR: {e}")
            draft_reply = ""
            evidence = [f"ERROR: {e}"]

        rows.append({
            "case_number": i + 1,
            "tweet_id": row["tweet_id"],
            "customer_text": customer_text,
            "main_agent_reply": draft_reply,
            "retrieved_evidence_1": (
                evidence[0] if len(evidence) > 0 else ""
            ),
            "retrieved_evidence_2": (
                evidence[1] if len(evidence) > 1 else ""
            ),
            "retrieved_evidence_3": (
                evidence[2] if len(evidence) > 2 else ""
            ),

            # Human ratings — intentionally blank.
            "human_groundedness": "",
            "human_correctness": "",
            "human_tone": "",
            "human_safety": "",
            "human_notes": "",
        })

    output_df = pd.DataFrame(rows)

    output_df.to_csv(output_path, index=False)

    print()
    print("=" * 80)
    print("HUMAN AGREEMENT CASES GENERATED")
    print("=" * 80)
    print(f"Examples: {len(output_df)}")
    print(f"Output:   {output_path}")
    print()
    print("Each case contains:")
    print("  - Customer query")
    print("  - Main Agent generated reply")
    print("  - Up to 3 retrieved historical resolutions")
    print("  - Blank human rating fields")
    print()
    print("Rating scale: 1 = poor, 5 = excellent")


if __name__ == "__main__":
    main()