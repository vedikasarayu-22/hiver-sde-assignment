import pandas as pd
import numpy as np

from src.llm_judge import LLMJudgeEvaluator


INPUT = "data/human_agreement_cases.csv"
OUTPUT = "data/human_llm_agreement_results.csv"
SUMMARY = "data/human_llm_agreement_summary.csv"


def main():
    # Load human review cases
    df = pd.read_csv(INPUT)

    # Remove blank rows accidentally added through Excel
    df = df[df["case_number"].notna()].copy()
    df = df.head(30).copy()

    print("=" * 80)
    print("HUMAN VS LLM-AS-JUDGE AGREEMENT")
    print("=" * 80)
    print(f"Cases to evaluate: {len(df)}")
    print()

    judge = LLMJudgeEvaluator()

    results = []

    for idx, row in df.iterrows():

        case_num = int(row["case_number"])

        customer_text = str(row["customer_text"])
        draft_reply = str(row["main_agent_reply"])

        retrieved_contexts = []

        # Collect the retrieved historical evidence.
        for i in range(1, 4):

            response_col = f"retrieved_{i}_response"
            score_col = f"retrieved_{i}_score"

            if response_col in df.columns:

                response_text = row.get(response_col, "")

                if pd.notna(response_text) and str(response_text).strip():

                    score = row.get(score_col, 0)

                    try:
                        score = float(score)
                    except (ValueError, TypeError):
                        score = 0.0

                    retrieved_contexts.append(
                        {
                            "brand_response": str(response_text),
                            "similarity_score": score
                        }
                    )

        print(f"Judging case {case_num}/{len(df)}...")

        # IMPORTANT:
        # reference_reply=None keeps the LLM judge blind to the
        # historical/gold reference answer.
        judge_result = judge.evaluate_reply(
            customer_text,
            draft_reply,
            retrieved_contexts,
            None
        )

        results.append(
            {
                "case_number": case_num,
                "customer_text": customer_text,
                "main_agent_reply": draft_reply,

                "human_groundedness": row.get(
                    "human_groundedness",
                    np.nan
                ),
                "human_correctness": row.get(
                    "human_correctness",
                    np.nan
                ),
                "human_tone": row.get(
                    "human_tone",
                    np.nan
                ),
                "human_safety": row.get(
                    "human_safety",
                    np.nan
                ),

                "llm_groundedness": judge_result.get(
                    "groundedness",
                    np.nan
                ),
                "llm_correctness": judge_result.get(
                    "correctness",
                    np.nan
                ),
                "llm_tone": judge_result.get(
                    "tone",
                    np.nan
                ),
                "llm_safety": judge_result.get(
                    "safety",
                    np.nan
                ),
                "llm_overall": judge_result.get(
                    "overall",
                    np.nan
                ),
                "llm_reason": judge_result.get(
                    "reason",
                    ""
                )
            }
        )

    result_df = pd.DataFrame(results)

    result_df.to_csv(
        OUTPUT,
        index=False
    )

    # ---------------------------------------------------------
    # Calculate agreement
    # ---------------------------------------------------------

    dimensions = [
        "groundedness",
        "correctness",
        "tone",
        "safety"
    ]

    summary_rows = []

    print()
    print("=" * 80)
    print("AGREEMENT RESULTS")
    print("=" * 80)

    for dimension in dimensions:

        human_col = f"human_{dimension}"
        llm_col = f"llm_{dimension}"

        valid = result_df[
            result_df[human_col].notna()
            & result_df[llm_col].notna()
        ].copy()

        if len(valid) == 0:
            print(
                f"{dimension.capitalize()}: "
                "No valid ratings"
            )
            continue

        human = valid[human_col].astype(float)
        llm = valid[llm_col].astype(float)

        # Exact agreement
        exact_agreement = (
            human == llm
        ).mean()

        # Agreement within one point
        within_one = (
            (human - llm).abs() <= 1
        ).mean()

        mean_abs_error = (
            (human - llm).abs()
        ).mean()

        print(
            f"{dimension.capitalize():15s} "
            f"Exact: {exact_agreement:.1%} | "
            f"Within ±1: {within_one:.1%} | "
            f"MAE: {mean_abs_error:.2f}"
        )

        summary_rows.append(
            {
                "dimension": dimension,
                "cases": len(valid),
                "exact_agreement": round(
                    float(exact_agreement),
                    4
                ),
                "within_1_agreement": round(
                    float(within_one),
                    4
                ),
                "mean_absolute_error": round(
                    float(mean_abs_error),
                    4
                )
            }
        )

    summary_df = pd.DataFrame(summary_rows)

    summary_df.to_csv(
        SUMMARY,
        index=False
    )

    print()
    print("=" * 80)
    print("FILES WRITTEN")
    print("=" * 80)
    print(f"Results : {OUTPUT}")
    print(f"Summary : {SUMMARY}")
    print("=" * 80)


if __name__ == "__main__":
    main()