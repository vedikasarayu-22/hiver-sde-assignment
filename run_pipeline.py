import os
import sys
import pandas as pd

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

from src.evaluator import EvaluationHarness

def main():
    print("=" * 80)
    print("HIVER SDE INTERN TAKE-HOME ASSIGNMENT — EVALUATION PIPELINE")
    print("Brand Target: @AppleSupport | Dataset Subset: Twitter Customer Support")
    print("=" * 80)

    harness = EvaluationHarness()
    benchmark_df = harness.run_full_benchmark()

    print("\n" + "=" * 80)
    print("FINAL BENCHMARK COMPARISON TABLE")
    print("=" * 80)
    print(benchmark_df.to_string(index=False))

    results_csv = os.path.join(os.path.dirname(__file__), "data", "benchmark_results.csv")
    benchmark_df.to_csv(results_csv, index=False)
    print(f"\nSaved benchmark results to: {results_csv}")

if __name__ == "__main__":
    main()
