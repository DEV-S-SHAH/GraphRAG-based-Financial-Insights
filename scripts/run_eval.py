"""
Benchmark Evaluation CLI.

Usage:
    python scripts/run_eval.py --max-questions 3
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.benchmark import FinancialBenchmark


def main():
    parser = argparse.ArgumentParser(description="Run Financial GraphRAG benchmark.")
    parser.add_argument("--max-questions", type=int, default=3, help="Max questions to evaluate")
    args = parser.parse_args()

    bm = FinancialBenchmark()
    bm.run_benchmark(max_questions=args.max_questions)


if __name__ == "__main__":
    main()
