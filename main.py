"""
Entry point for the Student Loan Plan 2 overpay-vs-invest simulator.

Usage:
    python main.py          # launches the web app at localhost:5000
    python main.py --cli    # runs the terminal interface
"""

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Student Loan Plan 2: Overpay vs Invest Simulator",
    )
    parser.add_argument(
        "--cli",
        action="store_true",
        help="Run in terminal mode instead of launching the web app",
    )
    args = parser.parse_args()

    if args.cli:
        from cli import run_cli
        run_cli()
    else:
        from app import run_web
        run_web()


if __name__ == "__main__":
    main()
