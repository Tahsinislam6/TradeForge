import argparse
from tradeforge.cli import optimize, analyze

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tf", description="TradeForge CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)
    optimize.register(subparsers)
    analyze.register(subparsers)
    return parser

def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)