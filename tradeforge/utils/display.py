"""Shared display utilities for CLI scripts."""


def print_header(title: str):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def print_success(message: str):
    print(f"✓ {message}")


def parse_number(value: str):
    """Parse an argument as an int when possible, otherwise as a float."""
    try:
        return int(value)
    except ValueError:
        return float(value)
