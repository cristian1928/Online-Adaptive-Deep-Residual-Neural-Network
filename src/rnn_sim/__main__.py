"""Main entry point for the RNN simulation package."""

import json
import sys
from typing import Any, Dict

from .apps.main import run_simulation


def main() -> None:
    """Main entry point for the RNN simulation."""
    if len(sys.argv) < 2:
        print("Usage: python -m rnn_sim.apps.main --config <config_file>")
        sys.exit(1)

    # Parse simple arguments
    config_path = "config.json"
    for i, arg in enumerate(sys.argv):
        if arg == "--config" and i + 1 < len(sys.argv):
            config_path = sys.argv[i + 1]
            break

    try:
        with open(config_path, "r") as config_file:
            config: Dict[str, Any] = json.load(config_file)
        run_simulation(config)
    except FileNotFoundError:
        print(f"Error: Configuration file '{config_path}' not found.")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in configuration file '{config_path}'.")
        sys.exit(1)


if __name__ == "__main__":
    main()
