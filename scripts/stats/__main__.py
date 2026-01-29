"""
Module entry point for command-line execution.
Allows running: python -m stats
"""

from .cli import main

if __name__ == "__main__":
    import sys
    sys.exit(main())
