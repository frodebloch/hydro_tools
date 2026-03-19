#!/usr/bin/env python3
"""Entry point for the DP Simulator Visualization.

Usage:
    python run.py --mock              # Run with synthetic demo data
    python run.py --port 9000         # Listen for live data on UDP port 9000
    python run.py --mock --fps 15     # Mock mode at 15 FPS
    python run.py --help              # Show all options
"""

from dp_sim_vis.main import main

if __name__ == "__main__":
    main()
