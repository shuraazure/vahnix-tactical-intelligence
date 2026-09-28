"""
Launcher script for NTRO Tactical Offline Dashboard.
Executes Streamlit web dashboard in standalone offline mode.
Authoritative Specifications: ORIGINAL_REQUEST.md § R3, PROJECT.md § 3
"""

from __future__ import annotations

import os
import subprocess
import sys


def main():
    dashboard_app = os.path.join(os.path.dirname(__file__), "src", "dashboard", "app.py")
    if not os.path.exists(dashboard_app):
        print(f"Error: Dashboard application not found at {dashboard_app}")
        sys.exit(1)

    print("=" * 70)
    print("Launching NTRO Tactical Thermal Intelligence Dashboard...")
    print("Mode: Standalone Offline Tactical Web Visualizer")
    print(f"Target: {dashboard_app}")
    print("=" * 70)

    # Set offline telemetry environment variables
    os.environ["STREAMLIT_SERVER_HEADLESS"] = "true"
    os.environ["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"

    cmd = [sys.executable, "-m", "streamlit", "run", dashboard_app, "--server.port=8501"]
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\nDashboard shutdown gracefully.")


if __name__ == "__main__":
    main()
