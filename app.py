"""
VahniX Platform - Root Entrypoint for Streamlit Community Cloud & Local Deployment.
Authoritative launcher ensuring modular imports and cloud compatibility.
"""

from __future__ import annotations

import os
import sys

# Ensure repository root is on sys.path
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# Delegate execution to the main VahniX dashboard application
import runpy

app_path = os.path.join(ROOT_DIR, "src", "dashboard", "app.py")
runpy.run_path(app_path, run_name="__main__")
