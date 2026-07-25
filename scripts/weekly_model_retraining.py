#!/usr/bin/env python3
"""
Weekly Model Retraining Pipeline — per-signal-type models
This script is a wrapper that calls the specialist training script.
"""
import subprocess
import sys
import os

# Run the specialist training script
script_path = os.path.join(os.path.dirname(__file__), "train_specialist_models.py")
result = subprocess.run([sys.executable, script_path], capture_output=True, text=True)
print(result.stdout)
if result.stderr:
    print(result.stderr, file=sys.stderr)
sys.exit(result.returncode)