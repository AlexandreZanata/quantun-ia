"""
Experiment template.
Copy this folder and adapt run.py, hypothesis.md, and results.md.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.training.structured_log import init_correlation_id, log_event

if __name__ == "__main__":
    init_correlation_id()
    log_event("warning", "template experiment — write hypothesis.md before running")
