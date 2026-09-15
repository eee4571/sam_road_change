"""Experiment entry point; algorithms live in the production engine."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'code'))
from engine.width.raw_boundary_core import *
