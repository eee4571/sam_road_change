"""Import the plugin engine without importing its pipeline or model runners."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PLUGIN = ROOT.parents[1]
sys.path.insert(0, str(PLUGIN / "code"))
