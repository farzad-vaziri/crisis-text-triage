"""Shared settings for the crisis-text-triage pipeline."""
from pathlib import Path
import hashlib

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"          # git-ignored: real post text lives only here
OUT = ROOT / "data"                  # committed: aggregates and predictions without text
MODELS = ROOT / "models"             # git-ignored: model weights are never published
SEED = 42
CLASSES = ["anxiety", "depression", "lonely", "relationships", "suicidewatch"]
HIGH_RISK = "suicidewatch"
PER_CLASS_2019 = 6000
EVAL_PER_CLASS = 80
VAL_PER_CLASS = 50
ZENODO = "https://zenodo.org/records/3941387/files/{f}?download=1"


def row_id(text: str) -> str:
    """Stable, non-reversible key for a post (so predictions carry no text)."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
