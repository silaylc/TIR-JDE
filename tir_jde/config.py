"""Project-wide paths and constants (cookiecutter-data-science convention).

Other modules import paths from here instead of hardcoding them, so moving
the project directory or renaming a data folder is a one-line change.
"""
from pathlib import Path

PROJ_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"
SPLITS_DIR = DATA_DIR / "splits"

MODELS_DIR = PROJ_ROOT / "models"
REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
