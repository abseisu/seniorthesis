from pathlib import Path

# Treat the repo root as "one directory above src/" so everything can be referenced consistently.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

def p(*parts) -> Path:
    """
    Build a path relative to the project root.

    Example:
        p("data", "raw", "crsp.parquet")
    """
    return PROJECT_ROOT.joinpath(*parts)

def load_config():
    """
    Central config for pull_crsp.py. This is where date ranges and output paths live.
    """
    return {
        "crsp": {
            "start": "2000-01-01",
            "end":   "2025-09-01",
        },
        "paths": {
            "raw_monthly":    "data/raw/crsp_msf_2000_2025.parquet",
            "interim_labeled":"data/interim/crsp_monthly_labeled.parquet",
            "processed_core": "data/processed/crsp_monthly_core.parquet",
        }
    }
