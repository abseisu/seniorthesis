# An Information-Theoretic Approach to Portfolio Diversification

This repository contains the complete codebase and Jupyter notebooks used to produce the empirical results in my senior thesis, *An Information-Theoretic Approach to Portfolio Diversification*. The primary objective of this repository is full reproducibility for researchers with appropriate data access.

---

## Repository Structure
```
.
├── README.md
├── online_references/
│   └── (references accessible online)
└── code/
    ├── config/
    │   └── data.yaml
    ├── scripts/
    │   └── pull_crsp.py
    ├── src/
    │   ├── paths.py
    │   ├── io_utils.py
    │   └── data_loading.py
    └── notebooks/
        ├── npeet/
        │   ├── npeet_properties.ipynb
        │   ├── npeet_benchmarking.ipynb
        │   └── npeet_tester.ipynb
        └── weekly_analysis/
            ├── ksg_tools.py
            ├── weekly_processing.ipynb
            ├── weekly_computation.ipynb
            ├── weekly_visualization.ipynb
            ├── Online_Learning_GMRP.ipynb
            ├── Experiment 1 - Kernel Comparison.ipynb
            ├── Experiment 2 - Frontiers.ipynb
            ├── Experiment 3 - Out-of-Sample Performance.ipynb
            ├── misc/
                ├── General Results.ipynb
                ├── weekly_data_preview.ipynb 
            └── robustness/
                ├── Estimation Robustness.ipynb
                ├── PSD_Ridge Robustness.ipynb
                └── General Robustness Checks.ipynb

```

All executable logic resides inside the `code/` directory.

---

## Data Requirements (Critical)

> **Important:** This project requires CRSP data accessed via WRDS. CRSP data is proprietary and not included in this repository. To obtain an educational CRSP license, I worked with Yale Library Services, which maintains a licensing partnership with the University of Pennsylvania. Researchers affiliated with an academic institution should be able to pursue access through their own university libraries in a similar manner.

You must be able to:

- Authenticate using `wrds.Connection()`
- Query CRSP tables (monthly and daily stock files, names, delistings)

**Without WRDS/CRSP access, the pipeline cannot be executed.**

---

## Python Environment

There is no pinned environment file. All required packages are imported explicitly within scripts and notebooks.

### Required Packages

- numpy
- pandas
- scipy
- matplotlib
- seaborn
- scikit-learn
- cvxpy
- wrds
- psycopg2
- pyarrow
- polars
- joblib
- tqdm

### NPEET (Required)

Mutual information estimation uses the KSG estimator from NPEET. Install via:
```bash
pip install git+https://github.com/gregversteeg/NPEET.git
```

---

## Path Configuration

Several notebooks contain absolute paths from the original execution environment. **You must update these paths before running anything.**

Each notebook typically defines paths such as:
```python
DATA = Path(".../data")
PROCESSED = DATA / "processed"
ANALYSIS = DATA / "analysis"
```

Choose a single project root on your machine and make all paths consistent.

> **Important:** `code/src/paths.py` defines `PROJECT_ROOT` relative to the `code/` directory. You must either:
> 
> - Place the `data/` directory inside `code/`, or
> - Modify `paths.py` so that `PROJECT_ROOT` points to the repository root.

---

## CRSP Data Construction

### Entry Script
```
code/scripts/pull_crsp.py
```

This script pulls CRSP data, applies cleaning and filtering rules, and writes parquet files using helpers in `code/src/`.

### Universe Design (Fixed)

- Top 1,000 equities by market equity
- Bottom 20% by dollar volume removed
- Minimum price: $5
- Minimum history: 12 months
- Monthly universe snapshots, forward-filled to weeks
- Weekly data anchored on Fridays (`W-FRI`)

### Training and Validation Windows

| Period     | Dates                    |
|------------|--------------------------|
| Training   | 1997-01-01 → 2018-06-29  |
| Validation | 2018-06-30 → 2024-12-30  |

Weekly files use exclusive end-date logic internally.

### Required CRSP Pull Commands

**Weekly — Training**
```bash
python code/scripts/pull_crsp.py \
  --freq weekly \
  --start 1997-01-01 \
  --end 2018-06-30 \
  --top-n 1000 \
  --liq-pctl 0.20 \
  --universe-rebalance monthly \
  --week-anchor W-FRI
```

**Weekly — Validation**
```bash
python code/scripts/pull_crsp.py \
  --freq weekly \
  --start 2018-06-30 \
  --end 2024-12-31 \
  --top-n 1000 \
  --liq-pctl 0.20 \
  --universe-rebalance monthly \
  --week-anchor W-FRI
```

**Monthly (for NPEET notebooks)**
```bash
python code/scripts/pull_crsp.py \
  --freq monthly \
  --start 1997-01-01 \
  --end 2025-01-01
```

---

## Expected Data Layout
```
data/
├── raw/
├── interim/
├── processed/
│   ├── crsp_monthly_core.parquet
│   ├── crsp_weekly_core_1997-01-01_to_2018-06-29.parquet
│   ├── crsp_weekly_core_2018-06-30_to_2024-12-30.parquet
│   ├── crsp_weekly_universe_1997-01-01_to_2018-06-29_top1000_liq20p_rebalance-monthly_W-FRI.parquet
│   └── crsp_weekly_universe_2018-06-30_to_2024-12-30_top1000_liq20p_rebalance-monthly_W-FRI.parquet
└── analysis/
```

---

## Notebook Execution Order

1. `weekly_processing.ipynb` — MI / NMI kernel construction
2. `weekly_computation.ipynb` — Kernel stabilization and GMRP weight computation
3. `weekly_visualization.ipynb` — Diagnostics and plots
4. `Online_Learning_GMRP.ipynb` — Rolling GMRP
5. `Experiment 1 - Kernel Comparison.ipynb`
6. `Experiment 2 - Frontiers.ipynb`
7. `Experiment 3 - Out-of-Sample Performance.ipynb`
8. Robustness notebooks
9. NPEET notebooks (validation only)

---

## Reproducibility Notes

- No simulated or mocked data
- All MI / NMI estimates computed directly from CRSP returns
- All kernels and portfolio weights written to disk
- All figures and tables correspond directly to notebook outputs

---

## Final Note

This repository is designed to be readable, auditable, and fully reproducible. With valid CRSP access and correct path configuration, all thesis results can be regenerated exactly.
