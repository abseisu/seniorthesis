from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Tuple

import pandas as pd
from pandas.tseries.offsets import MonthEnd

# Add src/ to sys.path so this can be run as a script from the repo root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "src"))

from paths import load_config, p
from io_utils import save_parquet
from data_loading import (
    connect_wrds,
    # Monthly
    load_crsp_monthly_base,
    load_crsp_msenames,
    load_crsp_stocknames,
    load_crsp_msedelist,
    attach_msenames,
    attach_stocknames,
    attach_dlret_and_total_return,
    filter_equities,
    add_eligibility_flags,
    universe_top_n,
    # Weekly (fast path, chunked with ETA)
    build_crsp_weekly_panel,
)

# Small helpers for CLI parsing and deterministic output naming.

def _parse_exchanges(s: str | None) -> Tuple[int, ...]:
    # Expected format is a comma-separated list like "1,2,3".
    if not s:
        return (1, 2, 3)
    try:
        return tuple(sorted({int(x.strip()) for x in s.split(",") if x.strip()}))
    except Exception as e:
        raise argparse.ArgumentError(None, f"Could not parse --exchanges '{s}': {e}")

def _cfg_path(cfg: dict, key: str, default_rel: str) -> Path:
    # Grab a path from config if present; otherwise fall back to a reasonable default.
    try:
        return p(cfg["paths"][key])
    except Exception:
        return p(default_rel)

def _safe_date_tag(start: str | None, end: str | None) -> str:
    """
    Build a compact date tag for filenames.

    end is treated as exclusive, so we label through (end - 1 day) when end is provided.
    """
    s = pd.to_datetime(start).date().isoformat() if start else "min"
    if end:
        e = (pd.to_datetime(end) - pd.Timedelta(days=1)).date().isoformat()
    else:
        e = "latest"
    return f"{s}_to_{e}"

def _universe_filename(
    freq: str,
    start: str | None,
    end: str | None,
    top_n: int,
    liq_pctl: float,
    extra: str | None = None,
) -> str:
    """
    Universe tables are parameter-sensitive (date range, top_n, liquidity cut, etc.),
    so the filename bakes those in to avoid accidental overwrites.
    """
    pctl = int(round(liq_pctl * 100))
    date_tag = _safe_date_tag(start, end)
    parts = [f"crsp_{freq}_universe", date_tag, f"top{top_n}", f"liq{pctl}p"]
    if extra:
        parts.append(extra)
    return "_".join(parts) + ".parquet"


# Universe construction helpers. These are intentionally tiny so the selection logic is readable.

def _one_class_per_firm(g: pd.DataFrame) -> pd.DataFrame:
    # Prefer the biggest market-cap share class within PERMCO.
    return g.sort_values(["permco", "me"], ascending=[True, False]).drop_duplicates("permco")

def _liquidity_cut(g: pd.DataFrame, p: float) -> pd.DataFrame:
    # Drop the bottom p fraction by dollar volume.
    cutoff = g["dollar_vol"].quantile(p)
    return g[g["dollar_vol"] >= cutoff]

def _topn_by_me(g: pd.DataFrame, n: int) -> pd.DataFrame:
    # Final ranking step by market cap (me).
    return g.nlargest(n, "me")

def _weekly_to_month_end(dts: pd.Series) -> pd.Series:
    # Map each weekly date to its corresponding month-end for grouping.
    return (pd.to_datetime(dts) + MonthEnd(0)).dt.normalize()
  
 # rebalancing "weekly", "monthly", "annual"
def build_universe_from_snapshots(
    panel: pd.DataFrame,
    top_n: int,
    liq_pctl: float,
    rebalance: str = "monthly",
    annual_month: int = 6,
) -> pd.DataFrame:
    """
    Build a universe membership flag off rebalance snapshots, then forward-fill that snapshot
    membership across weekly dates.

    Expects weekly columns: permno, permco, date, me, dollar_vol, ...
    Returns: ['permno','date','in_universe']
    """
    df = panel.copy().sort_values(["date", "permno"])

    # Weekly rebalance is straightforward: select membership independently each week.
    if rebalance == "weekly":
        def _weekly_xs(g):
            base = _one_class_per_firm(g)
            base = _liquidity_cut(base, liq_pctl)
            return _topn_by_me(base, top_n)

        snaps = df.groupby("date", observed=True, group_keys=False).apply(_weekly_xs)
        snaps = snaps[["permno", "date"]].assign(in_universe=True)
        out = df[["date"]].drop_duplicates().merge(snaps, on="date", how="left")
        out["in_universe"] = out["in_universe"].fillna(False)
        return out

    # For monthly/annual, pick a snapshot date and then carry that snapshot forward.
    df["date"] = pd.to_datetime(df["date"])
    df["month_end"] = _weekly_to_month_end(df["date"])

    # Snapshot date is the last weekly observation inside each month (or month matching annual_month).
    if rebalance == "monthly":
        snap_dates = df.groupby("month_end", observed=True)["date"].max().reset_index(name="snap_date")
    elif rebalance == "annual":
        months = df[df["month_end"].dt.month == annual_month]
        snap_dates = months.groupby("month_end", observed=True)["date"].max().reset_index(name="snap_date")
    else:
        raise ValueError("rebalance must be one of {'weekly','monthly','annual'}")

    # Identify rows that correspond exactly to snapshot dates.
    df = df.merge(snap_dates, left_on="date", right_on="snap_date", how="left", indicator=True)
    snaps_df = df[df["_merge"].eq("both")].drop(columns=["_merge"])

    def _xs(g: pd.DataFrame) -> pd.DataFrame:
        base = _one_class_per_firm(g)
        base = _liquidity_cut(base, liq_pctl)
        keep = _topn_by_me(base, top_n)
        return keep[["permno", "date"]]

    # Membership list at each snapshot date.
    members = (
        snaps_df.groupby("date", observed=True, group_keys=False)
        .apply(_xs)
        .reset_index(drop=True)
        .assign(in_universe=True)
        .rename(columns={"date": "snap_date"})
    )

    # For each weekly date, find the most recent snapshot date (merge_asof does the time alignment).
    snaplist = snap_dates["snap_date"].sort_values().unique()
    weekly_dates = df[["date"]].drop_duplicates().sort_values("date")
    ff = pd.merge_asof(
        weekly_dates.sort_values("date"),
        pd.DataFrame({"snap_date": pd.to_datetime(snaplist)}).sort_values("snap_date"),
        left_on="date",
        right_on="snap_date",
        direction="backward",
    )

    out = ff.merge(members, on="snap_date", how="left")[["date", "permno", "in_universe"]]
    out["in_universe"] = out["in_universe"].fillna(False)
    return out


# Monthly pipeline: pull msf, attach delistings + codes/names, compute eligibility, and build the monthly universe.

def run_monthly(
    db,
    start: str,
    end: str,
    include_adrs: bool,
    exchanges: Tuple[int, ...],
    top_n: int,
    liq_pctl: float,
    price_min: float,
    min_hist_months: int,
    liq_lookback: int,
    raw_path: Path,
    interim_path: Path,
    processed_path: Path,
    universe_path: Path,
):
    print(f"[INFO] Pulling CRSP monthly msf ({start} → {end})…")
    msf = load_crsp_monthly_base(db, start=start, end=end)
    dl = load_crsp_msedelist(db)
    msf = attach_dlret_and_total_return(msf, dl)
    save_parquet(msf, raw_path)
    print(f"[OK] Raw monthly (with ret_total) → {raw_path} | rows={msf.shape[0]:,}")

    # Attach share/exchange codes first so we can filter cleanly, then attach names/tickers.
    print("[INFO] Attaching msenames / filtering equities / attaching stocknames…")
    msenames = load_crsp_msenames(db)
    coded = attach_msenames(msf, msenames)
    coded_filt = filter_equities(coded, include_adrs=include_adrs, exchanges=exchanges, allow_missing_codes=False)
    names = load_crsp_stocknames(db)
    labeled = attach_stocknames(coded_filt, names)

    # One last safety dedupe on the panel keys.
    before, labeled = labeled.shape[0], labeled.drop_duplicates(["permno", "date"], keep="last")
    print(f"[INFO] Deduplicated {before - labeled.shape[0]:,} rows → {labeled.shape[0]:,}")

    print(f"[INFO] Eligibility flags (price_min={price_min}, min_hist={min_hist_months}m, liq_lookback={liq_lookback}m)…")
    labeled = add_eligibility_flags(
        labeled,
        min_hist_months=min_hist_months,
        liq_lookback=liq_lookback,
        price_min=price_min,
    )
    save_parquet(labeled, interim_path)
    print(f"[OK] Interim labeled → {interim_path}")

    # Universe is based on the eligibility-filtered panel.
    print(f"[INFO] Monthly Top-{top_n} universe (drop bottom {int(liq_pctl*100)}% by $-vol)…")
    uni = universe_top_n(labeled, n=top_n, liquidity_percentile=liq_pctl)
    save_parquet(uni, universe_path)
    print(f"[OK] Universe table → {universe_path} | rows={uni.shape[0]:,}")

    # Merge membership back into the labeled panel.
    print("[INFO] Merging universe flag into panel…")
    labeled = labeled.merge(uni, on=["permno", "date"], how="left")
    labeled["in_universe"] = labeled["in_universe"].fillna(False)

    # Keep a compact “core” panel for downstream work.
    keep = [
        "permno",
        "permco",
        "date",
        "ticker",
        "comnam",
        "ret",
        "dlret",
        "ret_total",
        "prc",
        "prc_abs",
        "shrout",
        "me",
        "vol",
        "dollar_vol",
        "med_dollar_vol",
        "exchcd",
        "shrcd",
        "hist_months",
        "elig_price",
        "elig_hist",
        "elig_liq",
        "in_universe",
    ]
    core = labeled[[c for c in keep if c in labeled.columns]].copy()
    save_parquet(core, processed_path)
    print(f"[OK] Processed core → {processed_path} | rows={core.shape[0]:,}")


# Weekly pipeline: build weekly panel from daily, then build a universe off snapshots and merge it back in.

def run_weekly(
    db,
    start: str,
    end: str,
    include_adrs: bool,
    exchanges: Tuple[int, ...],
    top_n: int,
    liq_pctl: float,
    price_min: float,
    min_hist_months: int,
    liq_lookback_weeks: int,
    week_anchor: str,
    rebalance: str,
    annual_month: int,
    raw_path: Path,
    interim_path: Path,
    processed_path: Path,
    universe_path: Path,
    preselect_from_monthly: bool,
    preselect_top_n: int,
    preselect_liq_pctl: float,
    progress: bool,
    chunk_freq: str,
):
    print(
        f"[INFO] Building CRSP weekly panel ({start} → {end}, anchor={week_anchor}) | "
        f"preselect_from_monthly={preselect_from_monthly}, preselect_top_n={preselect_top_n}, "
        f"preselect_liq_pctl={preselect_liq_pctl}…"
    )

    # build_crsp_weekly_panel handles the daily pull, delisting adjustment, and interval attaches.
    weekly_panel = build_crsp_weekly_panel(
        db,
        start=start,
        end=end,
        include_adrs=include_adrs,
        exchanges=exchanges,
        week_anchor=week_anchor,
        minimal_cols=False,
        preselect_from_monthly=preselect_from_monthly,
        top_n_for_preselect=preselect_top_n,
        liq_pctl_for_preselect=preselect_liq_pctl,
        progress=progress,
        chunk_freq=chunk_freq,
    )

    # Weekly liquidity uses a rolling window in weeks (not months), so recompute that median here.
    if "dollar_vol" in weekly_panel.columns:
        print(f"[INFO] Recomputing weekly liquidity median (lookback={liq_lookback_weeks} weeks)…")
        weekly_panel = weekly_panel.sort_values(["permno", "date"])
        weekly_panel["med_dollar_vol"] = (
            weekly_panel.groupby("permno", observed=True)["dollar_vol"]
            .rolling(window=liq_lookback_weeks, min_periods=1)
            .median()
            .reset_index(level=0, drop=True)
        )

    save_parquet(weekly_panel, raw_path)
    print(f"[OK] Raw weekly (post-attach/filter) → {raw_path} | rows={weekly_panel.shape[0]:,}")

    # Universe formation on weekly panels can be weekly/monthly/annual, depending on rebalance settings.
    print(f"[INFO] Building universe (rebalance='{rebalance}', top_n={top_n}, liq_pctl={liq_pctl})…")
    uni_w = build_universe_from_snapshots(
        weekly_panel, top_n=top_n, liq_pctl=liq_pctl, rebalance=rebalance, annual_month=annual_month
    )
    save_parquet(uni_w, universe_path)
    print(f"[OK] Weekly universe → {universe_path} | rows={uni_w.shape[0]:,}")

    print("[INFO] Merging universe flag back to weekly panel…")
    labeled = weekly_panel.merge(uni_w, on=["permno", "date"], how="left")
    labeled["in_universe"] = labeled["in_universe"].fillna(False)

    keep = [
        "permno",
        "permco",
        "date",
        "ticker",
        "comnam",
        "ret",
        "ret_total",
        "prc_abs",
        "shrout",
        "me",
        "vol",
        "dollar_vol",
        "med_dollar_vol",
        "exchcd",
        "shrcd",
        "hist_months",
        "elig_price",
        "elig_hist",
        "elig_liq",
        "in_universe",
    ]
    core = labeled[[c for c in keep if c in labeled.columns]].copy()
    save_parquet(core, processed_path)
    print(f"[OK] Weekly processed core → {processed_path} | rows={core.shape[0]:,}")


# Main entrypoint used by the CLI. Handles config defaults and file naming.

def main(
    freq: str,
    start: str | None,
    end: str | None,
    include_adrs: bool | None,
    exchanges: Tuple[int, ...] | None,
    top_n: int,
    liq_pctl: float,
    price_min: float,
    min_hist_months: int,
    liq_lookback: int,
    liq_lookback_weeks: int,
    week_anchor: str,
    universe_rebalance: str,
    annual_month: int,
    preselect_from_monthly: bool,
    preselect_top_n: int,
    preselect_liq_pctl: float,
    progress: bool,
    chunk_freq: str,
):
    cfg = load_config()
    start = start or cfg["crsp"]["start"]
    end = end or cfg["crsp"]["end"]
    cfg_include_adrs = bool(cfg.get("crsp", {}).get("include_adrs", False))
    include_adrs = cfg_include_adrs if include_adrs is None else include_adrs
    cfg_exchanges = tuple(cfg.get("crsp", {}).get("exchanges", (1, 2, 3)))
    exchanges = exchanges or cfg_exchanges

    # Weekly outputs are versioned by date range so multiple pulls can coexist.
    if freq == "weekly":
        date_tag = _safe_date_tag(start, end)

        raw_name = f"crsp_weekly_raw_{date_tag}.parquet"
        processed_name = f"crsp_weekly_core_{date_tag}.parquet"

        raw_path = Path(p("data/raw")) / raw_name
        interim_path = _cfg_path(cfg, "interim_weekly", "data/interim/crsp_weekly_labeled.parquet")
        processed_path = Path(p("data/processed")) / processed_name

        extra = f"rebalance-{universe_rebalance}_{week_anchor}"
        universe_name = _universe_filename(
            freq="weekly",
            start=start,
            end=end,
            top_n=top_n,
            liq_pctl=liq_pctl,
            extra=extra,
        )
        universe_path = Path(processed_path).with_name(universe_name)

    else:
        raw_path = _cfg_path(cfg, "raw_monthly", "data/raw/crsp_monthly_raw.parquet")
        interim_path = _cfg_path(cfg, "interim_labeled", "data/interim/crsp_monthly_labeled.parquet")
        processed_path = _cfg_path(cfg, "processed_core", "data/processed/crsp_monthly_core.parquet")

        universe_name = _universe_filename(
            freq="monthly",
            start=start,
            end=end,
            top_n=top_n,
            liq_pctl=liq_pctl,
        )
        universe_path = Path(processed_path).with_name(universe_name)

    print("[INFO] Connecting to WRDS…")
    db = connect_wrds()
    try:
        if freq == "weekly":
            run_weekly(
                db=db,
                start=start,
                end=end,
                include_adrs=include_adrs,
                exchanges=exchanges,
                top_n=top_n,
                liq_pctl=liq_pctl,
                price_min=price_min,
                min_hist_months=min_hist_months,
                liq_lookback_weeks=liq_lookback_weeks,
                week_anchor=week_anchor,
                rebalance=universe_rebalance,
                annual_month=annual_month,
                raw_path=raw_path,
                interim_path=interim_path,
                processed_path=processed_path,
                universe_path=universe_path,
                preselect_from_monthly=preselect_from_monthly,
                preselect_top_n=preselect_top_n,
                preselect_liq_pctl=preselect_liq_pctl,
                progress=progress,
                chunk_freq=chunk_freq,
            )
        else:
            run_monthly(
                db=db,
                start=start,
                end=end,
                include_adrs=include_adrs,
                exchanges=exchanges,
                top_n=top_n,
                liq_pctl=liq_pctl,
                price_min=price_min,
                min_hist_months=min_hist_months,
                liq_lookback=liq_lookback,
                raw_path=raw_path,
                interim_path=interim_path,
                processed_path=processed_path,
                universe_path=universe_path,
            )
        print("[DONE] CRSP pull pipeline completed successfully.")
        print("[NOTE] ADRs included (SHRCD=12)." if include_adrs else "[NOTE] ADRs excluded — pass --include-adrs to include ADRs.")
    finally:
        db.close()
        print("[INFO] WRDS connection closed.")


# CLI wiring. Keeps the script runnable without importing anything else.

if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description=(
            "Pull CRSP monthly/weekly, attach codes/names & delistings, build eligibility and a "
            "Russell-1000-like universe with liquidity screens, and emit parquet outputs."
        )
    )
    ap.add_argument("--freq", choices=["monthly", "weekly"], default="monthly", help="Output panel frequency.")
    ap.add_argument("--start", type=str, default=None, help="YYYY-MM-DD (inclusive)")
    ap.add_argument("--end", type=str, default=None, help="YYYY-MM-DD (exclusive)")
    ap.add_argument("--include-adrs", action="store_true", help="Include ADRs (SHRCD=12).")
    ap.add_argument("--exchanges", type=str, default=None, help="Comma-separated exchcds to keep (default: 1,2,3).")

    ap.add_argument("--top-n", type=int, default=1000, help="Top-N by market cap per snapshot.")
    ap.add_argument("--liq-pctl", type=float, default=0.20, help="Drop bottom p by $ volume per snapshot.")

    ap.add_argument("--price-min", type=float, default=5.0, help="Minimum price for eligibility.")
    ap.add_argument("--min-hist-months", type=int, default=12, help="Min trailing months of return history.")
    ap.add_argument("--liq-lookback", type=int, default=3, help="Liquidity lookback in MONTHS (monthly flow).")
    ap.add_argument("--liq-lookback-weeks", type=int, default=8, help="Liquidity lookback in WEEKS (weekly flow).")

    ap.add_argument("--week-anchor", type=str, default="W-FRI", help="Weekly anchor (e.g., W-FRI, W-WED).")
    ap.add_argument(
        "--universe-rebalance",
        choices=["weekly", "monthly", "annual"],
        default="monthly",
        help="Universe formation cadence for weekly panel.",
    )
    ap.add_argument("--annual-month", type=int, default=6, help="Month number for annual reconstitution (1–12).")

    ap.add_argument("--preselect-off", action="store_true", help="Disable monthly preselection of permnos.")
    ap.add_argument("--preselect-top-n", type=int, default=1000, help="Top-N used in monthly preselect.")
    ap.add_argument("--preselect-liq-pctl", type=float, default=0.20, help="Liquidity drop (bottom p) in preselect.")

    ap.add_argument("--progress", action="store_true", help="Print periodic ETA and stage timestamps (weekly only).")
    ap.add_argument("--chunk-freq", type=str, default="MS", help="Daily pull chunk frequency: MS/QS/YS. Default: MS.")

    args = ap.parse_args()

    main(
        freq=args.freq,
        start=args.start,
        end=args.end,
        include_adrs=args.include_adrs,
        exchanges=_parse_exchanges(args.exchanges),
        top_n=args.top_n,
        liq_pctl=args.liq_pctl,
        price_min=args.price_min,
        min_hist_months=args.min_hist_months,
        liq_lookback=args.liq_lookback,
        liq_lookback_weeks=args.liq_lookback_weeks,
        week_anchor=args.week_anchor,
        universe_rebalance=args.universe_rebalance,
        annual_month=args.annual_month,
        preselect_from_monthly=not args.preselect_off,
        preselect_top_n=args.preselect_top_n,
        preselect_liq_pctl=args.preselect_liq_pctl,
        progress=args.progress,
        chunk_freq=args.chunk_freq,
    )
