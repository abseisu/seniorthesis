from __future__ import annotations

import time
from datetime import datetime
from typing import Iterable, Tuple, Optional, List, Callable

import numpy as np
import pandas as pd
import wrds

import warnings
warnings.filterwarnings("ignore", message="Engine has switched to 'python'")

# Polars is used for fast multi-core groupby aggregation in the daily → weekly step.
import polars as pl

from psycopg2 import OperationalError


# WRDS connection helpers and quick CRSP discovery utilities.

def connect_wrds(wrds_username: str | None = None):
    """Open a WRDS connection. Prompts for credentials if username not provided."""
    return wrds.Connection() if wrds_username is None else wrds.Connection(wrds_username=wrds_username)

def list_crsp_tables(db) -> list[str]:
    """Return a list of available CRSP tables for your account."""
    return db.list_tables(library="crsp")

def _batches(seq, size):
    # Simple list chunker for WRDS-friendly IN() lists.
    for i in range(0, len(seq), size):
        yield seq[i:i+size]

def _raw_sql_retry(db, sql, date_cols=None, max_tries=5, base_sleep=2.0, progress_note: str | None = None):
    """
    Execute raw_sql with reconnection + exponential backoff for transient WRDS disconnects.
    """
    tries = 0
    while True:
        try:
            return db.raw_sql(sql, date_cols=date_cols)
        except OperationalError as e:
            tries += 1
            if tries >= max_tries:
                raise
            sleep = base_sleep * (2 ** (tries - 1))
            if progress_note:
                print(f"[WRDS retry {tries}/{max_tries}] {progress_note} | sleeping {sleep:.1f}s …")
            try:
                # Close + reopen tends to fix intermittent SSH/DB hiccups.
                db.close()
            except Exception:
                pass
            db.connect()
            time.sleep(sleep)


# Monthly CRSP loaders (msf + monthly delistings).

def load_crsp_monthly_base(db, start: str, end: str) -> pd.DataFrame:
    """
    Load CRSP monthly stock file (msf) with a stable, minimal column set.

    Output includes: permno, date, ret, prc, shrout, vol, prc_abs, me
    """
    sql = f"""
        select permno, date, ret, prc, shrout, vol
        from crsp.msf
        where date >= '{start}' and date < '{end}'
    """
    df = db.raw_sql(sql, date_cols=["date"])
    df["ret"]    = pd.to_numeric(df["ret"], errors="coerce")
    df["prc"]    = pd.to_numeric(df["prc"], errors="coerce")
    df["shrout"] = pd.to_numeric(df["shrout"], errors="coerce")
    df["vol"]    = pd.to_numeric(df["vol"], errors="coerce")
    df["prc_abs"] = df["prc"].abs()
    df["me"] = df["prc_abs"] * df["shrout"] * 1000.0
    return df

def load_crsp_msedelist(db) -> pd.DataFrame:
    """Monthly delisting file (join on (permno, dlstdt))."""
    sql = "select permno, dlstdt as date, dlret from crsp.msedelist"
    df = db.raw_sql(sql, date_cols=["date"])
    df["dlret"] = pd.to_numeric(df["dlret"], errors="coerce")
    return df


# Daily CRSP loaders (dsf + daily delistings), used for weekly aggregation.

def load_crsp_dsedelist(db) -> pd.DataFrame:
    """Daily delisting file (join on (permno, dlstdt))."""
    sql = "select permno, dlstdt as date, dlret from crsp.dsedelist"
    df = db.raw_sql(sql, date_cols=["date"])
    df["dlret"] = pd.to_numeric(df["dlret"], errors="coerce")
    return df

def load_crsp_daily_base_filtered(
    db,
    start: str,
    end: str,
    include_adrs: bool = False,
    exchanges: Tuple[int, ...] = (1, 2, 3),
    permnos: Optional[List[int]] = None,
) -> pd.DataFrame:
    """
    Daily dsf loader with server-side filtering via dsenames interval join.
    Optionally restrict to a provided permno list.

    Output includes: permno, date, ret, prc, vol, shrout, prc_abs, me
    """
    shrcds = (10, 11, 12) if include_adrs else (10, 11)
    exch   = "(" + ",".join(str(x) for x in sorted(set(exchanges))) + ")"
    shr    = "(" + ",".join(str(x) for x in sorted(set(shrcds))) + ")"

    base_sql = f"""
        from crsp.dsf d
        inner join crsp.dsenames n
            on d.permno = n.permno
           and d.date >= n.namedt
           and d.date <= n.nameendt
        where d.date >= '{start}' and d.date < '{end}'
          and n.exchcd in {exch}
          and n.shrcd in {shr}
    """
    if permnos:
        plist = "(" + ",".join(map(str, sorted(set(permnos)))) + ")"
        base_sql += f" and d.permno in {plist} "

    sql = f"select d.permno, d.date, d.ret, d.prc, d.vol, d.shrout {base_sql}"
    df = db.raw_sql(sql, date_cols=["date"])
    df["ret"]    = pd.to_numeric(df["ret"], errors="coerce")
    df["prc"]    = pd.to_numeric(df["prc"], errors="coerce")
    df["vol"]    = pd.to_numeric(df["vol"], errors="coerce")
    df["shrout"] = pd.to_numeric(df["shrout"], errors="coerce")
    df["prc_abs"] = df["prc"].abs()
    df["me"] = df["prc_abs"] * df["shrout"] * 1000.0
    return df


# Date chunking + chunked daily pulls (useful for long spans and WRDS stability).

def _date_chunks(start: str, end: str, freq: str = "MS") -> list[tuple[str, str]]:
    """
    Build [(chunk_start, chunk_end)] covering [start, end) with pandas-style freq.
    freq = 'MS' (month), 'QS' (quarter), 'YS' (year)
    """
    s, e = pd.to_datetime(start), pd.to_datetime(end)
    rng = pd.date_range(s, e, freq=freq)
    if len(rng) == 0 or rng[0] > s:
        rng = pd.DatetimeIndex([s]).append(rng)
    if rng[-1] < e:
        rng = rng.append(pd.DatetimeIndex([e]))
    return [(rng[i].strftime("%Y-%m-%d"), rng[i+1].strftime("%Y-%m-%d")) for i in range(len(rng) - 1)]

def load_crsp_daily_base_filtered_chunked(
    db,
    start: str,
    end: str,
    include_adrs: bool = False,
    exchanges: Tuple[int, ...] = (1, 2, 3),
    permnos: Optional[List[int]] = None,
    chunk_freq: str = "MS",
    progress_cb: Optional[Callable[[int, int, int, float, float], None]] = None,  # (i,n,rows,elapsed,eta)
    batch_size: int = 1000,   # WRDS-friendly IN() size
) -> pd.DataFrame:
    """
    Pull dsf in date chunks. If permnos are provided, also split into WRDS-friendly IN() batches.
    Uses retry logic to handle transient disconnects.
    """
    chunks = _date_chunks(start, end, freq=chunk_freq)
    n = len(chunks)
    t0 = time.time()
    frames: List[pd.DataFrame] = []

    shrcds = (10, 11, 12) if include_adrs else (10, 11)
    exch   = "(" + ",".join(str(x) for x in sorted(set(exchanges))) + ")"
    shr    = "(" + ",".join(str(x) for x in sorted(set(shrcds))) + ")"

    for i, (cs, ce) in enumerate(chunks, 1):
        batch_frames: List[pd.DataFrame] = []
        if permnos:
            for jb, pbatch in enumerate(_batches(permnos, batch_size), 1):
                plist = "(" + ",".join(map(str, pbatch)) + ")"
                sql = f"""
                    select d.permno, d.date, d.ret, d.prc, d.vol, d.shrout
                    from crsp.dsf d
                    inner join crsp.dsenames n
                        on d.permno = n.permno
                       and d.date >= n.namedt
                       and d.date <= n.nameendt
                    where d.date >= '{cs}' and d.date < '{ce}'
                      and n.exchcd in {exch}
                      and n.shrcd in {shr}
                      and d.permno in {plist}
                """
                dfb = _raw_sql_retry(db, sql, date_cols=["date"],
                                     progress_note=f"chunk {i}/{n}, permno batch {jb}")
                batch_frames.append(dfb)
        else:
            sql = f"""
                select d.permno, d.date, d.ret, d.prc, d.vol, d.shrout
                from crsp.dsf d
                inner join crsp.dsenames n
                    on d.permno = n.permno
                   and d.date >= n.namedt
                   and d.date <= n.nameendt
                where d.date >= '{cs}' and d.date < '{ce}'
                  and n.exchcd in {exch}
                  and n.shrcd in {shr}
            """
            dfb = _raw_sql_retry(db, sql, date_cols=["date"],
                                 progress_note=f"chunk {i}/{n}")
            batch_frames.append(dfb)

        dfc = pd.concat(batch_frames, ignore_index=True) if batch_frames else pd.DataFrame(
            columns=["permno","date","ret","prc","vol","shrout"]
        )

        # Force numerics and recompute derived columns consistently across chunks.
        dfc["ret"]    = pd.to_numeric(dfc["ret"], errors="coerce")
        dfc["prc"]    = pd.to_numeric(dfc["prc"], errors="coerce")
        dfc["vol"]    = pd.to_numeric(dfc["vol"], errors="coerce")
        dfc["shrout"] = pd.to_numeric(dfc["shrout"], errors="coerce")
        dfc["prc_abs"] = dfc["prc"].abs()
        dfc["me"] = dfc["prc_abs"] * dfc["shrout"] * 1000.0

        frames.append(dfc)

        if progress_cb:
            elapsed = time.time() - t0
            eta = (elapsed / i) * (n - i) if i else 0.0
            progress_cb(i, n, len(dfc), elapsed, eta)

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
        columns=["permno","date","ret","prc","vol","shrout","prc_abs","me"]
    )


# Attachers for interval metadata and delisting-adjusted returns.

def _normalize_numeric(panel: pd.DataFrame, cols: Iterable[str]) -> None:
    # Coerce common code columns to numeric without throwing on junk.
    for c in cols:
        if c in panel.columns and panel[c].dtype.kind not in "biufc":
            panel[c] = pd.to_numeric(panel[c], errors="coerce")

def _ffill_by_perm(panel: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    # Forward-fill selected columns within each permno.
    panel = panel.sort_values(["permno", "date"])
    panel[list(cols)] = panel.groupby("permno", observed=True)[list(cols)].ffill()
    return panel

def load_crsp_msenames(db) -> pd.DataFrame:
    # Share codes and exchange codes over time (interval data).
    sql = "select permno, permco, namedt, nameendt, shrcd, exchcd from crsp.msenames"
    return db.raw_sql(sql, date_cols=["namedt", "nameendt"])

def load_crsp_stocknames(db) -> pd.DataFrame:
    # Normalize the end-date column name for consistency with msenames.
    sql = "select permno, namedt, nameenddt as nameendt, comnam, ticker from crsp.stocknames"
    return db.raw_sql(sql, date_cols=["namedt", "nameendt"])

def attach_msenames(panel: pd.DataFrame, msenames: pd.DataFrame) -> pd.DataFrame:
    """
    Interval-attach permco/shrcd/exchcd per permno using merge_asof,
    then drop anything that falls outside the nameendt interval.
    """
    if panel.empty:
        return panel

    df = panel.dropna(subset=["permno", "date"]).copy()
    df["permno"] = pd.to_numeric(df["permno"], errors="coerce").astype("int64")
    df["date"]   = pd.to_datetime(df["date"])

    mn = msenames[["permno","permco","shrcd","exchcd","namedt","nameendt"]].dropna(subset=["permno","namedt"]).copy()
    mn["permno"]   = pd.to_numeric(mn["permno"], errors="coerce").astype("int64")
    mn["namedt"]   = pd.to_datetime(mn["namedt"])
    mn["nameendt"] = pd.to_datetime(mn["nameendt"], errors="coerce")

    parts = []
    # Doing this permno-by-permno keeps both frames sorted correctly for merge_asof.
    for perm, g in df.groupby("permno", sort=False, observed=True):
        mn_p = mn[mn["permno"] == perm]
        if mn_p.empty:
            gg = g.copy()
            gg[["permco","shrcd","exchcd"]] = np.nan
            parts.append(gg)
            continue

        g_sorted   = g.sort_values("date",   kind="mergesort")
        mn_sorted  = mn_p.sort_values("namedt", kind="mergesort")

        merged = pd.merge_asof(
            g_sorted,
            mn_sorted,
            left_on="date",
            right_on="namedt",
            direction="backward",
            allow_exact_matches=True,
        )

        # merge_asof can drop/overwrite keys in odd edge cases; force permno back in.
        merged["permno"] = perm
        mask = merged["nameendt"].isna() | (merged["date"] <= merged["nameendt"])
        merged.loc[~mask, ["permco","shrcd","exchcd"]] = np.nan
        parts.append(merged.drop(columns=["namedt","nameendt"], errors="ignore"))

    out = pd.concat(parts, ignore_index=True)

    # Re-coerce codes to numeric in case they came through as object.
    for c in ["permco","shrcd","exchcd"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    return out


def _attach_msenames_merge_mask(panel: pd.DataFrame, msenames: pd.DataFrame) -> pd.DataFrame:
    # Simpler (but heavier) interval attach: full merge then mask by (namedt, nameendt).
    df = panel.copy()
    mn = msenames[["permno","permco","namedt","nameendt","shrcd","exchcd"]].copy()
    out = df.merge(mn, on="permno", how="left", suffixes=("", "_mn"))
    mask = (out["date"] >= out["namedt"]) & (out["date"] <= out["nameendt"])
    out.loc[~mask, ["permco","shrcd","exchcd"]] = np.nan
    out = out.drop(columns=["namedt","nameendt"])
    for c in ["permco","shrcd","exchcd"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out

def attach_stocknames(panel: pd.DataFrame, stocknames: pd.DataFrame) -> pd.DataFrame:
    """
    Interval-attach comnam/ticker per permno using merge_asof,
    then blank out anything outside the nameendt interval.
    """
    if panel.empty:
        return panel

    df = panel.dropna(subset=["permno","date"]).copy()
    df["permno"] = pd.to_numeric(df["permno"], errors="coerce").astype("int64")
    df["date"]   = pd.to_datetime(df["date"])

    sn = stocknames[["permno","namedt","nameendt","comnam","ticker"]].dropna(subset=["permno","namedt"]).copy()
    sn["permno"]   = pd.to_numeric(sn["permno"], errors="coerce").astype("int64")
    sn["namedt"]   = pd.to_datetime(sn["namedt"])
    sn["nameendt"] = pd.to_datetime(sn["nameendt"], errors="coerce")

    parts = []
    for perm, g in df.groupby("permno", sort=False, observed=True):
        sn_p = sn[sn["permno"] == perm]
        if sn_p.empty:
            gg = g.copy()
            gg[["comnam","ticker"]] = np.nan
            parts.append(gg)
            continue

        g_sorted  = g.sort_values("date",   kind="mergesort")
        sn_sorted = sn_p.sort_values("namedt", kind="mergesort")

        # Drop permno on the right to avoid suffix collisions in merge_asof.
        sn_right = sn_sorted.drop(columns=["permno"])

        merged = pd.merge_asof(
            g_sorted,
            sn_right,
            left_on="date",
            right_on="namedt",
            direction="backward",
            allow_exact_matches=True,
            suffixes=("", "_sn"),
        )
        merged["permno"] = perm

        inside = merged["nameendt"].isna() | (merged["date"] <= merged["nameendt"])
        merged.loc[~inside, ["comnam","ticker"]] = np.nan

        parts.append(merged.drop(columns=["namedt","nameendt"], errors="ignore"))

    out = pd.concat(parts, ignore_index=True)

    # Reduce memory: treat tickers/comnam as categorical after normalization.
    if "ticker" in out.columns:
        out["ticker"] = out["ticker"].astype("string").str.strip().str.upper().astype("category")
    if "comnam" in out.columns:
        out["comnam"] = out["comnam"].astype("string").str.strip().astype("category")

    return out


def attach_dlret_and_total_return(msf: pd.DataFrame, msedelist: pd.DataFrame) -> pd.DataFrame:
    # Total return includes delisting returns when present.
    out = msf.merge(msedelist, on=["permno","date"], how="left")
    out["ret"] = pd.to_numeric(out["ret"], errors="coerce")
    out["dlret"] = pd.to_numeric(out["dlret"], errors="coerce")
    out["ret_total"] = (1.0 + out["ret"].fillna(0.0)) * (1.0 + out["dlret"].fillna(0.0)) - 1.0
    return out


# Equity filters and universe construction.

def filter_equities(
    panel: pd.DataFrame,
    include_adrs: bool = False,
    exchanges: Tuple[int, ...] = (1, 2, 3),
    allow_missing_codes: bool = False,
) -> pd.DataFrame:
    """Filter to equities by share code & exchanges."""
    panel = panel.copy()
    _normalize_numeric(panel, ["shrcd","exchcd"])
    if not allow_missing_codes:
        panel = panel.dropna(subset=["shrcd","exchcd"])
    shrcds = {10, 11}
    if include_adrs:
        shrcds.add(12)
    q = f"(shrcd in {tuple(sorted(shrcds))}) and (exchcd in {tuple(sorted(set(exchanges)))})"
    return panel.query(q).copy()


def add_eligibility_flags(
    panel: pd.DataFrame,
    min_hist_months: int = 12,
    liq_lookback: int = 3,
    price_min: float = 5.0,
) -> pd.DataFrame:
    """
    Adds:
      - dollar_vol, med_dollar_vol (rolling median)
      - hist_months (past non-null ret_total count)
      - elig_price, elig_hist, elig_liq

    Used for both monthly and weekly panels (history is month-count based).
    """
    df = panel.copy().sort_values(["permno","date"])
    if "vol" not in df.columns:
        raise ValueError("VOL not present; ensure 'vol' exists before eligibility.")
    for c in ["prc_abs","vol","me"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("float64")
    df["dollar_vol"] = pd.to_numeric(df["prc_abs"] * df["vol"], errors="coerce").astype("float64")
    df["med_dollar_vol"] = (
        df.groupby("permno", observed=True)["dollar_vol"]
          .rolling(window=liq_lookback, min_periods=1).median()
          .reset_index(level=0, drop=True)
    )
    def _hist_count(s: pd.Series) -> pd.Series:
        cnt = s.notna().astype(int).cumsum()
        return cnt.shift(1).fillna(0).astype(int)
    df["hist_months"] = (
        df.groupby("permno", observed=True)["ret_total"]
          .apply(_hist_count).reset_index(level=0, drop=True)
    )
    df["elig_price"] = df["prc_abs"] >= price_min
    df["elig_hist"]  = df["hist_months"] >= min_hist_months
    df["elig_liq"]   = df["med_dollar_vol"].notna()
    return df

def universe_top_n(
    panel_with_permco: pd.DataFrame,
    n: int = 1000,
    liquidity_percentile: float = 0.20,
) -> pd.DataFrame:
    """
    Per-date Top-N universe:
      1) pick the largest-ME share class per PERMCO
      2) apply price and history screens
      3) drop the least liquid tail by dollar volume
      4) take the top-N by market cap
    """
    df = panel_with_permco.copy()
    required = {"permco","me","elig_price","elig_hist","dollar_vol","date","permno"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"universe_top_n missing columns: {missing}")
    df = df.sort_values(["date","permco","me"], ascending=[True,True,False]) \
           .drop_duplicates(["date","permco"], keep="first")
    df = df[df["elig_price"] & df["elig_hist"]].copy()
    def _keep_liquid(g: pd.DataFrame) -> pd.DataFrame:
        cutoff = g["dollar_vol"].quantile(liquidity_percentile)
        return g[g["dollar_vol"] >= cutoff]
    df = df.groupby("date", observed=True, group_keys=False).apply(_keep_liquid)
    uni = df.groupby("date", observed=True, group_keys=False).apply(lambda g: g.nlargest(n, "me"))
    uni = uni[["permno","date"]].copy()
    uni["in_universe"] = True
    return uni


# Daily → weekly aggregation. This streams by date chunk and uses Polars for the heavy lifting.

def _week_ending(dt: pd.Series, week_anchor: str = "W-FRI") -> pd.Series:
    """Map calendar dates to week-ending stamps (robust chain to avoid .normalize attr issues)."""
    return dt.dt.to_period(week_anchor).dt.to_timestamp(how="end").dt.floor("D")

def aggregate_daily_to_weekly_streaming_polars(
    db,
    start: str,
    end: str,
    include_adrs: bool,
    exchanges: Tuple[int, ...],
    permnos: Optional[List[int]],
    dsedelist: pd.DataFrame,
    week_anchor: str = "W-FRI",
    chunk_freq: str = "MS",
    progress_cb: Optional[Callable[[int, int, int, int, float, float], None]] = None,  # (i,n,rows_in,rows_out,elapsed,eta)
) -> pd.DataFrame:
    """
    Stream per date chunk (with a 6-day overlap so weeks are complete). For each chunk:
      - pull daily data
      - attach daily dlret
      - compound to weekly returns
      - only emit weeks fully contained in the chunk, and carry the partial week forward
    """
    chunks = _date_chunks(start, end, freq=chunk_freq)
    n = len(chunks)
    t0 = time.time()
    weekly_parts: list[pd.DataFrame] = []
    carry_daily = pd.DataFrame(columns=["permno","date","ret","prc","vol","shrout","prc_abs","me"])

    dld_pl = pl.from_pandas(dsedelist)

    for i, (cs, ce) in enumerate(chunks, 1):
        overlap_start = (pd.to_datetime(cs) - pd.Timedelta(days=6)).strftime("%Y-%m-%d")

        df_chunk = load_crsp_daily_base_filtered_chunked(
            db,
            start=overlap_start,
            end=ce,
            include_adrs=include_adrs,
            exchanges=exchanges,
            permnos=permnos,
            chunk_freq="YS",
            progress_cb=None,
            batch_size=1000,
        )

        if not carry_daily.empty:
            df_chunk = pd.concat([carry_daily, df_chunk], ignore_index=True)

        df_pl = pl.from_pandas(df_chunk)
        df_pl  = df_pl.with_columns([pl.col("permno").cast(pl.Int64), pl.col("date").cast(pl.Date)])
        dld_pl = dld_pl.with_columns([pl.col("permno").cast(pl.Int64), pl.col("date").cast(pl.Date)])
        df_pl = df_pl.join(dld_pl, on=["permno","date"], how="left")
        df_pl = df_pl.with_columns([
            pl.col("ret").cast(pl.Float64),
            pl.col("dlret").cast(pl.Float64),
            (((pl.col("ret").fill_null(0.0) + 1.0) * (pl.col("dlret").fill_null(0.0) + 1.0)) - 1.0).alias("ret_total_d"),
            pl.col("prc_abs").cast(pl.Float64),
            pl.col("shrout").cast(pl.Float64),
            pl.col("vol").cast(pl.Float64),
        ])

        week_end = pd.to_datetime(df_chunk["date"]).dt.to_period(week_anchor).dt.to_timestamp(how="end").dt.floor("D")
        df_pl = df_pl.with_columns(pl.from_pandas(week_end).alias("week"))

        out_pl = (
            df_pl.group_by(["permno","week"])
                 .agg([
                     (pl.col("ret_total_d").fill_null(0.0) + 1.0).product() - 1.0,
                     (pl.col("ret").fill_null(0.0) + 1.0).product() - 1.0,
                     pl.col("vol").sum(),
                     pl.col("prc_abs").last(),
                     pl.col("shrout").last(),
                 ])
                 .rename({"week":"date","ret_total_d":"ret_total","ret":"ret",
                          "vol":"vol","prc_abs":"prc_abs","shrout":"shrout"})
                 .with_columns((pl.col("prc_abs") * pl.col("shrout") * 1000.0).alias("me"))
        )
        out = out_pl.to_pandas()

        ce_ts = pd.to_datetime(ce)
        full_weeks = out[out["date"] <= ce_ts].copy()
        partial_weeks = out[out["date"] > ce_ts].copy()

        weekly_parts.append(full_weeks)

        if not partial_weeks.empty:
            latest_week_end = partial_weeks["date"].max()
            mask = (week_end == latest_week_end)
            carry_daily = df_chunk.loc[mask.values].copy()
        else:
            carry_daily = pd.DataFrame(columns=df_chunk.columns)

        if progress_cb:
            elapsed = time.time() - t0
            rows_in = df_chunk.shape[0]
            rows_out = full_weeks.shape[0]
            eta = (elapsed / i) * (n - i) if i else 0.0
            progress_cb(i, n, rows_in, rows_out, elapsed, eta)

    if not carry_daily.empty:
        df_pl = pl.from_pandas(carry_daily).with_columns([
            pl.col("ret").cast(pl.Float64),
            pl.lit(None).cast(pl.Float64).alias("dlret"),
        ])
        df_pl = df_pl.with_columns((((pl.col("ret").fill_null(0.0) + 1.0) *
                                     (pl.col("dlret").fill_null(0.0) + 1.0)) - 1.0).alias("ret_total_d"))
        week_end = pd.to_datetime(carry_daily["date"]).dt.to_period(week_anchor).dt.to_timestamp(how="end").dt.floor("D")
        df_pl = df_pl.with_columns(pl.from_pandas(week_end).alias("week"))
        out_pl = (
            df_pl.group_by(["permno","week"])
                 .agg([
                     (pl.col("ret_total_d").fill_null(0.0) + 1.0).product() - 1.0,
                     (pl.col("ret").fill_null(0.0) + 1.0).product() - 1.0,
                     pl.col("vol").sum(),
                     pl.col("prc_abs").last(),
                     pl.col("shrout").last(),
                 ])
                 .rename({"week":"date","ret_total_d":"ret_total","ret":"ret",
                          "vol":"vol","prc_abs":"prc_abs","shrout":"shrout"})
                 .with_columns((pl.col("prc_abs") * pl.col("shrout") * 1000.0).alias("me"))
        )
        weekly_parts.append(out_pl.to_pandas())

    weekly = pd.concat(weekly_parts, ignore_index=True)

    for c in ["ret_total","ret","prc_abs","shrout","me","vol"]:
        if c in weekly.columns:
            weekly[c] = pd.to_numeric(weekly[c], errors="coerce")

    return weekly


# Monthly preselection to shrink the daily pull: build monthly Top-N universes and take the permno union.

def monthly_universe_permnos(
    db,
    start: str,
    end: str,
    top_n: int = 1000,
    liq_pctl: float = 0.20,
    include_adrs: bool = False,
    exchanges: Tuple[int, ...] = (1, 2, 3),
) -> List[int]:
    """
    Build a Top-N universe each month (after liquidity screen) and return the union of permnos.
    This intentionally skips stocknames to avoid a large interval merge that isn't needed here.
    """
    msf = load_crsp_monthly_base(db, start, end)
    dl  = load_crsp_msedelist(db)
    msf = attach_dlret_and_total_return(msf, dl)
    mn  = load_crsp_msenames(db)
    panel = _attach_msenames_merge_mask(msf, mn)
    panel = filter_equities(panel, include_adrs=include_adrs, exchanges=exchanges, allow_missing_codes=False)
    panel = panel.sort_values(["permno","date"]).reset_index(drop=True)
    panel = add_eligibility_flags(panel, min_hist_months=12, liq_lookback=3, price_min=5.0)
    uni = universe_top_n(panel, n=top_n, liquidity_percentile=liq_pctl)
    return sorted(uni["permno"].unique().tolist())


# High-level builders for monthly and weekly panels.

def build_crsp_monthly_panel(
    db,
    start: str,
    end: str,
    include_adrs: bool = False,
    exchanges: Tuple[int, ...] = (1, 2, 3),
    minimal_cols: bool = True,
) -> pd.DataFrame:
    """Build a monthly panel ready for downstream use."""
    msf = load_crsp_monthly_base(db, start, end)
    dl  = load_crsp_msedelist(db)
    msf = attach_dlret_and_total_return(msf, dl)
    mn  = load_crsp_msenames(db)
    sn  = load_crsp_stocknames(db)
    panel = attach_msenames(msf, mn)
    panel = attach_stocknames(panel, sn)
    panel = filter_equities(panel, include_adrs=include_adrs, exchanges=exchanges, allow_missing_codes=False)
    panel = panel.sort_values(["permno","date"]).reset_index(drop=True)
    panel = add_eligibility_flags(panel, min_hist_months=12, liq_lookback=3, price_min=5.0)

    if minimal_cols:
        keep = ["permno","permco","date","ret","dlret","ret_total","prc","prc_abs","shrout","me","vol",
                "dollar_vol","med_dollar_vol","shrcd","exchcd","ticker","comnam",
                "hist_months","elig_price","elig_hist","elig_liq"]
        panel = panel[[c for c in keep if c in panel.columns]]
    return panel


def build_crsp_weekly_panel(
    db,
    start: str,
    end: str,
    include_adrs: bool = False,
    exchanges: Tuple[int, ...] = (1, 2, 3),
    week_anchor: str = "W-FRI",
    minimal_cols: bool = True,
    preselect_from_monthly: bool = True,
    top_n_for_preselect: int = 1000,
    liq_pctl_for_preselect: float = 0.20,
    progress: bool = False,
    chunk_freq: str = "MS",
) -> pd.DataFrame:
    """
    Build a weekly panel from daily CRSP.
    Uses optional monthly Top-N preselection to reduce the daily pull, then aggregates daily → weekly.
    """
    permnos: Optional[List[int]] = None
    if preselect_from_monthly:
        if progress: print(f"[{datetime.now():%H:%M:%S}] Preselecting permnos from monthly Top-{top_n_for_preselect} (liq_pctl={liq_pctl_for_preselect})…")
        permnos = monthly_universe_permnos(
            db, start=start, end=end, top_n=top_n_for_preselect,
            liq_pctl=liq_pctl_for_preselect, include_adrs=include_adrs, exchanges=exchanges
        )
        if progress: print(f"[{datetime.now():%H:%M:%S}] Preselect permnos count: {len(permnos):,}")

    def _pull_progress(i, n, rows, elapsed, eta):
        if progress:
            print(f"[pull {i}/{n}] +{rows:,} rows | elapsed {elapsed:,.0f}s | ETA {eta:,.0f}s")

    if progress:
        print(f"[{datetime.now():%H:%M:%S}] Pulling daily from WRDS (chunk={chunk_freq})…")

    dld = load_crsp_dsedelist(db)

    if progress:
        print(f"[{datetime.now():%H:%M:%S}] Aggregating daily → weekly in chunks ({chunk_freq})…")

    wk = aggregate_daily_to_weekly_streaming_polars(
        db=db,
        start=start,
        end=end,
        include_adrs=include_adrs,
        exchanges=exchanges,
        permnos=permnos,
        dsedelist=dld,
        week_anchor=week_anchor,
        chunk_freq=chunk_freq,
        progress_cb=lambda i,n,rin,rout,el,eta: print(
            f"[agg {i}/{n}] {rin:,} daily → {rout:,} weekly | elapsed {el:,.0f}s | ETA {eta:,.0f}s"
        ) if progress else None,
    )

    if progress:
        print(f"[{datetime.now():%H:%M:%S}] Interval-attaching msenames/stocknames…")

    wk = wk.sort_values(["permno","date"]).reset_index(drop=True)

    mn = load_crsp_msenames(db)

    wk["permno"] = pd.to_numeric(wk["permno"], errors="coerce").astype("int64")
    wk["date"]   = pd.to_datetime(wk["date"])
    wk = wk.sort_values(["permno","date"], kind="mergesort").reset_index(drop=True)

    panel = attach_msenames(wk, mn)

    panel = filter_equities(panel, include_adrs=include_adrs, exchanges=exchanges, allow_missing_codes=False)

    sn = load_crsp_stocknames(db)
    panel = attach_stocknames(panel, sn)

    if progress:
        print(f"[{datetime.now():%H:%M:%S}] Computing eligibility flags…")

    panel = add_eligibility_flags(panel, min_hist_months=12, liq_lookback=3, price_min=5.0)

    if minimal_cols:
        keep = ["permno","permco","date","ret","ret_total","prc_abs","shrout","me","vol",
                "dollar_vol","med_dollar_vol","shrcd","exchcd","ticker","comnam",
                "hist_months","elig_price","elig_hist","elig_liq"]
        panel = panel[[c for c in keep if c in panel.columns]]
    return panel


# Basic parquet I/O helpers.

def write_panel_parquet(panel: pd.DataFrame, path: str) -> None:
    """Write compact parquet."""
    panel.to_parquet(path, index=False)

def read_panel_parquet(path: str) -> pd.DataFrame:
    """Read parquet."""
    return pd.read_parquet(path)
