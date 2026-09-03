#!/usr/bin/env python3
"""
French nuclear generation line chart – ENTSO-E Transparency Platform.

Lines produced
  solid blue   – this year's actual nuclear generation (7-day rolling avg, GW)
  dotted blue  – rest of year: nuclear capacity availability from planned outages
  solid orange – previous year actual
  dashed green – 5-year historical average (or 2-year if data is sparse)

Requirements
  pip install entsoe-py pandas matplotlib numpy

Usage
  export ENTSOE_API_KEY=your_key_here
  python french_nuclear.py

  Free API key: https://transparency.entsoe.eu/usrm/user/createPublicUser
"""

from __future__ import annotations
import os, sys, warnings
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker

warnings.filterwarnings("ignore")

try:
    from entsoe import EntsoePandasClient
except ImportError:
    sys.exit("Missing dependency – run:  pip install entsoe-py")


API_KEY      = os.environ.get("ENTSOE_API_KEY", "")
COUNTRY      = "FR"
TZ           = "Europe/Paris"
TODAY        = pd.Timestamp.now(tz=TZ).normalize()
CUR_YEAR     = TODAY.year
HIST_YEARS   = 5          # years to include in the historical average
SMOOTH_DAYS  = 7          # rolling-average window

# Approximate total installed nuclear capacity (GW).
# Used only when unavailability data cannot determine it dynamically.
FALLBACK_INSTALLED_GW = 56.0




def ts(year: int, month: int = 1, day: int = 1) -> pd.Timestamp:
    return pd.Timestamp(year=year, month=month, day=day, tz=TZ)


def smooth(s: pd.Series, w: int = SMOOTH_DAYS) -> pd.Series:
    return s.rolling(w, min_periods=1, center=True).mean()


def remap_to_year(s: pd.Series, target: int) -> pd.Series:
    def _remap(d: pd.Timestamp) -> pd.Timestamp | None:
        try:
            return d.replace(year=target)
        except ValueError:          # Feb 29 → non-leap year
            return None

    new_idx = [_remap(d) for d in s.index]
    valid   = [i is not None for i in new_idx]
    out     = s[valid].copy()
    out.index = pd.DatetimeIndex([i for i in new_idx if i is not None])
    return out




def _extract_nuclear(raw: pd.DataFrame | pd.Series) -> pd.Series:
    """Pull the 'Actual Aggregated' nuclear column from entsoe-py output."""
    if isinstance(raw, pd.Series):
        return raw
    if not isinstance(raw, pd.DataFrame):
        raise TypeError(f"Unexpected type from query_generation: {type(raw)}")
    # Flat column names
    for kw in ("Actual Aggregated", "Actual", "Nuclear"):
        if kw in raw.columns:
            return raw[kw]
    # MultiIndex columns e.g. ('Actual Aggregated', 'Nuclear')
    if isinstance(raw.columns, pd.MultiIndex):
        for col in raw.columns:
            if any("Actual" in str(p) for p in col):
                return raw[col]
    return raw.iloc[:, 0]          # last resort


def fetch_actual(client: EntsoePandasClient, year: int) -> pd.Series:
    """Daily mean nuclear generation (GW) for `year`, capped at today."""
    start = ts(year)
    end   = min(ts(year + 1), TODAY + pd.Timedelta("2D"))
    if start >= end:
        return pd.Series(dtype=float, name=str(year))
    try:
        raw    = client.query_generation(COUNTRY, start=start, end=end,
                                          psr_type="B14")
        series = _extract_nuclear(raw)
        daily  = (series / 1000).resample("D").mean().dropna()
        daily.name = str(year)
        return daily
    except Exception as exc:
        print(f"  ⚠  {year} actuals: {exc}")
        return pd.Series(dtype=float, name=str(year))


def fetch_availability(client: EntsoePandasClient,
                       cur_actual: pd.Series) -> pd.Series:
    """
    Daily available nuclear capacity (GW) from tomorrow through year-end.

    Primary method  – uses ENTSO-E generation unit outage data (doc type A80):
        available = total_installed − Σ(nominal_power − avail_qty) per day
    Permanently decommissioned units (e.g. Fessenheim, avail_qty=0 to 2099)
    appear in both installed capacity and unavailability, so they cancel out.

    Fallback         – flat projection of the 30-day trailing actual average.
    """
    start = TODAY + pd.Timedelta("1D")
    end   = ts(CUR_YEAR + 1)
    if start >= end:
        return pd.Series(dtype=float)

    dates = pd.date_range(start, end - pd.Timedelta("1D"), freq="D", tz=TZ)

    # ── primary: generation unit unavailability (A80) ─────────────────────────
    try:
        # Fetch registered installed nuclear capacity
        installed_mw = FALLBACK_INSTALLED_GW * 1000
        try:
            cap = client.query_installed_generation_capacity(
                COUNTRY, start=ts(CUR_YEAR), end=ts(CUR_YEAR + 1),
                psr_type="B14")
            installed_mw = float(cap.iloc[0] if isinstance(cap, pd.Series)
                                 else cap.iloc[0, 0])
            print(f"  Installed nuclear capacity: {installed_mw/1000:.1f} GW")
        except Exception:
            print(f"  Installed capacity query failed; using {FALLBACK_INSTALLED_GW} GW")

        # Fetch generation unit outages (A80 covers all plant types)
        df = client.query_unavailability_of_generation_units(
            COUNTRY, start=start, end=end
        )
        if df is None or df.empty:
            raise ValueError("Empty outage DataFrame")

        # Keep nuclear units only
        df = df[df["plant_type"] == "Nuclear"].copy()
        if df.empty:
            raise ValueError("No nuclear outage records found")

        # For each outage: unavailable_mw = nominal_power − avail_qty
        df["unavail_mw"] = df["nominal_power"].astype(float) - \
                           df["avail_qty"].astype(float)

        # Sum unavailable MW into a daily series
        unavail = pd.Series(0.0, index=dates)
        for _, row in df.iterrows():
            rs  = row["start"]
            re  = row["end"]
            mw  = float(row["unavail_mw"])
            if pd.isna(rs) or pd.isna(re) or mw <= 0:
                continue
            if getattr(rs, "tzinfo", None) is None:
                rs = rs.tz_localize(TZ)
            if getattr(re, "tzinfo", None) is None:
                re = re.tz_localize(TZ)
            mask = (unavail.index >= rs) & (unavail.index < re)
            unavail[mask] += mw

        avail = ((installed_mw - unavail) / 1000).clip(lower=0)
        print(f"  {len(df)} nuclear outage records → "
              f"mean available {avail.mean():.1f} GW "
              f"(range {avail.min():.1f}–{avail.max():.1f} GW)")
        return avail
    except Exception as exc:
        print(f"  ⚠  Unavailability query failed: {exc}")

    # ── fallback: project 30-day trailing mean ───────────────────────────────
    if not cur_actual.empty:
        window = min(30, len(cur_actual))
        level  = float(cur_actual.iloc[-window:].mean())
        print(f"  Fallback: projecting {window}-day trailing mean = {level:.1f} GW")
        return pd.Series(level, index=dates)

    return pd.Series(dtype=float)


def build_average(hist: dict[int, pd.Series]) -> tuple[pd.Series | None, str]:
    """
    Day-of-year mean across all available historical years, re-indexed onto
    the current year. Uses the year 2000 (a leap year) as a neutral pivot so
    Feb 29 is preserved until the final remap step.
    """
    if len(hist) < 2:
        return None, ""

    frames: dict[int, pd.Series] = {}
    for year, s in hist.items():
        frames[year] = remap_to_year(s, 2000)   # align to leap-year pivot

    aligned = pd.DataFrame(frames).sort_index().dropna(how="all")
    avg_doy = aligned.mean(axis=1)
    avg_cur = remap_to_year(avg_doy, CUR_YEAR).dropna()

    years = sorted(hist.keys())
    label = (f"{years[0]}–{years[-1]} average"
             if len(years) >= 4
             else f"{years[0]}–{years[-1]} avg")
    return avg_cur, label



BLUE   = "#1565C0"
ORANGE = "#BF360C"
GREEN  = "#1B5E20"


def draw(cur_actual: pd.Series,
         avail_fc:   pd.Series,
         prev_actual: pd.Series,
         avg_series:  pd.Series | None,
         avg_label:   str) -> None:

    fig, ax = plt.subplots(figsize=(13, 6.5))

    if not cur_actual.empty:
        ax.plot(cur_actual.index, smooth(cur_actual),
                color=BLUE, lw=2.2,
                label=f"{CUR_YEAR} actual")

    if not avail_fc.empty:
        ax.plot(avail_fc.index, smooth(avail_fc),
                color=BLUE, lw=2.2, ls=":",
                label=f"{CUR_YEAR} nuclear availability (forecast)")

    if not prev_actual.empty:
        s = remap_to_year(prev_actual, CUR_YEAR)
        ax.plot(s.index, smooth(s),
                color=ORANGE, lw=1.8,
                label=f"{CUR_YEAR - 1} actual")

    if avg_series is not None and not avg_series.empty:
        ax.plot(avg_series.index, smooth(avg_series),
                color=GREEN, lw=1.8, ls="--",
                label=avg_label)

    # "today" divider
    ax.axvline(TODAY, color="#757575", lw=0.9, ls="--", alpha=0.65, zorder=0)
    _, ymax = ax.get_ylim()
    ax.text(TODAY + pd.Timedelta("3D"), ymax * 0.96,
            f"Today\n{TODAY.strftime('%-d %b')}",
            color="#757575", fontsize=8, va="top")

    # axes & labels
    ax.set_xlim(ts(CUR_YEAR), ts(CUR_YEAR + 1))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax.set_ylabel("Nuclear output  (GW, 7-day rolling avg)", fontsize=10)
    ax.set_title("France – Nuclear Generation", fontsize=15,
                 fontweight="bold", pad=14)
    ax.legend(framealpha=0.93, fontsize=9, loc="lower right")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f"))
    ax.grid(axis="y", color="#BDBDBD", lw=0.5)
    ax.set_axisbelow(True)
    fig.text(0.99, 0.01,
             "Source: ENTSO-E Transparency Platform  •  transparency.entsoe.eu",
             ha="right", fontsize=7.5, color="#9E9E9E")

    plt.tight_layout()
    out = "french_nuclear.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nChart saved → {out}")
    plt.show()



def main() -> None:
    if not API_KEY:
        sys.exit(
            "\nENTSOE_API_KEY is not set.\n"
            "Register for a free key at:\n"
            "  https://transparency.entsoe.eu/usrm/user/createPublicUser\n\n"
            "Then export it before running:\n"
            "  export ENTSOE_API_KEY=your_key_here"
        )

    client = EntsoePandasClient(api_key=API_KEY)
    print(f"Date: {TODAY.date()}   Country: {COUNTRY}   Source: ENTSO-E\n")

    print(f"[1/4] {CUR_YEAR} actual generation …")
    cur_actual = fetch_actual(client, CUR_YEAR)
    print(f"      {len(cur_actual)} daily records")

    print(f"[2/4] {CUR_YEAR - 1} actual generation …")
    prev_actual = fetch_actual(client, CUR_YEAR - 1)
    print(f"      {len(prev_actual)} daily records")

    hist_range = range(CUR_YEAR - HIST_YEARS, CUR_YEAR)
    print(f"[3/4] Historical {list(hist_range)[0]}–{list(hist_range)[-1]} for average …")
    hist: dict[int, pd.Series] = {}
    for y in hist_range:
        s = fetch_actual(client, y)
        if not s.empty:
            hist[y] = s
            print(f"      {y}: {len(s)} days")

    avg_series, avg_label = build_average(hist)
    if avg_series is None:
        print("  Not enough historical data for average line (need ≥ 2 years).")

    print("[4/4] Nuclear availability forecast …")
    avail_fc = fetch_availability(client, cur_actual)

    print("\nRendering …")
    draw(cur_actual, avail_fc, prev_actual, avg_series, avg_label)


if __name__ == "__main__":
    main()
