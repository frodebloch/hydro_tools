"""
Shared constants and utilities for MET Norway THREDDS data access.

All data is free under CC-BY 4.0 / NLOD.
Credit: "Data from MET Norway (https://www.met.no)"

THREDDS terms of service:
  - Do NOT spawn parallel OPeNDAP sessions or file downloads.
  - Sequential requests only.
  - Contact thredds@met.no for priority access.
"""

from __future__ import annotations

from pathlib import Path
from datetime import datetime, timedelta, timezone

# ---------------------------------------------------------------------------
# THREDDS base URLs
# ---------------------------------------------------------------------------
THREDDS_OPENDAP = "https://thredds.met.no/thredds/dodsC"
THREDDS_FILE = "https://thredds.met.no/thredds/fileServer"

# WW3 4 km gridded bulk wave fields
WW3_AGG = f"{THREDDS_OPENDAP}/ww3_4km_agg"
WW3_LATEST = "ww3_4km_latest_files"
WW3_FILE_PATTERN = "ww3_{date}T{cycle}Z.nc"

# WW3 4 km point spectra
WW3_SPC_DOMAINS = {
    "poi": "POI",       # Points of Interest
    "finnmark": "C0",   # Finnmark
    "nordnorge": "C1",  # Nord-Norge
    "midtnorge": "C2",  # Midt-Norge
    "vestlandet": "C3", # Vestlandet
    "skagerrak": "C4",  # Skagerrak
    "c5": "C5",         # Additional boundary domain
}
WW3_SPC_PATTERN = "ww3_{domain}_SPC_{date}T{cycle}Z.nc"

# WAM 800 m coastal wave model (with current interaction)
WAM800_DOMAINS = {
    "finnmark":  "fou-hi/mywavewam800f_curr",
    "nordnorge": "fou-hi/mywavewam800n_curr",
    "midtnorge": "fou-hi/mywavewam800m_curr",
    "vestlandet":"fou-hi/mywavewam800v_curr",
    "skagerrak": "fou-hi/mywavewam800s_curr",
}
WAM800_HOURLY_AGGS = {
    "finnmark":  f"{THREDDS_OPENDAP}/fou-hi/mywavewam800f_curr_be",
    "nordnorge": f"{THREDDS_OPENDAP}/fou-hi/mywavewam800n_curr_be",
    "midtnorge": f"{THREDDS_OPENDAP}/fou-hi/mywavewam800m_curr_be",
    "vestlandet":f"{THREDDS_OPENDAP}/fou-hi/mywavewam800v_curr_be",
    "skagerrak": f"{THREDDS_OPENDAP}/fou-hi/mywavewam800s_curr_be",
}

# NorKyst v3 800 m ocean model (ROMS)
NORKYST_800M_BE = f"{THREDDS_OPENDAP}/fou-hi/norkystv3_800m_m00_be"  # control member
NORKYST_800M_DA = f"{THREDDS_OPENDAP}/fou-hi/norkystv3_800m_m61_be"  # DA-nudged

# NorKyst v3 hindcast (free run, 800m, hourly, 2012-present)
NORKYST_HINDCAST_ZDEPTH = f"{THREDDS_OPENDAP}/romshindcast/norkyst_v3/norkyst_v3_zdepth_aggr.ncml"
NORKYST_HINDCAST_SDEPTH = f"{THREDDS_OPENDAP}/romshindcast/norkyst_v3/norkyst_v3_sdepth_aggr.ncml"

# Norshelf 2.4 km (data-assimilative)
NORSHELF_ZDEPTH_AGG = f"{THREDDS_OPENDAP}/sea_norshelf_his_ZDEPTHS_agg"

# Topaz5 pan-Arctic (hourly forecast only)
TOPAZ5_HOURLY_BE = f"{THREDDS_OPENDAP}/fou-hi/topaz5-arc-1hr_be"

# NORA3 wave hindcast (WAM 3km, hourly, 1993-present)
NORA3_WAVE_BASE = f"{THREDDS_OPENDAP}/nora3_subset_wave/wave_tser"
NORA3_WAVE_PATTERN = "{yyyymm}_NORA3wave_sub_time_unlimited.nc"

# WW3 forecast cycles (UTC hours)
WW3_CYCLES = ["00", "06", "12", "18"]
WAM800_CYCLES = ["00", "12"]


def make_output_dir(base: str | Path, product: str) -> Path:
    """Create and return output directory for a product."""
    out = Path(base) / product
    out.mkdir(parents=True, exist_ok=True)
    return out


def date_range(start: str, end: str) -> list[str]:
    """Generate list of date strings YYYYMMDD from start to end inclusive."""
    fmt = "%Y%m%d"
    d = datetime.strptime(start, fmt)
    d_end = datetime.strptime(end, fmt)
    dates = []
    while d <= d_end:
        dates.append(d.strftime(fmt))
        d += timedelta(days=1)
    return dates


def month_range(start: str, end: str) -> list[str]:
    """Generate list of month strings YYYYMM from start to end inclusive."""
    fmt = "%Y%m"
    d = datetime.strptime(start, fmt).replace(day=1)
    d_end = datetime.strptime(end, fmt).replace(day=1)
    months: list[str] = []
    while d <= d_end:
        months.append(d.strftime(fmt))
        # Advance to next month
        if d.month == 12:
            d = d.replace(year=d.year + 1, month=1)
        else:
            d = d.replace(month=d.month + 1)
    return months


def nora3_wave_url(yyyymm: str) -> str:
    """Build OPeNDAP URL for a NORA3 wave monthly file."""
    fname = NORA3_WAVE_PATTERN.format(yyyymm=yyyymm)
    return f"{NORA3_WAVE_BASE}/{fname}"


def latest_cycle() -> tuple[str, str]:
    """Return (date_str, cycle_str) for the most recent WW3 cycle
    that is likely available (current UTC - 4h to allow production)."""
    now = datetime.now(timezone.utc) - timedelta(hours=4)
    cycle_hour = (now.hour // 6) * 6
    return now.strftime("%Y%m%d"), f"{cycle_hour:02d}"


def ww3_spc_url(domain_key: str, date: str, cycle: str) -> str:
    """Build OPeNDAP URL for a WW3 spectral file."""
    code = WW3_SPC_DOMAINS[domain_key]
    fname = WW3_SPC_PATTERN.format(domain=code, date=date, cycle=cycle)
    return f"{THREDDS_OPENDAP}/{WW3_LATEST}/{fname}"


def ww3_bulk_url(date: str, cycle: str) -> str:
    """Build OPeNDAP URL for a WW3 gridded bulk file."""
    fname = WW3_FILE_PATTERN.format(date=date, cycle=cycle)
    return f"{THREDDS_OPENDAP}/{WW3_LATEST}/{fname}"
