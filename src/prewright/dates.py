"""Turn messy publication-date metadata into a conservative date range.

Library catalogues record dates in many shapes: "1887", "[1899?]", "c. 1850",
"1890-1910", "18--", "189u", "n.d.". We parse each into the earliest and latest
dates the item could have been published. A document is eligible only if its
*latest* possible date falls before the cutoff, so any doubt counts against it.
"""

from __future__ import annotations

import calendar
import re
from dataclasses import dataclass
from datetime import date

# 17 December 1903: the Wright brothers' first powered, controlled flights at
# Kill Devil Hills. Documents must have been published strictly before this.
CUTOFF = date(1903, 12, 17)

# Widen approximate dates ("c. 1900", "1899?") by this many years either side.
DEFAULT_APPROX_MARGIN_YEARS = 5


@dataclass(frozen=True)
class DateRange:
    earliest: date
    latest: date
    approximate: bool = False


_APPROX_PREFIX = re.compile(r"^(c\.|ca\.|circa|approximately|about|c(?=\s*\d))\s*", re.I)
_FULL = re.compile(r"^(\d{4})-(\d{1,2})-(\d{1,2})$")
_YEAR_MONTH = re.compile(r"^(\d{4})-(\d{1,2})$")
_YEAR = re.compile(r"^(\d{4})$")
_RANGE = re.compile(r"^(\d{4})\s*[-–/]\s*(\d{2,4})$")
# "189-", "189u", "189x", "189?": some year in the 1890s.
_DECADE = re.compile(r"^(\d{3})[-ux_?]$", re.I)
# "18--", "18uu", "18??": some year in the 1800s.
_CENTURY = re.compile(r"^(\d{2})(--|uu|xx|\?\?)$", re.I)


def _last_day(year: int, month: int) -> date:
    return date(year, month, calendar.monthrange(year, month)[1])


def _candidates(s: str) -> list[tuple[date, date]]:
    out = []
    if m := _FULL.match(s):
        y, mo, d = map(int, m.groups())
        try:
            day = date(y, mo, d)
            out.append((day, day))
        except ValueError:
            pass
    if m := _YEAR_MONTH.match(s):
        y, mo = map(int, m.groups())
        if 1 <= mo <= 12:
            out.append((date(y, mo, 1), _last_day(y, mo)))
    if m := _YEAR.match(s):
        y = int(m.group(1))
        out.append((date(y, 1, 1), date(y, 12, 31)))
    if m := _RANGE.match(s):
        start, end = m.group(1), m.group(2)
        # "1890-95" means 1890 to 1895.
        end_year = int(start[: 4 - len(end)] + end)
        if end_year > int(start):
            out.append((date(int(start), 1, 1), date(end_year, 12, 31)))
    if m := _DECADE.match(s):
        base = int(m.group(1)) * 10
        out.append((date(base, 1, 1), date(base + 9, 12, 31)))
    if m := _CENTURY.match(s):
        base = int(m.group(1)) * 100
        out.append((date(base, 1, 1), date(base + 99, 12, 31)))
    return out


def _parse_exact(s: str) -> tuple[date, date] | None:
    # "1903-11" could be November 1903 or the years 1903-1911. When a string
    # reads both ways, take the widest span so doubt counts against eligibility.
    candidates = _candidates(s)
    if not candidates:
        return None
    return min(c[0] for c in candidates), max(c[1] for c in candidates)


def parse_date(
    raw: str | int | date | None,
    approx_margin_years: int = DEFAULT_APPROX_MARGIN_YEARS,
) -> DateRange | None:
    """Parse a catalogue date. Returns None when the date cannot be determined."""
    if raw is None:
        return None
    if isinstance(raw, date):
        return DateRange(raw, raw)
    if isinstance(raw, int):
        raw = str(raw)

    s = raw.strip().rstrip(".").strip()
    s = s.replace("[", "").replace("]", "").strip()
    if not s or s.lower() in {"n.d", "nd", "s.d", "undated", "unknown"}:
        return None

    approximate = False
    if _APPROX_PREFIX.match(s):
        approximate = True
        s = _APPROX_PREFIX.sub("", s)
    # A trailing "?" marks doubt about the whole date ("1899?"), but "189?"
    # (unknown final digit) is a decade pattern handled by _parse_exact.
    if s.endswith("?") and not _DECADE.match(s) and not _CENTURY.match(s):
        approximate = True
        s = s.rstrip("?").strip()

    parsed = _parse_exact(s)
    if parsed is None:
        return None
    earliest, latest = parsed
    if approximate:
        earliest = date(earliest.year - approx_margin_years, 1, 1)
        latest = date(latest.year + approx_margin_years, 12, 31)
    return DateRange(earliest, latest, approximate)


def is_before_cutoff(
    raw: str | int | date | None,
    cutoff: date = CUTOFF,
    approx_margin_years: int = DEFAULT_APPROX_MARGIN_YEARS,
) -> bool:
    """True only if the item was certainly published before the cutoff."""
    rng = parse_date(raw, approx_margin_years)
    return rng is not None and rng.latest < cutoff
