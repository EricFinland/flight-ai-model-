from datetime import date

import pytest

from prewright.dates import CUTOFF, is_before_cutoff, parse_date


@pytest.mark.parametrize(
    "raw, earliest, latest",
    [
        ("1887", date(1887, 1, 1), date(1887, 12, 31)),
        ("1903-10-08", date(1903, 10, 8), date(1903, 10, 8)),
        ("1903-02", date(1903, 2, 1), date(1903, 2, 28)),  # can't be a range
        ("1903-11", date(1903, 1, 1), date(1911, 12, 31)),  # month or range: widest
        ("1890-95", date(1890, 1, 1), date(1895, 12, 31)),
        ("1890-1910", date(1890, 1, 1), date(1910, 12, 31)),
        ("189-", date(1890, 1, 1), date(1899, 12, 31)),
        ("189u", date(1890, 1, 1), date(1899, 12, 31)),
        ("18--", date(1800, 1, 1), date(1899, 12, 31)),
        ("19uu", date(1900, 1, 1), date(1999, 12, 31)),
        ("[1850]", date(1850, 1, 1), date(1850, 12, 31)),
        ("c. 1850", date(1845, 1, 1), date(1855, 12, 31)),
        ("[1899?]", date(1894, 1, 1), date(1904, 12, 31)),
        (1776, date(1776, 1, 1), date(1776, 12, 31)),
    ],
)
def test_parse_date(raw, earliest, latest):
    rng = parse_date(raw)
    assert rng is not None
    assert (rng.earliest, rng.latest) == (earliest, latest)


@pytest.mark.parametrize("raw", [None, "", "n.d.", "[n.d.]", "unknown", "sometime", "1910-1890"])
def test_unparseable_dates(raw):
    assert parse_date(raw) is None


def test_cutoff_is_first_flight_day():
    assert CUTOFF == date(1903, 12, 17)


@pytest.mark.parametrize(
    "raw, eligible",
    [
        ("1902", True),
        ("1903-12-16", True),  # day before the flights
        ("1903-12-17", False),  # day of the flights
        ("1903", False),  # could be 18-31 December
        ("1903-02", True),
        ("1903-11", False),  # ambiguous: could mean 1903-1911
        ("18--", True),
        ("19--", False),
        ("1895-1905", False),  # multi-volume run crossing the cutoff
        ("c. 1895", True),
        ("c. 1899", False),  # could be as late as 1904
        ("n.d.", False),
        (None, False),
    ],
)
def test_is_before_cutoff(raw, eligible):
    assert is_before_cutoff(raw) is eligible
