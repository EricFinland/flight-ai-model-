import pytest

from prewright.leakage import Decision, filter_document, strip_boilerplate

GUTENBERG = """The Project Gutenberg eBook of Progress in Flying Machines

This eBook is for the use of anyone anywhere.

*** START OF THE PROJECT GUTENBERG EBOOK PROGRESS IN FLYING MACHINES ***

Produced by a volunteer and proofread at Distributed Proofreaders.

The problem of aerial navigation has attracted the attention of ingenious men
in all ages.

*** END OF THE PROJECT GUTENBERG EBOOK PROGRESS IN FLYING MACHINES ***

Updated editions will replace the previous one.
"""


def test_strip_gutenberg_and_credits():
    text, removed = strip_boilerplate(GUTENBERG)
    assert text.startswith("The problem of aerial navigation")
    assert "Gutenberg" not in text
    assert removed == 1


def test_scanner_notice_paragraph_removed():
    text, removed = strip_boilerplate("Digitized by Google\n\nCHAPTER I.\n\nOf the steam-engine.")
    assert text == "CHAPTER I.\n\nOf the steam-engine."
    assert removed == 1


@pytest.mark.parametrize(
    "text",
    [
        # Genuine period writing that must survive.
        "M. Santos-Dumont rounded the Eiffel Tower in his dirigible balloon in 1901.",
        "The aeroplane surfaces of Lilienthal's apparatus were of willow and cloth.",
        "Mr. Wilbur Wright described the gliding experiments made near Kitty Hawk.",
        "Professor Langley's aerodrome fell into the Potomac on the 8th of December.",
        "Planck's hypothesis treats the energy of resonators as made up of finite elements.",
        "The relativity of all human knowledge was a favourite doctrine of Hamilton.",
        "Professor Thomson's corpuscles, or electrons, are far smaller than the atom.",
        "The nuclear division of the cell was described by Flemming.",
        "Aristotle calls this error the proton pseudos of the argument.",
        "It was a titanic struggle between the two parties.",
        "The word television has lately been proposed for seeing at a distance.",
    ],
)
def test_period_text_is_kept(text):
    result = filter_document(text)
    assert result.decision is Decision.KEEP, result.reasons()


@pytest.mark.parametrize(
    "text",
    [
        "Einstein's theory of relativity overturned these notions.",
        "Bohr applied quantum theory to the hydrogen atom.",
        "The nucleus contains protons and neutrons.",
        "Before the First World War such machines were rare.",
        "The Wright Flyer is preserved at the Smithsonian.",
        "Their first powered flight lasted twelve seconds.",
        "The Russo-Japanese War ended in 1905.",
        "Survivors of the Titanic disaster were landed at New York.",
        "Rationing of vitamins was discussed.",
    ],
)
def test_anachronisms_rejected(text):
    result = filter_document(text)
    assert result.decision is Decision.REJECT
    assert result.hard_hits


def test_soft_signals_accumulate_to_review():
    text = "The airplane company was founded in 1909 and moved in 1912 and again in 1919."
    result = filter_document(text)
    assert result.decision is Decision.REVIEW
    assert result.soft_score == pytest.approx(2.0 + 3 * 0.5)


def test_single_later_year_is_tolerated():
    # Bond maturities and the like name future years in genuine period text.
    result = filter_document("The bonds bear four per cent and fall due in 1925.")
    assert result.decision is Decision.KEEP
    assert result.soft_score == 0.5


def test_modern_copyright_page_is_stripped_not_rejected():
    text = "Copyright 1923 by the Riverside Press.\n\nPREFACE.\n\nThe author offers these essays."
    result = filter_document(text)
    assert result.decision is Decision.KEEP
    assert result.text == "PREFACE.\n\nThe author offers these essays."
