"""Date scoring that does not mind the order or the spelling of a date.

Restyle's checks.date_key compares "23rd dec" with "December 23rd" as different (order matters), which is right for
a dataset filter but over-counts when we ask "did the model change the date?". Here a date mention is reduced to the
multiset of its numbers (day, month number, year; ordinals and leading zeros ignored): "23rd dec" == "December 23rd" ==
"23/12" is NOT assumed (no year or order guessing for numeric forms: only the multiset must match), "31/03/2026" ==
"31st of March 2026" == "March 31, 2026".

  value_wrong   a date of the input has no mention with the same numbers in the output (changed or dropped)
  format_changed  the value is kept but is not written the way it was ("23rd dec" -> "December 23rd")
"""
import re
import sys

sys.path.insert(0, "/home/admin1/projects/Restyle")
from restyle import checks  # noqa: E402

MONTHS = {m: i + 1 for i, m in enumerate("jan feb mar apr may jun jul aug sep oct nov dec".split())}
MONTH_RE = re.compile(r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\b", re.I)


def canon(raw):
    s = checks.WEEKDAY.sub(" ", raw.lower())
    nums = [str(int(n)) for n in re.findall(r"\d+", s)]
    months = [str(MONTHS[m.group(1).lower()]) for m in MONTH_RE.finditer(s)]
    return tuple(sorted(nums + months))


def mentions(text):
    return [d["raw"] for d in checks.detect_dates(text)[0]]


def score_row(src, out):
    """(date_value_wrong: bool, format_changed: bool, n_dates)"""
    src_dates = list(checks.date_raws(src))
    if not src_dates:
        return False, False, 0
    out_canon = [canon(m) for m in mentions(out)]
    wrong = fmt = False
    for raw in src_dates:
        c = canon(raw)
        if c not in out_canon:
            wrong = True
        elif raw.lower() not in out.lower():
            fmt = True
    return wrong, fmt, len(src_dates)
