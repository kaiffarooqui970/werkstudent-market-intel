"""
Skill extraction, language-gate detection, seniority tagging, and salary parsing.

Design: pure functions, no I/O. Testable in isolation. The skill taxonomy is a
plain Python dict so it's easy to extend without touching the extractor.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

# ─────────────────────────────────────────────────────────────────────────────
# Skill taxonomy
# ─────────────────────────────────────────────────────────────────────────────
# Each entry: canonical_name -> (category, [regex patterns matched case-insensitively])
# Patterns use word boundaries; add aliases as needed. Precision > recall.

SKILL_TAXONOMY: dict[str, tuple[str, list[str]]] = {
    # Languages
    "Python":       ("language",  [r"\bpython\b"]),
    "R":            ("language",  [r"(?<![A-Za-z])R(?:\s|,|\.|/|$)"]),  # tricky
    "SQL":          ("language",  [r"\bSQL\b"]),
    "Java":         ("language",  [r"\bjava\b(?!\s*script)"]),
    "JavaScript":   ("language",  [r"\bjava\s*script\b", r"\bjs\b"]),
    "TypeScript":   ("language",  [r"\btype\s*script\b", r"\bts\b"]),
    "Scala":        ("language",  [r"\bscala\b"]),
    "Go":           ("language",  [r"\bgolang\b", r"\bgo\s+lang\b"]),
    "C++":          ("language",  [r"\bc\+\+"]),
    "Rust":         ("language",  [r"\brust\b"]),

    # Data
    "Pandas":       ("data",      [r"\bpandas\b"]),
    "NumPy":        ("data",      [r"\bnumpy\b"]),
    "Spark":        ("data",      [r"\b(pyspark|apache\s+spark|\bspark)\b"]),
    "dbt":          ("data",      [r"\bdbt\b"]),
    "Airflow":      ("data",      [r"\bairflow\b"]),
    "Snowflake":    ("data",      [r"\bsnowflake\b"]),
    "BigQuery":     ("data",      [r"\bbig\s*query\b"]),
    "Databricks":   ("data",      [r"\bdatabricks\b"]),
    "Kafka":        ("data",      [r"\bkafka\b"]),
    "Redshift":     ("data",      [r"\bredshift\b"]),
    "Postgres":     ("data",      [r"\bpostgres(ql)?\b"]),

    # ML / AI
    "PyTorch":      ("ml",        [r"\bpytorch\b"]),
    "TensorFlow":   ("ml",        [r"\btensor\s*flow\b"]),
    "scikit-learn": ("ml",        [r"\bscikit[- ]learn\b", r"\bsklearn\b"]),
    "XGBoost":      ("ml",        [r"\bxg\s*boost\b"]),
    "LLM":          ("ml",        [r"\bLLM(s)?\b", r"\blarge language model"]),
    "RAG":          ("ml",        [r"\bRAG\b", r"retrieval[- ]augmented"]),
    "LangChain":    ("ml",        [r"\blangchain\b"]),
    "HuggingFace":  ("ml",        [r"\bhugging\s*face\b"]),
    "MLOps":        ("ml",        [r"\bmlops\b"]),

    # BI / analytics
    "Tableau":      ("bi",        [r"\btableau\b"]),
    "Power BI":     ("bi",        [r"\bpower\s*bi\b"]),
    "Looker":       ("bi",        [r"\blooker\b"]),
    "Metabase":     ("bi",        [r"\bmetabase\b"]),
    "Excel":        ("bi",        [r"\bexcel\b"]),

    # Cloud / infra
    "AWS":          ("cloud",     [r"\bAWS\b", r"\bamazon web services\b"]),
    "GCP":          ("cloud",     [r"\bGCP\b", r"\bgoogle cloud\b"]),
    "Azure":        ("cloud",     [r"\bazure\b"]),
    "Docker":       ("cloud",     [r"\bdocker\b"]),
    "Kubernetes":   ("cloud",     [r"\bkubernetes\b", r"\bk8s\b"]),
    "Terraform":    ("cloud",     [r"\bterraform\b"]),

    # Web
    "React":        ("web",       [r"\breact(?:\.js)?\b"]),
    "FastAPI":      ("web",       [r"\bfast\s*api\b"]),
    "Django":       ("web",       [r"\bdjango\b"]),
    "Next.js":      ("web",       [r"\bnext\.?js\b"]),
}


# ─────────────────────────────────────────────────────────────────────────────
# German-language gate heuristics
# ─────────────────────────────────────────────────────────────────────────────

GERMAN_REQUIRED_PATTERNS = [
    r"\bmuttersprachlich(e[rns]?)?\s+deutsch",
    r"deutsch\s*(auf|in|:)?\s*(mutter|c[12]|b[12]|native)",
    r"verhandlungssicher(e|es)?\s+deutsch",
    r"fließend(e[rns]?)?\s+deutsch",
    r"deutsch(kenntnisse|sprachig)\s+(erforderlich|zwingend|notwendig|Voraussetzung)",
    r"sehr\s+gute\s+deutschkenntnisse",
    r"native[- ]level\s+german",
    r"fluent\s+german",
    r"business[- ]fluent\s+german",
    r"german\s+(is\s+)?(a\s+)?(must|required|mandatory)",
]

ENGLISH_FRIENDLY_PATTERNS = [
    r"english[- ]speaking",
    r"english\s+only",
    r"working\s+language\s+is\s+english",
    r"no\s+german\s+(required|needed)",
    r"english\s+is\s+sufficient",
]


def detect_language_gate(text: str) -> str:
    """Returns one of: 'german_required', 'english_ok', 'unclear'.

    English-friendly patterns are checked first so phrases like 'No German
    required' aren't misclassified by the german_required matcher.
    """
    t = text.lower()
    for pat in ENGLISH_FRIENDLY_PATTERNS:
        if re.search(pat, t):
            return "english_ok"
    for pat in GERMAN_REQUIRED_PATTERNS:
        if re.search(pat, t):
            return "german_required"
    return "unclear"


# ─────────────────────────────────────────────────────────────────────────────
# Seniority
# ─────────────────────────────────────────────────────────────────────────────

SENIORITY_PATTERNS = {
    "werkstudent":  [r"\bwerk\s*student(in)?\b", r"\bworking\s+student\b"],
    "intern":       [r"\bintern(ship)?\b", r"\bpraktikum\b", r"\bprakti\s*kant"],
    "junior":       [r"\bjunior\b", r"\beinsteiger\b", r"\bentry[- ]level\b"],
    "senior":       [r"\bsenior\b", r"\blead\b", r"\bstaff\b", r"\bprincipal\b"],
    "mid":          [r"\bmid[- ]level\b"],
}


def detect_seniority(title: str, description: str) -> str:
    haystack = f"{title}\n{description}".lower()
    # Priority order: werkstudent > intern > senior > junior > mid > unknown
    for level in ["werkstudent", "intern", "senior", "junior", "mid"]:
        for pat in SENIORITY_PATTERNS[level]:
            if re.search(pat, haystack):
                return level
    return "unknown"


# ─────────────────────────────────────────────────────────────────────────────
# Salary parsing
# ─────────────────────────────────────────────────────────────────────────────
# Handles: "€45.000-55.000", "45k-55k EUR", "€25/Stunde", "25 EUR pro Stunde",
# "50.000 – 65.000 €", "€60,000 - 80,000 per year"

_NUM = r"(\d{1,3}(?:[.,]\d{3})*(?:[.,]\d+)?|\d+)"
_CURRENCY = r"(?:€|EUR|eur)"

# Annual: "€45.000-55.000" or "45k-55k EUR" or "45.000 – 55.000 €" or "€60,000 - €80,000"
ANNUAL_RANGE = re.compile(
    rf"{_CURRENCY}?\s*{_NUM}(k)?\s*(?:-|–|to|bis)\s*{_CURRENCY}?\s*{_NUM}(k)?\s*{_CURRENCY}?",
    re.IGNORECASE,
)

# Hourly: "€25/hour", "25 EUR pro Stunde", "€18/h"
HOURLY = re.compile(
    rf"{_CURRENCY}?\s*{_NUM}\s*{_CURRENCY}?\s*(?:/|pro|per)\s*(?:h(?:our)?|Stunde|Std)\.?",
    re.IGNORECASE,
)


@dataclass
class SalaryParse:
    min_eur: float | None
    max_eur: float | None
    period: str | None  # 'year' | 'hour'


def _to_float(s: str, k_suffix: bool) -> float:
    # German: '.' is thousands sep, ',' is decimal. English mirror image.
    # Heuristic: if both '.' and ',' present, last one is decimal.
    if "." in s and "," in s:
        if s.rfind(",") > s.rfind("."):
            s = s.replace(".", "").replace(",", ".")
        else:
            s = s.replace(",", "")
    elif "," in s:
        # Ambiguous; if there are exactly 3 digits after, treat as thousands
        parts = s.split(",")
        if len(parts) == 2 and len(parts[1]) == 3:
            s = s.replace(",", "")
        else:
            s = s.replace(",", ".")
    else:
        # Only '.' — could be thousands or decimal
        parts = s.split(".")
        if len(parts) == 2 and len(parts[1]) == 3:
            s = s.replace(".", "")
    val = float(s)
    if k_suffix:
        val *= 1000
    return val


def parse_salary(text: str) -> SalaryParse:
    if not text:
        return SalaryParse(None, None, None)

    # Try hourly first (more specific)
    m = HOURLY.search(text)
    if m:
        val = _to_float(m.group(1), False)
        # sanity: hourly rates 5–150 EUR
        if 5 <= val <= 150:
            return SalaryParse(val, val, "hour")

    # Then annual range
    m = ANNUAL_RANGE.search(text)
    if m:
        low_raw, low_k, high_raw, high_k = m.group(1), m.group(2), m.group(3), m.group(4)
        low = _to_float(low_raw, bool(low_k))
        high = _to_float(high_raw, bool(high_k))
        # sanity: annual salaries 10k–500k
        if 10_000 <= low <= 500_000 and 10_000 <= high <= 500_000 and low <= high:
            return SalaryParse(low, high, "year")

    return SalaryParse(None, None, None)


# ─────────────────────────────────────────────────────────────────────────────
# Skill extraction
# ─────────────────────────────────────────────────────────────────────────────

_COMPILED: dict[str, tuple[str, list[re.Pattern]]] = {
    name: (cat, [re.compile(p, re.IGNORECASE) for p in pats])
    for name, (cat, pats) in SKILL_TAXONOMY.items()
}


def extract_skills(text: str) -> list[str]:
    """Return a sorted list of canonical skill names found in text."""
    if not text:
        return []
    found = set()
    for name, (_cat, patterns) in _COMPILED.items():
        for p in patterns:
            if p.search(text):
                found.add(name)
                break
    return sorted(found)


def skill_categories() -> dict[str, str]:
    return {name: cat for name, (cat, _) in SKILL_TAXONOMY.items()}


# ─────────────────────────────────────────────────────────────────────────────
# City normalization
# ─────────────────────────────────────────────────────────────────────────────

CITY_ALIASES = {
    "berlin": "Berlin", "munich": "Munich", "münchen": "Munich",
    "muenchen": "Munich", "hamburg": "Hamburg", "cologne": "Cologne",
    "köln": "Cologne", "koeln": "Cologne", "frankfurt": "Frankfurt",
    "stuttgart": "Stuttgart", "leipzig": "Leipzig", "dresden": "Dresden",
    "düsseldorf": "Düsseldorf", "duesseldorf": "Düsseldorf",
    "hannover": "Hannover", "nuremberg": "Nuremberg", "nürnberg": "Nuremberg",
    "bremen": "Bremen", "essen": "Essen", "dortmund": "Dortmund",
    "karlsruhe": "Karlsruhe", "mannheim": "Mannheim",
}


def normalize_city(location: str, remote: bool) -> str:
    if not location:
        return "Remote" if remote else "Unknown"
    loc = location.lower()
    for alias, canonical in CITY_ALIASES.items():
        if alias in loc:
            return canonical
    if remote:
        return "Remote"
    return location.split(",")[0].strip().title() or "Unknown"
