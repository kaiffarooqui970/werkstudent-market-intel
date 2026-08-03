"""
Generate realistic sample job postings so the demo runs offline.
Not for production — for the README screenshots and initial dashboard load.
"""

from __future__ import annotations

import json
import random
from datetime import date, datetime, timedelta
from pathlib import Path

random.seed(42)

CITIES = [
    ("Berlin", "Berlin, Germany"),
    ("Munich", "Munich, Germany"),
    ("Hamburg", "Hamburg, Germany"),
    ("Frankfurt", "Frankfurt, Germany"),
    ("Cologne", "Cologne, Germany"),
    ("Leipzig", "Leipzig, Germany"),
    ("Stuttgart", "Stuttgart, Germany"),
    ("Dresden", "Dresden, Germany"),
]

ROLE_TEMPLATES = [
    ("Werkstudent Data Analytics (m/w/d)", "Analytics", ["Python", "SQL", "Excel", "Tableau"]),
    ("Werkstudent:in Data Science", "DS/ML", ["Python", "SQL", "Pandas", "scikit-learn"]),
    ("Working Student - Machine Learning", "DS/ML", ["Python", "PyTorch", "MLOps"]),
    ("Data Analyst (Werkstudent)", "Analytics", ["SQL", "Power BI", "Excel"]),
    ("Junior Data Engineer", "Data Eng", ["Python", "SQL", "Airflow", "Snowflake"]),
    ("Werkstudent Business Intelligence", "Analytics", ["SQL", "Looker", "dbt"]),
    ("AI Engineering Intern", "AI", ["Python", "LLM", "RAG", "LangChain"]),
    ("Data Scientist (m/w/d) - Werkstudent", "DS/ML", ["Python", "XGBoost", "SQL"]),
    ("Software Engineering Intern - Data Platform", "Software Eng", ["Python", "Kafka", "Docker"]),
    ("Werkstudent GenAI / LLM Engineering", "AI", ["Python", "LLM", "HuggingFace", "PyTorch"]),
    ("Analytics Engineer (Junior)", "Data Eng", ["dbt", "SQL", "BigQuery"]),
    ("Werkstudent BI & Reporting", "Analytics", ["SQL", "Power BI", "Excel"]),
]

COMPANIES = [
    "Zalando", "Delivery Hero", "N26", "Trade Republic", "Personio", "Celonis",
    "Wayfair", "HelloFresh", "SAP", "Siemens Healthineers", "Mercedes-Benz Tech",
    "Allianz", "Otto Group", "About You", "Flink", "GetYourGuide", "IONOS",
    "1&1", "T-Systems", "Deutsche Bahn Digital", "Bosch", "Continental",
]

GERMAN_PHRASES = [
    "Sehr gute Deutschkenntnisse erforderlich (mind. C1).",
    "Verhandlungssicheres Deutsch in Wort und Schrift.",
    "Fließende Deutschkenntnisse zwingend notwendig.",
    "Deutsch auf Muttersprachenniveau.",
]
ENGLISH_PHRASES = [
    "Our working language is English — no German required.",
    "This is an English-speaking role.",
    "English is sufficient; German is a nice-to-have.",
]

SALARY_TEMPLATES_ANNUAL = [
    "Gehalt: €{lo}.000–{hi}.000 pro Jahr",
    "Salary range: €{lo},000 - €{hi},000 per year",
    "Wir bieten {lo}k–{hi}k EUR p.a.",
]
SALARY_TEMPLATES_HOURLY = [
    "Vergütung: €{rate}/Stunde",
    "€{rate} per hour",
    "{rate} EUR pro Stunde",
]


def _make_description(role_family: str, skills: list[str], senior: str, city: str) -> tuple[str, str]:
    lang = random.choices(["german_required", "english_ok", "unclear"], weights=[0.35, 0.4, 0.25])[0]
    lang_line = ""
    if lang == "german_required":
        lang_line = random.choice(GERMAN_PHRASES)
    elif lang == "english_ok":
        lang_line = random.choice(ENGLISH_PHRASES)

    sal_line = ""
    if random.random() < 0.30:  # ~30% report salary; realistic-ish
        if senior in ("werkstudent", "intern") and random.random() < 0.6:
            rate = random.randint(15, 28)
            sal_line = random.choice(SALARY_TEMPLATES_HOURLY).format(rate=rate)
        else:
            lo = random.randint(35, 70)
            hi = lo + random.randint(10, 30)
            sal_line = random.choice(SALARY_TEMPLATES_ANNUAL).format(lo=lo, hi=hi)

    intro = f"Join our {role_family} team in {city}. "
    stack = "Our stack includes " + ", ".join(skills) + "."
    resp = ("You'll work on production data pipelines, dashboards for stakeholders, "
            "and experiments that shape product decisions. ")
    parts = [intro, resp, stack, lang_line, sal_line]
    return " ".join(p for p in parts if p), lang


def make_posting(i: int) -> dict:
    city, location = random.choice(CITIES)
    title, role_family, skills = random.choice(ROLE_TEMPLATES)
    company = random.choice(COMPANIES)
    remote = random.random() < 0.35
    if random.random() < 0.2 and "Werkstudent" not in title:
        title = "Werkstudent " + title

    senior = "unknown"
    if "Werkstudent" in title:
        senior = "werkstudent"
    elif "Intern" in title:
        senior = "intern"
    elif "Junior" in title:
        senior = "junior"

    desc, _lang = _make_description(role_family, skills, senior, city)
    posted = datetime.now() - timedelta(days=random.randint(0, 40))

    return {
        "slug": f"{company.lower().replace(' ', '-')}-{title.lower().replace(' ', '-')[:40]}-{i}",
        "title": title,
        "company_name": company,
        "location": location,
        "remote": remote,
        "url": f"https://arbeitnow.example/{i}",
        "tags": [role_family.lower(), city.lower()],
        "job_types": ["werkstudent"] if senior == "werkstudent" else ["intern"] if senior == "intern" else ["full-time"],
        "description": desc,
        "created_at": int(posted.timestamp()),
    }


def main() -> None:
    n = 250
    postings = [make_posting(i) for i in range(n)]
    out = {"snapshot_date": date.today().isoformat(), "data": postings}

    Path("sample_data").mkdir(exist_ok=True)
    Path("sample_data/seed.json").write_text(json.dumps(out, indent=2))
    print(f"wrote {n} postings to sample_data/seed.json")


if __name__ == "__main__":
    main()
