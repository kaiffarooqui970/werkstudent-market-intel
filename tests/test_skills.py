"""Unit tests for the transform layer."""

from __future__ import annotations

import pytest

from src.transform.skills import (
    detect_language_gate,
    detect_seniority,
    extract_skills,
    normalize_city,
    parse_salary,
)


class TestSkillExtraction:
    def test_python_and_sql(self):
        skills = extract_skills("We use Python and SQL heavily.")
        assert "Python" in skills
        assert "SQL" in skills

    def test_case_insensitive(self):
        assert "Python" in extract_skills("python developer wanted")
        assert "AWS" in extract_skills("experience with aws is a plus")

    def test_javascript_variants(self):
        assert "JavaScript" in extract_skills("Strong JS skills required")
        assert "JavaScript" in extract_skills("JavaScript experience")

    def test_avoids_false_positive_java(self):
        # 'JavaScript' should NOT trigger 'Java'
        skills = extract_skills("JavaScript is our main language")
        assert "JavaScript" in skills
        assert "Java" not in skills

    def test_empty_text(self):
        assert extract_skills("") == []
        assert extract_skills(None) == []  # type: ignore[arg-type]

    def test_kubernetes_alias(self):
        assert "Kubernetes" in extract_skills("we run everything on k8s")

    def test_deduplication(self):
        # Same skill mentioned twice returns once
        skills = extract_skills("Python. Python. Python.")
        assert skills.count("Python") == 1


class TestLanguageGate:
    @pytest.mark.parametrize("text", [
        "Sehr gute Deutschkenntnisse erforderlich",
        "Verhandlungssicheres Deutsch",
        "Fluent German is a must",
        "German is required for this role",
        "Deutsch auf Muttersprachenniveau",
    ])
    def test_detects_german_required(self, text):
        assert detect_language_gate(text) == "german_required"

    @pytest.mark.parametrize("text", [
        "Our working language is English",
        "This is an english-speaking team",
        "No German required",
    ])
    def test_detects_english_ok(self, text):
        assert detect_language_gate(text) == "english_ok"

    def test_unclear(self):
        assert detect_language_gate("Great team, cool product") == "unclear"


class TestSeniority:
    def test_werkstudent_wins(self):
        assert detect_seniority("Werkstudent Data Analyst", "junior role") == "werkstudent"

    def test_intern(self):
        assert detect_seniority("Data Science Intern", "") == "intern"
        assert detect_seniority("Praktikum Data", "") == "intern"

    def test_senior(self):
        assert detect_seniority("Senior Data Engineer", "") == "senior"

    def test_unknown(self):
        assert detect_seniority("Data Scientist", "cool role") == "unknown"


class TestSalaryParse:
    def test_annual_range_euro_german(self):
        r = parse_salary("Gehalt: €45.000–55.000 pro Jahr")
        assert r.min_eur == 45000
        assert r.max_eur == 55000
        assert r.period == "year"

    def test_annual_range_k_suffix(self):
        r = parse_salary("Salary: 60k - 80k EUR per year")
        assert r.min_eur == 60000
        assert r.max_eur == 80000
        assert r.period == "year"

    def test_annual_range_english_comma(self):
        r = parse_salary("€60,000 - €80,000 per year")
        assert r.min_eur == 60000
        assert r.max_eur == 80000

    def test_hourly_german(self):
        r = parse_salary("Vergütung: €18/Stunde")
        assert r.min_eur == 18
        assert r.period == "hour"

    def test_hourly_english(self):
        r = parse_salary("€25 per hour")
        assert r.min_eur == 25
        assert r.period == "hour"

    def test_no_salary(self):
        r = parse_salary("Great benefits and team")
        assert r.min_eur is None
        assert r.max_eur is None

    def test_rejects_out_of_range(self):
        # €500/hour is unrealistic for Werkstudent, rejected by sanity check
        r = parse_salary("€500 per hour")
        assert r.min_eur is None


class TestCityNormalize:
    def test_german_umlaut_variants(self):
        assert normalize_city("Munich, Germany", False) == "Munich"
        assert normalize_city("München", False) == "Munich"
        assert normalize_city("Muenchen, DE", False) == "Munich"

    def test_koeln_variants(self):
        assert normalize_city("Köln", False) == "Cologne"
        assert normalize_city("Koeln", False) == "Cologne"

    def test_remote_no_location(self):
        assert normalize_city("", True) == "Remote"

    def test_unknown_no_location(self):
        assert normalize_city("", False) == "Unknown"
