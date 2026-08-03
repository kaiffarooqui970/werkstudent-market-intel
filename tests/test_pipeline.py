"""End-to-end pipeline test using bundled sample data."""

from __future__ import annotations

import tempfile
from pathlib import Path

import duckdb
import pytest

from src.ingest.load_sample import load
from src.transform.run_sql import run_all


@pytest.fixture
def db_path():
    with tempfile.TemporaryDirectory() as d:
        yield str(Path(d) / "test.duckdb")


def test_load_sample_creates_raw_jobs(db_path):
    n = load(db_path, sample_dir="sample_data")
    assert n > 0
    con = duckdb.connect(db_path, read_only=True)
    count = con.execute("SELECT COUNT(*) FROM raw_jobs").fetchone()[0]
    assert count == n
    con.close()


def test_full_pipeline_builds_marts(db_path):
    load(db_path, sample_dir="sample_data")
    run_all(db_path)

    con = duckdb.connect(db_path, read_only=True)
    tables = [r[0] for r in con.execute("SHOW TABLES").fetchall()]
    for expected in [
        "raw_jobs", "enriched_jobs", "dim_skills", "bridge_job_skill",
        "fct_jobs", "mart_skill_demand", "mart_werkstudent_pay", "mart_language_gate",
    ]:
        assert expected in tables, f"missing {expected}"

    # sanity: skills bridge is non-empty
    n_bridge = con.execute("SELECT COUNT(*) FROM bridge_job_skill").fetchone()[0]
    assert n_bridge > 0

    # sanity: at least one language gate categorization exists
    n_gated = con.execute(
        "SELECT COUNT(*) FROM fct_jobs WHERE language_gate <> 'unclear'"
    ).fetchone()[0]
    assert n_gated > 0
    con.close()
