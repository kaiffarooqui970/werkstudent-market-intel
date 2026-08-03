"""
Executes SQL models against DuckDB in dependency order.
Plus one Python-side step: parsing skills / seniority / salary / language gate
from the description column, since regex is easier here than in SQL.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import duckdb
import pandas as pd

from src.db.bootstrap import ensure_schema
from src.transform.skills import (
    detect_language_gate,
    detect_seniority,
    extract_skills,
    normalize_city,
    parse_salary,
    skill_categories,
)

logger = logging.getLogger(__name__)

SQL_DIR = Path("sql")


def _run_sql_file(con: duckdb.DuckDBPyConnection, path: Path) -> None:
    logger.info("running %s", path)
    con.execute(path.read_text())


def _enrich_python(con: duckdb.DuckDBPyConnection) -> None:
    """Python-side enrichment: skill extraction, seniority, salary, language gate."""
    df = con.execute(
        "SELECT slug, title, description, location, remote FROM raw_jobs"
    ).fetch_df()

    if df.empty:
        logger.warning("raw_jobs is empty; skipping enrichment")
        return

    logger.info("enriching %d rows in Python", len(df))

    df["city"] = df.apply(lambda r: normalize_city(r["location"], bool(r["remote"])), axis=1)
    df["seniority"] = df.apply(
        lambda r: detect_seniority(r["title"] or "", r["description"] or ""), axis=1
    )
    df["language_gate"] = df["description"].fillna("").map(detect_language_gate)

    sal = df["description"].fillna("").map(parse_salary)
    df["salary_min_eur"] = [s.min_eur for s in sal]
    df["salary_max_eur"] = [s.max_eur for s in sal]
    df["salary_period"] = [s.period for s in sal]

    skills_lists = df["description"].fillna("").map(extract_skills)
    df["skills_json"] = skills_lists.map(json.dumps)

    enriched = df[[
        "slug", "city", "seniority", "language_gate",
        "salary_min_eur", "salary_max_eur", "salary_period", "skills_json",
    ]]

    con.execute("DROP TABLE IF EXISTS enriched_jobs")
    con.register("enriched_df", enriched)
    con.execute("CREATE TABLE enriched_jobs AS SELECT * FROM enriched_df")
    con.unregister("enriched_df")

    # Also build dim_skills + bridge_job_skill
    cats = skill_categories()
    dim_rows = [{"skill_name": n, "category": c} for n, c in cats.items()]
    con.execute("DROP TABLE IF EXISTS dim_skills")
    con.register("dim_df", pd.DataFrame(dim_rows))
    con.execute("CREATE TABLE dim_skills AS SELECT * FROM dim_df")
    con.unregister("dim_df")

    bridge = []
    for slug, sk_list in zip(df["slug"], skills_lists):
        for s in sk_list:
            bridge.append({"slug": slug, "skill_name": s})
    con.execute("DROP TABLE IF EXISTS bridge_job_skill")
    con.register(
        "bridge_df",
        pd.DataFrame(bridge) if bridge else pd.DataFrame(columns=["slug", "skill_name"]),
    )
    con.execute("CREATE TABLE bridge_job_skill AS SELECT * FROM bridge_df")
    con.unregister("bridge_df")


def run_all(db_path: str) -> None:
    con = duckdb.connect(db_path)
    ensure_schema(con)

    # 1. Python enrichment
    _enrich_python(con)

    # 2. SQL models in order
    for sql_path in sorted((SQL_DIR / "staging").glob("*.sql")):
        _run_sql_file(con, sql_path)
    for sql_path in sorted((SQL_DIR / "marts").glob("*.sql")):
        _run_sql_file(con, sql_path)

    con.close()
    logger.info("transform complete")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--db", default="data/warehouse.duckdb")
    args = p.parse_args()
    run_all(args.db)


if __name__ == "__main__":
    main()
