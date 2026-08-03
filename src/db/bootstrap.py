"""Schema bootstrap. Idempotent: safe to run on every connection."""

from __future__ import annotations

import duckdb

SCHEMA = """
CREATE TABLE IF NOT EXISTS raw_jobs (
    slug          VARCHAR PRIMARY KEY,
    title         VARCHAR,
    company_name  VARCHAR,
    location      VARCHAR,
    remote        BOOLEAN,
    url           VARCHAR,
    tags          VARCHAR,           -- json array
    job_types     VARCHAR,           -- json array
    description   VARCHAR,
    created_at    VARCHAR,
    snapshot_date DATE,
    raw_payload   VARCHAR,
    ingested_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_raw_jobs_snapshot ON raw_jobs(snapshot_date);
"""


def ensure_schema(con: duckdb.DuckDBPyConnection) -> None:
    con.execute(SCHEMA)
