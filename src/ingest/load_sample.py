"""Load bundled sample_data/*.json into raw_jobs so the demo works offline."""

from __future__ import annotations

import argparse
import json
import logging
from datetime import date
from pathlib import Path

import duckdb

from src.db.bootstrap import ensure_schema
from src.ingest.arbeitnow import _normalize

logger = logging.getLogger(__name__)


def load(db_path: str, sample_dir: str = "sample_data") -> int:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(db_path)
    ensure_schema(con)

    files = sorted(Path(sample_dir).glob("*.json"))
    if not files:
        raise SystemExit(f"no sample files in {sample_dir}")

    total = 0
    for f in files:
        raw = json.loads(f.read_text())
        snapshot = date.fromisoformat(raw.get("snapshot_date", str(date.today())))
        rows = [_normalize(j, snapshot) for j in raw["data"]]
        con.executemany(
            """
            INSERT OR REPLACE INTO raw_jobs
              (slug, title, company_name, location, remote, url, tags, job_types,
               description, created_at, snapshot_date, raw_payload)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    r["slug"], r["title"], r["company_name"], r["location"], r["remote"],
                    r["url"], r["tags"], r["job_types"], r["description"], r["created_at"],
                    r["snapshot_date"], r["raw_payload"],
                )
                for r in rows
            ],
        )
        total += len(rows)
        logger.info("loaded %d rows from %s", len(rows), f.name)

    con.close()
    return total


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    p = argparse.ArgumentParser()
    p.add_argument("--db", default="data/warehouse.duckdb")
    p.add_argument("--sample-dir", default="sample_data")
    args = p.parse_args()
    n = load(args.db, args.sample_dir)
    print(f"loaded {n} sample rows")


if __name__ == "__main__":
    main()
