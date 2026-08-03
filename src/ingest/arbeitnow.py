"""
Arbeitnow job board API client.

Docs: https://documenter.getpostman.com/view/18545278/UVJbJdKh
Endpoint: https://www.arbeitnow.com/api/job-board-api

The API is unauthenticated and paginated. We fetch all pages, dedup by slug,
and insert into DuckDB's `raw_jobs` table with a snapshot_date column so we
can track appearance/disappearance of postings over time.
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Iterator

import duckdb
import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.db.bootstrap import ensure_schema

logger = logging.getLogger(__name__)

API_URL = "https://www.arbeitnow.com/api/job-board-api"
MAX_PAGES = 50  # safety cap
PAGE_TIMEOUT_S = 20


@retry(
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    retry=retry_if_exception_type((httpx.HTTPError, httpx.TimeoutException)),
    reraise=True,
)
def _fetch_page(client: httpx.Client, page: int) -> dict:
    resp = client.get(API_URL, params={"page": page}, timeout=PAGE_TIMEOUT_S)
    resp.raise_for_status()
    return resp.json()


def iter_pages(max_pages: int = MAX_PAGES) -> Iterator[list[dict]]:
    """Yield lists of job dicts, one list per page, until API returns empty."""
    with httpx.Client(headers={"User-Agent": "werkstudent-intel/0.1 (portfolio project)"}) as c:
        for page in range(1, max_pages + 1):
            payload = _fetch_page(c, page)
            data = payload.get("data", [])
            if not data:
                logger.info("empty page %d, stopping", page)
                return
            logger.info("fetched page %d (%d jobs)", page, len(data))
            yield data


def _normalize(raw: dict, snapshot: date) -> dict:
    """Coerce raw API row into our raw_jobs schema. Keep it forgiving."""
    created = raw.get("created_at")
    if isinstance(created, (int, float)):
        created_at = datetime.fromtimestamp(created, tz=timezone.utc).isoformat()
    else:
        created_at = str(created) if created else None

    return {
        "slug": raw.get("slug") or "",
        "title": raw.get("title") or "",
        "company_name": raw.get("company_name") or "",
        "location": raw.get("location") or "",
        "remote": bool(raw.get("remote")),
        "url": raw.get("url") or "",
        "tags": json.dumps(raw.get("tags") or []),
        "job_types": json.dumps(raw.get("job_types") or []),
        "description": raw.get("description") or "",
        "created_at": created_at,
        "snapshot_date": snapshot.isoformat(),
        "raw_payload": json.dumps(raw),
    }


def ingest(db_path: str, max_pages: int = MAX_PAGES) -> int:
    """Fetch all pages and upsert into raw_jobs. Returns count inserted."""
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(db_path)
    ensure_schema(con)

    snapshot = date.today()
    rows: list[dict] = []
    for page_jobs in iter_pages(max_pages=max_pages):
        rows.extend(_normalize(j, snapshot) for j in page_jobs)

    if not rows:
        logger.warning("no rows fetched")
        return 0

    # dedup within batch by (slug, snapshot_date)
    seen = set()
    deduped = []
    for r in rows:
        key = (r["slug"], r["snapshot_date"])
        if key in seen or not r["slug"]:
            continue
        seen.add(key)
        deduped.append(r)

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
            for r in deduped
        ],
    )
    con.close()
    logger.info("inserted %d rows into raw_jobs", len(deduped))
    return len(deduped)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--db", default="data/warehouse.duckdb")
    p.add_argument("--max-pages", type=int, default=MAX_PAGES)
    args = p.parse_args()
    n = ingest(args.db, max_pages=args.max_pages)
    print(f"ingested {n} rows")


if __name__ == "__main__":
    main()
