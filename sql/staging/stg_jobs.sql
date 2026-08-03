-- stg_jobs: one row per posting, cleaned types, joined with Python enrichments.
-- Grain: (slug)

CREATE OR REPLACE VIEW stg_jobs AS
SELECT
    r.slug,
    r.title,
    r.company_name,
    r.location AS raw_location,
    e.city,
    r.remote,
    r.url,
    TRY_CAST(r.created_at AS TIMESTAMP)      AS posted_at,
    r.snapshot_date,
    r.tags,
    r.job_types,
    r.description,
    e.seniority,
    e.language_gate,
    e.salary_min_eur,
    e.salary_max_eur,
    e.salary_period,
    e.skills_json
FROM raw_jobs r
LEFT JOIN enriched_jobs e USING (slug);
