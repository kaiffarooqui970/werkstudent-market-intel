-- fct_jobs: analytics-ready fact. Grain: (slug).
-- Adds derived columns useful for the dashboard and salary model.

CREATE OR REPLACE TABLE fct_jobs AS
SELECT
    slug,
    title,
    company_name,
    city,
    remote,
    url,
    posted_at,
    snapshot_date,
    seniority,
    language_gate,
    salary_min_eur,
    salary_max_eur,
    salary_period,
    -- annualize hourly rates assuming 20h/week Werkstudent, 52 weeks
    CASE
        WHEN salary_period = 'hour' THEN salary_min_eur * 20 * 52
        WHEN salary_period = 'year' THEN salary_min_eur
    END AS salary_min_annualized_eur,
    -- role family: coarse bucket from title
    CASE
        WHEN LOWER(title) LIKE '%data scien%' OR LOWER(title) LIKE '%machine learn%' OR LOWER(title) LIKE '%ml engineer%' THEN 'DS/ML'
        WHEN LOWER(title) LIKE '%data analyst%' OR LOWER(title) LIKE '%analytics%' OR LOWER(title) LIKE '%bi %' THEN 'Analytics'
        WHEN LOWER(title) LIKE '%data engineer%' OR LOWER(title) LIKE '%data platform%' THEN 'Data Eng'
        WHEN LOWER(title) LIKE '%ai %' OR LOWER(title) LIKE '%llm%' OR LOWER(title) LIKE '%genai%' THEN 'AI'
        WHEN LOWER(title) LIKE '%backend%' OR LOWER(title) LIKE '%frontend%' OR LOWER(title) LIKE '%fullstack%' OR LOWER(title) LIKE '%software%' THEN 'Software Eng'
        ELSE 'Other'
    END AS role_family,
    description,
    skills_json
FROM stg_jobs;
