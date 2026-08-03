-- mart_skill_demand: skill × city × week counts, plus a "% of postings requiring German"
-- overlay so you can spot skills where German is disproportionately gated.

CREATE OR REPLACE TABLE mart_skill_demand AS
WITH exploded AS (
    SELECT
        f.slug,
        f.city,
        f.role_family,
        f.language_gate,
        DATE_TRUNC('week', f.posted_at) AS week,
        b.skill_name
    FROM fct_jobs f
    JOIN bridge_job_skill b USING (slug)
),
counts AS (
    SELECT
        skill_name,
        city,
        role_family,
        week,
        COUNT(*) AS n_postings,
        SUM(CASE WHEN language_gate = 'german_required' THEN 1 ELSE 0 END) AS n_german_required,
        SUM(CASE WHEN language_gate = 'english_ok'       THEN 1 ELSE 0 END) AS n_english_ok
    FROM exploded
    GROUP BY 1, 2, 3, 4
)
SELECT
    c.*,
    d.category AS skill_category,
    ROUND(100.0 * n_german_required / NULLIF(n_postings, 0), 1) AS pct_german_required
FROM counts c
LEFT JOIN dim_skills d ON d.skill_name = c.skill_name;
