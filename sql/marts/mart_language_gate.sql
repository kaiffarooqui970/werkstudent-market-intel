-- mart_language_gate: what fraction of postings require German, by city × role_family.
-- The core "English-friendly filter" data for international students.

CREATE OR REPLACE TABLE mart_language_gate AS
SELECT
    city,
    role_family,
    COUNT(*)                                                              AS n_postings,
    SUM(CASE WHEN language_gate = 'german_required' THEN 1 ELSE 0 END)    AS n_german_required,
    SUM(CASE WHEN language_gate = 'english_ok'       THEN 1 ELSE 0 END)   AS n_english_ok,
    SUM(CASE WHEN language_gate = 'unclear'          THEN 1 ELSE 0 END)   AS n_unclear,
    ROUND(100.0 * SUM(CASE WHEN language_gate = 'german_required' THEN 1 ELSE 0 END) / NULLIF(COUNT(*), 0), 1) AS pct_german_required,
    ROUND(100.0 * SUM(CASE WHEN language_gate = 'english_ok'       THEN 1 ELSE 0 END) / NULLIF(COUNT(*), 0), 1) AS pct_english_ok
FROM fct_jobs
GROUP BY 1, 2
HAVING n_postings >= 3;
