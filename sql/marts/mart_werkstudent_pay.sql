-- mart_werkstudent_pay: pay bands for Werkstudent-tagged roles by city and role_family.
-- Uses annualized figures for comparability across hourly/yearly quoted salaries.

CREATE OR REPLACE TABLE mart_werkstudent_pay AS
SELECT
    city,
    role_family,
    COUNT(*)                                                   AS n_postings,
    COUNT(salary_min_annualized_eur)                           AS n_with_salary,
    ROUND(AVG(salary_min_annualized_eur), 0)                   AS avg_min_eur,
    ROUND(quantile_cont(salary_min_annualized_eur, 0.25), 0)   AS p25_min_eur,
    ROUND(quantile_cont(salary_min_annualized_eur, 0.50), 0)   AS median_min_eur,
    ROUND(quantile_cont(salary_min_annualized_eur, 0.75), 0)   AS p75_min_eur
FROM fct_jobs
WHERE seniority IN ('werkstudent', 'intern', 'junior')
GROUP BY 1, 2
HAVING n_postings >= 2;
