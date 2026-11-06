-- GROUP BY and CASE: descriptive rates, not an approval or causal rule.
SELECT CASE WHEN limit_bal < 50000 THEN '01_under_50k'
            WHEN limit_bal < 150000 THEN '02_50k_to_150k'
            WHEN limit_bal < 300000 THEN '03_150k_to_300k'
            ELSE '04_300k_plus' END AS limit_band,
       COUNT(*) AS customers, SUM(target) AS defaults,
       ROUND(AVG(target), 4) AS default_rate
FROM customers c JOIN split_manifest s ON s.id = c.id WHERE s.split = 'train'
GROUP BY limit_band ORDER BY limit_band;
