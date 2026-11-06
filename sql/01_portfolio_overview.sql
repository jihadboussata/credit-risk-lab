-- Business question: what is the observed base rate and scale of the cohort?
SELECT COUNT(*) AS customers,
       SUM(target) AS next_month_defaults,
       ROUND(AVG(target), 4) AS default_rate,
       ROUND(AVG(limit_bal), 2) AS average_credit_limit_ntd
FROM customers c JOIN split_manifest s ON s.id = c.id WHERE s.split = 'train';
