-- HAVING and a scalar subquery: segments above the cohort's observed base rate.
SELECT education, marriage, COUNT(*) AS customers,
       ROUND(AVG(target), 4) AS default_rate
FROM customers c JOIN split_manifest s ON s.id = c.id WHERE s.split = 'train'
GROUP BY education, marriage
HAVING COUNT(*) >= 100 AND AVG(target) > (
    SELECT AVG(c2.target) FROM customers c2
    JOIN split_manifest s2 ON s2.id = c2.id WHERE s2.split = 'train'
)
ORDER BY default_rate DESC;
