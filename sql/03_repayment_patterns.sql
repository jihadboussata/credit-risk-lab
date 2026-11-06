-- JOIN and CTE: one historical feature row per customer, avoiding multiplication.
WITH behavior AS (
    SELECT customer_id,
           SUM(CASE WHEN repayment_status > 0 THEN 1 ELSE 0 END) AS delayed_months,
           AVG(payment_amount) AS mean_recorded_payment
    FROM repayment_history GROUP BY customer_id
)
SELECT b.delayed_months, COUNT(*) AS customers,
       ROUND(AVG(c.target), 4) AS default_rate,
       ROUND(AVG(b.mean_recorded_payment), 2) AS average_payment_ntd
FROM behavior b JOIN customers c ON c.id = b.customer_id
JOIN split_manifest s ON s.id = c.id WHERE s.split = 'train'
GROUP BY b.delayed_months ORDER BY b.delayed_months;
