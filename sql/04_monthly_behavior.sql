-- Temporal aggregation is descriptive: all months precede the outcome.
SELECT h.month, c.target AS next_month_default,
       COUNT(*) AS accounts, ROUND(AVG(h.bill_amount), 2) AS mean_bill_ntd,
       ROUND(AVG(h.payment_amount), 2) AS mean_payment_ntd,
       ROUND(AVG(CASE WHEN h.repayment_status > 0 THEN 1.0 ELSE 0.0 END), 4) AS delayed_share
FROM repayment_history h JOIN customers c ON c.id = h.customer_id
JOIN split_manifest s ON s.id = c.id WHERE s.split = 'train'
GROUP BY h.month, c.target ORDER BY h.month, c.target;
