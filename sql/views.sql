CREATE OR REPLACE VIEW features_view AS

/* CTE which aggregates order_items table and reduces the order and the items it comprises
to a single row, including the id of a most valuable item in the order. */
WITH order_items_agg AS (
	SELECT 
		order_id, 
		MAX(order_item_id) AS num_items,
		SUM(price) AS total_price,
		SUM(freight_value) AS total_freight, 
		MAX(shipping_limit_date) AS limit_date,
		
	-- creating product_id array, sorting prices in a descending order - first element is the most important product
		(ARRAY_AGG(product_id ORDER BY price DESC))[1] AS most_imp_product_id
	FROM order_items
	GROUP BY order_id
)

SELECT 
	-- target feature
	r.review_score,
	CASE WHEN r.review_score <= 3 THEN 1 ELSE 0 END AS bad_score,
	
	-- features from aggregated item_orders table
	i.num_items,
	i.total_price,
	i.total_freight,
	ROUND((i.total_freight / i.total_price),4) AS freight_ratio,

	-- features from product table
	COALESCE(p.product_name_lenght, 0) AS name_len,
	COALESCE(p.product_description_lenght, 0) AS desc_len,
	COALESCE(c.product_category_name_english, p.product_category_name) AS category,
	COALESCE(p.product_photos_qty, 0) AS prod_photos,
	COALESCE(p.product_weight_g, 0) AS weight_g, 
	
	-- reducing 3 dimensions into a volume variable
	COALESCE(ROUND((p.product_length_cm * p.product_height_cm * p.product_width_cm / 1000.0),2), 0) AS volume_l, 
	CASE WHEN p.product_name_lenght IS NULL OR p.product_description_lenght IS NULL OR
		p.product_photos_qty IS NULL or p.product_weight_g IS NULL THEN 1 ELSE 0 
	END AS has_missing_details,

	/* extracting important features from timestamps concernig delivery length, differnce between the actual delivery date and the estimated one, and the time it took the seller to process the order */
	EXTRACT (DAY FROM(o.order_delivered_customer_date - o.order_purchase_timestamp)) AS delivery_days,
	EXTRACT (DAY FROM(o.order_delivered_customer_date - o.order_estimated_delivery_date)) AS estimated_delivery_diff,
	EXTRACT (DAY FROM(o.order_delivered_carrier_date - i.limit_date)) AS seller_disp_diff,
	EXTRACT (DAY FROM(o.order_delivered_carrier_date - o.order_approved_at)) AS processing_days,

	CASE WHEN o.order_delivered_carrier_date > i.limit_date THEN 1 ELSE 0 
		END AS is_seller_late,
	CASE WHEN o.order_delivered_customer_date::DATE > o.order_estimated_delivery_date::DATE THEN 1 ELSE 0 
		END AS is_delivery_late
	
FROM orders AS o
JOIN order_items_agg AS i ON o.order_id = i.order_id
JOIN products AS p ON i.most_imp_product_id = p.product_id
JOIN reviews AS r ON o.order_id = r.order_id
LEFT JOIN category_names AS c ON p.product_category_name = c.product_category_name
WHERE o.order_status = 'delivered' AND o.order_delivered_customer_date IS NOT NULL 
