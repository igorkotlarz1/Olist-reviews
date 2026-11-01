/*
One row per delivered order, describing what is known at the moment the parcel reaches the customer.
That is the prediction point: the model flags orders likely to get a 1-3 star review so support can
step in before the customer writes it.

Two rules keep the labels honest:
  * reviews answered BEFORE the parcel was delivered are excluded. Those customers complained about a
    parcel that had not arrived yet (about nine in ten of them are 1-3 stars), and every delivery feature
    would be information from after the review was written.
  * an order with several reviews keeps only its latest one, chosen deterministically.
  * seller and product history only counts reviews answered before this order was delivered.
*/
DROP VIEW IF EXISTS features_view;
CREATE VIEW features_view AS
WITH last_review AS (
    SELECT DISTINCT ON (order_id) order_id, review_score, review_answer_timestamp
    FROM reviews
    ORDER BY order_id, review_answer_timestamp DESC, review_id
),
items AS (
    SELECT
        order_id,
        COUNT(*)                                      AS num_items,
        COUNT(DISTINCT product_id)                    AS num_products,
        COUNT(DISTINCT seller_id)                     AS num_sellers,
        SUM(price)                                    AS total_price,
        SUM(freight_value)                            AS total_freight,
        MAX(shipping_limit_date)                      AS shipping_limit_date,
        -- the most expensive item represents the order
        (ARRAY_AGG(product_id ORDER BY price DESC, order_item_id))[1] AS main_product_id,
        (ARRAY_AGG(seller_id  ORDER BY price DESC, order_item_id))[1] AS main_seller_id
    FROM order_items
    GROUP BY order_id
),
payments AS (
    SELECT
        order_id,
        COUNT(*)                                                    AS num_payments,
        MAX(payment_installments)                                   AS installments,
        (ARRAY_AGG(payment_type ORDER BY payment_value DESC))[1]    AS payment_type
    FROM order_payments
    GROUP BY order_id
),
delivered AS (
    SELECT o.order_id, i.main_seller_id, i.main_product_id, o.order_delivered_customer_date::DATE AS delivered_on
    FROM orders o
    JOIN items i ON i.order_id = o.order_id
    WHERE o.order_status = 'delivered' AND o.order_delivered_customer_date IS NOT NULL
),
/* Reputation features: how many reviews the seller (product) had received before this order was
   delivered, and what share of them was bad. Computed as a running total over a timeline that mixes
   answered reviews (kind 1) and deliveries (kind 0). A review counts from the day it was answered,
   because that is when its score becomes known. Deliveries sort before reviews on the same day, so
   only reviews answered strictly before the delivery date are counted. */
seller_history AS (
    SELECT order_id, n_reviews, n_bad
    FROM (
        SELECT order_id, kind,
               SUM(kind) OVER w AS n_reviews,
               SUM(bad)  OVER w AS n_bad
        FROM (
            SELECT main_seller_id AS seller_id, delivered_on AS day, 0 AS kind, order_id, 0 AS bad FROM delivered
            UNION ALL
            SELECT seller_id, day, 1, NULL, bad
            FROM (  -- each review once per seller, even if the order had several of its items
                SELECT DISTINCT oi.seller_id, rv.review_id, rv.order_id,
                       rv.review_answer_timestamp::DATE AS day, (rv.review_score <= 3)::INT AS bad
                FROM reviews rv JOIN order_items oi ON oi.order_id = rv.order_id
            ) reviewed
        ) timeline
        WINDOW w AS (PARTITION BY seller_id ORDER BY day, kind ROWS UNBOUNDED PRECEDING)
    ) running
    WHERE kind = 0
),
product_history AS (
    SELECT order_id, n_reviews, n_bad
    FROM (
        SELECT order_id, kind,
               SUM(kind) OVER w AS n_reviews,
               SUM(bad)  OVER w AS n_bad
        FROM (
            SELECT main_product_id AS product_id, delivered_on AS day, 0 AS kind, order_id, 0 AS bad FROM delivered
            UNION ALL
            SELECT product_id, day, 1, NULL, bad
            FROM (  -- each review once per product, even if the order had several of its items
                SELECT DISTINCT oi.product_id, rv.review_id, rv.order_id,
                       rv.review_answer_timestamp::DATE AS day, (rv.review_score <= 3)::INT AS bad
                FROM reviews rv JOIN order_items oi ON oi.order_id = rv.order_id
            ) reviewed
        ) timeline
        WINDOW w AS (PARTITION BY product_id ORDER BY day, kind ROWS UNBOUNDED PRECEDING)
    ) running
    WHERE kind = 0
)
SELECT
    -- identifiers and target (not model features)
    o.order_id,
    o.order_purchase_timestamp                                     AS purchase_ts,
    r.review_score,
    (r.review_score <= 3)::INT                                     AS bad_review,

    -- basket
    i.num_items,
    i.num_products,
    i.num_sellers,
    i.total_price::FLOAT                                           AS total_price,
    i.total_freight::FLOAT                                         AS total_freight,
    ROUND(i.total_freight / NULLIF(i.total_price, 0), 4)::FLOAT    AS freight_ratio,
    pay.num_payments,
    pay.installments,
    pay.payment_type,

    -- main product
    COALESCE(cn.product_category_name_english, p.product_category_name, 'unknown') AS category,
    p.product_name_lenght                                          AS name_len,
    p.product_description_lenght                                   AS desc_len,
    p.product_photos_qty                                           AS photos,
    p.product_weight_g                                             AS weight_g,
    ROUND(p.product_length_cm * p.product_height_cm * p.product_width_cm / 1000.0, 2)::FLOAT AS volume_l,

    -- geography
    c.customer_state,
    s.seller_state,
    (c.customer_state = s.seller_state)::INT                       AS same_state,

    -- reputation: reviews the seller / product had received before this delivery
    sh.n_reviews                                                   AS seller_prior_reviews,
    (sh.n_bad::FLOAT / NULLIF(sh.n_reviews, 0))                    AS seller_prior_bad_rate,
    ph.n_reviews                                                   AS product_prior_reviews,
    (ph.n_bad::FLOAT / NULLIF(ph.n_reviews, 0))                    AS product_prior_bad_rate,

    -- timeline, in days
    EXTRACT(EPOCH FROM o.order_estimated_delivery_date - o.order_purchase_timestamp) / 86400 AS promised_days,
    EXTRACT(EPOCH FROM o.order_approved_at - o.order_purchase_timestamp) / 3600              AS approval_hours,
    EXTRACT(EPOCH FROM o.order_delivered_carrier_date - o.order_approved_at) / 86400         AS handling_days,
    EXTRACT(EPOCH FROM o.order_delivered_carrier_date - i.shipping_limit_date) / 86400       AS seller_delay_days,
    EXTRACT(EPOCH FROM o.order_delivered_customer_date - o.order_delivered_carrier_date) / 86400 AS transit_days,
    EXTRACT(EPOCH FROM o.order_delivered_customer_date - o.order_purchase_timestamp) / 86400 AS delivery_days,
    (o.order_delivered_customer_date::DATE - o.order_estimated_delivery_date::DATE)          AS delay_days,
    (o.order_delivered_customer_date::DATE > o.order_estimated_delivery_date::DATE)::INT     AS is_late

FROM orders AS o
JOIN last_review    AS r   ON r.order_id = o.order_id
JOIN items          AS i   ON i.order_id = o.order_id
JOIN products       AS p   ON p.product_id = i.main_product_id
JOIN sellers        AS s   ON s.seller_id = i.main_seller_id
JOIN customers      AS c   ON c.customer_id = o.customer_id
LEFT JOIN payments  AS pay ON pay.order_id = o.order_id
LEFT JOIN category_names AS cn ON cn.product_category_name = p.product_category_name
JOIN seller_history  AS sh  ON sh.order_id = o.order_id
JOIN product_history AS ph  ON ph.order_id = o.order_id
WHERE o.order_status = 'delivered'
  AND o.order_delivered_customer_date IS NOT NULL
  AND r.review_answer_timestamp >= o.order_delivered_customer_date;
