-- Raw Olist tables. The ETL drops and recreates them on every run, so loading is idempotent.
DROP VIEW IF EXISTS features_view;
DROP TABLE IF EXISTS order_payments, order_items, reviews, orders, customers, sellers, products, category_names CASCADE;

CREATE TABLE customers (
    customer_id              TEXT PRIMARY KEY,
    customer_unique_id       TEXT,
    customer_zip_code_prefix VARCHAR(5),
    customer_city            TEXT,
    customer_state           VARCHAR(2)
);

CREATE TABLE sellers (
    seller_id              TEXT PRIMARY KEY,
    seller_zip_code_prefix VARCHAR(5),
    seller_city            TEXT,
    seller_state           VARCHAR(2)
);

-- column names keep the original dataset's spelling ("lenght")
CREATE TABLE products (
    product_id                 TEXT PRIMARY KEY,
    product_category_name      TEXT,
    product_name_lenght        INT,
    product_description_lenght INT,
    product_photos_qty         INT,
    product_weight_g           INT,
    product_length_cm          INT,
    product_height_cm          INT,
    product_width_cm           INT
);

CREATE TABLE category_names (
    product_category_name         TEXT PRIMARY KEY,
    product_category_name_english TEXT
);

CREATE TABLE orders (
    order_id                      TEXT PRIMARY KEY,
    customer_id                   TEXT REFERENCES customers (customer_id),
    order_status                  VARCHAR(15),
    order_purchase_timestamp      TIMESTAMP,
    order_approved_at             TIMESTAMP,
    order_delivered_carrier_date  TIMESTAMP,
    order_delivered_customer_date TIMESTAMP,
    order_estimated_delivery_date TIMESTAMP
);

-- review_id is not unique in the source data, and a few orders have several reviews
CREATE TABLE reviews (
    review_id               TEXT,
    order_id                TEXT REFERENCES orders (order_id),
    review_score            INT,
    review_comment_title    TEXT,
    review_comment_message  TEXT,
    review_creation_date    TIMESTAMP,
    review_answer_timestamp TIMESTAMP
);

CREATE TABLE order_items (
    order_id            TEXT REFERENCES orders (order_id),
    order_item_id       INT,
    product_id          TEXT REFERENCES products (product_id),
    seller_id           TEXT REFERENCES sellers (seller_id),
    shipping_limit_date TIMESTAMP,
    price               NUMERIC(10, 2),
    freight_value       NUMERIC(10, 2),
    PRIMARY KEY (order_id, order_item_id)
);

CREATE TABLE order_payments (
    order_id             TEXT REFERENCES orders (order_id),
    payment_sequential   INT,
    payment_type         TEXT,
    payment_installments INT,
    payment_value        NUMERIC(10, 2),
    PRIMARY KEY (order_id, payment_sequential)
);

CREATE INDEX ON order_items (seller_id);
CREATE INDEX ON order_items (product_id);
CREATE INDEX ON reviews (order_id);
