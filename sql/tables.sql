CREATE TABLE IF NOT EXISTS customers (
	customer_id TEXT PRIMARY KEY,
	customer_unique_id TEXT,
	customer_zip_code_prefix VARCHAR(5),
	customer_city TEXT,
	customer_state VARCHAR(2)
);

CREATE TABLE IF NOT EXISTS products (
	product_id TEXT PRIMARY KEY,
	product_category_name TEXT,
	product_name_lenght INT,
	product_description_lenght INT,
	product_photos_qty INT,
	product_weight_g INT,
	product_length_cm INT,
	product_height_cm INT,
	product_width_cm INT
);

CREATE TABLE IF NOT EXISTS orders (
	order_id TEXT PRIMARY KEY,
	customer_id TEXT REFERENCES customers(customer_id),
	order_status VARCHAR(15),
	order_purchase_timestamp TIMESTAMP,
	order_approved_at TIMESTAMP,
	order_delivered_carrier_date TIMESTAMP,
	order_delivered_customer_date TIMESTAMP,
	order_estimated_delivery_date TIMESTAMP
);

CREATE TABLE IF NOT EXISTS reviews (
	review_id TEXT, -- values aren't unique
	order_id TEXT REFERENCES orders(order_id),
	review_score INT,
	review_comment_title TEXT,
	review_comment_message TEXT,
	review_creation_date TIMESTAMP,
	review_answer_timestamp TIMESTAMP
);

CREATE TABLE IF NOT EXISTS order_items (
	order_id TEXT REFERENCES orders(order_id),
	order_item_id INT,
	product_id TEXT REFERENCES products(product_id),
	seller_id TEXT,
	shipping_limit_date TIMESTAMP,
	price NUMERIC(10, 2),
	freight_value NUMERIC(10,2),
	PRIMARY KEY (order_id, product_id, order_item_id)	
);
