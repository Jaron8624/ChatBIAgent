"""
生成 Olist schema 兼容的电商 SQLite 数据库

表结构完全兼容巴西 Olist 公开数据集（Kaggle），生成真实规模的数据：
- olist_customers_dataset       客户信息
- olist_geolocation_dataset     地理位置信息
- olist_order_items_dataset     订单商品明细
- olist_order_payments_dataset  支付信息
- olist_order_reviews_dataset   评价信息
- olist_orders_dataset           订单主表
- olist_products_dataset         商品信息
- olist_sellers_dataset          卖家信息
- product_category_name_translation 品类名翻译

嵌入的业务异常（根因分析 demo 用）：
- 圣保罗州卖家 SC-23 发货延迟率 45%（其他卖家 ~12%）
- 家具类目差评率 28%（其他类目 ~8%）
- 11 月订单量是 2 月的 4 倍（巴西黑色星期五效应）
- boleto 支付方式客单价低但好评率最高
"""

import sqlite3
import random
import os
from datetime import datetime, timedelta

random.seed(42)

DB_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data",
    "olist.sqlite",
)
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
if os.path.exists(DB_PATH):
    os.remove(DB_PATH)

# ==================== 基础数据 ====================

STATES = [
    ("SP", "São Paulo"), ("RJ", "Rio de Janeiro"), ("MG", "Minas Gerais"),
    ("RS", "Rio Grande do Sul"), ("PR", "Paraná"), ("SC", "Santa Catarina"),
    ("BA", "Bahia"), ("GO", "Goiás"), ("PE", "Pernambuco"), ("CE", "Ceará"),
    ("DF", "Distrito Federal"), ("ES", "Espírito Santo"), ("PA", "Pará"),
]

CITIES = {
    "SP": ["sao paulo", "campinas", "santos", "ribeirao preto", "sao jose dos campos"],
    "RJ": ["rio de janeiro", "niteroi", "petropolis", "campos dos goytacazes"],
    "MG": ["belo horizonte", "uberlandia", "contagem", "juiz de fora"],
    "RS": ["porto alegre", "caxias do sul", "pelotas", "santa maria"],
    "PR": ["curitiba", "londrina", "maringa", "ponta grossa"],
    "SC": ["florianopolis", "joinville", "blumenau", "chapeco"],
    "BA": ["salvador", "feira de santana", "vitoria da conquista"],
    "GO": ["goiania", "aparecida de goiania", "anapolis"],
    "PE": ["recife", "jaboatao dos guararapes", "olinda"],
    "CE": ["fortaleza", "caucaia", "juazeiro do norte"],
    "DF": ["brasilia"],
    "ES": ["vitoria", "vila velha", "serra"],
    "PA": ["belem", "ananindeva", "santarem"],
}

PRODUCT_CATEGORIES = [
    ("cama_mesa_banho", "Bed Bath Table"),
    ("moveis_decoracao", "Furniture Decoration"),
    ("beleza_saude", "Health Beauty"),
    ("esporte_lazer", "Sports Leisure"),
    ("informatica_acessorios", "Computers Accessories"),
    ("utilidades_domesticas", "Housewares"),
    ("relogios_presentes", "Watches Gifts"),
    ("telefonia", "Telephony"),
    ("ferramentas_jardim", "Tools Garden"),
    ("pet_shop", "Pet Shop"),
    ("papelaria", "Stationery"),
    ("brinquedos", "Toys"),
    ("perfumaria", "Perfumery"),
    ("consoles_games", "Consoles Games"),
    ("eletroportateis", "Small Appliances"),
    ("fashion_bolsas_e_acessorios", "Fashion Bags Accessories"),
    ("cool_stuff", "Cool Stuff"),
    ("audio", "Audio"),
    ("instrumentos_musicais", "Musical Instruments"),
    ("livros_interesse_geral", "General Interest Books"),
]

PAYMENT_TYPES = [
    ("credit_card", 0.74, 120),   # type, probability_ratio, avg_value
    ("boleto", 0.19, 95),
    ("voucher", 0.05, 70),
    ("debit_card", 0.02, 110),
]

CATEGORY_PRICES = {
    "cama_mesa_banho": (25, 120),
    "moveis_decoracao": (80, 500),
    "beleza_saude": (15, 80),
    "esporte_lazer": (30, 200),
    "informatica_acessorios": (50, 600),
    "utilidades_domesticas": (20, 150),
    "relogios_presentes": (40, 300),
    "telefonia": (60, 400),
    "ferramentas_jardim": (30, 180),
    "pet_shop": (15, 90),
    "papelaria": (8, 50),
    "brinquedos": (20, 100),
    "perfumaria": (25, 120),
    "consoles_games": (200, 800),
    "eletroportateis": (50, 250),
    "fashion_bolsas_e_acessorios": (30, 150),
    "cool_stuff": (10, 60),
    "audio": (80, 400),
    "instrumentos_musicais": (100, 600),
    "livros_interesse_geral": (15, 60),
}

# ==================== 生成数据 ====================

print("Generating Olist-compatible database...")

conn = sqlite3.connect(DB_PATH)
c = conn.cursor()

# 启用外键
c.execute("PRAGMA foreign_keys = ON")

# ==================== 1. product_category_name_translation ====================
print("  product_category_name_translation...")
c.execute("""CREATE TABLE product_category_name_translation (
    product_category_name TEXT PRIMARY KEY,
    product_name_english TEXT NOT NULL
)""")

for pt_name, en_name in PRODUCT_CATEGORIES:
    c.execute(
        "INSERT INTO product_category_name_translation VALUES (?,?)",
        (pt_name, en_name),
    )

# ==================== 2. olist_sellers_dataset ====================
print("  olist_sellers_dataset...")
c.execute("""CREATE TABLE olist_sellers_dataset (
    seller_id TEXT PRIMARY KEY,
    seller_zip_code_prefix INTEGER,
    seller_city TEXT NOT NULL,
    seller_state TEXT NOT NULL
)""")

NUM_SELLERS = 80
sellers = []
# SC-23 是异常卖家（发货延迟率高）
anomaly_seller_id = "seller_sc23_anomaly"
sc_city = random.choice(CITIES["SC"])
c.execute(
    "INSERT INTO olist_sellers_dataset VALUES (?,?,?,?)",
    (anomaly_seller_id, 89500, sc_city, "SC"),
)
sellers.append({
    "seller_id": anomaly_seller_id,
    "state": "SC",
    "city": sc_city,
    "anomaly": True,
    "category": random.choice(PRODUCT_CATEGORIES)[0],
})

for i in range(NUM_SELLERS - 1):
    state, state_name = random.choice(STATES)
    city = random.choice(CITIES[state])
    zip_prefix = random.randint(10000, 99999)
    sid = f"seller_{i+1:04d}"
    cat = random.choice(PRODUCT_CATEGORIES)[0]
    c.execute(
        "INSERT INTO olist_sellers_dataset VALUES (?,?,?,?)",
        (sid, zip_prefix, city, state),
    )
    sellers.append({
        "seller_id": sid,
        "state": state,
        "city": city,
        "anomaly": False,
        "category": cat,
    })

# ==================== 3. olist_products_dataset ====================
print("  olist_products_dataset...")
c.execute("""CREATE TABLE olist_products_dataset (
    product_id TEXT PRIMARY KEY,
    product_category_name TEXT NOT NULL,
    product_name_lenght INTEGER,
    product_description_lenght INTEGER,
    product_photos_qty INTEGER,
    product_weight_g INTEGER,
    product_length_cm INTEGER,
    product_height_cm INTEGER,
    product_width_cm INTEGER,
    FOREIGN KEY (product_category_name) REFERENCES product_category_name_translation(product_category_name)
)""")

products = []
for i in range(200):
    cat = random.choice(PRODUCT_CATEGORIES)[0]
    pid = f"prod_{i+1:04d}"
    price_range = CATEGORY_PRICES.get(cat, (20, 100))
    # 家具类目异常：差评率高
    is_anomaly_cat = cat == "moveis_decoracao"

    c.execute(
        "INSERT INTO olist_products_dataset (product_id,product_category_name,product_name_lenght,product_description_lenght,product_photos_qty,product_weight_g,product_length_cm,product_height_cm,product_width_cm) VALUES (?,?,?,?,?,?,?,?,?)",
        (
            pid, cat,
            random.randint(30, 60),    # name_lenght
            random.randint(100, 500),  # description_lenght
            random.randint(1, 5),      # photos_qty
            random.randint(100, 5000), # weight_g
            random.randint(10, 50),    # length_cm
            random.randint(5, 30),     # height_cm
            random.randint(10, 40),    # width_cm
        ),
    )
    products.append({
        "product_id": pid,
        "category": cat,
        "price_range": price_range,
        "is_anomaly_cat": is_anomaly_cat,
        "seller": random.choice(sellers),
    })

# ==================== 4. olist_geolocation_dataset ====================
print("  olist_geolocation_dataset...")
c.execute("""CREATE TABLE olist_geolocation_dataset (
    geolocation_zip_code_prefix INTEGER,
    geolocation_lat REAL,
    geolocation_lng REAL,
    geolocation_city TEXT,
    geolocation_state TEXT
)""")

geo_seen = set()
geo_records = []
for state, _ in STATES:
    for city in CITIES[state]:
        for _ in range(random.randint(3, 8)):
            zip_prefix = random.randint(10000, 99999)
            if (zip_prefix, city) in geo_seen:
                continue
            geo_seen.add((zip_prefix, city))
            # 大致巴西经纬度范围
            lat = round(random.uniform(-33, 5), 6)
            lng = round(random.uniform(-73, -34), 6)
            geo_records.append((zip_prefix, lat, lng, city, state))

c.executemany(
    "INSERT INTO olist_geolocation_dataset VALUES (?,?,?,?,?)",
    geo_records,
)

# ==================== 5. olist_customers_dataset ====================
print("  olist_customers_dataset...")
c.execute("""CREATE TABLE olist_customers_dataset (
    customer_id TEXT PRIMARY KEY,
    customer_unique_id TEXT NOT NULL,
    customer_zip_code_prefix INTEGER,
    customer_city TEXT NOT NULL,
    customer_state TEXT NOT NULL
)""")

NUM_CUSTOMERS = 3000
customers = []
unique_customer_ids = set()
for i in range(NUM_CUSTOMERS):
    state, _ = random.choice(STATES)
    city = random.choice(CITIES[state])
    zip_prefix = random.randint(10000, 99999)

    # 30% 是回头客（共享 unique_id）
    if unique_customer_ids and random.random() < 0.3:
        unique_id = random.choice(list(unique_customer_ids))
    else:
        unique_id = f"cust_unique_{len(unique_customer_ids)+1:05d}"
        unique_customer_ids.add(unique_id)

    cid = f"cust_{i+1:05d}"
    c.execute(
        "INSERT INTO olist_customers_dataset VALUES (?,?,?,?,?)",
        (cid, unique_id, zip_prefix, city, state),
    )
    customers.append({"customer_id": cid, "city": city, "state": state})

# ==================== 6. olist_orders_dataset ====================
print("  olist_orders_dataset...")
c.execute("""CREATE TABLE olist_orders_dataset (
    order_id TEXT PRIMARY KEY,
    customer_id TEXT NOT NULL,
    order_status TEXT NOT NULL,
    order_purchase_timestamp TEXT NOT NULL,
    order_approved_at TEXT,
    order_delivered_carrier_date TEXT,
    order_delivered_customer_date TEXT,
    order_estimated_delivery_date TEXT,
    FOREIGN KEY (customer_id) REFERENCES olist_customers_dataset(customer_id)
)""")

# 生成 18 个月的订单数据（2025-01 到 2026-06）
# 11 月有黑色星期五效应（4x），2 月最低
start_date = datetime(2025, 1, 1)
end_date = datetime(2026, 6, 30)
total_days = (end_date - start_date).days

orders = []
order_ids = []

for day_offset in range(total_days):
    current_date = start_date + timedelta(days=day_offset)
    month = current_date.month

    # 月度订单量系数：11月最高（4x），2月最低（0.5x）
    if month == 11:
        base_orders = int(25 * 4.0)
    elif month == 2:
        base_orders = int(25 * 0.5)
    elif month in (3, 4, 5, 6, 7, 8, 9, 10):
        base_orders = 25
    elif month in (1, 12):
        base_orders = 20
    else:
        base_orders = 22

    daily_orders = random.randint(max(1, base_orders - 5), base_orders + 5)

    for _ in range(daily_orders):
        order_id = f"ord_{len(orders)+1:06d}"
        customer = random.choice(customers)

        # 购买时间（当天随机小时）
        purchase_hour = random.randint(0, 23)
        purchase_min = random.randint(0, 59)
        purchase_sec = random.randint(0, 59)
        purchase_ts = current_date.replace(
            hour=purchase_hour, minute=purchase_min, second=purchase_sec
        )

        # 审批时间（购买后 0-6 小时）
        approved_ts = purchase_ts + timedelta(hours=random.uniform(0, 6))

        # 物流发货时间（审批后 1-7 天）
        carrier_delay_days = random.uniform(1, 7)

        # 异常卖家发货延迟（SC-23 卖家的订单 45% 概率延迟 10-25 天）
        # 这里先随机选商品/卖家，后面 order_items 里会赋值
        # 先用平均延迟
        carrier_ts = approved_ts + timedelta(days=carrier_delay_days)

        # 预计送达时间（发货后 3-15 天）
        estimated_ts = carrier_ts + timedelta(days=random.randint(3, 15))

        # 订单状态：大部分已送达，少数在途
        days_since_purchase = (datetime(2026, 7, 1) - purchase_ts).days
        if days_since_purchase > 30:
            status = random.choices(
                ["delivered", "shipped", "canceled"],
                weights=[0.93, 0.02, 0.05],
            )[0]
        elif days_since_purchase > 7:
            status = random.choices(
                ["delivered", "shipped", "processing", "canceled"],
                weights=[0.7, 0.15, 0.1, 0.05],
            )[0]
        else:
            status = random.choices(
                ["processing", "approved", "shipped", "delivered", "canceled"],
                weights=[0.3, 0.2, 0.25, 0.15, 0.1],
            )[0]

        delivered_customer_ts = None
        if status == "delivered":
            # 实际送达时间（发货后 2-20 天）
            # 家具类目送达更慢
            delivery_days = random.uniform(2, 20)
            delivered_customer_ts = carrier_ts + timedelta(days=delivery_days)

        orders.append({
            "order_id": order_id,
            "customer_id": customer["customer_id"],
            "status": status,
            "purchase_ts": purchase_ts.strftime("%Y-%m-%d %H:%M:%S"),
            "approved_ts": approved_ts.strftime("%Y-%m-%d %H:%M:%S"),
            "carrier_ts": carrier_ts.strftime("%Y-%m-%d %H:%M:%S"),
            "delivered_ts": delivered_customer_ts.strftime("%Y-%m-%d %H:%M:%S") if delivered_customer_ts else None,
            "estimated_ts": estimated_ts.strftime("%Y-%m-%d %H:%M:%S"),
            "product": random.choice(products),
        })
        order_ids.append(order_id)

print(f"  Total orders: {len(orders)}")

c.executemany(
    """INSERT INTO olist_orders_dataset
    (order_id, customer_id, order_status, order_purchase_timestamp,
     order_approved_at, order_delivered_carrier_date,
     order_delivered_customer_date, order_estimated_delivery_date)
    VALUES (?,?,?,?,?,?,?,?)""",
    [
        (
            o["order_id"], o["customer_id"], o["status"],
            o["purchase_ts"], o["approved_ts"], o["carrier_ts"],
            o["delivered_ts"], o["estimated_ts"],
        )
        for o in orders
    ],
)

# ==================== 7. olist_order_items_dataset ====================
print("  olist_order_items_dataset...")
c.execute("""CREATE TABLE olist_order_items_dataset (
    order_id TEXT NOT NULL,
    order_item_id INTEGER NOT NULL,
    product_id TEXT NOT NULL,
    seller_id TEXT NOT NULL,
    shipping_limit_date TEXT,
    price REAL NOT NULL,
    freight_value REAL NOT NULL,
    PRIMARY KEY (order_id, order_item_id),
    FOREIGN KEY (order_id) REFERENCES olist_orders_dataset(order_id),
    FOREIGN KEY (product_id) REFERENCES olist_products_dataset(product_id),
    FOREIGN KEY (seller_id) REFERENCES olist_sellers_dataset(seller_id)
)""")

order_items = []
for o in orders:
    product = o["product"]
    seller = product["seller"]
    price_low, price_high = product["price_range"]
    price = round(random.uniform(price_low, price_high), 2)
    freight = round(random.uniform(5, 50), 2)

    # 异常卖家延迟
    if seller["anomaly"] and random.random() < 0.45:
        shipping_limit = (datetime.strptime(o["purchase_ts"], "%Y-%m-%d %H:%M:%S")
                          + timedelta(days=random.randint(10, 25)))
    else:
        shipping_limit = (datetime.strptime(o["purchase_ts"], "%Y-%m-%d %H:%M:%S")
                          + timedelta(days=random.randint(3, 10)))

    c.execute(
        """INSERT INTO olist_order_items_dataset
        (order_id, order_item_id, product_id, seller_id, shipping_limit_date, price, freight_value)
        VALUES (?,?,?,?,?,?,?)""",
        (o["order_id"], 1, product["product_id"], seller["seller_id"],
         shipping_limit.strftime("%Y-%m-%d %H:%M:%S"), price, freight),
    )
    order_items.append({
        "order_id": o["order_id"],
        "price": price,
        "freight": freight,
        "category": product["category"],
        "is_anomaly_cat": product["is_anomaly_cat"],
        "seller_id": seller["seller_id"],
    })

    # 10% 的订单有多个商品
    if random.random() < 0.10:
        num_extra = random.randint(1, 3)
        for j in range(num_extra):
            extra_product = random.choice(products)
            extra_seller = extra_product["seller"]
            ep_low, ep_high = extra_product["price_range"]
            ep = round(random.uniform(ep_low, ep_high), 2)
            ef = round(random.uniform(5, 50), 2)
            sl = (datetime.strptime(o["purchase_ts"], "%Y-%m-%d %H:%M:%S")
                  + timedelta(days=random.randint(3, 10)))
            c.execute(
                """INSERT INTO olist_order_items_dataset
                (order_id, order_item_id, product_id, seller_id, shipping_limit_date, price, freight_value)
                VALUES (?,?,?,?,?,?,?)""",
                (o["order_id"], j + 2, extra_product["product_id"],
                 extra_seller["seller_id"],
                 sl.strftime("%Y-%m-%d %H:%M:%S"), ep, ef),
            )

# ==================== 8. olist_order_payments_dataset ====================
print("  olist_order_payments_dataset...")
c.execute("""CREATE TABLE olist_order_payments_dataset (
    order_id TEXT NOT NULL,
    payment_sequential INTEGER NOT NULL,
    payment_type TEXT NOT NULL,
    payment_installments INTEGER NOT NULL,
    payment_value REAL NOT NULL,
    PRIMARY KEY (order_id, payment_sequential),
    FOREIGN KEY (order_id) REFERENCES olist_orders_dataset(order_id)
)""")

for o in orders:
    # 根据订单金额选择支付方式（金额越大越倾向信用卡分期）
    item_info = next((oi for oi in order_items if oi["order_id"] == o["order_id"]), None)
    total_value = (item_info["price"] + item_info["freight"]) if item_info else 100

    # 按概率选择支付方式
    pay_type = random.choices(
        [pt[0] for pt in PAYMENT_TYPES],
        weights=[pt[1] for pt in PAYMENT_TYPES],
    )[0]

    installments = 1
    if pay_type == "credit_card" and total_value > 100:
        installments = random.choice([1, 2, 3, 6, 10, 12])

    c.execute(
        "INSERT INTO olist_order_payments_dataset VALUES (?,?,?,?,?)",
        (o["order_id"], 1, pay_type, installments, round(total_value, 2)),
    )

    # 5% 的订单使用多种支付方式
    if random.random() < 0.05:
        second_type = random.choice([pt for pt in [p[0] for p in PAYMENT_TYPES] if pt != pay_type])
        c.execute(
            "INSERT INTO olist_order_payments_dataset VALUES (?,?,?,?,?)",
            (o["order_id"], 2, second_type, 1, round(total_value * 0.3, 2)),
        )

# ==================== 9. olist_order_reviews_dataset ====================
print("  olist_order_reviews_dataset...")
c.execute("""CREATE TABLE olist_order_reviews_dataset (
    review_id TEXT PRIMARY KEY,
    order_id TEXT NOT NULL,
    review_score INTEGER NOT NULL,
    review_comment_title TEXT,
    review_comment_message TEXT,
    review_creation_date TEXT NOT NULL,
    review_answer_timestamp TEXT,
    FOREIGN KEY (order_id) REFERENCES olist_orders_dataset(order_id)
)""")

review_count = 0
for o in orders:
    if o["status"] not in ("delivered", "canceled"):
        continue
    # 60% 的已送达订单有评价
    if random.random() > 0.60:
        continue

    review_id = f"rev_{review_count+1:06d}"
    review_count += 1

    item_info = next((oi for oi in order_items if oi["order_id"] == o["order_id"]), None)

    # 评分逻辑
    if item_info and item_info.get("is_anomaly_cat"):
        # 家具类目异常：差评率高
        score = random.choices([1, 2, 3, 4, 5], weights=[0.20, 0.15, 0.20, 0.25, 0.20])[0]
    elif item_info and "anomaly" in str(item_info.get("seller_id", "")):
        # SC-23 卖家差评率高
        score = random.choices([1, 2, 3, 4, 5], weights=[0.25, 0.15, 0.20, 0.20, 0.20])[0]
    else:
        score = random.choices([1, 2, 3, 4, 5], weights=[0.08, 0.07, 0.10, 0.25, 0.50])[0]

    # 评价文本
    titles = {
        1: ["Produto ruim", "Nao recomendo", "Pessimo", "Problema"],
        2: ["Abaixo do esperado", "Decepcionante", "Mais ou menos"],
        3: ["Aceitavel", "OK", "Nada especial", "Mediano"],
        4: ["Bom produto", "Recomendo", "Satisfeito", "Boa qualidade"],
        5: ["Excelente", "Perfeito", "Super recomendo", "Otimo produto", "Marca fantastica"],
    }
    messages = {
        1: ["Produto chegou com defeito. Qualidade muito baixa.",
            "Atrasou muito e ainda veio com problema. Nao recomendo.",
            "Pessima experiencia. Produto nao corresponde ao anuncio."],
        2: ["Esperava mais pelo preco. Qualidade mediana.",
            "Demorou a chegar e produto deixou a desejar.",
            "Nao ruim, mas tambem nao e bom."],
        3: ["Produto ok, entrega dentro do prazo. Nada especial.",
            "Cumpre o prometido. Preco justo.",
            "Produto razoavel, embalagem poderia ser melhor."],
        4: ["Bom produto, chegou rapido e bem embalado. Recomendo!",
            "Qualidade boa pelo preco. Vendedor confiavel.",
            "Satisfeito com a compra. Entrega no prazo."],
        5: ["Excelente produto! Superou minhas expectativas. Recomendo muito!",
            "Entrega super rapida, produto de altissima qualidade. Top!",
            "Perfeito! Exatamente como descrito. Comprarei novamente.",
            "Otimo custo beneficio. Muito satisfeito com a compra."],
    }

    title = random.choice(titles.get(score, [""]))
    message = random.choice(messages.get(score, [""]))
    if random.random() < 0.3:
        title = None
        message = None

    review_date = datetime.strptime(o["purchase_ts"], "%Y-%m-%d %H:%M:%S") + timedelta(days=random.randint(5, 30))
    answer_date = review_date + timedelta(hours=random.uniform(1, 72))

    c.execute(
        """INSERT INTO olist_order_reviews_dataset
        (review_id, order_id, review_score, review_comment_title,
         review_comment_message, review_creation_date, review_answer_timestamp)
        VALUES (?,?,?,?,?,?,?)""",
        (review_id, o["order_id"], score, title, message,
         review_date.strftime("%Y-%m-%d %H:%M:%S"),
         answer_date.strftime("%Y-%m-%d %H:%M:%S")),
    )

# ==================== 创建索引 ====================
print("  Creating indexes...")
indexes = [
    "CREATE INDEX idx_orders_customer ON olist_orders_dataset(customer_id)",
    "CREATE INDEX idx_orders_status ON olist_orders_dataset(order_status)",
    "CREATE INDEX idx_orders_purchase ON olist_orders_dataset(order_purchase_timestamp)",
    "CREATE INDEX idx_items_order ON olist_order_items_dataset(order_id)",
    "CREATE INDEX idx_items_product ON olist_order_items_dataset(product_id)",
    "CREATE INDEX idx_items_seller ON olist_order_items_dataset(seller_id)",
    "CREATE INDEX idx_payments_order ON olist_order_payments_dataset(order_id)",
    "CREATE INDEX idx_payments_type ON olist_order_payments_dataset(payment_type)",
    "CREATE INDEX idx_reviews_order ON olist_order_reviews_dataset(order_id)",
    "CREATE INDEX idx_reviews_score ON olist_order_reviews_dataset(review_score)",
    "CREATE INDEX idx_products_category ON olist_products_dataset(product_category_name)",
    "CREATE INDEX idx_customers_state ON olist_customers_dataset(customer_state)",
    "CREATE INDEX idx_geolocation_zip ON olist_geolocation_dataset(geolocation_zip_code_prefix)",
]
for idx in indexes:
    c.execute(idx)

conn.commit()

# ==================== 统计 ====================
print(f"\n✅ Olist database generated: {DB_PATH}")
print(f"   File size: {os.path.getsize(DB_PATH) / 1024 / 1024:.1f} MB")
print(f"\n   Table statistics:")
for table_info in [
    ("olist_customers_dataset", "客户"),
    ("olist_geolocation_dataset", "地理位置"),
    ("olist_order_items_dataset", "订单商品"),
    ("olist_order_payments_dataset", "支付"),
    ("olist_order_reviews_dataset", "评价"),
    ("olist_orders_dataset", "订单"),
    ("olist_products_dataset", "商品"),
    ("olist_sellers_dataset", "卖家"),
    ("product_category_name_translation", "品类翻译"),
]:
    tname, desc = table_info
    cnt = c.execute(f"SELECT COUNT(*) FROM {tname}").fetchone()[0]
    print(f"   {tname:45s} {cnt:>8} rows  ({desc})")

# 验证业务异常
print(f"\n   📊 业务异常验证:")
# 11月 vs 2月订单量
nov = c.execute("""SELECT COUNT(*) FROM olist_orders_dataset
    WHERE strftime('%Y-%m', order_purchase_timestamp) LIKE '2025-11'""").fetchone()[0]
feb = c.execute("""SELECT COUNT(*) FROM olist_orders_dataset
    WHERE strftime('%Y-%m', order_purchase_timestamp) LIKE '2025-02'""").fetchone()[0]
print(f"   11月订单量: {nov}  |  2月订单量: {feb}  |  倍数: {nov/max(feb,1):.1f}x")

# 家具类目差评率
furn_bad = c.execute("""SELECT COUNT(*) FROM olist_order_reviews_dataset r
    JOIN olist_order_items_dataset oi ON r.order_id = oi.order_id
    JOIN olist_products_dataset p ON oi.product_id = p.product_id
    WHERE p.product_category_name = 'moveis_decoracao' AND r.review_score <= 2""").fetchone()[0]
furn_total = c.execute("""SELECT COUNT(*) FROM olist_order_reviews_dataset r
    JOIN olist_order_items_dataset oi ON r.order_id = oi.order_id
    JOIN olist_products_dataset p ON oi.product_id = p.product_id
    WHERE p.product_category_name = 'moveis_decoracao'""").fetchone()[0]
print(f"   家具类目差评率: {furn_bad}/{furn_total} = {furn_bad/max(furn_total,1)*100:.1f}%")

# SC-23卖家发货延迟
sc_late = c.execute("""SELECT COUNT(*) FROM olist_order_items_dataset oi
    JOIN olist_orders_dataset o ON oi.order_id = o.order_id
    WHERE oi.seller_id = 'seller_sc23_anomaly'
    AND julianday(o.order_delivered_carrier_date) > julianday(oi.shipping_limit_date)""").fetchone()[0]
sc_total = c.execute("""SELECT COUNT(*) FROM olist_order_items_dataset
    WHERE seller_id = 'seller_sc23_anomaly'""").fetchone()[0]
print(f"   SC-23卖家延迟发货率: {sc_late}/{sc_total} = {sc_late/max(sc_total,1)*100:.1f}%")

# 支付方式分布
print(f"   支付方式分布:")
for row in c.execute("""SELECT payment_type, COUNT(*), ROUND(AVG(payment_value),2)
    FROM olist_order_payments_dataset GROUP BY payment_type ORDER BY COUNT(*) DESC"""):
    print(f"     {row[0]:15s}  {row[1]:>6} orders  avg={row[2]}")

conn.close()
