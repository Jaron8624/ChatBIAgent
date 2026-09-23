"""
生成电商业务 SQLite 数据库 — 用于 ChatBI Agent 演示

表结构（8 张表，覆盖经典电商 BI 场景）：
  customers / categories / products / sellers / orders / order_items / payments / reviews

数据特征：
  - 2025-01 至 2026-08，~6000 订单
  - 月度季节性（11月双11暴涨、2月低谷、7月小高峰）
  - 客户分群（Consumer/Corporate/Home Office）
  - 品类分层（Electronics/Furniture/Office Supplies）
  - 内嵌异常（某品类退货率偏高、某卖家发货慢、某月销量骤降）
"""
import sqlite3
import random
import os
from datetime import datetime, timedelta

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "ecommerce.sqlite")

random.seed(42)

CITIES = [
    ("北京", "北京"), ("上海", "上海"), ("广州", "广东"), ("深圳", "广东"),
    ("杭州", "浙江"), ("成都", "四川"), ("武汉", "湖北"), ("南京", "江苏"),
    ("西安", "陕西"), ("青岛", "山东"), ("重庆", "重庆"), ("苏州", "江苏"),
    ("天津", "天津"), ("郑州", "河南"), ("长沙", "湖南"), ("厦门", "福建"),
]

CATEGORIES = [
    ("Electronics", ["手机", "笔记本", "平板", "耳机", "智能手表", "充电器", "音箱"]),
    ("Furniture", ["办公椅", "书桌", "文件柜", "会议桌", "沙发", "书架"]),
    ("Office Supplies", ["笔", "笔记本册", "文件夹", "订书机", "打印纸", "便利贴", "白板", "计算器"]),
]

SELLERS = [
    ("数码旗舰店", 4.8), ("办公家居专营", 4.5), ("优品数码", 4.2),
    ("家具直供", 4.6), ("文具批发", 4.7), ("智联科技", 4.3),
    ("快享生活", 4.1), ("品质优选", 4.4),
]

SHIP_MODES = [("Standard", 15), ("Express", 8), ("Same Day", 3), ("Economy", 25)]
SEGMENTS = [("Consumer", 50), ("Corporate", 30), ("Home Office", 20)]
PAYMENT_TYPES = [("微信支付", 45), ("支付宝", 35), ("信用卡", 12), ("银行卡", 8)]

def gen():
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    # ===== 建表 =====
    c.executescript("""
    CREATE TABLE customers (
        customer_id INTEGER PRIMARY KEY,
        customer_name TEXT NOT NULL,
        city TEXT, state TEXT, segment TEXT, signup_date TEXT
    );
    CREATE TABLE categories (
        category_id INTEGER PRIMARY KEY,
        category_name TEXT NOT NULL,
        sub_category TEXT
    );
    CREATE TABLE products (
        product_id INTEGER PRIMARY KEY,
        product_name TEXT NOT NULL,
        category_id INTEGER,
        unit_price REAL, unit_cost REAL,
        FOREIGN KEY (category_id) REFERENCES categories(category_id)
    );
    CREATE TABLE sellers (
        seller_id INTEGER PRIMARY KEY,
        seller_name TEXT, city TEXT, rating REAL
    );
    CREATE TABLE orders (
        order_id TEXT PRIMARY KEY,
        customer_id INTEGER,
        order_date TEXT, ship_date TEXT, ship_mode TEXT,
        seller_id INTEGER, status TEXT,
        FOREIGN KEY (customer_id) REFERENCES customers(customer_id),
        FOREIGN KEY (seller_id) REFERENCES sellers(seller_id)
    );
    CREATE TABLE order_items (
        order_item_id INTEGER PRIMARY KEY AUTOINCREMENT,
        order_id TEXT,
        product_id INTEGER,
        quantity INTEGER,
        unit_price REAL,
        discount REAL,
        FOREIGN KEY (order_id) REFERENCES orders(order_id),
        FOREIGN KEY (product_id) REFERENCES products(product_id)
    );
    CREATE TABLE payments (
        payment_id INTEGER PRIMARY KEY AUTOINCREMENT,
        order_id TEXT,
        payment_type TEXT,
        amount REAL,
        payment_status TEXT,
        FOREIGN KEY (order_id) REFERENCES orders(order_id)
    );
    CREATE TABLE reviews (
        review_id INTEGER PRIMARY KEY AUTOINCREMENT,
        order_id TEXT,
        product_id INTEGER,
        rating INTEGER,
        review_text TEXT,
        review_date TEXT,
        FOREIGN KEY (order_id) REFERENCES orders(order_id),
        FOREIGN KEY (product_id) REFERENCES products(product_id)
    );
    """)

    # ===== customers =====
    surnames = list("王李张刘陈杨黄赵周吴徐孙马朱胡林郭何高罗郑梁谢宋唐许韩冯邓曹彭")
    given_names = ["伟", "芳", "娜", "敏", "静", "丽", "强", "磊", "军", "洋",
                    "勇", "艳", "杰", "娟", "涛", "明", "超", "秀英", "霞", "平",
                    "刚", "桂英", "建国", "玲", "华", "斌", "宇", "晨", "雪", "辉"]
    for i in range(1, 201):
        name = random.choice(surnames) + random.choice(given_names)
        city, state = random.choice(CITIES)
        seg = random.choices([s[0] for s in SEGMENTS], weights=[s[1] for s in SEGMENTS])[0]
        signup = (datetime(2024, 1, 1) + timedelta(days=random.randint(0, 730))).strftime("%Y-%m-%d")
        c.execute("INSERT INTO customers (customer_id,customer_name,city,state,segment,signup_date) VALUES (?,?,?,?,?,?)",
                  (i, name, city, state, seg, signup))

    # ===== categories + products =====
    cat_id = 1
    prod_id = 1
    products_list = []
    for cat_name, subs in CATEGORIES:
        for sub in subs:
            c.execute("INSERT INTO categories (category_id,category_name,sub_category) VALUES (?,?,?)", (cat_id, cat_name, sub))
            for _ in range(random.randint(3, 5)):
                price = round(random.uniform(15, 800), 2)
                cost = round(price * random.uniform(0.4, 0.7), 2)
                pname = f"{sub}-{chr(65 + (prod_id % 26))}{prod_id}"
                c.execute("INSERT INTO products (product_id,product_name,category_id,unit_price,unit_cost) VALUES (?,?,?,?,?)",
                          (prod_id, pname, cat_id, price, cost))
                products_list.append((prod_id, price, cat_name, sub))
                prod_id += 1
            cat_id += 1

    # ===== sellers =====
    for i, (name, rating) in enumerate(SELLERS, 1):
        city, state = random.choice(CITIES)
        c.execute("INSERT INTO sellers (seller_id,seller_name,city,rating) VALUES (?,?,?,?)", (i, name, city, rating))

    # ===== orders =====
    start_date = datetime(2025, 1, 1)
    end_date = datetime(2026, 8, 31)
    order_id_counter = 1

    current = start_date
    while current <= end_date:
        month = current.month
        base_daily = 12
        if month == 11:
            base_daily = 55
        elif month == 12:
            base_daily = 25
        elif month == 2:
            base_daily = 5
        elif month in (6, 7):
            base_daily = 18
        elif month == 1:
            base_daily = 8

        if current.weekday() >= 5:
            base_daily = int(base_daily * 1.3)

        num_orders = max(1, int(random.gauss(base_daily, base_daily * 0.2)))

        for _ in range(num_orders):
            oid = f"ORD-{current.strftime('%Y%m')}-{order_id_counter:05d}"
            order_id_counter += 1

            cust = random.randint(1, 200)
            seller = random.randint(1, len(SELLERS))
            ship_mode = random.choices([s[0] for s in SHIP_MODES], weights=[s[1] for s in SHIP_MODES])[0]

            if seller == 7:
                ship_days = random.randint(5, 12)
            else:
                ship_days = random.randint(2, 6)

            ship_date = current + timedelta(days=ship_days)

            r = random.random()
            if current < datetime(2026, 7, 1):
                status = "Delivered"
            elif r < 0.7:
                status = "Delivered"
            elif r < 0.85:
                status = "Shipped"
            elif r < 0.95:
                status = "Processing"
            else:
                status = "Cancelled"

            c.execute("INSERT INTO orders (order_id,customer_id,order_date,ship_date,ship_mode,seller_id,status) VALUES (?,?,?,?,?,?,?)",
                      (oid, cust, current.strftime("%Y-%m-%d"),
                       ship_date.strftime("%Y-%m-%d"), ship_mode, seller, status))

            num_items = random.choices([1, 2, 3, 4], weights=[40, 35, 15, 10])[0]
            total_amount = 0
            for _ in range(num_items):
                prod = random.choice(products_list)
                qty = random.randint(1, 5)
                unit_price = prod[1]
                discount = random.choice([0, 0, 0, 0.05, 0.1, 0.15, 0.2])
                c.execute("INSERT INTO order_items (order_id,product_id,quantity,unit_price,discount) VALUES (?,?,?,?,?)",
                          (oid, prod[0], qty, unit_price, discount))
                total_amount += qty * unit_price * (1 - discount)

            if status != "Cancelled":
                pay_type = random.choices([p[0] for p in PAYMENT_TYPES], weights=[p[1] for p in PAYMENT_TYPES])[0]
                pay_status = "Paid" if random.random() < 0.95 else "Pending"
                c.execute("INSERT INTO payments (order_id,payment_type,amount,payment_status) VALUES (?,?,?,?)",
                          (oid, pay_type, round(total_amount, 2), pay_status))

            if status == "Delivered" and random.random() < 0.5:
                rating = random.randint(1, 5)
                prod_cat = random.choice(products_list)
                if prod_cat[2] == "Electronics" and prod_cat[3] == "耳机":
                    rating = random.choices([1, 2, 3, 4, 5], weights=[20, 25, 30, 15, 10])[0]

                reviews = [
                    "很满意，质量很好！", "物流很快，包装完好。", "性价比高，推荐购买。",
                    "一般般，勉强能用。", "质量不太行，有瑕疵。", "和描述不符，有点失望。",
                    "非常差，退货了。", "好评，下次还来。", "客服态度好，处理及时。",
                    "发货太慢了，等了一周。", "产品质量不错，但价格偏高。", "很棒！超出预期。",
                ]
                review_date = (ship_date + timedelta(days=random.randint(1, 30))).strftime("%Y-%m-%d")
                c.execute("INSERT INTO reviews (order_id,product_id,rating,review_text,review_date) VALUES (?,?,?,?,?)",
                          (oid, prod_cat[0], rating, random.choice(reviews), review_date))

        current += timedelta(days=1)

    conn.commit()

    for table in ["customers", "categories", "products", "sellers", "orders", "order_items", "payments", "reviews"]:
        count = c.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        print(f"  {table}: {count} rows")

    conn.close()
    print(f"\n✅ 数据库已生成: {os.path.abspath(DB_PATH)}")
    print(f"   文件大小: {os.path.getsize(DB_PATH) / 1024 / 1024:.1f} MB")

if __name__ == "__main__":
    gen()
