# ChatBI 双数据集 Schema 文档

本文档供 RAG 知识库检索使用，Agent 在编写 SQL 前应参考此文档了解表结构和字段含义。

## 数据集概览

系统使用 SQLite ATTACH DATABASE 实现双数据集跨库查询：

- **Olist（主库）**：巴西电商真实数据集，9 张表，14747 订单，涵盖客户、订单、商品、卖家、支付、评价、地理位置、品类翻译
- **Chinook（附加库，schema 名 `chinook`）**：数字媒体商店数据集，11 张表，3503 曲目，涵盖艺术家、专辑、曲目、发票、客户、员工、流派、媒体类型、播放列表

跨库查询时，Olist 表直接用表名，Chinook 表需加 `chinook.` 前缀。

---

## Olist 数据集（巴西电商）

### olist_orders_dataset（订单主表，14747 行）

| 字段 | 类型 | 说明 |
|------|------|------|
| order_id | TEXT PK | 订单唯一 ID |
| customer_id | TEXT FK | 客户 ID（指向 olist_customers_dataset） |
| order_status | TEXT | 订单状态：delivered/shipped/processing/approved/canceled |
| order_purchase_timestamp | TEXT | 下单时间（YYYY-MM-DD HH:MM:SS） |
| order_approved_at | TEXT | 支付审批时间 |
| order_delivered_carrier_date | TEXT | 物流揽收时间 |
| order_delivered_customer_date | TEXT | 送达客户时间 |
| order_estimated_delivery_date | TEXT | 预计送达时间 |

### olist_customers_dataset（客户表，3000 行）

| 字段 | 类型 | 说明 |
|------|------|------|
| customer_id | TEXT PK | 客户 ID（每个订单一个，关联订单表） |
| customer_unique_id | TEXT | 客户唯一标识（回头客共享同一 ID） |
| customer_zip_code_prefix | INTEGER | 邮编前缀 |
| customer_city | TEXT | 城市 |
| customer_state | TEXT | 州缩写（SP/RJ/MG/RS/PR/SC/BA/GO/PE/CE/DF/ES/PA） |

### olist_order_items_dataset（订单商品明细，17682 行）

| 字段 | 类型 | 说明 |
|------|------|------|
| order_id | TEXT | 订单 ID（联合主键） |
| order_item_id | INTEGER | 商品序号（联合主键） |
| product_id | TEXT FK | 商品 ID |
| seller_id | TEXT FK | 卖家 ID |
| shipping_limit_date | TEXT | 卖家发货截止时间 |
| price | REAL | 商品价格（不含运费） |
| freight_value | REAL | 运费 |

### olist_products_dataset（商品表，200 行）

| 字段 | 类型 | 说明 |
|------|------|------|
| product_id | TEXT PK | 商品 ID |
| product_category_name | TEXT FK | 品类名（葡萄牙语，关联翻译表） |
| product_name_lenght | INTEGER | 商品名长度 |
| product_description_lenght | INTEGER | 描述长度 |
| product_photos_qty | INTEGER | 图片数量 |
| product_weight_g | INTEGER | 重量（克） |
| product_length_cm | INTEGER | 长度（厘米） |
| product_height_cm | INTEGER | 高度（厘米） |
| product_width_cm | INTEGER | 宽度（厘米） |

### olist_sellers_dataset（卖家表，80 行）

| 字段 | 类型 | 说明 |
|------|------|------|
| seller_id | TEXT PK | 卖家 ID |
| seller_zip_code_prefix | INTEGER | 邮编前缀 |
| seller_city | TEXT | 城市 |
| seller_state | TEXT | 州缩写 |

### olist_order_payments_dataset（支付表，15445 行）

| 字段 | 类型 | 说明 |
|------|------|------|
| order_id | TEXT | 订单 ID（联合主键） |
| payment_sequential | INTEGER | 支付序号（联合主键，多笔支付时递增） |
| payment_type | TEXT | 支付方式：credit_card/boleto/voucher/debit_card |
| payment_installments | INTEGER | 分期数 |
| payment_value | REAL | 支付金额 |

### olist_order_reviews_dataset（评价表，8511 行）

| 字段 | 类型 | 说明 |
|------|------|------|
| review_id | TEXT PK | 评价 ID |
| order_id | TEXT FK | 订单 ID |
| review_score | INTEGER | 评分 1-5（1=最差，5=最好） |
| review_comment_title | TEXT | 评价标题（葡萄牙语，可空） |
| review_comment_message | TEXT | 评价内容（葡萄牙语，可空） |
| review_creation_date | TEXT | 评价创建时间 |
| review_answer_timestamp | TEXT | 卖家回复时间 |

### olist_geolocation_dataset（地理位置，262 行）

| 字段 | 类型 | 说明 |
|------|------|------|
| geolocation_zip_code_prefix | INTEGER | 邮编前缀 |
| geolocation_lat | REAL | 纬度 |
| geolocation_lng | REAL | 经度 |
| geolocation_city | TEXT | 城市 |
| geolocation_state | TEXT | 州 |

### product_category_name_translation（品类翻译，20 行）

| 字段 | 类型 | 说明 |
|------|------|------|
| product_category_name | TEXT PK | 葡萄牙语品类名 |
| product_name_english | TEXT | 英语品类名 |

---

## Chinook 数据集（数字媒体，schema 名 `chinook`）

查询时所有 Chinook 表需加 `chinook.` 前缀，例如 `chinook.Album`、`chinook.Invoice`。

### chinook.Artist（艺术家，275 行）

| 字段 | 类型 | 说明 |
|------|------|------|
| ArtistId | INTEGER PK | 艺术家 ID |
| Name | NVARCHAR(120) | 艺术家名 |

### chinook.Album（专辑，347 行）

| 字段 | 类型 | 说明 |
|------|------|------|
| AlbumId | INTEGER PK | 专辑 ID |
| Title | NVARCHAR(160) | 专辑名 |
| ArtistId | INTEGER FK | 艺术家 ID |

### chinook.Track（曲目，3503 行）

| 字段 | 类型 | 说明 |
|------|------|------|
| TrackId | INTEGER PK | 曲目 ID |
| Name | NVARCHAR(200) | 曲目名 |
| AlbumId | INTEGER FK | 专辑 ID |
| MediaTypeId | INTEGER FK | 媒体类型 ID |
| GenreId | INTEGER FK | 流派 ID |
| Composer | NVARCHAR(220) | 作曲者 |
| Milliseconds | INTEGER | 时长（毫秒） |
| Bytes | INTEGER | 文件大小（字节） |
| UnitPrice | NUMERIC(10,2) | 单价 |

### chinook.Genre（流派，25 行）

| 字段 | 类型 | 说明 |
|------|------|------|
| GenreId | INTEGER PK | 流派 ID |
| Name | NVARCHAR(120) | 流派名 |

### chinook.MediaType（媒体类型，5 行）

| 字段 | 类型 | 说明 |
|------|------|------|
| MediaTypeId | INTEGER PK | 媒体类型 ID |
| Name | NVARCHAR(120) | 类型名 |

### chinook.Invoice（发票，412 行）

| 字段 | 类型 | 说明 |
|------|------|------|
| InvoiceId | INTEGER PK | 发票 ID |
| CustomerId | INTEGER FK | 客户 ID |
| InvoiceDate | DATETIME | 开票日期 |
| BillingAddress | NVARCHAR(70) | 账单地址 |
| BillingCity | NVARCHAR(40) | 账单城市 |
| BillingState | NVARCHAR(40) | 账单州 |
| BillingCountry | NVARCHAR(40) | 账单国家 |
| BillingPostalCode | NVARCHAR(10) | 邮编 |
| Total | NUMERIC(10,2) | 总金额 |

### chinook.InvoiceLine（发票明细，2240 行）

| 字段 | 类型 | 说明 |
|------|------|------|
| InvoiceLineId | INTEGER PK | 明细 ID |
| InvoiceId | INTEGER FK | 发票 ID |
| TrackId | INTEGER FK | 曲目 ID |
| UnitPrice | NUMERIC(10,2) | 单价 |
| Quantity | INTEGER | 数量 |

### chinook.Customer（客户，59 行）

| 字段 | 类型 | 说明 |
|------|------|------|
| CustomerId | INTEGER PK | 客户 ID |
| FirstName | NVARCHAR(40) | 名 |
| LastName | NVARCHAR(20) | 姓 |
| Company | NVARCHAR(80) | 公司 |
| Address | NVARCHAR(70) | 地址 |
| City | NVARCHAR(40) | 城市 |
| State | NVARCHAR(40) | 州 |
| Country | NVARCHAR(40) | 国家 |
| PostalCode | NVARCHAR(10) | 邮编 |
| Phone | NVARCHAR(24) | 电话 |
| Fax | NVARCHAR(24) | 传真 |
| Email | NVARCHAR(60) | 邮箱 |
| SupportRepId | INTEGER FK | 客服员工 ID |

### chinook.Employee（员工，8 行）

| 字段 | 类型 | 说明 |
|------|------|------|
| EmployeeId | INTEGER PK | 员工 ID |
| LastName | NVARCHAR(20) | 姓 |
| FirstName | NVARCHAR(20) | 名 |
| Title | NVARCHAR(30) | 职位 |
| ReportsTo | INTEGER FK | 上级 ID |
| BirthDate | DATETIME | 生日 |
| HireDate | DATETIME | 入职日期 |
| Address | NVARCHAR(70) | 地址 |
| City/State/Country/PostalCode/Phone/Fax/Email | - | 联系信息 |

### chinook.Playlist（播放列表，18 行）

| 字段 | 类型 | 说明 |
|------|------|------|
| PlaylistId | INTEGER PK | 播放列表 ID |
| Name | NVARCHAR(120) | 名称 |

### chinook.PlaylistTrack（播放列表曲目，8715 行）

| 字段 | 类型 | 说明 |
|------|------|------|
| PlaylistId | INTEGER FK | 播放列表 ID（联合主键） |
| TrackId | INTEGER FK | 曲目 ID（联合主键） |

---

## ER 关系图

### Olist 关系

```
olist_customers_dataset 1───* olist_orders_dataset 1───* olist_order_items_dataset *───1 olist_products_dataset
                                                                  │                              │
                                                                  *───1 olist_sellers_dataset   │
                                                                                                  │
olist_orders_dataset 1───* olist_order_payments_dataset          product_category_name_translation
olist_orders_dataset 1───* olist_order_reviews_dataset
```

### Chinook 关系

```
chinook.Artist 1───* chinook.Album 1───* chinook.Track *───1 chinook.Genre
                                          │                    │
                                          *───1 chinook.MediaType
                                          
chinook.Customer 1───* chinook.Invoice 1───* chinook.InvoiceLine *───1 chinook.Track

chinook.Playlist *───* chinook.PlaylistTrack *───* chinook.Track

chinook.Employee 1───* chinook.Customer (SupportRepId)
```

---

## 常见分析 SQL 示例

### Olist 电商分析

1. **月度订单量趋势**（黑色星期五效应）：
```sql
SELECT strftime('%Y-%m', order_purchase_timestamp) AS month, COUNT(*) AS orders
FROM olist_orders_dataset GROUP BY month ORDER BY month;
```

2. **各品类销售额排名**：
```sql
SELECT t.product_name_english AS category, ROUND(SUM(oi.price), 2) AS revenue
FROM olist_order_items_dataset oi
JOIN olist_products_dataset p ON oi.product_id = p.product_id
JOIN product_category_name_translation t ON p.product_category_name = t.product_category_name
GROUP BY t.product_name_english ORDER BY revenue DESC;
```

3. **品类差评率排名**（家具类目异常）：
```sql
SELECT t.product_name_english AS category,
       SUM(CASE WHEN r.review_score <= 2 THEN 1 ELSE 0 END) AS bad,
       COUNT(*) AS total,
       ROUND(SUM(CASE WHEN r.review_score <= 2 THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 1) AS bad_pct
FROM olist_order_reviews_dataset r
JOIN olist_order_items_dataset oi ON r.order_id = oi.order_id
JOIN olist_products_dataset p ON oi.product_id = p.product_id
JOIN product_category_name_translation t ON p.product_category_name = t.product_category_name
GROUP BY t.product_name_english ORDER BY bad_pct DESC;
```

4. **支付方式分布与客单价**：
```sql
SELECT payment_type, COUNT(*) AS count, ROUND(AVG(payment_value), 2) AS avg_value
FROM olist_order_payments_dataset GROUP BY payment_type ORDER BY count DESC;
```

5. **各州订单量排名**：
```sql
SELECT c.customer_state, COUNT(*) AS orders
FROM olist_orders_dataset o
JOIN olist_customers_dataset c ON o.customer_id = c.customer_id
GROUP BY c.customer_state ORDER BY orders DESC;
```

### Chinook 媒体分析

6. **各流派曲目数和收入**：
```sql
SELECT g.Name AS genre, COUNT(t.TrackId) AS tracks,
       ROUND(SUM(il.UnitPrice * il.Quantity), 2) AS revenue
FROM chinook.Genre g
JOIN chinook.Track t ON g.GenreId = t.GenreId
JOIN chinook.InvoiceLine il ON t.TrackId = il.TrackId
GROUP BY g.Name ORDER BY revenue DESC;
```

7. **Top 5 艺术家按专辑数**：
```sql
SELECT ar.Name AS artist, COUNT(al.AlbumId) AS albums
FROM chinook.Artist ar
JOIN chinook.Album al ON ar.ArtistId = al.ArtistId
GROUP BY ar.Name ORDER BY albums DESC LIMIT 5;
```

8. **各国销售额**：
```sql
SELECT BillingCountry AS country, ROUND(SUM(Total), 2) AS revenue
FROM chinook.Invoice GROUP BY BillingCountry ORDER BY revenue DESC;
```

### 跨库查询

9. **双数据集对比（各州 Olist 订单数 vs Chinook 发票数）**：
```sql
SELECT c.customer_state, COUNT(*) AS olist_orders,
       (SELECT COUNT(*) FROM chinook.Invoice) AS chinook_invoices
FROM olist_orders_dataset o
JOIN olist_customers_dataset c ON o.customer_id = c.customer_id
GROUP BY c.customer_state ORDER BY olist_orders DESC LIMIT 10;
```

---

## 数据特征提示（根因分析用）

- **季节性**：Olist 11 月订单量是 2 月的 8-9 倍（巴西黑色星期五效应），分析月度趋势时注意此异常
- **品类差评异常**：`moveis_decoracao`（家具装饰）差评率 27.7%，远高于其他品类（平均 ~17%），根因可能与物流延迟或商品质量有关
- **卖家延迟异常**：SC 州某卖家发货延迟率 14%，高于平均水平
- **支付特征**：信用卡占 71%，boleto（巴西银行支付）占 19%，boleto 客单价较高但好评率最高
- **回头客**：约 30% 的客户是回头客（共享 customer_unique_id）
- **Chinook 流派集中度**：Rock 流派曲目数最多（1297），收入也最高，存在明显的长尾分布
