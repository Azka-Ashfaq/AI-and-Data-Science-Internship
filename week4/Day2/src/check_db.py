import sqlite3
import pandas as pd

conn = sqlite3.connect("data/properties.db")

print("=== DHA Lahore (For Sale, first 10 by price) ===")
q1 = """
SELECT property_id, location, price, bedrooms, baths, area_raw
FROM properties
WHERE city='Lahore' AND location LIKE '%DHA%' AND purpose='For Sale'
ORDER BY price ASC LIMIT 10
"""
print(pd.read_sql(q1, conn))

print("\n=== All Lahore locations (For Sale) ===")
q2 = """
SELECT location, COUNT(*) as cnt, MIN(price) as min_price, MAX(price) as max_price
FROM properties
WHERE city='Lahore' AND purpose='For Sale'
GROUP BY location
ORDER BY cnt DESC
"""
print(pd.read_sql(q2, conn))

print("\n=== Lahore 3-bedroom houses under 3 crore ===")
q3 = """
SELECT property_id, location, price, bedrooms, area_raw
FROM properties
WHERE city='Lahore' AND purpose='For Sale'
  AND bedrooms >= 3 AND price <= 30000000
ORDER BY price ASC LIMIT 10
"""
print(pd.read_sql(q3, conn))

conn.close()