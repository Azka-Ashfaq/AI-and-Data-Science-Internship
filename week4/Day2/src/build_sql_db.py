"""
Day 2 - Task 3: Structured retrieval store.
Everything that is an exact fact you'd filter/sort/aggregate on -- price,
availability(purpose), plot size, bedrooms, agent names -- goes into SQL.
This is queried directly with WHERE/ORDER BY, never via the LLM guessing.
"""
import sqlite3
import pandas as pd

conn = sqlite3.connect("data/properties.db")

props = pd.read_csv("data/properties_clean.csv")
props.to_sql("properties", conn, if_exists="replace", index=False)

pd.read_csv("data/payment_plans.csv").to_sql("payment_plans", conn, if_exists="replace", index=False)
pd.read_csv("data/schools.csv").to_sql("schools", conn, if_exists="replace", index=False)
pd.read_csv("data/hospitals.csv").to_sql("hospitals", conn, if_exists="replace", index=False)
pd.read_csv("data/amenities.csv").to_sql("amenities", conn, if_exists="replace", index=False)

conn.execute("CREATE INDEX IF NOT EXISTS idx_city ON properties(city)")
conn.execute("CREATE INDEX IF NOT EXISTS idx_purpose ON properties(purpose)")
conn.execute("CREATE INDEX IF NOT EXISTS idx_price ON properties(price)")
conn.execute("CREATE INDEX IF NOT EXISTS idx_bedrooms ON properties(bedrooms)")
conn.commit()

cur = conn.execute("SELECT COUNT(*) FROM properties")
print("properties rows in SQLite:", cur.fetchone()[0])

# sanity query matching a real conversation ("budget 3 crore, DHA, Lahore")
q = """
SELECT property_id, location, property_type, price, bedrooms, baths, area_raw, agency, agent
FROM properties
WHERE city = 'Lahore' AND location LIKE '%DHA%' AND price <= 30000000 AND purpose = 'For Sale'
ORDER BY price DESC LIMIT 5
"""
print(pd.read_sql(q, conn))
conn.close()
