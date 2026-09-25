"""
Day 2 - Task 1: Knowledge Base Design
Builds a clean, demo-sized property dataset from the Kaggle Zameen.com
Property.csv, plus synthetic (clearly-labelled) supplementary tables for
amenities, nearby schools/hospitals, payment plans, and company FAQs --
none of which exist in the raw Kaggle data but are required by the
capstone brief.
"""
import pandas as pd
import numpy as np
import re
import random
import os

random.seed(42)
np.random.seed(42)

RAW = "raw/Property.csv"
OUT_DIR = "data"
os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_csv(RAW, sep=";")
df = df[df["price"] > 10000].copy()

def parse_area(a):
    a = str(a).strip()
    m = re.match(r"([\d.]+)\s*(Kanal|Marla|Sq\.?\s?Yd\.?|Sq\.?\s?Ft\.?)", a, re.I)
    if not m:
        return None, a
    val, unit = float(m.group(1)), m.group(2).lower()
    if "kanal" in unit:
        marla = val * 20
    elif "marla" in unit:
        marla = val
    elif "yd" in unit:
        marla = val / 30.25
    elif "ft" in unit:
        marla = val / 272.25
    else:
        marla = val
    return round(marla, 1), a

df["area_marla"], df["area_raw"] = zip(*df["area"].map(parse_area))
df = df[df["area_marla"].notna()]

# Scope to all 5 cities present in the Kaggle data (3 major + the 2 remaining)
MAJOR_CITIES = ["Lahore", "Karachi", "Islamabad", "Rawalpindi", "Faisalabad"]
df = df[df["city"].isin(MAJOR_CITIES)].copy()

TARGET_N = 700
frames = []
for city, grp in df.groupby("city"):
    frac = TARGET_N / len(df)
    n = max(40, int(len(grp) * frac))
    frames.append(grp.sample(n=min(n, len(grp)), random_state=42))
sample = pd.concat(frames).reset_index(drop=True)
sample = sample.sample(frac=1, random_state=1).reset_index(drop=True)
sample = sample.head(TARGET_N)

sample["agency"] = sample["agency"].replace("", np.nan).fillna("RealEstate Hub Direct")
sample["agent"] = sample["agent"].replace("", np.nan).fillna("Unassigned")
sample["property_id"] = sample["property_id"].astype(int)

print(f"Loaded raw: {len(df)} rows -> demo sample: {len(sample)} rows")
print(sample["city"].value_counts())
print(sample["purpose"].value_counts())
print(sample["property_type"].value_counts())

sample.to_csv(f"{OUT_DIR}/properties_clean.csv", index=False)
