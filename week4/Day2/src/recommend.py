"""
Day 2 - Task 4: Property Recommendation Engine.
Combines the SQL structured filter (hard constraints: budget, city, purpose,
bedrooms) with a semantic re-rank (soft preferences: amenities, "investment
goals" phrasing) via retriever.hybrid_search, then adds a simple explainable
score so results aren't just "first N that match".
"""
import pandas as pd
from retriever import structured_search, hybrid_search


def score_property(row, budget, target_bedrooms=None):
    """Simple, explainable scoring: reward being under budget with room to
    spare, and matching the requested bedroom count closely."""
    score = 0.0
    if budget:
        headroom = (budget - row["price"]) / budget
        score += max(0, min(headroom, 0.3)) * 100  # up to 30 pts for value-for-budget
    if target_bedrooms:
        score += max(0, 20 - abs(row["bedrooms"] - target_bedrooms) * 8)
    return round(score, 1)


def recommend(city=None, purpose="For Sale", budget=None, bedrooms=None,
              area_min_marla=None, amenity_query=None, limit=5):
    """
    budget            -> max_price (hard SQL filter)
    bedrooms           -> min_bedrooms (hard SQL filter, +/-1 tolerance handled by score)
    amenity_query      -> free text like "near good schools, gated, quiet" (semantic re-rank)
    """
    if amenity_query:
        df, _ = hybrid_search(city=city, purpose=purpose, max_price=budget,
                               min_bedrooms=max(0, (bedrooms or 1) - 1),
                               query_text=amenity_query, limit=max(limit * 3, 15))
    else:
        df = structured_search(city=city, purpose=purpose, max_price=budget,
                                min_bedrooms=max(0, (bedrooms or 1) - 1), limit=max(limit * 3, 15))

    if df.empty:
        return df

    if area_min_marla:
        df = df[df["area_marla"] >= area_min_marla]
    if df.empty:
        return df

    df = df.copy()
    df["match_score"] = df.apply(lambda r: score_property(r, budget, bedrooms), axis=1)
    df = df.sort_values("match_score", ascending=False).head(limit)
    return df[["property_id", "city", "location", "property_type", "price",
               "bedrooms", "baths", "area_raw", "agency", "agent", "match_score"]]


if __name__ == "__main__":
    print("Buyer: Lahore, budget 3 crore, 3 bedrooms, wants gated + near schools")
    res = recommend(city="Lahore", purpose="For Sale", budget=30_000_000,
                     bedrooms=3, amenity_query="gated community near good schools quiet family friendly")
    print(res.to_string(index=False))

    print("\nInvestor: Islamabad, budget 5 crore, no bedroom preference, ROI-oriented society")
    res2 = recommend(city="Islamabad", purpose="For Sale", budget=50_000_000,
                      amenity_query="strong appreciation potential developing society investment")
    print(res2.to_string(index=False))
