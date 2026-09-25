"""
Day 2 - Task 3: Structured Retrieval vs Semantic Retrieval.

Split rationale (documented fully in README.md):
  SQL (structured_search)  -> exact facts you filter/sort/aggregate on:
      price, availability (purpose), plot size, bedrooms, baths, agent/agency
      names, city/location. These are categorical/numeric -- a WHERE clause
      is faster, cheaper, and 100% accurate; embedding a number for
      "similarity" search is the wrong tool and risks the LLM guessing.
  Vector (semantic_search) -> fuzzy, descriptive, free-text material:
      amenities phrasing, "why this society is nice", FAQs, payment-plan
      explanations -- things a caller asks in natural language that don't
      map to a single column.
"""
import sqlite3
import os
import chromadb
import pandas as pd
from embeddings import LocalEmbedder

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
DB_PATH = os.path.join(_ROOT, "data", "properties.db")
CHROMA_PATH = os.path.join(_ROOT, "data", "chroma")
_EMBEDDER_PATH = os.path.join(_ROOT, "data", "embedder.pkl")

_embedder = None
_client = None
_collections = {}  # cache by collection name -- avoid re-fetching collection
                    # metadata from disk on every single semantic_search call


def _get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = LocalEmbedder.load(_EMBEDDER_PATH)
    return _embedder


def _get_client():
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(path=CHROMA_PATH)
    return _client


def _get_collection(name):
    if name not in _collections:
        _collections[name] = _get_client().get_collection(name, embedding_function=_get_embedder())
    return _collections[name]


def structured_search(city=None, purpose=None, min_price=None, max_price=None,
                       min_bedrooms=None, property_type=None, location_contains=None,
                       order_by="price", limit=5):
    """Exact-fact SQL lookup. Returns a DataFrame straight from properties.db."""
    conn = sqlite3.connect(DB_PATH)
    clauses, params = [], []
    if city:
        clauses.append("city = ?"); params.append(city)
    if purpose:
        clauses.append("purpose = ?"); params.append(purpose)
    if min_price is not None:
        clauses.append("price >= ?"); params.append(min_price)
    if max_price is not None:
        clauses.append("price <= ?"); params.append(max_price)
    if min_bedrooms is not None:
        clauses.append("bedrooms >= ?"); params.append(min_bedrooms)
    if property_type:
        clauses.append("property_type = ?"); params.append(property_type)
    if location_contains:
        clauses.append("location LIKE ?"); params.append(f"%{location_contains}%")

    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""

    # Smart ordering: if the caller specified a bedroom count, sort by how
    # close each match is to that count (then cheapest first). Otherwise
    # cheapest-first so affordable options surface instead of mansions.
    # NOTE: `order_by` param is still accepted for backwards compatibility
    # but is intentionally ignored in favor of this smarter clause.
    if min_bedrooms is not None:
        order_clause = f"ORDER BY ABS(bedrooms - {int(min_bedrooms)}) ASC, price ASC"
    else:
        order_clause = "ORDER BY price ASC"

    sql = f"""
        SELECT property_id, city, location, property_type, purpose, price,
               bedrooms, baths, area_raw, area_marla, agency, agent, date_added
        FROM properties
        {where}
        {order_clause}
        LIMIT ?
    """
    params.append(limit)
    df = pd.read_sql(sql, conn, params=params)
    conn.close()
    return df


def semantic_search(query, collection="property_docs", n_results=4):
    """Fuzzy natural-language lookup over the vector store."""
    col = _get_collection(collection)
    res = col.query(query_texts=[query], n_results=n_results)
    hits = []
    for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
        hits.append({"text": doc, "metadata": meta, "distance": dist})
    return hits


def hybrid_search(city=None, purpose=None, max_price=None, min_bedrooms=None, query_text=None, limit=5):
    """Filter with SQL first (fast, exact), then rank/explain with semantic search
    over the *filtered* property_ids only -- avoids the vector store returning
    a perfect-sounding property that's outside budget."""
    exact = structured_search(city=city, purpose=purpose, max_price=max_price,
                               min_bedrooms=min_bedrooms, limit=50)
    if exact.empty:
        return exact.head(0), []
    allowed_ids = set(exact["property_id"].tolist())

    if not query_text:
        return exact.head(limit), []

    col = _get_collection("property_docs")
    # restrict the semantic search to just the SQL-filtered candidates via a
    # metadata `where` filter, so ranking never pulls in an out-of-budget match
    res = col.query(
        query_texts=[query_text],
        n_results=min(50, len(allowed_ids) * 2),
        where={"property_id": {"$in": list(allowed_ids)}},
    )
    ranked_ids = []
    for meta in res["metadatas"][0]:
        pid = meta["property_id"]
        if pid in allowed_ids and pid not in ranked_ids:
            ranked_ids.append(pid)
    ranked_ids = ranked_ids[:limit]
    ranked_df = exact[exact["property_id"].isin(ranked_ids)].set_index("property_id").loc[ranked_ids].reset_index()
    return ranked_df, res


if __name__ == "__main__":
    print("== structured_search: Lahore, For Sale, <= 5 crore, 3+ bed ==")
    print(structured_search(city="Lahore", purpose="For Sale", max_price=50_000_000, min_bedrooms=3, limit=5))

    print("\n== semantic_search: 'gated society with good schools nearby' ==")
    for h in semantic_search("gated society with good schools nearby", n_results=3):
        print(round(h["distance"], 3), h["metadata"], "|", h["text"][:100])

    print("\n== hybrid_search: budget 3 crore, DHA feel, family friendly ==")
    df, _ = hybrid_search(city="Lahore", purpose="For Sale", max_price=30_000_000,
                           query_text="family friendly gated community near good schools", limit=5)
    print(df[["property_id", "location", "price", "bedrooms"]] if not df.empty else "no matches")