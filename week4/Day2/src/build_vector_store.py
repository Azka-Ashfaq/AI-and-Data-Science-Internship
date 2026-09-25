"""
Day 2 - Task 1 (documents) + Task 2 (RAG pipeline: loader, chunking, embedding, vector store).

Two Chroma collections are built:
  - "property_docs": one chunked document per property (facts + amenities +
    nearby schools/hospitals folded into natural language) -- semantic layer
    that complements the SQL structured layer.
  - "faqs": company FAQ knowledge base.

Chunk-size evaluation (Task 2) is run separately in evaluate_chunking.py.
"""
import pandas as pd
import chromadb
from embeddings import LocalEmbedder
import re
import json
import os

CHUNK_SIZE = 400   # chars; re-evaluated after adding all 5 cities (see README) --
CHUNK_OVERLAP = 60   # 400 now edges out 800 (80.0% vs 73.3% precision@1) now that
                     # denser, more varied listings across 5 cities compete for
                     # retrieval -- smaller chunks discriminate better here


def simple_chunk(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Lightweight recursive-ish splitter: break on sentence boundaries first,
    pack into ~size-char windows with overlap. No external deps needed."""
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    chunks, cur = [], ""
    for s in sentences:
        if len(cur) + len(s) + 1 <= size:
            cur = (cur + " " + s).strip()
        else:
            if cur:
                chunks.append(cur)
            cur = s
    if cur:
        chunks.append(cur)
    # add overlap between consecutive chunks
    if overlap and len(chunks) > 1:
        overlapped = [chunks[0]]
        for i in range(1, len(chunks)):
            tail = overlapped[-1][-overlap:]
            overlapped.append((tail + " " + chunks[i]).strip())
        chunks = overlapped
    return chunks


def property_to_document(row, amenities_map, schools_map, hospitals_map, plans_map):
    key = (row["city"], row["location"])
    amenities = amenities_map.get(key, "standard society amenities")
    schools = schools_map.get(key, "nearby schools on request")
    hospitals = hospitals_map.get(key, "nearby hospitals on request")
    plan = plans_map.get(key)
    plan_txt = plan["plan_summary"] if plan is not None else "Payment plan available on request."

    price_str = f"PKR {row['price']:,.0f}"
    bedrooms = int(row["bedrooms"]) if row["bedrooms"] else 0
    baths = int(row["baths"]) if row["baths"] else 0

    text = (
        f"{row['property_type']} {row['purpose'].lower()} in {row['location']}, {row['city']}. "
        f"Price: {price_str}. Size: {row['area_raw']} ({row['area_marla']} marla). "
        f"Bedrooms: {bedrooms}, Bathrooms: {baths}. "
        f"Listed by {row['agency']} (agent: {row['agent']}). "
        f"Amenities in this society: {amenities}. "
        f"Nearby schools: {schools}. Nearby hospitals: {hospitals}. "
        f"Payment plan: {plan_txt}"
    )
    return text


def main():
    props = pd.read_csv("data/properties_clean.csv")
    amenities_df = pd.read_csv("data/amenities.csv")
    schools_df = pd.read_csv("data/schools.csv")
    hospitals_df = pd.read_csv("data/hospitals.csv")
    plans_df = pd.read_csv("data/payment_plans.csv")
    faqs_df = pd.read_csv("data/faqs.csv")

    amenities_map = {(r.city, r.location): r.amenities for r in amenities_df.itertuples()}
    schools_map = (schools_df.groupby(["city", "location"])["school_name"]
                   .apply(lambda s: ", ".join(s)).to_dict())
    hospitals_map = (hospitals_df.groupby(["city", "location"])["hospital_name"]
                      .apply(lambda s: ", ".join(s)).to_dict())
    plans_map = {(r.city, r.location): {"plan_summary": r.plan_summary} for r in plans_df.itertuples()}

    # ---- build documents (Task 1 corpus) ----
    docs, metas, ids = [], [], []
    for row in props.itertuples():
        r = row._asdict() if hasattr(row, "_asdict") else dict(zip(props.columns, row[1:]))
        text = property_to_document(r, amenities_map, schools_map, hospitals_map, plans_map)
        for ci, chunk in enumerate(simple_chunk(text)):
            docs.append(chunk)
            metas.append({"property_id": int(r["property_id"]), "city": r["city"],
                           "location": r["location"], "chunk_index": ci})
            ids.append(f"prop-{r['property_id']}-{ci}")

    print(f"Property corpus: {len(props)} properties -> {len(docs)} chunks "
          f"(avg {len(docs)/len(props):.2f} chunks/property)")

    faq_docs, faq_metas, faq_ids = [], [], []
    for i, r in faqs_df.iterrows():
        text = f"Q: {r['question']} A: {r['answer']}"
        faq_docs.append(text)
        faq_metas.append({"faq_id": i})
        faq_ids.append(f"faq-{i}")

    # ---- fit local embedder on the full corpus (properties + FAQs) ----
    embedder = LocalEmbedder(n_components=128).fit(docs + faq_docs)
    embedder.save("data/embedder.pkl")

    client = chromadb.PersistentClient(path="data/chroma")
    for name in ("property_docs", "faqs"):
        try:
            client.delete_collection(name)
        except Exception:
            pass

    prop_col = client.create_collection("property_docs", embedding_function=embedder)
    faq_col = client.create_collection("faqs", embedding_function=embedder)

    BATCH = 200
    for i in range(0, len(docs), BATCH):
        prop_col.add(documents=docs[i:i+BATCH], metadatas=metas[i:i+BATCH], ids=ids[i:i+BATCH])
    faq_col.add(documents=faq_docs, metadatas=faq_metas, ids=faq_ids)

    print(f"Chroma 'property_docs' collection: {prop_col.count()} chunks")
    print(f"Chroma 'faqs' collection: {faq_col.count()} chunks")


if __name__ == "__main__":
    main()
