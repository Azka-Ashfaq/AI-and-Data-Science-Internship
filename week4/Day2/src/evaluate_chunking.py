"""
Day 2 - Task 2: Evaluate different chunk sizes.
Rebuilds the property_docs collection 3 times at different chunk sizes and
measures, for a fixed set of test queries, whether the top-1 retrieved
chunk's property_id is the one we know (from the query) is the intended
answer -- i.e. retrieval precision@1, plus average chunks/property (cost proxy).
"""
import pandas as pd
import chromadb
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from embeddings import LocalEmbedder
from build_vector_store import simple_chunk, property_to_document

props = pd.read_csv("data/properties_clean.csv")
amenities_df = pd.read_csv("data/amenities.csv")
schools_df = pd.read_csv("data/schools.csv")
hospitals_df = pd.read_csv("data/hospitals.csv")
plans_df = pd.read_csv("data/payment_plans.csv")

amenities_map = {(r.city, r.location): r.amenities for r in amenities_df.itertuples()}
schools_map = (schools_df.groupby(["city", "location"])["school_name"].apply(lambda s: ", ".join(s)).to_dict())
hospitals_map = (hospitals_df.groupby(["city", "location"])["hospital_name"].apply(lambda s: ", ".join(s)).to_dict())
plans_map = {(r.city, r.location): {"plan_summary": r.plan_summary} for r in plans_df.itertuples()}

# Build test queries from 15 random real properties: query = a paraphrase of
# their own facts, target = their own property_id (ground truth).
sample_props = props.sample(15, random_state=3)
test_cases = []
for r in sample_props.itertuples():
    q = f"{r.bedrooms} bedroom {r.property_type.lower()} for {r.purpose.split()[-1].lower()} in {r.location} {r.city} around {r.price:,} rupees"
    test_cases.append((q, r.property_id))

def build_and_eval(chunk_size, overlap):
    docs, metas, ids = [], [], []
    for row in props.itertuples():
        r = dict(zip(props.columns, row[1:]))
        text = property_to_document(r, amenities_map, schools_map, hospitals_map, plans_map)
        for ci, chunk in enumerate(simple_chunk(text, size=chunk_size, overlap=overlap)):
            docs.append(chunk)
            metas.append({"property_id": int(r["property_id"]), "chunk_index": ci})
            ids.append(f"p{r['property_id']}-{ci}")

    embedder = LocalEmbedder(n_components=128).fit(docs)
    client = chromadb.EphemeralClient()
    name = f"eval_{chunk_size}"
    try:
        client.delete_collection(name)
    except Exception:
        pass
    col = client.create_collection(name, embedding_function=embedder)
    B = 300
    for i in range(0, len(docs), B):
        col.add(documents=docs[i:i+B], metadatas=metas[i:i+B], ids=ids[i:i+B])

    hits = 0
    for q, true_pid in test_cases:
        res = col.query(query_texts=[q], n_results=1)
        pred_pid = res["metadatas"][0][0]["property_id"]
        hits += int(pred_pid == true_pid)

    precision_at_1 = hits / len(test_cases)
    avg_chunks = len(docs) / len(props)
    return precision_at_1, avg_chunks, len(docs)


print(f"{'chunk_size':>10} {'overlap':>8} {'precision@1':>12} {'avg_chunks/prop':>16} {'total_chunks':>13}")
for size, overlap in [(150, 20), (400, 60), (800, 100)]:
    p1, avgc, total = build_and_eval(size, overlap)
    print(f"{size:>10} {overlap:>8} {p1:>12.2%} {avgc:>16.2f} {total:>13}")
