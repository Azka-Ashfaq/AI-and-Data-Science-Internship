"""
Local, offline embedding backend (TF-IDF + Truncated SVD -> dense vectors).

Why not a hosted embedding API here: this sandbox has no OpenAI/Gemini key
and no network route to those providers, so a real Day-2 demo needs an
embedding step that runs with zero external dependencies. TF-IDF+SVD
("LSA") is a legitimate, classic dense-embedding technique and is a drop-in
placeholder: swap `LocalEmbedder` for `OpenAIEmbeddingFunction` /
`GoogleGenerativeAiEmbeddingFunction` in build_vector_store.py once an API
key is available -- nothing else in the pipeline changes.
"""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import numpy as np
import pickle


class LocalEmbedder:
    def __init__(self, n_components=128):
        self.vectorizer = TfidfVectorizer(max_features=20000, stop_words="english", ngram_range=(1, 2))
        self.svd = TruncatedSVD(n_components=n_components, random_state=42)
        self.fitted = False

    def fit(self, texts):
        tfidf = self.vectorizer.fit_transform(texts)
        self.svd.fit(tfidf)
        self.fitted = True
        return self

    def encode(self, texts):
        if not self.fitted:
            raise RuntimeError("Call fit() first")
        tfidf = self.vectorizer.transform(texts)
        vecs = self.svd.transform(tfidf)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1
        return vecs / norms

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump({"vectorizer": self.vectorizer, "svd": self.svd}, f)

    @classmethod
    def load(cls, path):
        obj = cls()
        with open(path, "rb") as f:
            d = pickle.load(f)
        obj.vectorizer, obj.svd = d["vectorizer"], d["svd"]
        obj.fitted = True
        return obj

    # chromadb EmbeddingFunction protocol
    def __call__(self, input):
        return self.encode(input).tolist()
    def embed_query(self, input):
        return self.__call__(input)
    def name(self):
        return "local-tfidf-svd"
