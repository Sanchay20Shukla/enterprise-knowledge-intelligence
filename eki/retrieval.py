"""Dense + BM25 retrieval with reciprocal rank fusion and explicit reranking."""
import hashlib
import re
import sqlite3
import uuid
import numpy as np
from qdrant_client import QdrantClient, models
from rank_bm25 import BM25Okapi
from .models import Chunk, Evidence

STOP = set("a an the is are was were what which how why when who for to of in and or with does do please tell me about explain describe policy".split())


def terms(text):
    return [x for x in re.findall(r"[a-z0-9]+", text.lower()) if x not in STOP]


class Embedder:
    def __init__(self, mode="hash"):
        self.semantic = mode == "fastembed"
        self.dimension = 384
        self.model_id = "BAAI/bge-small-en-v1.5" if self.semantic else "hash-baseline-v1"
        if self.semantic:
            from fastembed import TextEmbedding
            self.model = TextEmbedding(model_name=self.model_id, threads=2)

    def encode(self, texts):
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)
        if self.semantic:
            matrix = np.array(list(self.model.embed(texts)), dtype=np.float32)
        else:
            matrix = np.zeros((len(texts), self.dimension), dtype=np.float32)
            for i, text in enumerate(texts):
                for word in terms(text):
                    digest = hashlib.sha256(word.encode()).digest()
                    matrix[i, int.from_bytes(digest[:4], "big") % self.dimension] += 1
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        return matrix / np.maximum(norms, 1e-9)


class Index:
    def __init__(self, settings):
        self.settings = settings
        settings.data_dir.mkdir(parents=True, exist_ok=True)
        self.embedder = Embedder(settings.embedding)
        self.collection = "chunks_" + settings.embedding + "_v1"
        self.client = (QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key, timeout=15)
                       if settings.qdrant_url else QdrantClient(path=str(settings.data_dir / "qdrant")))
        if not self.client.collection_exists(self.collection):
            self.client.create_collection(self.collection,
                vectors_config=models.VectorParams(size=384, distance=models.Distance.COSINE))
        self.db = sqlite3.connect(settings.data_dir / "catalog.sqlite", check_same_thread=False)
        self.db.execute("CREATE TABLE IF NOT EXISTS chunks (id TEXT PRIMARY KEY, body TEXT NOT NULL)")
        self.db.execute("CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        stored = self.db.execute("SELECT value FROM meta WHERE key='embedding'").fetchone()
        if stored and stored[0] != self.embedder.model_id:
            raise ValueError("Embedding model changed: use a fresh EKI_DATA_DIR and re-ingest")
        self.db.execute("INSERT OR IGNORE INTO meta VALUES ('embedding', ?)", (self.embedder.model_id,))
        self.db.commit()
        self.reranker = None
        if settings.reranker == "fastembed":
            from fastembed.rerank.cross_encoder import TextCrossEncoder
            self.reranker = TextCrossEncoder(model_name="Xenova/ms-marco-MiniLM-L-6-v2", threads=2)
        self.reload()

    def reload(self):
        self.chunks = [Chunk.model_validate_json(r[0]) for r in self.db.execute("SELECT body FROM chunks ORDER BY id")]
        self.by_id = {c.id: c for c in self.chunks}
        self.bm25 = BM25Okapi([terms(c.content) or ["empty"] for c in self.chunks]) if self.chunks else None
        self.revision = hashlib.sha256("".join(c.id for c in self.chunks).encode()).hexdigest()

    def add(self, chunks):
        new = [c for c in chunks if c.id not in self.by_id]
        if len(self.chunks) + len(new) > self.settings.max_corpus_chunks:
            raise ValueError("Corpus limit reached")
        vectors = self.embedder.encode([c.content for c in new])
        # Upsert first: orphan vectors after an interrupted commit are ignored by catalog membership.
        for start in range(0, len(new), 100):
            self.client.upsert(self.collection, points=[models.PointStruct(
                id=str(uuid.UUID(c.id)), vector=vector.tolist(), payload={"chunk_id": c.id})
                for c, vector in zip(new[start:start+100], vectors[start:start+100])], wait=True)
        with self.db:
            self.db.executemany("INSERT OR IGNORE INTO chunks VALUES (?, ?)",
                                [(c.id, c.model_dump_json()) for c in new])
        self.reload()
        return len(new)

    def search(self, query, limit=5):
        if not self.chunks:
            return []
        scores = self.bm25.get_scores(terms(query))
        keyword_ids = [self.chunks[i].id for i in np.argsort(-scores)[:20] if scores[i] > 0]
        vector = self.embedder.encode([query])[0].tolist()
        hits = self.client.query_points(self.collection, query=vector, limit=20).points
        dense_ids = [h.payload['chunk_id'] for h in hits if h.payload['chunk_id'] in self.by_id]
        rrf = {}
        for ranking in (keyword_ids, dense_ids):
            for rank, cid in enumerate(ranking, 1):
                rrf[cid] = rrf.get(cid, 0) + 1 / (60 + rank)
        candidates = sorted(rrf, key=lambda cid: (-rrf[cid], cid))[:20]
        if self.reranker:
            relevance = list(self.reranker.rerank(query, [self.by_id[c].content for c in candidates]))
            ranked = sorted(zip(candidates, relevance), key=lambda item: -item[1])
        else:
            q = set(terms(query))
            ranked = sorted([(cid, len(q & set(terms(self.by_id[cid].content))) / max(len(q), 1))
                             for cid in candidates], key=lambda item: (-item[1], -rrf[item[0]]))
        evidence = []
        for cid, score in ranked[:limit]:
            c = self.by_id[cid]
            evidence.append(Evidence(id=cid, source=c.source, text=c.content, page=c.page, row=c.row,
                                     score=float(score), url=f"/evidence/{cid}"))
        return evidence

    def close(self):
        self.db.close()
        self.client.close()
