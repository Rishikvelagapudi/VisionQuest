"""
Qdrant Vector Database Integration Engine.

Provides native Qdrant vector indexing and HNSW cosine similarity search:
- In-memory or local persistent Qdrant engine via `qdrant-client`.
- Native payload metadata language filtering.
- Dual compatibility with FAISSIndexManager interface.
"""

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import (
        Distance,
        VectorParams,
        PointStruct,
        Filter,
        FieldCondition,
        MatchValue,
    )
    HAS_QDRANT = True
except ImportError:
    HAS_QDRANT = False

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from doc_chunking.metadata import Chunk
from vector_search.embed import get_embedder

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", force=True)
logger = logging.getLogger(__name__)


class StrategyQdrantIndex:
    """
    Manages a Qdrant collection for a specific chunking strategy.
    """
    def __init__(
        self,
        strategy_name: str,
        client: "QdrantClient",
        dim: int = config.EMBEDDING_DIM,
    ):
        if not HAS_QDRANT:
            raise ImportError("qdrant-client is not installed. Run `pip install qdrant-client`")
            
        self.strategy_name = strategy_name
        self.client = client
        self.dim = dim
        self.collection_name = f"vector_{strategy_name}"
        self.chunks: List[Chunk] = []

        # Create or recreate collection in Qdrant
        if not self.client.collection_exists(self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=self.dim, distance=Distance.COSINE),
            )
            logger.info("Created Qdrant collection: '%s' (dim=%d)", self.collection_name, self.dim)

    @property
    def size(self) -> int:
        """Returns number of vectors in collection."""
        try:
            info = self.client.get_collection(self.collection_name)
            return info.points_count or len(self.chunks)
        except Exception:
            return len(self.chunks)

    def add_chunks(self, chunks: List[Chunk], vectors: np.ndarray):
        """Add chunks and vector embeddings into Qdrant index."""
        if not chunks or len(chunks) == 0:
            return
        if len(chunks) != len(vectors):
            raise ValueError(f"Chunk count ({len(chunks)}) != vector count ({len(vectors)})")

        start_idx = len(self.chunks)
        points: List[PointStruct] = []
        
        for idx_offset, (chunk, vec) in enumerate(zip(chunks, vectors)):
            global_id = start_idx + idx_offset
            # Unit normalize for cosine distance
            norm = np.linalg.norm(vec)
            unit_vec = (vec / norm).tolist() if norm > 1e-9 else vec.tolist()
            
            payload = {
                "chunk_id": chunk.chunk_id,
                "doc_id": chunk.doc_id,
                "text": chunk.text,
                "source_lang": chunk.source_lang.lower(),
                "metadata": chunk.metadata,
                "idx": global_id,
            }
            points.append(PointStruct(id=global_id, vector=unit_vec, payload=payload))
            self.chunks.append(chunk)

        self.client.upsert(collection_name=self.collection_name, points=points)
        logger.info("Indexed %d vectors into Qdrant collection '%s'", len(points), self.collection_name)

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 5,
        lang_filter: Optional[str] = None,
    ) -> List[Tuple[Chunk, float]]:
        """
        Execute Qdrant cosine vector search with optional payload language filtering.
        """
        if self.size == 0:
            return []

        norm = np.linalg.norm(query_vector)
        unit_q = (query_vector / norm).tolist() if norm > 1e-9 else query_vector.tolist()

        qdrant_filter = None
        if lang_filter and lang_filter.strip():
            qdrant_filter = Filter(
                must=[
                    FieldCondition(
                        key="source_lang",
                        match=MatchValue(value=lang_filter.lower().strip()),
                    )
                ]
            )

        if hasattr(self.client, "query_points"):
            search_result = self.client.query_points(
                collection_name=self.collection_name,
                query=unit_q,
                query_filter=qdrant_filter,
                limit=top_k,
            ).points
        else:
            search_result = self.client.search(
                collection_name=self.collection_name,
                query_vector=unit_q,
                query_filter=qdrant_filter,
                limit=top_k,
            )

        results: List[Tuple[Chunk, float]] = []
        for hit in search_result:
            payload = hit.payload or {}
            idx = payload.get("idx", 0)
            
            if 0 <= idx < len(self.chunks):
                chunk = self.chunks[idx]
            else:
                chunk_text = payload.get("text", "")
                chunk = Chunk(
                    chunk_id=payload.get("chunk_id", ""),
                    doc_id=payload.get("doc_id", ""),
                    text=chunk_text,
                    embed_text=chunk_text,
                    source_lang=payload.get("source_lang", "en"),
                    chunk_strategy=self.strategy_name,
                    token_count=len(chunk_text.split()),
                    metadata=payload.get("metadata", {}),
                )
            
            # Qdrant score is cosine similarity [0, 1]
            score = float(hit.score)
            results.append((chunk, score))

        return results


class QdrantIndexManager:
    """
    Manager for Qdrant Vector Engine across all chunking strategies.
    """
    def __init__(self, in_memory: bool = True, storage_path: Optional[str] = None):
        if not HAS_QDRANT:
            raise ImportError("qdrant-client package is required for Qdrant vector backend.")
            
        qdrant_url = getattr(config, "QDRANT_URL", "")
        qdrant_key = getattr(config, "QDRANT_API_KEY", "")

        if qdrant_url and qdrant_url.strip():
            self.client = QdrantClient(url=qdrant_url.strip(), api_key=qdrant_key.strip() if qdrant_key else None)
            logger.info("Initialized Qdrant Cloud Vector Engine at '%s'.", qdrant_url)
        elif in_memory:
            self.client = QdrantClient(":memory:")
            logger.info("Initialized Qdrant In-Memory Vector Storage Engine.")
        else:
            path = storage_path or str(config.DATA_DIR / "qdrant_db")
            os.makedirs(path, exist_ok=True)
            self.client = QdrantClient(path=path)
            logger.info("Initialized Qdrant Persistent Storage Engine at '%s'.", path)

        self.strategy_indices: Dict[str, StrategyQdrantIndex] = {}
        self.centroids: Dict[str, np.ndarray] = {}

    def get_or_create_index(self, strategy_name: str) -> StrategyQdrantIndex:
        if strategy_name not in self.strategy_indices:
            self.strategy_indices[strategy_name] = StrategyQdrantIndex(
                strategy_name=strategy_name,
                client=self.client,
            )
        return self.strategy_indices[strategy_name]

    def search(
        self,
        strategy_name: str,
        query_vector: np.ndarray,
        top_k: int = 5,
        lang_filter: Optional[str] = None,
    ) -> List[Tuple[Chunk, float]]:
        if strategy_name not in self.strategy_indices:
            logger.warning("Strategy '%s' not indexed in Qdrant", strategy_name)
            return []
        return self.strategy_indices[strategy_name].search(
            query_vector=query_vector,
            top_k=top_k,
            lang_filter=lang_filter,
        )

    def compute_centroids(self):
        """Compute corpus language centroids for guardrails."""
        lang_vectors: Dict[str, List[np.ndarray]] = {}
        embedder = get_embedder()
        
        for idx_obj in self.strategy_indices.values():
            for chunk in idx_obj.chunks:
                lang = chunk.source_lang.lower()
                if lang not in lang_vectors:
                    lang_vectors[lang] = []
                vec = embedder.embed_text(chunk.text)
                lang_vectors[lang].append(vec)
                
        for lang, vecs in lang_vectors.items():
            arr = np.array(vecs)
            mean_v = np.mean(arr, axis=0)
            norm = np.linalg.norm(mean_v)
            self.centroids[lang] = (mean_v / norm) if norm > 1e-9 else mean_v
            
        logger.info("Computed Qdrant corpus centroids for languages: %s", list(self.centroids.keys()))

    def get_centroid(self, lang: str) -> Optional[np.ndarray]:
        return self.centroids.get(lang.lower().strip())

    def get_all_centroids(self) -> Dict[str, np.ndarray]:
        return self.centroids


_QDRANT_MANAGER_INSTANCE: Optional[QdrantIndexManager] = None


def get_qdrant_manager(in_memory: bool = True) -> QdrantIndexManager:
    global _QDRANT_MANAGER_INSTANCE
    if _QDRANT_MANAGER_INSTANCE is None:
        _QDRANT_MANAGER_INSTANCE = QdrantIndexManager(in_memory=in_memory)
    return _QDRANT_MANAGER_INSTANCE
