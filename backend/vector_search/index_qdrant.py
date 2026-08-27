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
        PayloadSchemaType,
    )
    HAS_QDRANT = True
except ImportError:
    HAS_QDRANT = False

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from doc_chunking.metadata import Chunk
from doc_chunking.passage_native import process_corpus_passage_native
from doc_chunking.sentence_window import process_longdocs_sentence_window
from doc_chunking.semantic import process_longdocs_semantic
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

        try:
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="source_lang",
                field_schema=PayloadSchemaType.KEYWORD,
            )
            logger.info("Ensured 'source_lang' keyword payload index for '%s'", self.collection_name)
        except Exception as e:
            logger.debug("Note creating payload index for '%s': %s", self.collection_name, e)

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
        query_vec: np.ndarray,
        target_lang: Optional[str] = None,
        top_k: int = 15,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """
        Execute Qdrant cosine vector search with optional payload language filtering.
        Returns candidate dicts identical to FAISS StrategyVectorIndex.search interface.
        """
        q_vec = query_vec if query_vec is not None else kwargs.get("query_vector")
        lang_filter = target_lang if target_lang is not None else kwargs.get("lang_filter")
        
        if q_vec is None or self.size == 0:
            return []

        # Ensure query vector is 1D float array (384,) to prevent multi-vector 400 error in Qdrant
        if isinstance(q_vec, np.ndarray):
            q_vec = q_vec.reshape(-1)
        elif isinstance(q_vec, list) and len(q_vec) > 0 and isinstance(q_vec[0], list):
            q_vec = q_vec[0]

        norm = np.linalg.norm(q_vec)
        unit_q = (np.array(q_vec) / (norm + 1e-12)).astype(float).tolist()

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

        search_result = []
        try:
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
        except Exception as e:
            logger.warning("Filtered Qdrant query encounter (%s); executing fallback query...", e)
            try:
                if hasattr(self.client, "query_points"):
                    search_result = self.client.query_points(
                        collection_name=self.collection_name,
                        query=unit_q,
                        limit=top_k,
                    ).points
                else:
                    search_result = self.client.search(
                        collection_name=self.collection_name,
                        query_vector=unit_q,
                        limit=top_k,
                    )
            except Exception as e2:
                logger.error("Qdrant query execution error: %s", e2)
                return []

        results: List[Dict[str, Any]] = []
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
            
            score = float(hit.score)
            results.append({
                "chunk_id": chunk.chunk_id,
                "text": chunk.text,
                "score": score,
                "source_lang": chunk.source_lang,
                "chunk_strategy": chunk.chunk_strategy,
                "source_query_ids": getattr(chunk, "source_query_ids", []),
                "doc_id": chunk.doc_id,
                "context_window": getattr(chunk, "context_window", None),
                "metadata": chunk.metadata,
            })

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
        self.load_all_indexes()

    @property
    def indexes(self) -> Dict[str, StrategyQdrantIndex]:
        """Alias for compatibility with FAISSIndexManager."""
        return self.strategy_indices

    @property
    def global_centroid(self) -> Optional[np.ndarray]:
        """Compute and return global centroid across all corpus centroids."""
        if not self.centroids:
            return None
        all_vecs = list(self.centroids.values())
        mean_v = np.mean(np.array(all_vecs), axis=0)
        norm = np.linalg.norm(mean_v)
        return (mean_v / norm) if norm > 1e-9 else mean_v

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
        query_vec: np.ndarray,
        target_lang: Optional[str] = None,
        top_k: int = 15,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        if strategy_name not in self.strategy_indices:
            logger.warning("Strategy '%s' not indexed in Qdrant", strategy_name)
            return []
        return self.strategy_indices[strategy_name].search(
            query_vec=query_vec,
            target_lang=target_lang,
            top_k=top_k,
            **kwargs,
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
                vec = embedder.encode_passages([chunk.text])[0]
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

    def build_all_indexes(self, max_passages_per_lang: Optional[int] = None):
        """
        Build and populate all chunking strategies into Qdrant Cloud Cluster.
        """
        logger.info("[QdrantManager] Building and indexing all corpora into Qdrant Cloud...")
        embedder = get_embedder()
        
        # 1. Passage Native Strategy
        passage_index = self.get_or_create_index("passage_native")
        for lang in config.LANGUAGES:
            corpus_file = config.PROCESSED_DATA_DIR / f"{lang}_corpus.jsonl"
            if not corpus_file.exists():
                continue
            records = []
            with open(corpus_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        records.append(json.loads(line))
            if records:
                chunks = process_corpus_passage_native(records)
                texts = [c.embed_text for c in chunks]
                vecs = embedder.encode_passages(texts)
                passage_index.add_chunks(chunks, vecs)
                logger.info("[Qdrant] Indexed %d passage-native chunks for language '%s'", len(chunks), lang)

        # 2. Semantic Longdoc Strategy
        longdoc_index = self.get_or_create_index("semantic_longdoc")
        for lang in config.LANGUAGES:
            longdoc_file = config.PROCESSED_DATA_DIR / f"{lang}_longdocs.jsonl"
            if not longdoc_file.exists():
                continue
            records = []
            with open(longdoc_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        records.append(json.loads(line))
            if records:
                chunks = process_longdocs_sentence_window(records) + process_longdocs_semantic(records)
                if chunks:
                    texts = [c.embed_text for c in chunks]
                    vecs = embedder.encode_passages(texts)
                    longdoc_index.add_chunks(chunks, vecs)
                    logger.info("[Qdrant] Indexed %d longdoc chunks for language '%s'", len(chunks), lang)

        self.compute_centroids()
        logger.info("[QdrantManager] Qdrant Cloud indexing complete across all strategies!")

    def load_all_indexes(self):
        """Auto-recovers or populates Qdrant Cloud indexes."""
        for strat in ["passage_native", "semantic_longdoc"]:
            self.get_or_create_index(strat)
        logger.info("[QdrantManager] Loaded active strategy indexes: %s", list(self.strategy_indices.keys()))


_QDRANT_MANAGER_INSTANCE: Optional[QdrantIndexManager] = None


def get_qdrant_manager(in_memory: bool = True) -> QdrantIndexManager:
    global _QDRANT_MANAGER_INSTANCE
    if _QDRANT_MANAGER_INSTANCE is None:
        _QDRANT_MANAGER_INSTANCE = QdrantIndexManager(in_memory=in_memory)
    return _QDRANT_MANAGER_INSTANCE
