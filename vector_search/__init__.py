import logging
import config

logger = logging.getLogger(__name__)


def get_index_manager():
    """
    Dynamically returns the active vector index manager (Qdrant or FAISS)
    based on `config.VECTOR_STORE_BACKEND`.
    """
    backend = getattr(config, "VECTOR_STORE_BACKEND", "faiss").lower()
    if backend == "qdrant":
        from vector_search.index_qdrant import get_qdrant_manager
        return get_qdrant_manager()
    else:
        from vector_search.index_faiss import get_index_manager as get_faiss_manager
        return get_faiss_manager()
