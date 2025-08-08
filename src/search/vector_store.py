import os
import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

import faiss
from sentence_transformers import SentenceTransformer

from ..shared.logging_utils import get_logger
from ..ingest.github_ingest import Document

logger = get_logger(__name__)


class VectorStore:
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        persist_dir: str = "./index/faiss_index",
        dimension: Optional[int] = None
    ):
        self.model_name = model_name
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Loading embedding model: {model_name}")
        self.encoder = SentenceTransformer(model_name)
        
        # Get embedding dimension
        if dimension is None:
            test_embedding = self.encoder.encode(["test"])
            self.dimension = test_embedding.shape[1]
        else:
            self.dimension = dimension
        
        logger.info(f"Embedding dimension: {self.dimension}")
        
        # Initialize FAISS index
        self.index = faiss.IndexFlatIP(self.dimension)  # Inner product (cosine similarity)
        
        # Store document metadata
        self.documents: List[Document] = []
        self.id_to_doc: Dict[int, Document] = {}
        
        # Load existing index if available
        self._load_index()
    
    def add_documents(self, documents: List[Document]) -> None:
        logger.info(f"Adding {len(documents)} documents to vector store")
        
        if not documents:
            return
        
        # Extract text content for embedding
        texts = [doc.content for doc in documents]
        
        # Generate embeddings in batches
        batch_size = 32
        embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_embeddings = self.encoder.encode(
                batch_texts,
                convert_to_numpy=True,
                show_progress_bar=True if len(texts) > 100 else False
            )
            embeddings.append(batch_embeddings)
        
        embeddings = np.vstack(embeddings)
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Add to FAISS index
        start_id = len(self.documents)
        self.index.add(embeddings)
        
        # Store documents with IDs
        for i, doc in enumerate(documents):
            doc_id = start_id + i
            self.documents.append(doc)
            self.id_to_doc[doc_id] = doc
        
        logger.info(f"Total documents in index: {len(self.documents)}")
    
    def search(
        self, 
        query: str, 
        k: int = 10,
        score_threshold: float = 0.5
    ) -> List[Tuple[Document, float]]:
        if not self.documents:
            logger.warning("No documents in vector store")
            return []
        
        # Generate query embedding
        query_embedding = self.encoder.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(query_embedding)
        
        # Search
        scores, indices = self.index.search(query_embedding, k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx != -1 and score >= score_threshold:
                document = self.id_to_doc[idx]
                results.append((document, float(score)))
        
        return results
    
    def save_index(self) -> None:
        logger.info(f"Saving vector index to {self.persist_dir}")
        
        # Save FAISS index
        index_path = self.persist_dir / "index.faiss"
        faiss.write_index(self.index, str(index_path))
        
        # Save documents and metadata
        docs_path = self.persist_dir / "documents.pkl"
        with open(docs_path, "wb") as f:
            pickle.dump({
                "documents": self.documents,
                "id_to_doc": self.id_to_doc,
                "model_name": self.model_name,
                "dimension": self.dimension
            }, f)
        
        logger.info("Vector index saved successfully")
    
    def _load_index(self) -> None:
        index_path = self.persist_dir / "index.faiss"
        docs_path = self.persist_dir / "documents.pkl"
        
        if index_path.exists() and docs_path.exists():
            logger.info("Loading existing vector index")
            
            try:
                # Load FAISS index
                self.index = faiss.read_index(str(index_path))
                
                # Load documents
                with open(docs_path, "rb") as f:
                    data = pickle.load(f)
                
                self.documents = data["documents"]
                self.id_to_doc = data["id_to_doc"]
                
                # Verify model compatibility
                saved_model = data.get("model_name", "")
                if saved_model != self.model_name:
                    logger.warning(f"Model mismatch: saved={saved_model}, current={self.model_name}")
                
                logger.info(f"Loaded {len(self.documents)} documents from index")
                
            except Exception as e:
                logger.error(f"Failed to load index: {e}")
                logger.info("Starting with empty index")
                self.index = faiss.IndexFlatIP(self.dimension)
                self.documents = []
                self.id_to_doc = {}
        else:
            logger.info("No existing index found, starting fresh")
    
    def clear_index(self) -> None:
        logger.info("Clearing vector index")
        self.index = faiss.IndexFlatIP(self.dimension)
        self.documents = []
        self.id_to_doc = {}
    
    def get_stats(self) -> Dict[str, Any]:
        doc_types = {}
        sources = {}
        
        for doc in self.documents:
            doc_type = doc.doc_type
            doc_types[doc_type] = doc_types.get(doc_type, 0) + 1
            
            source = doc.metadata.get("source", "unknown")
            sources[source] = sources.get(source, 0) + 1
        
        return {
            "total_documents": len(self.documents),
            "index_size": self.index.ntotal,
            "embedding_dimension": self.dimension,
            "model_name": self.model_name,
            "document_types": doc_types,
            "sources": sources
        }


class KeywordSearch:
    def __init__(self):
        self.documents: List[Document] = []
        self.doc_to_id: Dict[str, int] = {}
    
    def add_documents(self, documents: List[Document]) -> None:
        start_id = len(self.documents)
        for i, doc in enumerate(documents):
            doc_id = start_id + i
            self.documents.append(doc)
            self.doc_to_id[doc.source] = doc_id
    
    def search(self, query: str, k: int = 10) -> List[Tuple[Document, float]]:
        query_terms = set(query.lower().split())
        
        scores = []
        for i, doc in enumerate(self.documents):
            content_terms = set(doc.content.lower().split())
            
            # Simple TF-IDF-like scoring
            matches = query_terms.intersection(content_terms)
            if matches:
                score = len(matches) / len(query_terms)
                scores.append((doc, score, i))
        
        # Sort by score and return top k
        scores.sort(key=lambda x: x[1], reverse=True)
        return [(doc, score) for doc, score, _ in scores[:k]]
    
    def clear_index(self) -> None:
        self.documents = []
        self.doc_to_id = {}