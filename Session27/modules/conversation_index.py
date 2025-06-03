from typing import List, Dict, Any, Optional
import json
import os
from datetime import datetime
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from dataclasses import dataclass
import pickle


@dataclass
class ConversationEntry:
    timestamp: float
    session_id: str
    query: str
    response: str
    metadata: Dict[str, Any]
    embedding: Optional[np.ndarray] = None


class ConversationIndex:
    def __init__(self, index_dir: str = "faiss_index"):
        self.index_dir = index_dir
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.dimension = 384  # Dimension of the embeddings
        self.index = None
        self.entries: List[ConversationEntry] = []
        self._initialize_index()

    def _initialize_index(self):
        """Initialize or load existing FAISS index."""
        os.makedirs(self.index_dir, exist_ok=True)
        index_path = os.path.join(self.index_dir, "conversation_index.faiss")
        entries_path = os.path.join(self.index_dir, "conversation_entries.pkl")

        if os.path.exists(index_path) and os.path.exists(entries_path):
            try:
                self.index = faiss.read_index(index_path)
                with open(entries_path, 'rb') as f:
                    self.entries = pickle.load(f)
            except Exception as e:
                print(f"Error loading existing index: {e}")
                self._create_new_index()
        else:
            self._create_new_index()

    def _create_new_index(self):
        """Create a new empty index."""
        self.index = faiss.IndexFlatL2(self.dimension)
        self.entries = []
        self._save_index()

    def _save_index(self):
        """Save the FAISS index and entries to disk."""
        try:
            index_path = os.path.join(self.index_dir, "conversation_index.faiss")
            entries_path = os.path.join(self.index_dir, "conversation_entries.pkl")

            faiss.write_index(self.index, index_path)
            with open(entries_path, 'wb') as f:
                pickle.dump(self.entries, f)
        except Exception as e:
            print(f"Error saving index: {e}")

    def add_conversation(self, session_id: str, query: str, response: str, metadata: Dict[str, Any] = None):
        """Add a new conversation to the index."""
        try:
            # Create conversation entry
            entry = ConversationEntry(
                timestamp=datetime.now().timestamp(),
                session_id=session_id,
                query=query,
                response=response,
                metadata=metadata or {}
            )

            # Generate embedding for the query
            query_embedding = self.model.encode([query])[0]
            entry.embedding = query_embedding

            # Add to FAISS index
            self.index.add(np.array([query_embedding]))
            self.entries.append(entry)

            # Save updated index
            self._save_index()
        except Exception as e:
            print(f"Error adding conversation: {e}")

    def search_similar_conversations(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar conversations based on query."""
        if not self.entries:
            return []

        try:
            # Generate query embedding
            query_embedding = self.model.encode([query])[0]

            # Search in FAISS index
            distances, indices = self.index.search(np.array([query_embedding]), k)

            # Get results
            results = []
            for i, idx in enumerate(indices[0]):
                if idx < len(self.entries):
                    entry = self.entries[idx]
                    results.append({
                        'query': entry.query,
                        'response': entry.response,
                        'session_id': entry.session_id,
                        'timestamp': entry.timestamp,
                        'similarity_score': float(1 / (1 + distances[0][i])),  # Convert distance to similarity
                        'metadata': entry.metadata
                    })

            return results
        except Exception as e:
            print(f"Error searching conversations: {e}")
            return []

    def get_session_history(self, session_id: str) -> List[Dict[str, Any]]:
        """Get all conversations for a specific session."""
        try:
            return [
                {
                    'query': entry.query,
                    'response': entry.response,
                    'timestamp': entry.timestamp,
                    'metadata': entry.metadata
                }
                for entry in self.entries
                if entry.session_id == session_id
            ]
        except Exception as e:
            print(f"Error getting session history: {e}")
            return []

    def get_recent_conversations(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get most recent conversations across all sessions."""
        try:
            sorted_entries = sorted(self.entries, key=lambda x: x.timestamp, reverse=True)
            return [
                {
                    'query': entry.query,
                    'response': entry.response,
                    'session_id': entry.session_id,
                    'timestamp': entry.timestamp,
                    'metadata': entry.metadata
                }
                for entry in sorted_entries[:limit]
            ]
        except Exception as e:
            print(f"Error getting recent conversations: {e}")
            return []
