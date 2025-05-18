from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os
import logging
from utils import save_to_cache, load_from_cache, ensure_cache_dir

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmbeddingService:
    def __init__(self):
        # Try to load cached model
        self.model = load_from_cache('embedding', 'model')
        if not self.model:
            logger.info("Loading new embedding model...")
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            save_to_cache(self.model, 'embedding', 'model')
        
        # Initialize FAISS index
        self.index = None
        self.initialize_index()

    def get_faiss_index_path(self):
        """Get the path for the FAISS index file"""
        cache_dir = ensure_cache_dir()
        return os.path.join(cache_dir, 'faiss_index.bin')

    def initialize_index(self):
        """Initialize or load FAISS index"""
        index_path = self.get_faiss_index_path()
        
        # Try to load existing index
        if os.path.exists(index_path):
            try:
                logger.info("Loading existing FAISS index...")
                self.index = faiss.read_index(index_path)
                return
            except Exception as e:
                logger.error(f"Error loading FAISS index: {str(e)}")
                logger.info("Creating new index...")

        # Create new index if loading failed or no index exists
        embedding_dim = self.model.get_sentence_embedding_dimension()
        self.index = faiss.IndexFlatL2(embedding_dim)
        
        # Save the new index
        try:
            faiss.write_index(self.index, index_path)
            logger.info("Created and saved new FAISS index")
        except Exception as e:
            logger.error(f"Error saving FAISS index: {str(e)}")

    def generate_embeddings(self, texts):
        """Generate embeddings for a list of texts"""
        try:
            logger.info(f"Generating embeddings for {len(texts)} texts")
            embeddings = self.model.encode(texts)
            logger.info(f"Generated embeddings with shape: {embeddings.shape}")
            return embeddings
        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            raise

    def find_similar_emails(self, query_embedding, email_embeddings, emails, k=2):
        """Find similar emails using FAISS"""
        try:
            # Input validation
            if email_embeddings is None or emails is None:
                logger.warning("Email embeddings or emails list is None")
                return []

            if isinstance(email_embeddings, list) and len(email_embeddings) == 0:
                logger.warning("Empty email embeddings list")
                return []

            if len(emails) == 0:
                logger.warning("Empty emails list")
                return []

            # Log shapes for debugging
            logger.info(f"Query embedding shape: {query_embedding.shape}")
            logger.info(f"Email embeddings shape: {np.array(email_embeddings).shape}")
            logger.info(f"Number of emails: {len(emails)}")

            # Convert embeddings to float32 and ensure correct shape
            query_embedding = np.array(query_embedding, dtype='float32')
            email_embeddings = np.array(email_embeddings, dtype='float32')

            # Reshape if necessary
            if query_embedding.ndim == 1:
                query_embedding = query_embedding.reshape(1, -1)
            if email_embeddings.ndim == 1:
                email_embeddings = email_embeddings.reshape(1, -1)

            logger.info(f"Reshaped query embedding: {query_embedding.shape}")
            logger.info(f"Reshaped email embeddings: {email_embeddings.shape}")

            # Reset and rebuild index
            self.index.reset()
            self.index.add(email_embeddings)

            # Search for similar emails
            distances, indices = self.index.search(query_embedding, k)
            logger.info(f"Search results - distances: {distances}, indices: {indices}")

            # Get similar emails
            similar_emails = []
            for i, idx in enumerate(indices[0]):
                if idx >= 0 and idx < len(emails):
                    similar_emails.append(emails[idx])
                    logger.info(f"Added email {idx} with distance {distances[0][i]}")
                else:
                    logger.warning(f"Invalid index {idx} found in search results")

            # Save updated index
            index_path = self.get_faiss_index_path()
            faiss.write_index(self.index, index_path)
            logger.info(f"Found {len(similar_emails)} similar emails")

            return similar_emails

        except Exception as e:
            logger.error(f"Error in find_similar_emails: {str(e)}")
            logger.error(f"Query embedding type: {type(query_embedding)}")
            logger.error(f"Email embeddings type: {type(email_embeddings)}")
            if isinstance(query_embedding, np.ndarray):
                logger.error(f"Query embedding shape: {query_embedding.shape}")
            if isinstance(email_embeddings, np.ndarray):
                logger.error(f"Email embeddings shape: {email_embeddings.shape}")
            raise 