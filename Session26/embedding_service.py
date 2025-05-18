from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os
import logging
from utils import save_to_cache, load_from_cache, ensure_cache_dir
from sklearn.metrics.pairwise import cosine_similarity
from concurrent.futures import ThreadPoolExecutor
import torch
from models import Email, init_db
import json
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmbeddingService:
    def __init__(self):
        self.model = None
        self.cache = {}
        self.max_workers = min(4, os.cpu_count() or 1)
        self.batch_size = 32
        self.use_cache = True
        self.db = init_db()
        self.faiss_index = None
        self.initialize_faiss()

    def _load_model(self):
        """Load the embedding model"""
        try:
            if self.use_cache:
                cached_model = load_from_cache('embedding', 'model')
                if cached_model:
                    logger.info("Loaded embedding model from cache")
                    return cached_model

            logger.info("Loading embedding model...")
            model = SentenceTransformer('all-MiniLM-L6-v2')
            
            if self.use_cache:
                save_to_cache(model, 'embedding', 'model')
            logger.info("Loaded and cached new embedding model")
            return model
        except Exception as e:
            logger.error(f"Error loading embedding model: {str(e)}")
            raise

    def initialize_faiss(self):
        """Initialize FAISS index from database"""
        try:
            if not self.model:
                self.model = self._load_model()

            # Get all emails with embeddings
            emails = self.db.query(Email).filter(Email.embedding.isnot(None)).all()
            if not emails:
                logger.info("No emails with embeddings found in database")
                return

            # Create FAISS index
            embedding_dim = self.model.get_sentence_embedding_dimension()
            self.faiss_index = faiss.IndexFlatL2(embedding_dim)
            
            # Add embeddings to index
            embeddings = []
            for email in emails:
                if email.embedding:
                    embeddings.append(np.array(json.loads(email.embedding)))
            
            if embeddings:
                self.faiss_index.add(np.array(embeddings))
                logger.info(f"Initialized FAISS index with {len(embeddings)} embeddings")
        except Exception as e:
            logger.error(f"Error initializing FAISS index: {str(e)}")

    def update_embeddings(self, emails):
        """Update embeddings for new or modified emails"""
        try:
            if not self.model:
                self.model = self._load_model()

            for email in emails:
                try:
                    # Generate embedding
                    text = f"{email['subject']} {email['content']}"
                    embedding = self.model.encode(text, show_progress_bar=False)
                    
                    # Update or create email record
                    db_email = self.db.query(Email).filter_by(id=email['id']).first()
                    if db_email:
                        db_email.subject = email['subject']
                        db_email.from_address = email['from']
                        db_email.date = datetime.fromisoformat(email['date']) if email.get('date') else None
                        db_email.content = email['content']
                        db_email.embedding = json.dumps(embedding.tolist())
                    else:
                        db_email = Email(
                            id=email['id'],
                            subject=email['subject'],
                            from_address=email['from'],
                            date=datetime.fromisoformat(email['date']) if email.get('date') else None,
                            content=email['content'],
                            embedding=json.dumps(embedding.tolist())
                        )
                        self.db.add(db_email)

                except Exception as e:
                    logger.error(f"Error processing email {email.get('id', 'unknown')}: {str(e)}")
                    continue

            self.db.commit()
            self.initialize_faiss()  # Reinitialize FAISS index
            logger.info(f"Updated embeddings for {len(emails)} emails")
        except Exception as e:
            logger.error(f"Error updating embeddings: {str(e)}")
            self.db.rollback()
            raise

    def search_emails(self, query, use_faiss=True, top_k=5, similarity_threshold=0.3):
        """Search emails using either FAISS or cosine similarity"""
        try:
            if not self.model:
                self.model = self._load_model()

            # Generate query embedding
            query_embedding = self.model.encode(query, show_progress_bar=False)

            similar_emails = []
            if use_faiss and self.faiss_index:
                # FAISS search
                distances, indices = self.faiss_index.search(
                    np.array([query_embedding]), 
                    top_k
                )
                
                # Get emails from database
                all_emails = self.db.query(Email).all()
                for idx, distance in zip(indices[0], distances[0]):
                    if idx != -1 and idx < len(all_emails):  # FAISS returns -1 for empty slots
                        email = all_emails[idx]
                        similarity = 1 / (1 + distance)  # Convert distance to similarity
                        if similarity >= similarity_threshold:
                            email_dict = email.to_dict()
                            email_dict['similarity_score'] = float(similarity)
                            similar_emails.append(email_dict)
            else:
                # Cosine similarity search
                emails = self.db.query(Email).all()
                email_embeddings = []
                for email in emails:
                    if email.embedding:
                        email_embeddings.append(np.array(json.loads(email.embedding)))
                
                if email_embeddings:
                    similarities = cosine_similarity([query_embedding], email_embeddings)[0]
                    top_indices = np.argsort(similarities)[::-1][:top_k]
                    
                    for idx in top_indices:
                        similarity = similarities[idx]
                        if similarity >= similarity_threshold:
                            email = emails[idx]
                            email_dict = email.to_dict()
                            email_dict['similarity_score'] = float(similarity)
                            similar_emails.append(email_dict)

            logger.info(f"Found {len(similar_emails)} similar emails")
            return similar_emails

        except Exception as e:
            logger.error(f"Error searching emails: {str(e)}")
            raise 