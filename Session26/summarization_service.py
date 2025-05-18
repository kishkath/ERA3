import logging
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from utils import save_to_cache, load_from_cache, ensure_cache_dir
import hashlib
import os
import torch
from concurrent.futures import ThreadPoolExecutor
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

torch.set_num_threads(4)  # Limit PyTorch threads


class SummarizationService:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.cache_dir = ensure_cache_dir()
        self.summarizer = None
        self.max_workers = min(4, os.cpu_count() or 1)  # Limit threads for CPU
        self.chunk_size = 512  # Smaller chunks for faster processing

    def _load_model(self):
        """Load the summarization model"""
        try:
            # Try to load from cache first
            cached_model = load_from_cache('summarization', 'model')
            cached_tokenizer = load_from_cache('summarization', 'tokenizer')

            if cached_model and cached_tokenizer:
                logger.info("Loaded summarization model from cache")
                return cached_model, cached_tokenizer

            # Load smaller model for CPU
            logger.info("Loading summarization model...")
            model_name = "facebook/bart-base"  # Smaller than bart-large-cnn
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

            # Save to cache
            save_to_cache(tokenizer, 'summarization', 'tokenizer')
            save_to_cache(model, 'summarization', 'model')
            logger.info("Loaded and cached new summarization model")
            return model, tokenizer
        except Exception as e:
            logger.error(f"Error loading summarization model: {str(e)}")
            raise

    def _get_cache_path(self, email_id):
        """Get the cache path for an email summary"""
        return os.path.join(self.cache_dir, f'summary_{hashlib.md5(email_id.encode()).hexdigest()}.txt')

    def _summarize_chunk(self, chunk):
        """Summarize a single chunk of text"""
        try:
            if len(chunk.strip()) < 50:
                return None

            # Tokenize
            inputs = self.tokenizer(chunk, max_length=self.chunk_size, truncation=True, return_tensors="pt")

            # Generate summary
            summary_ids = self.model.generate(
                inputs["input_ids"],
                max_length=100,
                min_length=30,
                num_beams=2,  # Reduced for speed
                early_stopping=True
            )

            return self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        except Exception as e:
            logger.error(f"Error summarizing chunk: {str(e)}")
            return None

    def summarize_emails(self, emails):
        """Summarize a list of emails using parallel processing"""
        try:
            if not self.model or not self.tokenizer:
                self.model, self.tokenizer = self._load_model()

            summarized_emails = []
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                for email in emails:
                    try:
                        email_id = email.get('id', '')
                        logger.info(f"Processing email {email_id}")

                        # Check cache first
                        cache_path = self._get_cache_path(email_id)
                        if os.path.exists(cache_path):
                            with open(cache_path, 'r', encoding='utf-8') as f:
                                summary = f.read().strip()
                            logger.info(f"Using cached summary for email {email_id}")
                            email['summary'] = summary
                            summarized_emails.append(email)
                            continue

                        # Generate new summary
                        logger.info(f"Generating summary for email {email_id}")
                        content = email.get('content', '').strip()

                        if not content:
                            logger.warning(f"Empty content for email {email_id}")
                            email['summary'] = "No content available to summarize."
                            continue

                        # Split content into smaller chunks
                        chunks = [content[i:i + self.chunk_size] for i in range(0, len(content), self.chunk_size)]

                        # Process chunks in parallel
                        chunk_summaries = list(filter(None, executor.map(self._summarize_chunk, chunks)))

                        if chunk_summaries:
                            final_summary = " ".join(chunk_summaries)
                            # Cache the summary
                            with open(cache_path, 'w', encoding='utf-8') as f:
                                f.write(final_summary)
                            logger.info(f"Caching summary for email {email_id}")
                            email['summary'] = final_summary
                        else:
                            email['summary'] = "Unable to generate summary for this email."
                            logger.warning(f"Could not generate summary for email {email_id}")

                        summarized_emails.append(email)

                    except Exception as e:
                        logger.error(f"Error in summarization for email {email_id}: {str(e)}")
                        email['summary'] = "Error generating summary."
                        summarized_emails.append(email)
                        continue

            return summarized_emails

        except Exception as e:
            logger.error(f"Error in summarize_emails: {str(e)}")
            raise

    def summarize_email(self, content):
        """Summarize a single email content"""
        try:
            if not self.model or not self.tokenizer:
                self.model, self.tokenizer = self._load_model()

            if not content or len(content.strip()) < 50:
                return "Content too short to summarize."

            # Split content into smaller chunks
            chunks = [content[i:i + self.chunk_size] for i in range(0, len(content), self.chunk_size)]

            # Process chunks in parallel
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                chunk_summaries = list(filter(None, executor.map(self._summarize_chunk, chunks)))

            if chunk_summaries:
                return " ".join(chunk_summaries)
            else:
                return "Unable to generate summary for this content."

        except Exception as e:
            logger.error(f"Error in summarize_email: {str(e)}")
            return "Error generating summary."
