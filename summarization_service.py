import logging
from transformers import pipeline
from utils import save_to_cache, load_from_cache

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SummarizationService:
    def __init__(self):
        # Try to load cached model
        self.summarizer = load_from_cache('summarization', 'model')
        if not self.summarizer:
            logger.info("Loading new summarization model...")
            self.summarizer = pipeline(
                "summarization",
                model="facebook/bart-large-cnn",
                device=-1  # Use CPU
            )
            save_to_cache(self.summarizer, 'summarization', 'model')
            logger.info("Summarization model loaded successfully")

    def _prepare_text_for_summarization(self, text, max_words=1024):
        """Prepare text for summarization by truncating and cleaning."""
        # Split into words and truncate
        words = text.split()
        if len(words) > max_words:
            text = ' '.join(words[:max_words])
        
        # Ensure text is not empty
        if not text.strip():
            return "No content available for summarization."
        
        return text

    def summarize_emails(self, emails, max_length=150, min_length=50):
        """Summarize a list of emails."""
        summaries = []
        
        for email in emails:
            email_id = email.get('id', 'unknown')
            logger.info(f"Processing email {email_id}")
            
            # Check cache for existing summary
            cache_key = f"summary_{email_id}"
            cached_summary = load_from_cache('summarization', cache_key)
            if cached_summary:
                logger.info(f"Using cached summary for email {email_id}")
                summaries.append(cached_summary)
                continue

            try:
                # Prepare content for summarization
                content = email.get('content', '')
                subject = email.get('subject', 'No Subject')
                
                # Combine subject and content for better context
                full_text = f"Subject: {subject}\n\n{content}"
                
                # Prepare text for summarization
                prepared_text = self._prepare_text_for_summarization(full_text)
                
                # Generate summary
                logger.info(f"Generating summary for email {email_id}")
                summary_result = self.summarizer(
                    prepared_text,
                    max_length=max_length,
                    min_length=min_length,
                    do_sample=False
                )
                
                # Extract summary text safely
                summary_text = summary_result[0]['summary_text'] if summary_result else "Unable to generate summary"
                
                # Create detailed summary object
                email_summary = {
                    'id': email_id,
                    'subject': subject,
                    'sender': email.get('sender', 'Unknown Sender'),
                    'date': email.get('date', 'Unknown Date'),
                    'summary': summary_text,
                    'body_preview': content[:500] + '...' if len(content) > 500 else content,
                    'full_body': content
                }
                
                # Save to cache
                logger.info(f"Caching summary for email {email_id}")
                save_to_cache(email_summary, 'summarization', cache_key)
                summaries.append(email_summary)
                
            except Exception as e:
                logger.error(f"Error summarizing email {email_id}: {str(e)}")
                # Add error information to the summary
                error_summary = {
                    'id': email_id,
                    'subject': email.get('subject', 'No Subject'),
                    'sender': email.get('sender', 'Unknown Sender'),
                    'date': email.get('date', 'Unknown Date'),
                    'summary': 'Error generating summary',
                    'error': str(e)
                }
                summaries.append(error_summary)
                continue
        
        logger.info(f"Completed summarization of {len(summaries)} emails")
        return summaries 