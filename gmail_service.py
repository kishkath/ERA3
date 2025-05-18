import base64
import email
import logging
from email.mime.text import MIMEText
from datetime import datetime, timedelta
import httplib2
from googleapiclient.discovery import build
from google.oauth2.credentials import Credentials
import ssl
import re
from utils import save_to_cache, load_from_cache

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GmailService:
    def __init__(self):
        self.service = None

    def build_service(self, credentials):
        """Build Gmail API service with proper SSL configuration"""
        try:
            # Build the service with credentials only
            self.service = build('gmail', 'v1', credentials=credentials)
            return self.service
        except Exception as e:
            logger.error(f"Error building Gmail service: {str(e)}")
            raise

    def get_emails(self, credentials, date_range):
        """Get emails from Gmail within the specified date range"""
        try:
            # Build the service if not already built
            if not self.service:
                self.build_service(credentials)

            # Calculate date range
            end_date = datetime.now()
            if date_range['type'] == 'custom':
                try:
                    start_date = datetime.strptime(date_range['start'], '%Y-%m-%d')
                    end_date = datetime.strptime(date_range['end'], '%Y-%m-%d')
                    logger.info(f"Using custom date range: {start_date} to {end_date}")
                except KeyError as e:
                    logger.error(f"Missing date key in date_range: {str(e)}")
                    raise ValueError(f"Invalid date range format: {date_range}")
                except ValueError as e:
                    logger.error(f"Invalid date format: {str(e)}")
                    raise ValueError(f"Invalid date format in date_range: {date_range}")
            elif date_range['type'] == 'month':
                start_date = end_date.replace(day=1)
                logger.info(f"Using month date range: {start_date} to {end_date}")
            elif date_range['type'] == 'year':
                start_date = end_date.replace(month=1, day=1)
                logger.info(f"Using year date range: {start_date} to {end_date}")
            else:
                logger.error(f"Invalid date range type: {date_range['type']}")
                raise ValueError(f"Invalid date range type: {date_range['type']}")

            # Check cache for emails in this date range
            cache_key = f"emails_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}"
            cached_emails = load_from_cache('gmail', cache_key)
            if cached_emails:
                logger.info(f"Using cached emails for date range {start_date} to {end_date}")
                return cached_emails

            # Format dates for Gmail query
            start_date_str = start_date.strftime('%Y/%m/%d')
            end_date_str = (end_date + timedelta(days=1)).strftime('%Y/%m/%d')
            
            # Construct the query
            query = f'after:{start_date_str} before:{end_date_str}'
            logger.info(f"Gmail query: {query}")

            # Get messages
            results = self.service.users().messages().list(
                userId='me',
                q=query,
                maxResults=100  # Limit to 100 messages for performance
            ).execute()

            messages = results.get('messages', [])
            if not messages:
                logger.info("No messages found in the specified date range")
                return []

            # Get full message details
            emails = []
            for message in messages:
                try:
                    # Check cache for individual message
                    msg_cache_key = f"message_{message['id']}"
                    cached_msg = load_from_cache('gmail', msg_cache_key)
                    if cached_msg:
                        emails.append(cached_msg)
                        continue

                    msg = self.service.users().messages().get(
                        userId='me',
                        id=message['id'],
                        format='full'
                    ).execute()

                    # Extract email data
                    headers = msg['payload']['headers']
                    subject = next((h['value'] for h in headers if h['name'].lower() == 'subject'), 'No Subject')
                    sender = next((h['value'] for h in headers if h['name'].lower() == 'from'), 'Unknown Sender')
                    date = next((h['value'] for h in headers if h['name'].lower() == 'date'), 'Unknown Date')

                    # Get email body
                    if 'parts' in msg['payload']:
                        parts = msg['payload']['parts']
                        body = ''
                        for part in parts:
                            if part['mimeType'] == 'text/plain':
                                body = base64.urlsafe_b64decode(part['body']['data']).decode()
                                break
                    else:
                        body = base64.urlsafe_b64decode(msg['payload']['body']['data']).decode()

                    email_data = {
                        'id': message['id'],
                        'subject': subject,
                        'sender': sender,
                        'date': date,
                        'content': f"Subject: {subject}\nFrom: {sender}\nDate: {date}\n\n{body}"
                    }

                    # Save individual message to cache
                    save_to_cache(email_data, 'gmail', msg_cache_key)
                    emails.append(email_data)

                except Exception as e:
                    logger.error(f"Error processing message {message['id']}: {str(e)}")
                    continue

            # Save all emails to cache
            save_to_cache(emails, 'gmail', cache_key)
            logger.info(f"Retrieved {len(emails)} emails from Gmail")
            return emails

        except Exception as e:
            logger.error(f"Error fetching emails: {str(e)}")
            raise

    def send_email(self, to, subject, message_text):
        """Send an email using Gmail API"""
        try:
            if not self.service:
                raise ValueError("Gmail service not initialized")

            message = MIMEText(message_text)
            message['to'] = to
            message['subject'] = subject

            raw = base64.urlsafe_b64encode(message.as_bytes()).decode()
            self.service.users().messages().send(
                userId='me',
                body={'raw': raw}
            ).execute()
            return True
        except Exception as e:
            print(f"Error sending email: {str(e)}")
            return False

    def _construct_date_query(self, date_range):
        """Construct Gmail search query for date range."""
        if date_range['type'] == 'custom':
            start_date = datetime.strptime(date_range['start'], '%Y-%m-%d')
            end_date = datetime.strptime(date_range['end'], '%Y-%m-%d')
            return f'after:{start_date.strftime("%Y/%m/%d")} before:{end_date.strftime("%Y/%m/%d")}'
        elif date_range['type'] == 'month':
            return f'after:{date_range["year"]}/{date_range["month"]}/1 before:{date_range["year"]}/{date_range["month"]}/31'
        elif date_range['type'] == 'year':
            return f'after:{date_range["year"]}/1/1 before:{date_range["year"]}/12/31'
        return ''

    def _extract_email_data(self, message):
        """Extract relevant data from email message."""
        headers = message['payload']['headers']
        subject = next((h['value'] for h in headers if h['name'] == 'Subject'), '')
        from_header = next((h['value'] for h in headers if h['name'] == 'From'), '')
        date = next((h['value'] for h in headers if h['name'] == 'Date'), '')
        
        # Extract email body
        body = self._get_email_body(message['payload'])
        if not body:
            return None
            
        return {
            'id': message['id'],
            'subject': subject,
            'from': from_header,
            'date': date,
            'content': f"Subject: {subject}\nFrom: {from_header}\nDate: {date}\n\n{body}"
        }

    def _get_email_body(self, payload):
        """Extract email body from message payload."""
        if 'parts' in payload:
            for part in payload['parts']:
                if part['mimeType'] == 'text/plain':
                    return base64.urlsafe_b64decode(part['body']['data']).decode()
                elif part['mimeType'] == 'text/html':
                    # Remove HTML tags for plain text
                    html_content = base64.urlsafe_b64decode(part['body']['data']).decode()
                    return re.sub('<[^<]+?>', '', html_content)
        
        # If no parts, try to get body directly
        if 'body' in payload and 'data' in payload['body']:
            return base64.urlsafe_b64decode(payload['body']['data']).decode()
        
        return '' 