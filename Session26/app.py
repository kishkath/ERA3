import logging
import os
import secrets
from datetime import datetime

from dotenv import load_dotenv
from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import Flow

from embedding_service import EmbeddingService
from gmail_service import GmailService
from summarization_service import SummarizationService
from utils import get_google_credentials

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

app = Flask(__name__)


# Get or generate secret key
def get_or_create_secret_key():
    secret_key = os.getenv('FLASK_SECRET_KEY')
    if not secret_key:
        # Generate a new secret key
        secret_key = secrets.token_hex(32)
        # Save it to .env file if it doesn't exist
        if not os.path.exists('.env'):
            with open('.env', 'w') as f:
                f.write(f'FLASK_SECRET_KEY={secret_key}\n')
        logger.info("Generated new Flask secret key")
    return secret_key


logger.info("Initializing application...")
app.secret_key = get_or_create_secret_key()

# Initialize services
logger.info("Initializing services...")
gmail_service = GmailService()
logger.info("GMAIL SERVICE IS DONE")
embedding_service = EmbeddingService()
logger.info("EMBEDDINGS ARE DONE")
summarization_service = SummarizationService()
logger.info("SUMMARINZATION IS DONE")

# OAuth2 configuration
SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']
CLIENT_SECRETS_FILE = "credentials.json"

# Get Google credentials and update .env
logger.info("GETTING GOOGLE CREDENTIALS")
client_id, client_secret = get_google_credentials()
if not client_id or not client_secret:
    logger.warning("Warning: Google credentials not found. Please ensure credentials.json is present.")


@app.route('/')
def index():
    if 'credentials' not in session:
        logger.info("User not authenticated, redirecting to authorization")
        return redirect(url_for('authorize'))
    return render_template('index.html')


@app.route('/authorize')
def authorize():
    logger.info("Starting OAuth2 authorization flow")
    flow = Flow.from_client_secrets_file(
        CLIENT_SECRETS_FILE,
        scopes=SCOPES,
        redirect_uri=url_for('oauth2callback', _external=True)
    )
    authorization_url, state = flow.authorization_url(
        access_type='offline',
        include_granted_scopes='true'
    )
    session['state'] = state
    return redirect(authorization_url)


@app.route('/oauth2callback')
def oauth2callback():
    logger.info("Processing OAuth2 callback")
    state = session['state']
    flow = Flow.from_client_secrets_file(
        CLIENT_SECRETS_FILE,
        scopes=SCOPES,
        state=state,
        redirect_uri=url_for('oauth2callback', _external=True)
    )

    authorization_response = request.url
    flow.fetch_token(authorization_response=authorization_response)
    credentials = flow.credentials
    session['credentials'] = {
        'token': credentials.token,
        'refresh_token': credentials.refresh_token,
        'token_uri': credentials.token_uri,
        'client_id': credentials.client_id,
        'client_secret': credentials.client_secret,
        'scopes': credentials.scopes
    }
    logger.info("OAuth2 authentication completed successfully")
    return redirect(url_for('index'))


@app.route('/search', methods=['POST'])
def search():
    try:
        if 'credentials' not in session:
            logger.warning("Unauthenticated search attempt")
            return jsonify({'error': 'Not authenticated'}), 401

        data = request.json
        query = data.get('query')
        date_range = data.get('dateRange')

        if not query or not date_range:
            logger.warning("Missing query or date range in search request")
            return jsonify({'error': 'Missing query or date range'}), 400

        logger.info(f"Processing search request: {query} for date range: {date_range}")

        # Convert credentials from session to Credentials object
        credentials = Credentials(**session['credentials'])

        # Check if credentials need refresh
        if credentials.expired and credentials.refresh_token:
            logger.info("Refreshing expired credentials")
            credentials.refresh(Request())

        # Get emails from Gmail
        try:
            logger.info("Fetching emails from Gmail")
            emails = gmail_service.get_emails(credentials, date_range)
            logger.info(f"Retrieved {len(emails)} emails")
        except Exception as e:
            logger.error(f"Error fetching emails: {str(e)}")
            return jsonify({'error': 'Failed to fetch emails. Please try again.'}), 500

        if not emails:
            logger.info("No emails found in specified date range")
            return jsonify({'message': 'No emails found in the specified date range'})

        # Generate embeddings for emails and query
        try:
            logger.info("Generating embeddings for emails and query")
            email_embeddings = embedding_service.generate_embeddings([email['content'] for email in emails])
            query_embedding = embedding_service.generate_embeddings([query])[0]
        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            return jsonify({'error': 'Failed to process emails. Please try again.'}), 500

        # Perform similarity search (limited to top 2 matches)
        try:
            logger.info("Finding similar emails")
            similar_emails = embedding_service.find_similar_emails(query_embedding, email_embeddings, emails)
            logger.info(f"Found {len(similar_emails)} similar emails")
        except Exception as e:
            logger.error(f"Error finding similar emails: {str(e)}")
            return jsonify({'error': 'Failed to find relevant emails. Please try again.'}), 500

        if not similar_emails:
            logger.info("No relevant emails found for query")
            return jsonify({'message': 'No relevant emails found for your query'})

        # Summarize the results
        try:
            logger.info("Summarizing emails")
            summaries = summarization_service.summarize_emails(similar_emails)
            logger.info("Email summarization completed")
        except Exception as e:
            logger.error(f"Error summarizing emails: {str(e)}")
            return jsonify({'error': 'Failed to summarize emails. Please try again.'}), 500

        return jsonify({
            'results': summaries,
            'timestamp': datetime.now().isoformat()
        })

    except Exception as e:
        logger.error(f"Unexpected error in search endpoint: {str(e)}")
        return jsonify({'error': 'An unexpected error occurred. Please try again.'}), 500


if __name__ == '__main__':
    logger.info("Starting Flask application")
    # For development only - remove in production
    os.environ['OAUTHLIB_INSECURE_TRANSPORT'] = '1'

    # Run the app without SSL in development
    app.run(host='127.0.0.1', port=5000, debug=True)
