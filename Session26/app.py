import logging
import os
import secrets
from datetime import datetime
from flask import Flask, render_template, request, jsonify, session, redirect, url_for, flash
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import Flow
from dotenv import load_dotenv

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
        secret_key = secrets.token_hex(32)
        if not os.path.exists('.env'):
            with open('.env', 'w') as f:
                f.write(f'FLASK_SECRET_KEY={secret_key}\n')
        logger.info("Generated new Flask secret key")
    return secret_key


app.secret_key = get_or_create_secret_key()

# Initialize services
gmail_service = GmailService()
embedding_service = EmbeddingService()
summarization_service = SummarizationService()

# OAuth2 configuration
SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']
CLIENT_SECRETS_FILE = "credentials.json"

# Get Google credentials and update .env
client_id, client_secret = get_google_credentials()
if not client_id or not client_secret:
    logger.warning("Warning: Google credentials not found. Please ensure credentials.json is present.")


def credentials_to_dict(credentials):
    """Convert credentials object to dictionary"""
    return {
        'token': credentials.token,
        'refresh_token': credentials.refresh_token,
        'token_uri': credentials.token_uri,
        'client_id': credentials.client_id,
        'client_secret': credentials.client_secret,
        'scopes': credentials.scopes
    }


def is_authenticated():
    """Check if user is authenticated"""
    if 'credentials' not in session:
        return False

    credentials = Credentials(**session['credentials'])
    if credentials.expired and credentials.refresh_token:
        try:
            credentials.refresh(Request())
            session['credentials'] = credentials_to_dict(credentials)
            return True
        except Exception as e:
            logger.error(f"Error refreshing credentials: {str(e)}")
            return False
    return True


@app.route('/')
def index():
    if not is_authenticated():
        logger.info("User not authenticated, redirecting to authorization")
        return redirect(url_for('authorize'))
    return render_template('index.html')


@app.route('/authorize')
def authorize():
    if is_authenticated():
        return redirect(url_for('index'))

    logger.info("Starting OAuth2 authorization flow")
    flow = Flow.from_client_secrets_file(
        CLIENT_SECRETS_FILE,
        scopes=SCOPES,
        redirect_uri=url_for('oauth2callback', _external=True)
    )

    authorization_url, state = flow.authorization_url(
        access_type='offline',
        include_granted_scopes='true',
        prompt='consent'  # Force consent screen to ensure we get refresh token
    )

    # Store only the state in session
    session['state'] = state
    # Store the redirect URI for callback
    session['redirect_uri'] = url_for('oauth2callback', _external=True)

    return redirect(authorization_url)


@app.route('/oauth2callback')
def oauth2callback():
    if 'state' not in session:
        logger.error("No state found in session")
        return redirect(url_for('authorize'))

    try:
        # Recreate the flow object
        flow = Flow.from_client_secrets_file(
            CLIENT_SECRETS_FILE,
            scopes=SCOPES,
            state=session['state'],
            redirect_uri=session['redirect_uri']
        )

        authorization_response = request.url
        flow.fetch_token(authorization_response=authorization_response)
        credentials = flow.credentials

        # Store credentials in session
        session['credentials'] = credentials_to_dict(credentials)
        logger.info("OAuth2 authentication completed successfully")

        # Clean up session
        session.pop('state', None)
        session.pop('redirect_uri', None)

        return redirect(url_for('index'))
    except Exception as e:
        logger.error(f"Error in OAuth2 callback: {str(e)}")
        flash('Authentication failed. Please try again.', 'error')
        return redirect(url_for('authorize'))


@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('index'))


@app.route('/check-auth')
def check_auth():
    """Check if user is authenticated"""
    return jsonify({'authenticated': is_authenticated()})


@app.route('/search', methods=['POST'])
def search():
    """Search emails endpoint"""
    try:
        if not is_authenticated():
            logger.warning("Unauthenticated search attempt")
            return jsonify({'error': 'Please authenticate first'}), 401

        data = request.get_json()
        query = data.get('query')
        date_range = data.get('dateRange', {})
        should_summarize = data.get('summarize', False)
        use_faiss = data.get('useFaiss', True)
        top_k = data.get('topK', 5)
        fetch_new_emails = data.get('fetchNewEmails', False)
        similarity_threshold = 0.3  # Hardcoded threshold

        if not query:
            return jsonify({'error': 'Query is required'}), 400

        # Get credentials from session
        credentials = Credentials(**session['credentials'])

        # Check if we need to update the database
        last_update = session.get('last_email_update')
        should_update = fetch_new_emails or (not last_update or (datetime.now() - datetime.fromisoformat(last_update)).days >= 1)
        
        if should_update:
            try:
                logger.info("Fetching new emails from Gmail")
                # Fetch all emails (no date range for initial fetch)
                emails = gmail_service.get_emails(credentials=credentials)
                if emails:
                    logger.info(f"Updating database with {len(emails)} emails")
                    embedding_service.update_embeddings(emails)
                    session['last_email_update'] = datetime.now().isoformat()
                    logger.info("Database update completed")
            except Exception as e:
                logger.error(f"Error updating email database: {str(e)}")
                if fetch_new_emails:
                    return jsonify({'error': 'Failed to fetch new emails. Please try again.'}), 500
                # Continue with existing database if update fails and fetch_new_emails is False

        # Search for similar emails
        try:
            logger.info(f"Searching emails with {'FAISS' if use_faiss else 'cosine similarity'}")
            similar_emails = embedding_service.search_emails(
                query=query,
                use_faiss=use_faiss,
                top_k=top_k,
                similarity_threshold=similarity_threshold
            )
            if not similar_emails:
                return jsonify({'message': 'No similar emails found'})
            logger.info(f"Found {len(similar_emails)} similar emails")
        except Exception as e:
            logger.error(f"Error searching emails: {str(e)}")
            return jsonify({'error': 'Failed to search emails. Please try again.'}), 500

        # Only summarize if requested
        if should_summarize:
            try:
                logger.info("Generating summaries for similar emails")
                similar_emails = summarization_service.summarize_emails(similar_emails)
                logger.info("Summarization completed")
            except Exception as e:
                logger.error(f"Error in summarization: {str(e)}")
                # Continue without summaries if summarization fails
                pass

        return jsonify({'results': similar_emails})

    except Exception as e:
        logger.error(f"Error in search: {str(e)}")
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    logger.info("Starting Flask application")
    # For development only - remove in production
    os.environ['OAUTHLIB_INSECURE_TRANSPORT'] = '1'

    # Run the app without SSL in development
    app.run(host='127.0.0.1', port=5000, debug=True)
