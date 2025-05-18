# Gmail RAG Application

A Retrieval-Augmented Generation (RAG) application that allows users to search and summarize their Gmail contents using semantic search.

## Features

- Custom date range selection for email search
- Semantic search using natural language queries
- Email content summarization
- Secure Gmail OAuth2 authentication
- Efficient vector search using FAISS

## Setup

1. Clone this repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up Google OAuth2 credentials:
   - Go to Google Cloud Console
   - Create a new project
   - Enable Gmail API
   - Create OAuth2 credentials
   - Download the credentials and save as `credentials.json` in the project root

4. Create a `.env` file with your configuration:
   ```
   FLASK_SECRET_KEY=your_secret_key
   GOOGLE_CLIENT_ID=your_client_id
   GOOGLE_CLIENT_SECRET=your_client_secret
   ```

5. Run the application:
   ```bash
   python app.py
   ```

## Usage

1. Access the application at `http://localhost:5000`
2. Authenticate with your Gmail account
3. Select date range and enter your search query
4. View summarized results of matching emails

## Project Structure

- `app.py`: Main Flask application
- `gmail_service.py`: Gmail API integration
- `embedding_service.py`: Text embedding and FAISS operations
- `summarization_service.py`: Email summarization logic
- `static/`: Frontend assets
- `templates/`: HTML templates 