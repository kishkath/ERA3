import json
import os
import pickle
from dotenv import load_dotenv
from datetime import datetime


def get_google_credentials():
    """
    Read Google OAuth2 credentials from credentials.json and update .env file
    """
    try:
        # Read the credentials.json file
        with open('credentials.json', 'r') as f:
            credentials = json.load(f)

        # Print the structure of credentials for debugging
        print("Credentials structure:", json.dumps(credentials, indent=2))

        # Extract client ID and client secret
        web_credentials = credentials.get('web', {})
        if not web_credentials:
            print("Error: 'web' section not found in credentials.json")
            print("Please ensure you've downloaded the correct OAuth 2.0 Client ID for a web application")
            return None, None

        client_id = web_credentials.get('client_id')
        client_secret = web_credentials.get('client_secret')

        if not client_id:
            print("Error: Client ID not found in credentials.json")
            print("Please ensure you've downloaded the correct OAuth 2.0 Client ID")
            return None, None

        if not client_secret:
            print("Error: Client Secret not found in credentials.json")
            print("Please follow these steps to get the client secret:")
            print("1. Go to Google Cloud Console (https://console.cloud.google.com)")
            print("2. Select your project")
            print("3. Go to 'APIs & Services' > 'Credentials'")
            print("4. Find your OAuth 2.0 Client ID")
            print("5. Click on the client ID to view details")
            print("6. Look for 'Client secret' section")
            print("7. If not visible, create a new OAuth 2.0 Client ID:")
            print("   - Click 'Create Credentials' > 'OAuth client ID'")
            print("   - Choose 'Web application'")
            print("   - Add redirect URI: http://localhost:5000/oauth2callback")
            print("   - Click 'Create'")
            print("   - Download the new credentials file")
            return None, None

        # Update .env file
        env_vars = {}
        if os.path.exists('.env'):
            with open('.env', 'r') as f:
                for line in f:
                    if '=' in line:
                        key, value = line.strip().split('=', 1)
                        env_vars[key] = value

        # Update or add Google credentials
        env_vars['GOOGLE_CLIENT_ID'] = client_id
        env_vars['GOOGLE_CLIENT_SECRET'] = client_secret

        # Write back to .env file
        with open('.env', 'w') as f:
            for key, value in env_vars.items():
                f.write(f'{key}={value}\n')

        print("Successfully updated .env file with Google credentials")
        return client_id, client_secret

    except FileNotFoundError:
        print("Error: credentials.json file not found")
        print("Please ensure you've downloaded the credentials file from Google Cloud Console")
        return None, None
    except json.JSONDecodeError:
        print("Error: Invalid JSON in credentials.json file")
        print("Please ensure you've downloaded a valid credentials file")
        return None, None
    except Exception as e:
        print(f"Error: {str(e)}")
        return None, None


# Cache management functions
def ensure_cache_dir():
    """Ensure the cache directory exists"""
    cache_dir = os.path.join(os.path.dirname(__file__), 'cache')
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    return cache_dir

def get_cache_path(service_name, cache_type):
    """Get the path for a specific cache file"""
    cache_dir = ensure_cache_dir()
    return os.path.join(cache_dir, f'{service_name}_{cache_type}.pkl')

def save_to_cache(data, service_name, cache_type):
    """Save data to cache"""
    cache_path = get_cache_path(service_name, cache_type)
    with open(cache_path, 'wb') as f:
        pickle.dump({
            'timestamp': datetime.now(),
            'data': data
        }, f)

def load_from_cache(service_name, cache_type, max_age_hours=24):
    """Load data from cache if it exists and is not too old"""
    cache_path = get_cache_path(service_name, cache_type)
    if not os.path.exists(cache_path):
        return None
    
    try:
        with open(cache_path, 'rb') as f:
            cached_data = pickle.load(f)
        
        # Check if cache is too old
        cache_age = datetime.now() - cached_data['timestamp']
        if cache_age.total_seconds() > max_age_hours * 3600:
            return None
        
        return cached_data['data']
    except Exception as e:
        print(f"Error loading cache for {service_name}_{cache_type}: {str(e)}")
        return None


if __name__ == "__main__":
    # When run directly, update the .env file
    get_google_credentials()
