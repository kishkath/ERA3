import requests
from flask import Flask, request, jsonify

app = Flask(__name__)

# URL of App1 (LLM server)
LLM_SERVER_URL = "http://app1:5000/generate"

@app.route("/send_prompt", methods=["POST"])
def send_prompt():
    """
    API Endpoint to send a prompt to App1.
    Expected JSON input: {"prompt": "Your prompt here", "max_length": 100, "temperature": 1.0}
    """
    try:
        data = request.get_json()
        response = requests.post(LLM_SERVER_URL, json=data)
        return jsonify(response.json())

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=6000, debug=True)
