from flask import Flask, render_template, send_from_directory
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app)

@app.route('/')
def index():
    return render_template('index.html')

# Route to serve images
@app.route('/images/<path:filename>')
def images(filename):
    return send_from_directory(os.path.join(app.root_path, 'images'), filename)

if __name__ == "__main__":
    app.run(debug=True,host="0.0.0.0", port=1702)

