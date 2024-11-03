from flask import Flask, render_template, request, jsonify
from flask_cors import CORS

from data_pipeline.pre_processor import text_processing

app = Flask(__name__)
CORS(app)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/media/processing", methods=["POST"])
def pre_processing():
    input_request = request.json
    text = input_request.get("text")
    tokens = text_processing(text)
    return jsonify({"tokens": tokens})


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=1702)
