import torch
import torch.nn.functional as F
from flask import Flask, request, jsonify
from transformers import AutoTokenizer
import pytorch_lightning as pl
from model import SMOLLLightningModule  # Ensure the model class is in model.py

# Initialize Flask App
app = Flask(__name__)

# Load Model Checkpoint
checkpoint_path = "pl_smoll_checkpoint.ckpt"
model = SMOLLLightningModule.load_from_checkpoint(checkpoint_path)
model.eval()
model.freeze()
MODEL_NAME = "HuggingFaceTB/SmolLM2-135M-Instruct"
# Load Tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)  # Replace with actual model name

@app.route("/generate", methods=["POST"])
def generate_text():
    """
    API Endpoint to generate text.
    Expected JSON input: {"prompt": "Your prompt here", "max_length": 100, "temperature": 1.0}
    """
    try:
        data = request.get_json()
        prompt = data.get("prompt", "Once upon a time")
        max_length = int(data.get("max_length", 100))
        temperature = float(data.get("temperature", 1.0))

        # Tokenize Input
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)

        # Generate Text
        with torch.no_grad():
            for _ in range(max_length):
                logits, _ = model(input_ids)
                next_token_logits = logits[:, -1, :] / temperature
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                input_ids = torch.cat([input_ids, next_token], dim=1)

        generated_text = tokenizer.decode(input_ids[0].tolist(), skip_special_tokens=True)

        return jsonify({"generated_text": generated_text})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
