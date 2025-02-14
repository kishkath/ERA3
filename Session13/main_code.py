import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
import math
import tiktoken


############################################
# Causal Self-Attention Module
############################################

class CasualSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Linear layer to project input embeddings into queries (Q), keys (K), and values (V)
        # The output size is 3 times the embedding size because we need Q, K, and V
        self.c_attn = nn.Linear(config.n_embed, 3 * config.n_embed)

        # Linear layer to project the attention output back to the original embedding size
        self.c_proj = nn.Linear(config.n_embed, config.n_embed)

        # Number of attention heads and embedding size
        self.n_head = config.n_head
        self.n_embed = config.n_embed

        # Create a lower triangular matrix to mask future tokens (for causal/self-attention)
        # This ensures the model cannot "peek" at future tokens during training
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.block_size, config.block_size))
            .view(1, 1, config.block_size, config.block_size)
        )

    def forward(self, x):
        B, T, C = x.size()  # B = Batch size, T = Sequence length, C = Embedding dimension

        # Project the input 'x' to get combined queries, keys, and values
        qkv = self.c_attn(x)

        # Split the combined projections into separate Q, K, and V tensors
        q, k, v = qkv.split(self.n_embed, dim=2)

        # Reshape Q, K, V to handle multiple attention heads
        # Shape: (Batch, Heads, Sequence Length, Head Dimension)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        # Compute scaled dot-product attention scores
        # (Q × Kᵀ) scaled by the square root of the key dimension to stabilize gradients
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))

        # Apply the causal mask to prevent attention to future tokens
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))

        # Apply softmax to convert attention scores to probabilities
        att = F.softmax(att, dim=-1)

        # Apply attention weights to the values (V)
        y = att @ v

        # Reshape back to the original input format
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # Final linear projection to match the input embedding size
        y = self.c_proj(y)

        return y


############################################
# Multi-Layer Perceptron (MLP)
############################################

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        # First linear layer expands the embedding dimension by 4x (common practice in transformers)
        self.c_fc = nn.Linear(config.n_embed, 4 * config.n_embed)

        # GELU activation introduces non-linearity to help the model learn complex patterns
        self.gelu = nn.GELU(approximate='tanh')  # 'approximate' speeds up computation with minimal loss in accuracy

        # Second linear layer projects the output back to the original embedding size
        self.c_proj = nn.Linear(4 * config.n_embed, config.n_embed)

        # Correcting the initialization (if scaling was intended)
        # Applying constant scaling to weights instead of using a non-existent attribute
        nn.init.constant_(self.c_proj.weight, 1)  # Scales weights to 1 (if needed)

    def forward(self, x):
        # Apply first linear transformation
        x = self.c_fc(x)
        # Apply GELU activation for non-linearity
        x = self.gelu(x)
        # Project back to original embedding dimension
        x = self.c_proj(x)
        return x


############################################
# Transformer Block
############################################

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Layer normalization to stabilize training and improve convergence
        self.ln_1 = nn.LayerNorm(config.n_embed)
        # Causal Self-Attention layer to capture relationships between tokens
        self.attn = CasualSelfAttention(config)
        # Another LayerNorm before feeding into the MLP
        self.ln_2 = nn.LayerNorm(config.n_embed)
        # Feedforward neural network (MLP) to process representations from the attention output
        self.mlp = MLP(config)

    def forward(self, x):
        # First sub-layer: Apply LayerNorm, then attention, followed by a residual connection
        x = x + self.attn(self.ln_1(x))  # Residual connection helps prevent vanishing gradients
        # Second sub-layer: Apply LayerNorm, then MLP, followed by another residual connection
        x = x + self.mlp(self.ln_2(x))  # Residual connection for stability
        return x


############################################
# Configuration Class for GPT
############################################

@dataclass
class GPTConfig:
    # Maximum number of tokens in a sequence (i.e., the context length)
    block_size: int = 1024  # Determines how much context the model can "see" at once
    # Size of the vocabulary (number of unique tokens the model can recognize)
    vocab_size: int = 50257  # Common in models like GPT-2 (includes special tokens)
    # Number of transformer blocks (layers) stacked in the model
    n_layer: int = 12  # More layers = deeper model, capable of learning complex patterns
    # Number of attention heads in the multi-head attention mechanism
    n_head: int = 12  # Allows the model to focus on different parts of the sequence simultaneously
    # Dimensionality of token embeddings and hidden states
    n_embed: int = 768  # Higher embedding size captures richer information but increases computation


############################################
# GPT Model
############################################

class GPT(nn.Module):
    def __init__(self, config):
        """
        Initialize the GPT model.

        Args:
            config: A configuration object containing model hyperparameters, such as:
                - vocab_size: Size of the vocabulary.
                - n_embed: Dimensionality of embeddings.
                - block_size: Maximum sequence length (context length).
                - n_layer: Number of transformer blocks.
                (Other parameters like n_head may be used inside Block.)
        """
        super().__init__()
        self.config = config

        # Build the transformer components in a ModuleDict for organized access.
        self.transformer = nn.ModuleDict({
            # Token embedding: maps token indices (0 to vocab_size-1) to embedding vectors.
            'wte': nn.Embedding(config.vocab_size, config.n_embed),
            # Positional embedding: provides a unique embedding for each position in the input sequence.
            'wpe': nn.Embedding(config.block_size, config.n_embed),
            # A list of transformer blocks (e.g., self-attention + feed-forward layers).
            'h': nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            # Final layer normalization applied to the output of the transformer blocks.
            'ln_f': nn.LayerNorm(config.n_embed),
        })

        # Output layer (language model head) to project the transformer output to logits over the vocabulary.
        self.lm_head = nn.Linear(config.n_embed, config.vocab_size, bias=False)

    def forward(self, idx, targets=None):
        """
        Perform a forward pass through the GPT model.

        Args:
            idx: Input tensor of token indices with shape (B, T), where B is the batch size and T is the sequence length.
            targets: (Optional) Tensor of target token indices with shape (B, T) used for computing the loss.

        Returns:
            logits: Tensor of shape (B, T, vocab_size) containing unnormalized log probabilities for each token.
            loss: Cross-entropy loss computed between logits and targets if targets are provided; otherwise, None.
        """
        # Get the batch size (B) and sequence length (T)
        B, T = idx.size()
        # Ensure the sequence length does not exceed the maximum block size defined in the configuration.
        assert T <= self.config.block_size, (
            f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        )

        # Create a tensor of position indices (0, 1, ..., T-1) on the same device as idx.
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)  # Shape: (T)

        # Obtain positional embeddings for each position in the sequence.
        pos_emb = self.transformer.wpe(pos)  # Shape: (T, n_embed)

        # Obtain token embeddings for the input tokens.
        tok_emb = self.transformer.wte(idx)  # Shape: (B, T, n_embed)

        # Combine token and positional embeddings.
        # The positional embeddings are automatically broadcasted over the batch dimension.
        x = tok_emb + pos_emb

        # Pass the embeddings through each transformer block sequentially.
        for block in self.transformer.h:
            x = block(x)

        # Apply the final layer normalization.
        x = self.transformer.ln_f(x)

        # Compute logits by projecting the transformer output to the size of the vocabulary.
        logits = self.lm_head(x)  # Shape: (B, T, vocab_size)

        loss = None
        if targets is not None:
            # Flatten the logits and targets for computing cross-entropy loss.
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss

    @classmethod
    def from_pretrained(cls, model_type):
        """
        Create a GPT model instance with weights loaded from a pretrained GPT-2 model from Hugging Face.

        Args:
            model_type (str): One of {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}.

        Returns:
            model: A GPT instance with pretrained weights.
        """
        # Validate that the requested model type is supported.
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}

        # Import the pretrained model class from Hugging Face Transformers.
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # Define model hyperparameters based on the GPT-2 variant.
        config_args = {
            'gpt2': dict(n_layer=12, n_head=12, n_embed=768),  # 124M parameters
            'gpt2-medium': dict(n_layer=24, n_head=16, n_embed=1024),  # 350M parameters
            'gpt2-large': dict(n_layer=36, n_head=20, n_embed=1280),  # 774M parameters
            'gpt2-xl': dict(n_layer=48, n_head=25, n_embed=1600),  # 1558M parameters
        }[model_type]
        # Set constant parameters for all GPT-2 models.
        config_args['vocab_size'] = 50257  # Fixed vocabulary size for GPT-2 checkpoints.
        config_args['block_size'] = 1024  # Fixed maximum sequence length for GPT-2 checkpoints.

        # Create a configuration object and initialize a new GPT model.
        config = GPTConfig(**config_args)
        model = GPT(config)

        # Retrieve the state dictionary (all parameters) of the new model.
        sd = model.state_dict()
        # Exclude keys that correspond to buffers (e.g., attention bias) that are not actual parameters.
        sd_keys = [k for k in sd.keys() if not k.endswith('.attn.bias')]

        # Load the Hugging Face GPT-2 model with pretrained weights.
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # Filter out buffer keys from the Hugging Face state dictionary.
        sd_keys_hf = [k for k in sd_hf.keys() if not k.endswith('.attn.masked_bias')]
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')]

        # List of parameter names that require transposition.
        # This is necessary because the original GPT-2 weights use a Conv1D implementation,
        # whereas our implementation uses a standard Linear layer.
        transposed = [
            'attn.c_attn.weight',
            'attn.c_proj.weight',
            'mlp.c_fc.weight',
            'mlp.c_proj.weight'
        ]

        # Ensure the number of parameters matches between our model and the Hugging Face model.
        assert len(sd_keys_hf) == len(sd_keys), (
            f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        )

        # Copy the weights from the Hugging Face model into our model's state dictionary.
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # For these weights, verify that transposing the Hugging Face weight gives the correct shape.
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    # Transpose the weight before copying.
                    sd[k].copy_(sd_hf[k].t())
            else:
                # For all other weights, ensure the shapes match and copy directly.
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model


############################################
# Device Setup
############################################

# Choose device: CUDA if available, otherwise CPU (or MPS for Apple silicon)
device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"
print(f"Using device: {device}")

############################################
# Data Preparation
############################################

# Hyperparameters for training data
B, T = 4, 32  # Batch size and sequence length for the training batch

# Load the tokenizer (using tiktoken for GPT-2)
enc = tiktoken.get_encoding('gpt2')

# Read a sample text file (make sure 'input.txt' exists in your working directory)
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# Optionally trim the text (here we use the first 1000 characters)
text = text[:1000]
# Tokenize the text
tokens = enc.encode(text)

# Ensure there are enough tokens to form a training batch (B * T + 1 tokens)
if len(tokens) < B * T + 1:
    raise ValueError("Not enough tokens in input.txt for one batch.")

# Create a tensor from tokens and move it to the selected device
buf = torch.tensor(tokens[:B * T + 1], dtype=torch.long, device=device)
# x: input tokens, y: target tokens (shifted one position)
x = buf[:-1].view(B, T)
y = buf[1:].view(B, T)

############################################
# Model Initialization and Training
############################################

# Create a GPTConfig instance and initialize the model.
config = GPTConfig(vocab_size=50257, block_size=1024, n_layer=4, n_head=4, n_embed=128)
model = GPT(config)
model.to(device)

# Define an optimizer (e.g., Adam)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# A simple training loop (for demonstration purposes)
model.train()
num_steps = 50  # Adjust the number of training steps as needed
for step in range(num_steps):
    optimizer.zero_grad()
    logits, loss = model(x, targets=y)
    loss.backward()
    optimizer.step()

    if step % 10 == 0:
        print(f"Step {step}, Loss: {loss.item()}")


############################################
# Inference: Text Generation
############################################

def generate(model, prompt, max_length=30, temperature=1.0):
    """
    Generate a sequence of tokens from the model given a prompt.

    Args:
        model (GPT): The trained GPT model.
        prompt (str): The text prompt to start generation.
        max_length (int): The number of new tokens to generate.
        temperature (float): Temperature parameter for sampling diversity.

    Returns:
        generated (list): List of token IDs representing the generated sequence.
    """
    model.eval()
    # Encode the prompt into token IDs and add a batch dimension.
    input_ids = torch.tensor(enc.encode(prompt), dtype=torch.long, device=device).unsqueeze(0)

    with torch.no_grad():
        for _ in range(max_length):
            logits, _ = model(input_ids)
            # Focus on the logits of the last token and apply temperature scaling.
            next_token_logits = logits[:, -1, :] / temperature
            # Convert logits to probabilities.
            probs = F.softmax(next_token_logits, dim=-1)
            # Sample the next token (you can also use greedy sampling with argmax)
            next_token = torch.multinomial(probs, num_samples=1)
            # Append the sampled token to the input_ids.
            input_ids = torch.cat([input_ids, next_token], dim=1)

    # Remove the batch dimension and return the list of token IDs.
    return input_ids[0].tolist()


# Generate several sequences from a given prompt.
num_return_sequences = 5
max_gen_length = 30
prompt = "Once upon a time"

generated_sequences = []
for i in range(num_return_sequences):
    seq_ids = generate(model, prompt, max_length=max_gen_length, temperature=1.0)
    # Decode the token IDs back to text.
    generated_text = enc.decode(seq_ids)
    generated_sequences.append(generated_text)

print("\nGenerated sequences:")
for idx, seq in enumerate(generated_sequences, start=1):
    print(f"\nSequence {idx}:\n{seq}")


