# SmolLM2-135M: Training, Checkpointing, and Deployment

**SmolLM2-135M** is a lightweight language model designed for efficiency while maintaining strong text generation capabilities. This document describes the model architecture, training process, optimization techniques, and deployment steps.


## 📚 Project Overview

This project demonstrates the process of training and optimizing the **SmolLM2-135M** model, a lightweight causal language model. The primary objective is to:

1. Train the model for **5000 steps** with predictions logged every **500 steps**.
2. Save a checkpoint at 5000 steps and resume training for **50 additional steps**.
3. Leverage advanced performance optimizations (e.g., **FlashAttention**, **Dynamic Quantization**, **Weight Sharing**).
4. Reverse engineer model details using the YAML configuration and **Hugging Face** checkpoints.
5. Upload the trained and quantized model to **Hugging Face Spaces** for inference.

---

## 🧠 SmolLM2-135M Model Definition

The **SmolLM2-135M** model is based on a **decoder-only transformer** architecture with the following key specifications:

- **Model Size:** 135 million parameters
- **Vocabulary Size:** 49,152
- **Context Length:** 1,024 tokens
- **Transformer Layers:** 30
- **Attention Heads:** 9
- **Embedding Dimension:** 576
- **Intermediate Size:** 1536

### Model Components Breakdown:

1. **Embedding Layers:**
   - `wte`: Token embeddings
   - `wpe`: Positional embeddings

2. **Attention Mechanism (Without RoPE):**
   - Implements causal self-attention.
   - Uses `scaled_dot_product_attention` for fast matrix multiplications.
   - **RoPE (Rotary Position Embedding)** is omitted for simplicity.

3. **RMSNorm (Root Mean Square Normalization):**
   - Applied instead of LayerNorm for better numerical stability and efficiency.

4. **Residual Scaling:**
   - Introduced in the transformer blocks to stabilize training and reduce gradient explosion.
   - Residuals are scaled by \(\frac{1}{\sqrt{2}}\) to prevent gradient accumulation.

5. **Weight Sharing:**
   - The `lm_head` shares weights with the input embedding layer to reduce model size.

---

## ⚡️ Optimization Techniques

1. **Precision Optimization:**
   - `torch.set_float32_matmul_precision('high')` for faster matrix multiplications using Tensor Cores (on compatible GPUs).

2. **Gradient Clipping:**
   - Mitigates exploding gradients by capping gradient norms.

3. **Dynamic Quantization:**
   - Converts linear layers to 8-bit integers for faster inference and reduced memory usage:

```python
lit_model.model = quantize_dynamic(lit_model.model, {nn.Linear}, dtype=torch.qint8)
```

4. **Custom Learning Rate Scheduler:**
   - Implements a custom warmup and decay schedule:

```python
def lr_lambda(step):
    warmup_steps = 2000
    decay_start = 1600000
    decay_steps = 400000
    if step < warmup_steps:
        return step / warmup_steps
    elif step < decay_start:
        return 1.0
    return max(0.0, 1 - (step - decay_start) / decay_steps)
```

5. **FlashAttention:**
   - Uses `scaled_dot_product_attention` for efficient multi-head attention.

---

## 🛠️ Training Process

1. **Initial Training:**
   - Train the model for **5000 steps** with checkpoints saved every 500 steps.

```python
trainer = Trainer(max_steps=5000, log_every_n_steps=500, accelerator="auto", devices=1)
trainer.fit(lit_model, dataloader)
```

2. **Resuming Training:**
   - Load the saved checkpoint and continue training for **50 steps**.

```python
lit_model = SMOLLLightningModule.load_from_checkpoint("smollm2_135m.ckpt")
trainer = Trainer(max_steps=50)
trainer.fit(lit_model, dataloader)
```

Why 50 More Steps? 🤔
> This additional training step serves as a sanity check to confirm that model convergence is stable across resumed checkpoints.

---

## 📊 Parameter Calculation

### Total Parameters Calculation:

1. **Embedding Layers:**
   - Token Embedding: \( 49152 \times 576 = 28,303,872 \)
   - Positional Embedding: \( 1024 \times 576 = 589,824 \)

2. **Transformer Blocks (30 layers):**
   - Attention: \( 3 \times 576 \times 576 = 995,328 \)
   - MLP: \( 576 \times 1536 = 884,736 \)

   For 30 layers: \( (995,328 + 884,736) \times 30 = 56,402,880 \)

3. **Final LM Head:**
   - \( 576 \times 49152 = 28,303,872 \)

**Total:** ~135 million parameters.

---

## 📤 Deployment to Hugging Face Spaces

1. **Saving Checkpoints:**

```python
trainer.save_checkpoint("smollm2_135m_final.ckpt")
```

2. **Quantized Model Saving:**

```python
torch.save({"state_dict": lit_model.state_dict()}, "smollm2_135m_quantized.ckpt")
```

3. **Uploading to Hugging Face:**
   - Use `huggingface_hub` package to upload model artifacts.

```bash
pip install huggingface_hub
huggingface-cli login
huggingface-cli repo create smollm2-135m
huggingface-cli upload smollm2-135m smollm2_135m_quantized.ckpt
```

---

## 📎 Links to Submit

1. **GitHub Repository:** Include all training scripts and documentation.
2. **Hugging Face Space:** Host the quantized model for public inference.

Share both links for evaluation. ✅

---

## 🎉 Final Thoughts

This demonstration highlights advanced techniques to improve transformer-based model efficiency and usability. The project emphasizes:

1. Custom architecture exploration.
2. Optimized training with quantization.
3. Reproducibility via detailed checkpoints.

Enjoy experimenting with SmolLM2-135M! 🚀
