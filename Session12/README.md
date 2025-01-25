# Shakespeare GPT Text Generator

A GPT-2 based text generator fine-tuned on Shakespeare's works, deployed as a Hugging Face Space with a Gradio interface.

## Aim

To achieve loss less than 0.09999. 

                              step9597, loss: 0.006384422071278095
                              step9598, loss: 0.007867483422160149
                              step9599, loss: 0.006179888732731342

## Project Overview

This project implements a custom GPT model trained on Shakespeare's complete works to generate Shakespeare-style text. The model uses the GPT-2 architecture and is trained from scratch on Shakespeare's texts.

hugging face space app : https://huggingface.co/spaces/kishkath/GPT2

![Screenshot from 2025-01-25 09-42-36](https://github.com/user-attachments/assets/f55347f3-2f88-46a9-899a-d12e949cd351)


## Model Architecture

The model follows the GPT-2 architecture with the following specifications:
- 12 transformer layers
- 768 embedding dimensions
- 12 attention heads
- 50,257 vocabulary size (using GPT-2 tokenizer)
- Maximum sequence length of 1024 tokens

### Key Components
- **Causal Self Attention**: Implements masked self-attention mechanism
- **MLP Blocks**: Feed-forward neural networks with GELU activation
- **Layer Normalization**: Applied before attention and MLP blocks
- **Positional Embeddings**: Learned position embeddings up to 1024 positions

## Training Process

The training was conducted in the `gpt2_training.ipynb` notebook with the following steps:

1. **Data Preparation**
   - Used Shakespeare's complete works from input.txt
   - Tokenized using GPT-2 tokenizer (tiktoken)
   - Implemented custom DataLoader for efficient batch processing

2. **Training Configuration**
   - Batch size: 8
   - Sequence length: 512
   - Learning rate: 3e-4
   - Optimizer: AdamW
   - Training iterations: 9600steps (only 1 Epoch)

3. **Model Training**
   - Trained on GPU (Tesla T4) (Kaggle)
   - Used cross-entropy loss for next token prediction
   - Implemented gradient clipping for stability
   - Model checkpoints saved as model_state_dict.pth

### Architecture Deep Dive

1. **Transformer Blocks**
   - Multi-head self-attention mechanism
   - Layer normalization for training stability
   - Residual connections for gradient flow
   - GELU activation functions

2. **Attention Mechanism**
   - Scaled dot-product attention
   - Causal masking for autoregressive generation
   - Multi-head attention for parallel processing

3. **Position Embeddings**
   - Learned positional encodings
   - Maximum sequence length of 1024 tokens
   - Combined with token embeddings

### Training Details

1. **Loss Function**
   - Cross-entropy loss for next token prediction
   - Implemented with PyTorch's nn.CrossEntropyLoss

2. **Optimization**
   - AdamW optimizer with β1=0.9, β2=0.999
   - Learning rate of 3e-4
   - Gradient clipping to prevent exploding gradients

3. **Data Processing**
   - Custom DataLoader for efficient batch processing
   - Dynamic sequence length handling
   - GPT-2 tokenizer for consistent tokenization

## Deployment

The model is deployed as a Hugging Face Space with a Gradio interface:
## Performance and Results

### Training Metrics
- Final training loss: [Your final loss value]
- Training time: [Training duration]
- Hardware used: NVIDIA Tesla T4 GPU

### Generation Examples

1. Input: "To be, or not to be,"
   Output: [Example generated text]

2. Input: "All the world's a stage,"
   Output: [Example generated text]

## Limitations and Future Work

### Current Limitations
1. Limited training data (only Shakespeare's works)
2. Fixed maximum sequence length
3. No fine-grained control over generation sty
