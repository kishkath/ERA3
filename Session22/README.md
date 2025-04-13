**Phi-2 Fine‑Tuning & qLoRA Compression Demo**

This repository demonstrates the conceptual process of fine‑tuning the Phi‑2 (un‑SFTed) language model using the GRPO Trainer from Hugging Face, compressing the resulting model with qLoRA, and deploying a live inference demo on Hugging Face Spaces.

---

## 🚀 Project Overview

1. **Fine‑Tune Phi‑2 with GRPO**
   - GRPO (Guided Reinforcement Policy Optimization) is used to fine‑tune the Phi‑2 model using preference-based training strategies.

2. **Compress with qLoRA**
   - After training, the model is compressed using qLoRA, enabling efficient deployment and inference.

3. **Deploy on Hugging Face Spaces**
   - The compressed model is wrapped in a Gradio application and deployed as a live demo on Hugging Face Spaces. (Running on CPU)

4. **Repository & Demo Links**
   - **Spaces Demo**: [https://huggingface.co/spaces/kishkath/phi2-grpo-qlora]

---

## 📦 Repository Structure (Conceptual)

- **data/**: Contains the dataset used for fine‑tuning.
- **grpo_training/**: Contains fine‑tuning logic using the GRPO Trainer.
- **quantization/**: Details the qLoRA compression process.
- **app/**: Hosts the Gradio-based web demo for the model.
- **README.md**: Provides theoretical overview and project documentation.

---

## 🎓 Fine‑Tuning with GRPO Trainer

### What is GRPO?

GRPO (Guided Reinforcement Policy Optimization) is a method that merges reinforcement learning and supervised fine-tuning with the goal of improving language models based on preference or reward signals. GRPO extends PPO (Proximal Policy Optimization) by incorporating guidance from a reward model or curated feedback. This helps the language model to learn more human-aligned responses without straying too far from its original capabilities.

**Key Features:**
- **Guided Rewards:** Enables fine‑tuning based on scalar rewards, human feedback, or synthetic signals.
- **Stability:** Employs mechanisms similar to PPO’s trust region to prevent performance degradation.
- **Hybrid Learning:** Balances reinforcement learning and supervised objectives.

---

## 🔍 GRPO vs. DPO vs. PPO

| Aspect               | GRPO                                | DPO (Direct Preference Optimization) | PPO (Proximal Policy Optimization)      |
|----------------------|-------------------------------------|---------------------------------------|-----------------------------------------|
| **Optimization Goal** | Maximize reward with guidance       | Maximize preference likelihood        | Maximize expected reward                 |
| **Feedback Type**    | Scalar rewards or ranked outputs    | Pairwise preferences only             | Scalar rewards                           |
| **Learning Method**  | Hybrid: supervised + policy updates | Supervised (no RL)                    | Reinforcement learning with clipping     |
| **Policy Stability** | Maintains stability via constraints | No explicit trust region              | Trust-region-like objective              |
| **Sample Efficiency**| Moderate to high                    | High                                  | Moderate                                 |
| **Complexity**       | Moderate                            | Low                                   | Moderate                                 |
| **Use Case**         | Aligning models with detailed reward signals | Pairwise preference alignment         | General RL with language models         |

---

## 🧊 qLoRA Compression

qLoRA (quantized Low-Rank Adaptation) is an optimization technique for compressing large language models by combining quantization and parameter-efficient fine-tuning.

**Benefits:**
- **Reduced Memory Usage:** Achieves 4‑bit precision while maintaining accuracy.
- **Faster Inference:** Lower memory consumption translates to faster predictions.
- **Efficient Deployment:** Ideal for deploying models on resource-constrained environments like Hugging Face Spaces.

qLoRA is particularly useful when deploying fine‑tuned models at scale or in interactive demos where inference speed and memory efficiency are crucial.

---

## 🌐 Deployment on Hugging Face Spaces

The qLoRA‑compressed model is integrated into a Gradio app that allows users to interact with the model live via Hugging Face Spaces. The app provides a simple user interface where users can input prompts and observe model responses in real time.

---

## 📖 Recalculated Responses Demo

The base model response:


The finetuned (GRPO) model response:


This showcases the effectiveness of GRPO training combined with qLoRA compression.

---

## 🔗 Links

- **Hugging Face Space:** [https://huggingface.co/spaces/<your-username>/phi2-grpo-qLoRA-demo](https://huggingface.co/spaces/kishkath/phi2-grpo-qlora)

---

## 📜 License

This project is licensed under the Educational Purposes.
