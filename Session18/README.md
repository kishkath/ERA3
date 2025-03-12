# LLM Finetuning and Optimization

This project showcases the fine-tuning of Microsoft's **phi-2** language model using the innovative **QLoRA** (Quantized Low-Rank Adaptation) strategy. The aim is to enhance the model's performance while ensuring efficient resource usage.

---

## Overview

- **Model:** Microsoft's phi-2, a state-of-the-art language model known for its natural text generation.
- **Strategy:** **QLoRA**  
  - **Selective Fine-Tuning:** Only a few key modules (e.g., specific projection layers) are updated, reducing computational overhead.
  - **4-bit Quantization:** Model weights are compressed to 4 bits to significantly decrease memory usage while maintaining performance.

---

## Dataset

The project uses the **OpenAssistant Conversations (OASST1)** dataset, which was created to democratize research on large-scale alignment. This dataset includes:

- **161,443 messages** across **35 languages**
- **461,292 quality ratings**
- Over **10,000 fully annotated conversation trees**
- Contributions from more than **13,500 volunteers** worldwide

---

## Workflow

1. **Dataset Loading:**  
   The OpenAssistant Conversations dataset is loaded from Hugging Face.

2. **Data Preparation:**  
   Data is reformatted into system instructions and prompt formats to suit the model’s training requirements.

3. **Tokenization:**  
   The text data is tokenized using a suitable tokenizer, preparing it for model ingestion.

4. **Data Mapping:**  
   The tokenized data is mapped into batches using a data collator integrated with the SFTTrainer, ensuring smooth batch processing.

5. **Parameter Tuning:**  
   The training configuration includes the following parameters:
   - **per_device_train_batch_size:** 4
   - **gradient_accumulation_steps:** 4
   - **optimizer:** paged_adamw_32bit
   - **save_steps:** 10
   - **logging_steps:** 10
   - **learning_rate:** 2e-4
   - **max_grad_norm:** 0.3
   - **max_steps:** 500
   - **warmup_ratio:** 0.03
   - **lr_scheduler_type:** cosine
   - Additional settings include fp16 precision and gradient checkpointing for efficient memory usage.

6. **Training Initiation:**  
   The fine-tuning process is initiated and completed within 500 training steps.

---

## Deployment

The final model is hosted on [[Hugging Face Spaces](https://huggingface.co/spaces/kishkath/phi2-qlora)
![Screenshot from 2025-03-12 22-39-30](https://github.com/user-attachments/assets/fc4ad336-f431-461d-9a8d-b9a27df18894)
![Screenshot from 2025-03-12 22-26-38](https://github.com/user-attachments/assets/1432eae7-2f2d-4bf3-846f-8b32fdc06600)
![Screenshot from 2025-03-12 22-26-24](https://github.com/user-attachments/assets/27eb8689-17b0-449a-8789-1ae252ac6fac)
![Screenshot from 2025-03-11 15-29-23](https://github.com/user-attachments/assets/b2b52ee2-3d3f-45bf-84e3-cec8d632b5ff)
](#) and is optimized for efficient CPU-based inference. This makes it accessible to everyone without the need for specialized hardware. Continuous improvements are planned to further boost inference speed and overall performance.

---

## Future Improvements

- **Enhanced Inference Speed:** Ongoing optimizations to reduce latency during CPU inferencing.
- **Performance Tuning:** Further adjustments and refinements to achieve even better model performance.

