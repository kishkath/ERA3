import torch
import torch.nn as nn
import pytorch_lightning as pl
from transformers import AutoModel


MODEL_NAME = "HuggingFaceTB/SmolLM2-135M-Instruct"
class SMOLLLightningModule(pl.LightningModule):
    def __init__(self, model_name=MODEL_NAME):  # Replace with actual model name
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, input_ids, labels=None):
        outputs = self.model(input_ids)
        logits = outputs.logits

        if labels is not None:
            loss = self.loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
            return loss, logits
        return logits

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=2e-5)
