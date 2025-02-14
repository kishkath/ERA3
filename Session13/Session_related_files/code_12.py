import math
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, RichProgressBar
from lightning.pytorch.callbacks.progress.rich_progress import RichProgressBarTheme
import tiktoken
from pytorch_lightning.loggers import TensorBoardLogger

# Dataset
class TextDataset(Dataset):
    def __init__(self, filepath, tokenizer, block_size=1024):
        with open(filepath, 'r', encoding='utf-8') as f:
            text = f.read()
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.tokens = tokenizer.encode(text)

    def __len__(self):
        return len(self.tokens) - self.block_size

    def __getitem__(self, idx):
        input_ids = self.tokens[idx:idx + self.block_size]
        target_ids = self.tokens[idx + 1:idx + self.block_size + 1]
        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(target_ids, dtype=torch.long)

# Model
class DecoderOnlyTransformer(nn.Module):
    def __init__(self, vocab_size, embed_dim=768, num_heads=12, num_layers=12, block_size=1024):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(block_size, embed_dim)
        self.layers = nn.ModuleList([
            nn.TransformerDecoderLayer(embed_dim, num_heads, dim_feedforward=4 * embed_dim, activation='gelu')
            for _ in range(num_layers)
        ])
        self.ln = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size, bias=False)
        # weight sharing
        self.token_embedding.weight = self.head.weight

        # weight initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean = 0.0, std = std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std = 0.02)

    def forward(self, x):
        b, t = x.size()
        positions = torch.arange(t, device=x.device).unsqueeze(0)
        x = self.token_embedding(x) + self.position_embedding(positions)
        for layer in self.layers:
            x = layer(x, x)
        x = self.ln(x)
        return self.head(x)

# Lightning Module
class TransformerLightning(pl.LightningModule):
    def __init__(self, vocab_size, block_size, lr, warmup_steps, max_steps):
        super().__init__()
        self.save_hyperparameters()
        self.model = DecoderOnlyTransformer(vocab_size, block_size=block_size)
        self.criterion = nn.CrossEntropyLoss()
        self.tokenizer = tiktoken.get_encoding("gpt2")

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        input_ids, target_ids = batch
        logits = self(input_ids)
        loss = self.criterion(logits.view(-1, logits.size(-1)), target_ids.view(-1))
        # Log the loss with 4 decimal precision
        self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=False, logger=True)
        # Optionally print for observation
        # self.print(f"Train Loss: {loss.item():.4f}")
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)

        def lr_lambda(current_step):
            if current_step < self.hparams.warmup_steps:
                return self.hparams.lr * (current_step + 1) / self.hparams.warmup_steps
            elif current_step > self.hparams.max_steps:
                return self.hparams.lr * 0.1
            decay_ratio = (current_step - self.hparams.warmup_steps) / (self.hparams.max_steps - self.hparams.warmup_steps)
            coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
            return self.hparams.lr * 0.1 + coeff * (self.hparams.lr - self.hparams.lr * 0.1)

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return [optimizer], [scheduler]

# Training Script
if __name__ == "__main__":
    # Parameters
    block_size = 768
    batch_size = 16
    max_lr = 6e-4
    warmup_steps = 10
    max_steps = 25000

    tokenizer = tiktoken.get_encoding("gpt2")
    vocab_size = tokenizer.n_vocab

    # Dataset and DataLoader
    dataset = TextDataset("input.txt", tokenizer, block_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Model
    model = TransformerLightning(vocab_size, block_size, max_lr, warmup_steps, max_steps)

    torch.set_float32_matmul_precision('high')

    # Set up TensorBoard logger
    logger = TensorBoardLogger("logs/", name="transformer_experiment")
    # tensorboard --logdir logs/
    # create your own theme!
    progress_bar = RichProgressBar(
            refresh_rate=1,
            leave=False,
            theme=RichProgressBarTheme(
                description='',
                progress_bar='#6206E0',
                progress_bar_finished='#6206E0',
                progress_bar_pulse='#6206E0',
                batch_progress='',
                time='dim',
                processing_speed='dim underline',
                metrics='italic',
                metrics_text_delimiter=' ',
                metrics_format='.3f'),
            console_kwargs=None
    )

    # Trainer
    trainer = pl.Trainer(
        max_steps=max_steps,
        accelerator="gpu",
        devices=1,
        callbacks=[LearningRateMonitor(logging_interval='step'), progress_bar],
        precision='bf16-mixed', # 16-bit floating point, many other options are there
        log_every_n_steps=1,
        enable_progress_bar=True, # show progress bar
        enable_model_summary=True, # show model summary
        logger=logger
    )

    # https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.trainer.trainer.Trainer.html
    # Training
    trainer.fit(model, dataloader)
