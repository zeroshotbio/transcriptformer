# smoke_finetune.py  – one-batch LoRA sanity check
from pathlib import Path

import pytorch_lightning as pl
import torch

from transcriptformer.data.dataloader import AnnDataset  # path in repo
from transcriptformer.model.lora import LoRAConfig, apply_lora, lora_state_dict
from transcriptformer.model.model import Transcriptformer

CKPT = Path("./checkpoints/tf_sapiens")
ADATA = Path("test/data/human_val.h5ad")

# 1. load frozen backbone ----------------------------------------------------
model = Transcriptformer.load_from_checkpoint(str(CKPT / "pytorch_model.bin"))

# 2. patch LoRA adapters -----------------------------------------------------
apply_lora(model, LoRAConfig(r=4, alpha=16, dropout=0.05, target_modules=("linear1", "linear2", "linears")))

# 3. build tiny dataset ------------------------------------------------------
dataset = AnnDataset([str(ADATA)], gene_vocab=model.gene_vocab, max_len=2047)
loader = torch.utils.data.DataLoader(dataset, batch_size=1, collate_fn=dataset.collate_fn)

# 4. one optimisation step ---------------------------------------------------
trainer = pl.Trainer(
    max_epochs=1, limit_train_batches=1, devices=1, accelerator="auto", precision="16-mixed", logger=False
)
trainer.fit(model, loader)

# 5. save adapter weights ----------------------------------------------------
torch.save(lora_state_dict(model), "adapter_smoke.pt")
print("LoRA smoke-test finished ✓")
