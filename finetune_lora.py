"""Minimal LoRA fine-tuning utility for TranscriptFormer.

Example
-------
>>> python finetune_lora.py \
...     --checkpoint-path ./checkpoints/tf_sapiens \
...     --train-files train.h5ad \
...     --output-path lora_weights.pt \
...     --lora-r 4 --lora-alpha 16 --lora-dropout 0.0
"""

from __future__ import annotations

import argparse
import json
import os
from types import SimpleNamespace

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from transcriptformer.data.dataclasses import DataConfig, LossConfig, ModelConfig
from transcriptformer.data.dataloader import AnnDataset
from transcriptformer.model.lora import (
    LoRAConfig,
    apply_lora,
    lora_state_dict,
)
from transcriptformer.model.model import Transcriptformer
from transcriptformer.tokenizer.vocab import load_vocabs_and_embeddings


def load_configs(checkpoint_path: str) -> tuple[DataConfig, ModelConfig, LossConfig]:
    """Load dataclass configs from a checkpoint directory."""
    with open(os.path.join(checkpoint_path, "config.json")) as f:
        cfg = json.load(f)["model"]
    return (
        DataConfig(**cfg["data_config"]),
        ModelConfig(**cfg["model_config"]),
        LossConfig(**cfg["loss_config"]),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="LoRA fine-tuning for TranscriptFormer")
    parser.add_argument("--checkpoint-path", required=True, help="Directory with model checkpoint")
    parser.add_argument("--train-files", nargs="+", required=True, help="Training h5ad files")
    parser.add_argument("--output-path", default="lora_weights.pt", help="File to save LoRA weights")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lora-r", type=int, default=4)
    parser.add_argument("--lora-alpha", type=float, default=16.0)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    args = parser.parse_args()

    data_cfg, model_cfg, loss_cfg = load_configs(args.checkpoint_path)

    # Update paths relative to checkpoint
    if not os.path.isabs(data_cfg.aux_vocab_path):
        data_cfg.aux_vocab_path = os.path.join(args.checkpoint_path, data_cfg.aux_vocab_path)
    if data_cfg.esm2_mappings_path and not os.path.isabs(data_cfg.esm2_mappings_path):
        data_cfg.esm2_mappings_path = os.path.join(args.checkpoint_path, data_cfg.esm2_mappings_path)

    dummy_cfg = SimpleNamespace(model=SimpleNamespace(data_config=data_cfg))
    (gene_vocab, aux_vocab), emb_matrix = load_vocabs_and_embeddings(dummy_cfg)

    model = Transcriptformer(
        data_config=data_cfg,
        model_config=model_cfg,
        loss_config=loss_cfg,
        gene_vocab_dict=gene_vocab,
        aux_vocab_dict=aux_vocab,
        emb_matrix=emb_matrix,
    )

    weights_path = os.path.join(args.checkpoint_path, "model_weights.pt")
    if os.path.exists(weights_path):
        state = torch.load(weights_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state)

    lora_cfg = LoRAConfig(r=args.lora_r, alpha=args.lora_alpha, dropout=args.lora_dropout)
    apply_lora(model, lora_cfg)

    dataset = AnnDataset(
        files_list=args.train_files,
        gene_vocab=gene_vocab,
        aux_vocab=aux_vocab,
        max_len=model_cfg.seq_len,
        normalize_to_scale=data_cfg.normalize_to_scale,
        sort_genes=data_cfg.sort_genes,
        randomize_order=data_cfg.randomize_genes,
        pad_zeros=data_cfg.pad_zeros,
        gene_col_name=data_cfg.gene_col_name,
        filter_to_vocab=data_cfg.filter_to_vocabs,
        filter_outliers=data_cfg.filter_outliers,
        min_expressed_genes=data_cfg.min_expressed_genes,
        pad_token=data_cfg.gene_pad_token,
        clip_counts=data_cfg.clip_counts,
        obs_keys=None,
        use_raw=data_cfg.use_raw,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=data_cfg.n_data_workers,
        collate_fn=dataset.collate_fn,
        pin_memory=data_cfg.pin_memory,
    )

    trainer = pl.Trainer(max_epochs=args.epochs, devices=1, accelerator="auto")
    trainer.fit(model, loader)

    torch.save(lora_state_dict(model), args.output_path)


if __name__ == "__main__":
    main()
