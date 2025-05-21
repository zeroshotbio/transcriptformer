r"""Minimal LoRA fine-tuning utility for TranscriptFormer.

CPU smoke-test
--------------
python finetune_lora.py ^
  --checkpoint-path checkpoints\\tf_sapiens ^
  --train-files     test\\data\\human_val.h5ad ^
  --epochs          1 ^
  --batch-size      4 ^
  --lora-r          4 --lora-alpha 16 ^
  --devices         0 ^
  --output-path     adapters_epoch1.pt
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytorch_lightning as pl
import torch
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from torch.utils.data import DataLoader

from transcriptformer.data.dataclasses import DataConfig, LossConfig, ModelConfig
from transcriptformer.data.dataloader import AnnDataset
from transcriptformer.model.lora import LoRAConfig, apply_lora, lora_state_dict
from transcriptformer.model.model import Transcriptformer
from transcriptformer.tokenizer.vocab import SPECIAL_TOKENS, load_vocabs_and_embeddings

# ────────────────────────────────────────────────────────────────────────────
# 1.  Make flex-attention completely CPU-safe  ⬅️ NEW
# ────────────────────────────────────────────────────────────────────────────
try:
    import torch.nn.attention.flex_attention as _fa

    # Patch BOTH helpers so they always run on CPU and skip CUDA kernels
    if hasattr(_fa, "create_mask") and hasattr(_fa, "create_block_mask"):
        _orig_create_mask = _fa.create_mask
        _orig_block_mask = _fa.create_block_mask

        def _cpu_create_mask(mask_mod, B, H, Q, KV, _device, _compile=False):
            return _orig_create_mask(mask_mod, B, H, Q, KV, device="cpu", _compile=False)

        def _cpu_block_mask(*args, **kwargs):
            kwargs["_compile"] = False
            kwargs["device"] = "cpu"
            return _orig_block_mask(*args, **kwargs)

        _fa.create_mask = _cpu_create_mask  # type: ignore[attr-defined]
        _fa.create_block_mask = _cpu_block_mask  # type: ignore[attr-defined]
except ModuleNotFoundError:
    pass  # old Torch versions don’t have flex-attention – nothing to patch

# ────────────────────────────────────────────────────────────────────────────
# 2.  Aux-vocab is optional (same as before)
# ────────────────────────────────────────────────────────────────────────────
import transcriptformer.tokenizer.vocab as _tvocab

_orig_open_vocabs = _tvocab.open_vocabs
_tvocab.open_vocabs = lambda path, cols: None if not path or not Path(path).exists() else _orig_open_vocabs(path, cols)  # type: ignore[attr-defined]

console = Console()


# ───────────────────────── helper utilities ────────────────────────────────
def _strip_hydra(d: dict[str, Any]) -> dict[str, Any]:
    return {
        k: _strip_hydra(v) if isinstance(v, dict) else v for k, v in d.items() if k not in {"_target_", "_partial_"}
    }


def _load_cfg(ckpt: Path) -> tuple[DataConfig, ModelConfig, LossConfig]:
    if ckpt.is_file():
        raw = torch.load(ckpt, map_location="cpu").get("model", {})
    else:
        cfg_p = ckpt / "config.json"
        raw = json.loads(cfg_p.read_text())["model"]
    return (
        DataConfig(**_strip_hydra(raw["data_config"])),
        ModelConfig(**_strip_hydra(raw["model_config"])),
        LossConfig(**_strip_hydra(raw["loss_config"])),
    )


def _find_weights(ckpt: Path) -> Path:
    if ckpt.is_file():
        return ckpt
    for p in (
        ckpt / "pytorch_model.bin",
        ckpt / "model_weights.pt",
        *ckpt.glob("*.ckpt"),
        *ckpt.glob("*.pt"),
        *ckpt.glob("*.bin"),
    ):
        if p.exists():
            return p
    raise FileNotFoundError(f"No weight file in {ckpt}")


def _random_vocab_and_emb(vocab_size: int, dim: int):
    vocab = {tok: i for i, tok in enumerate(SPECIAL_TOKENS)}
    while len(vocab) < vocab_size:
        vocab[f"rand_{len(vocab)}"] = len(vocab)
    return vocab, torch.randn(vocab_size, dim)


# ─────────────────── Lightning helper adds missing hooks ───────────────────
class LoRAFineTuner(pl.LightningModule):
    def __init__(self, backbone: Transcriptformer):
        super().__init__()
        self.backbone = backbone

    def training_step(self, batch, _):
        out = self.backbone(batch)
        if isinstance(out, torch.Tensor):
            return out
        if isinstance(out, dict) and "loss" in out:
            return out["loss"]
        if isinstance(out, tuple):
            return out[0]
        return torch.tensor(0.0, requires_grad=True)

    def configure_optimizers(self):
        return torch.optim.AdamW(
            (p for p in self.parameters() if p.requires_grad),
            lr=1e-3,
            weight_decay=0.0,
        )


# ─────────────────────────────────── main ──────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint-path", required=True)
    ap.add_argument("--train-files", nargs="+", required=True)
    ap.add_argument("--output-path", default="lora_weights.pt")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument(
        "--devices",
        default="cpu",
        help="Device count or 'cpu' to force CPU.",
    )
    ap.add_argument("--precision", default="16-mixed")
    ap.add_argument("--lora-r", type=int, default=4)
    ap.add_argument("--lora-alpha", type=float, default=16.0)
    ap.add_argument("--lora-dropout", type=float, default=0.0)
    ap.add_argument("--lora-target-modules", nargs="+", default=("linear1", "linear2", "linears"))
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Instantiate model/dataloader and exit before training (used by CI smoke tests).",
    )
    args = ap.parse_args()

    ckpt_dir = Path(args.checkpoint_path).expanduser().resolve()
    data_cfg, model_cfg, loss_cfg = _load_cfg(ckpt_dir)

    for fld in ("aux_vocab_path", "esm2_mappings_path", "gene_vocab_path"):
        pth = getattr(data_cfg, fld, None)
        if pth and pth not in ("", None) and not Path(pth).is_absolute():
            setattr(data_cfg, fld, ckpt_dir / pth)
    data_cfg.esm2_mappings_path = str(getattr(data_cfg, "esm2_mappings_path", ""))

    console.log("[cyan]Loading vocabularies & embeddings…")
    dummy = SimpleNamespace(model=SimpleNamespace(data_config=data_cfg))
    try:
        (gene_vocab, aux_vocab), emb = load_vocabs_and_embeddings(dummy)
    except Exception as exc:
        w = _find_weights(ckpt_dir)
        cp_gene_emb = torch.load(w, map_location="cpu")["gene_embeddings.embedding.weight"]
        emb_dim, vocab_size = cp_gene_emb.shape[1], cp_gene_emb.shape[0]
        warnings.warn(
            f"Falling back to RANDOM embeddings – {exc}",
            RuntimeWarning,
            stacklevel=2,
        )
        gene_vocab, emb = _random_vocab_and_emb(vocab_size, emb_dim)
        aux_vocab = None

    if not hasattr(model_cfg, "use_aux"):
        model_cfg.use_aux = bool(aux_vocab)

    data_cfg.filter_outliers = data_cfg.filter_outliers or 0
    data_cfg.min_expressed_genes = data_cfg.min_expressed_genes or 0

    backbone = Transcriptformer(
        data_config=data_cfg,
        model_config=model_cfg,
        loss_config=loss_cfg,
        gene_vocab_dict=gene_vocab,
        aux_vocab_dict=aux_vocab,
        emb_matrix=emb,
    )

    try:
        backbone.load_state_dict(torch.load(_find_weights(ckpt_dir), map_location="cpu"), strict=True)
    except RuntimeError as e:
        console.log(f"[yellow]Checkpoint mismatch – loading compatible tensors only ({e})")
        backbone.load_state_dict(torch.load(_find_weights(ckpt_dir), map_location="cpu"), strict=False)

    apply_lora(
        backbone,
        LoRAConfig(
            r=args.lora_r,
            alpha=args.lora_alpha,
            dropout=args.lora_dropout,
            target_modules=tuple(args.lora_target_modules),
        ),
    )
    console.log("[green]LoRA adapters injected[/]")

    with Progress(SpinnerColumn(), TextColumn("{task.description}")):
        ds = AnnDataset(
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

    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=data_cfg.n_data_workers,
        collate_fn=ds.collate_fn,
        pin_memory=data_cfg.pin_memory,
    )

    if str(args.devices).lower() == "cpu" or str(args.devices) == "0":
        accelerator = "cpu"
        devices_flag = 1
    else:
        accelerator = "gpu"
        devices_flag = int(args.devices)

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator=accelerator,
        devices=devices_flag,
        precision=args.precision,
        log_every_n_steps=5,
    )
    if args.dry_run:
        console.log("[bold yellow]Dry-run complete – exiting before training loop.[/]")
        sys.exit(0)

    console.log("[bold]Starting fine-tune (smoke)…[/]")
    trainer.fit(LoRAFineTuner(backbone), dl)

    torch.save(lora_state_dict(backbone), args.output_path)
    console.log(f"[bold green]✅ LoRA weights saved → {args.output_path}[/]")


if __name__ == "__main__":
    main()
