from __future__ import annotations

import functools
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.attention.flex_attention as flex_attn
from torch.utils.data import DataLoader

import transcriptformer.model.model as model_mod
from test.utils_data import make_fake_anndata
from transcriptformer.data.dataclasses import DataConfig, LossConfig, ModelConfig
from transcriptformer.data.dataloader import AnnDataset
from transcriptformer.model.lora import LoRAConfig, apply_lora
from transcriptformer.model.model import Transcriptformer
from transcriptformer.tokenizer.vocab import SPECIAL_TOKENS, build_gene_vocab_from_list


def test_finetune_single_step(tmp_path: Path) -> None:
    """End-to-end training smoke test with synthetic data."""
    os.environ.setdefault("TORCH_NN_FLEX_ATTENTION_DISABLE_COMPILE", "1")
    os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
    os.environ.setdefault("TORCH_COMPILE", "0")
    model_mod.create_block_mask = functools.partial(flex_attn.create_block_mask, device="cpu", _compile=False)

    torch.manual_seed(0)
    np.random.seed(0)

    adata = make_fake_anndata(n_cells=8, n_genes=128, max_count=10, add_raw=True)
    special_tokens = SPECIAL_TOKENS

    mdl_cfg = ModelConfig(
        log_counts_eps=1e-6,
        num_layers=1,
        num_heads=2,
        model_dim=32,
        embed_dim=32,
        seq_len=32,
        block_len=16,
        dropout=0.0,
        activation="gelu",
        attn_bias=False,
        fw_bias=False,
        mu_link_fn="softmax",
        softcap=10,
        aux_len=0,
    )
    mdl_cfg.use_aux = False
    data_cfg = DataConfig(
        aux_vocab_path=".",
        pin_memory=False,
        aux_cols=[],
        gene_col_name="ensembl_id",
        clip_counts=30,
        filter_to_vocabs=True,
        filter_outliers=0.0,
        pad_zeros=True,
        normalize_to_scale=0,
        n_data_workers=0,
        sort_genes=False,
        randomize_genes=False,
        min_expressed_genes=0,
        gene_pad_token="<PAD>",
        aux_pad_token="<PAD>",
        esm2_mappings=None,
        special_tokens=special_tokens,
        esm2_mappings_path=None,
        use_raw=None,
    )
    loss_cfg = LossConfig(gene_id_loss_weight=0.0)

    gene_vocab = build_gene_vocab_from_list(list(adata.var["ensembl_id"]), special_tokens)
    emb = torch.randn(len(gene_vocab), mdl_cfg.embed_dim)

    model = Transcriptformer(
        data_config=data_cfg,
        model_config=mdl_cfg,
        loss_config=loss_cfg,
        gene_vocab_dict=gene_vocab,
        aux_vocab_dict=None,
        emb_matrix=emb,
    )

    for param in model.parameters():
        param.requires_grad_(False)
    apply_lora(model, LoRAConfig(r=2, alpha=4, target_modules=("linear1", "linear2")))

    ds = AnnDataset([adata], gene_vocab=gene_vocab, max_len=mdl_cfg.seq_len)
    loader = DataLoader(ds, batch_size=4, collate_fn=ds.collate_fn)

    batch = next(iter(loader))
    out1 = model(batch)
    loss1 = model.criterion(mu=out1["mu"], input_counts=out1["input_counts"], mask=out1["mask"])

    opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
    opt.zero_grad()
    loss1.backward()
    opt.step()

    out2 = model(batch)
    loss2 = model.criterion(mu=out2["mu"], input_counts=out2["input_counts"], mask=out2["mask"])

    assert loss2.item() < loss1.item(), "loss did not decrease after a training step"

    adapter_path = tmp_path / "tiny_adapter.pt"
    torch.save({k: v.cpu() for k, v in model.state_dict().items() if "lora_" in k}, adapter_path)
    assert adapter_path.is_file()
