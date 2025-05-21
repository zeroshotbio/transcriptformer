from __future__ import annotations

import functools
import os
from pathlib import Path

import pytest
import torch
import torch.nn.attention.flex_attention as flex_attn

import transcriptformer.model.model as model_mod
from transcriptformer.data.dataclasses import BatchData, DataConfig, LossConfig, ModelConfig
from transcriptformer.model.lora import LoRAConfig, apply_lora
from transcriptformer.model.model import Transcriptformer
from transcriptformer.tokenizer.vocab import SPECIAL_TOKENS, build_gene_vocab_from_list

# ---------------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
CKPT = ROOT / "test" / "assets" / "mini_ckpt.pt"

BATCH = 4
SEQ = 32
VOCAB = 256
DIMS = 128


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def model_with_lora() -> Transcriptformer:
    os.environ.setdefault("TORCH_NN_FLEX_ATTENTION_DISABLE_COMPILE", "1")
    os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
    os.environ.setdefault("TORCH_COMPILE", "0")
    model_mod.create_block_mask = functools.partial(flex_attn.create_block_mask, device="cpu", _compile=False)

    torch.manual_seed(0)

    gene_vocab = build_gene_vocab_from_list(
        [f"g{i}" for i in range(VOCAB - len(SPECIAL_TOKENS))],
        SPECIAL_TOKENS,
    )

    model_cfg = ModelConfig(
        log_counts_eps=1e-6,
        num_heads=4,
        num_layers=1,
        model_dim=DIMS,
        embed_dim=DIMS,
        dropout=0.0,
        activation="gelu",
        attn_bias=False,
        fw_bias=False,
        mu_link_fn="softmax",
        softcap=10,
        seq_len=SEQ,
        aux_len=0,
        block_len=16,
    )
    model_cfg.use_aux = False

    data_cfg = DataConfig(
        aux_vocab_path=".",
        pin_memory=False,
        aux_cols=[],
        gene_col_name="id",
        clip_counts=30,
        filter_to_vocabs=True,
        filter_outliers=0.0,
        pad_zeros=True,
        normalize_to_scale=0,
        n_data_workers=0,
        sort_genes=False,
        randomize_genes=False,
        min_expressed_genes=0,
        gene_pad_token="[PAD]",
        aux_pad_token="[PAD]",
        esm2_mappings=None,
        special_tokens=SPECIAL_TOKENS,
        esm2_mappings_path=None,
        use_raw=None,
    )
    loss_cfg = LossConfig(gene_id_loss_weight=0.0)

    emb_matrix = torch.randn(len(gene_vocab), DIMS)

    model = Transcriptformer(
        data_config=data_cfg,
        model_config=model_cfg,
        loss_config=loss_cfg,
        gene_vocab_dict=gene_vocab,
        aux_vocab_dict=None,
        emb_matrix=emb_matrix,
    )

    if CKPT.exists():
        state = torch.load(CKPT, map_location="cpu")
        model.load_state_dict(state.get("state_dict", state), strict=False)

    for p in model.parameters():
        p.requires_grad = False

    apply_lora(
        model,
        LoRAConfig(r=2, alpha=4, target_modules=("linear1", "linear2", "linears")),
    )

    model.train()
    return model


# ---------------------------------------------------------------------------
# test cases
# ---------------------------------------------------------------------------


def test_lora_params_require_grad(model_with_lora: Transcriptformer) -> None:
    lora_params = {n for n, p in model_with_lora.named_parameters() if "weight_a" in n or "weight_b" in n}
    non_lora_trainables = [n for n, p in model_with_lora.named_parameters() if p.requires_grad and n not in lora_params]
    assert non_lora_trainables == [], f"found unexpected trainables: {non_lora_trainables}"
    assert len(lora_params) > 0, "no LoRA parameters detected"


def test_forward_backward_ok(model_with_lora: Transcriptformer) -> None:
    batch = BatchData(
        gene_counts=torch.ones(BATCH, SEQ),
        gene_token_indices=torch.randint(VOCAB, (BATCH, SEQ)),
    )

    out = model_with_lora(batch)
    assert "mu" in out
    assert out["mu"].shape == (BATCH, SEQ)

    loss = out["mu"].pow(2).mean()
    loss.backward()

    grads = [p.grad for n, p in model_with_lora.named_parameters() if p.requires_grad]
    assert any(g is not None and torch.any(g != 0) for g in grads), "LoRA grads missing"
