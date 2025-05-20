import functools
import os

import anndata as ad
import torch
import torch.nn.attention.flex_attention as flex_attn
from torch.utils.data import DataLoader

import transcriptformer.model.model as model_mod
from transcriptformer.data.dataclasses import (
    DataConfig,
    InferenceConfig,
    LossConfig,
    ModelConfig,
)
from transcriptformer.data.dataloader import AnnDataset
from transcriptformer.model.model import Transcriptformer
from transcriptformer.tokenizer.vocab import SPECIAL_TOKENS, build_gene_vocab_from_list


def _build_model(inference: bool = False):
    os.environ.setdefault("TORCH_NN_FLEX_ATTENTION_DISABLE_COMPILE", "1")
    os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
    os.environ.setdefault("TORCH_COMPILE", "0")
    model_mod.create_block_mask = functools.partial(flex_attn.create_block_mask, device="cpu", _compile=False)

    adata = ad.read_h5ad("test/data/human_val.h5ad")[:2, :50].copy()
    adata.var["ensembl_id"] = [g.split(".")[0] for g in adata.var["ensembl_id"]]

    gene_vocab = build_gene_vocab_from_list(list(adata.var["ensembl_id"]), SPECIAL_TOKENS)

    model_cfg = ModelConfig(
        log_counts_eps=1e-6,
        num_heads=2,
        num_layers=1,
        model_dim=8,
        embed_dim=8,
        dropout=0.0,
        activation="gelu",
        attn_bias=False,
        fw_bias=False,
        mu_link_fn="softmax",
        softcap=10,
        seq_len=20,
        aux_len=0,
        block_len=10,
    )
    model_cfg.use_aux = False

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
        gene_pad_token="[PAD]",
        aux_pad_token="[PAD]",
        esm2_mappings=None,
        special_tokens=SPECIAL_TOKENS,
        esm2_mappings_path=None,
        use_raw=None,
    )
    loss_cfg = LossConfig(gene_id_loss_weight=0.0)
    emb_matrix = torch.randn(len(gene_vocab), model_cfg.embed_dim)

    if inference:
        inf_cfg = InferenceConfig(
            output_keys=["embeddings"],
            batch_size=2,
            obs_keys=[],
            data_files=None,
            load_checkpoint=None,
            output_path=None,
            output_filename="embeddings.h5ad",
        )
    else:
        inf_cfg = None

    model = Transcriptformer(
        data_config=data_cfg,
        model_config=model_cfg,
        loss_config=loss_cfg,
        inference_config=inf_cfg,
        gene_vocab_dict=gene_vocab,
        aux_vocab_dict=None,
        emb_matrix=emb_matrix,
    )

    dataset = AnnDataset([adata], gene_vocab=gene_vocab, max_len=model_cfg.seq_len)
    loader = DataLoader(dataset, batch_size=2, collate_fn=dataset.collate_fn)
    batch = next(iter(loader))
    return model, batch


def test_forward_smoke():
    model, batch = _build_model()
    output = model(batch)
    assert "mu" in output
    assert output["mu"].shape[0] == 2


def test_inference_smoke():
    model, batch = _build_model(inference=True)
    output = model.inference(batch)
    assert "embeddings" in output
    assert output["embeddings"].shape[0] == 2
