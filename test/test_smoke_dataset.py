import anndata as ad
from torch.utils.data import DataLoader

from transcriptformer.data.dataloader import AnnDataset
from transcriptformer.tokenizer.vocab import SPECIAL_TOKENS, build_gene_vocab_from_list


def test_dataset_smoke():
    """Basic AnnDataset loading and collation"""
    adata = ad.read_h5ad("test/data/human_val.h5ad")[:2, :50].copy()
    adata.var["ensembl_id"] = [g.split(".")[0] for g in adata.var["ensembl_id"]]

    gene_vocab = build_gene_vocab_from_list(list(adata.var["ensembl_id"]), SPECIAL_TOKENS)
    dataset = AnnDataset(
        [adata],
        gene_vocab=gene_vocab,
        max_len=20,
        gene_col_name="ensembl_id",
        pad_zeros=True,
        filter_to_vocab=True,
    )

    assert len(dataset) == 2
    item = dataset[0]
    assert item.gene_counts.shape[-1] <= 20

    loader = DataLoader(dataset, batch_size=2, collate_fn=dataset.collate_fn)
    batch = next(iter(loader))
    assert batch.gene_counts.shape[0] == 2
