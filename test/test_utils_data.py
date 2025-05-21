from test.utils_data import make_fake_anndata


def test_fake_anndata_shapes():
    adata = make_fake_anndata(n_cells=8, n_genes=12, max_count=5)
    assert adata.n_obs == 8
    assert adata.n_vars == 12
    assert adata.X.shape == (8, 12)
    # verify mandatory columns
    assert "cell_id" in adata.obs
    assert "ensembl_id" in adata.var
