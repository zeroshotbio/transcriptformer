import anndata as ad
import numpy as np


def _generate_counts(n_cells: int, n_genes: int, max_count: int, rng: np.random.Generator) -> np.ndarray:
    """Create random dense count matrix."""
    lam = rng.integers(1, max_count, size=(n_genes,))
    return rng.poisson(lam, size=(n_cells, n_genes)).astype(np.float32)


def _make_ids(prefix: str, n: int) -> list[str]:
    """Return list of incrementing IDs."""
    return [f"{prefix}{i}" for i in range(n)]


def make_fake_anndata(
    n_cells: int = 16,
    n_genes: int = 256,
    max_count: int = 20,
    *,
    add_raw: bool = False,
) -> "ad.AnnData":
    r"""Create an *in-memory* AnnData object suitable for Transcriptformer tests.

    Parameters
    ----------
    n_cells
        Number of observations (rows / cells).
    n_genes
        Number of variables (columns / genes **excluding** special tokens).
    max_count
        Upper bound for synthetic UMI counts (Poisson λ is drawn in [1, max_count]).
    add_raw
        If *True* adds an identical `.raw` copy – some data loaders require it.

    Returns
    -------
    AnnData
        * ``adata.X``  – dense ``float32`` count matrix shaped ``(n_cells, n_genes)``
        * ``adata.obs``— column ``"cell_id"`` with unique strings
        * ``adata.var``— column ``"ensembl_id"`` with strings ``g0`` … ``g{n_genes-1}``

    Notes
    -----
    * The helper never touches disk – perfect for fast CI.
    * Random seed *must* be set by the caller if determinism is required.
    """
    rng = np.random.default_rng()
    counts = _generate_counts(n_cells, n_genes, max_count, rng)
    obs = {"cell_id": _make_ids("c", n_cells)}
    var = {"ensembl_id": _make_ids("g", n_genes)}
    adata = ad.AnnData(X=counts, obs=obs, var=var)

    if add_raw:
        adata.raw = adata.copy()

    return adata
