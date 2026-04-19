"""PCA, neighbors, and UMAP."""

from __future__ import annotations

from typing import TYPE_CHECKING

import scanpy as sc
from anndata import AnnData

if TYPE_CHECKING:
    from arc_me_sc.config import EmbeddingParams


def run_embedding(
    adata: AnnData,
    params: "EmbeddingParams",
    flavor: str = "seurat",
    copy: bool = False,
) -> AnnData | None:
    """
    Highly variable genes -> scale -> PCA -> neighbors -> UMAP.

    Expects log-normalized ``adata.X`` (as in Campbell processed releases).
    """
    if copy:
        adata = adata.copy()

    sc.pp.highly_variable_genes(
        adata,
        n_top_genes=params.n_top_genes,
        flavor=flavor,
        subset=False,
    )
    sc.pp.scale(adata, max_value=10)
    sc.tl.pca(adata, n_comps=params.n_pcs, svd_solver="arpack")
    sc.pp.neighbors(
        adata,
        n_neighbors=params.n_neighbors,
        n_pcs=params.n_pcs,
        random_state=params.random_state,
    )
    sc.tl.umap(adata, random_state=params.random_state)

    return adata if copy else None
