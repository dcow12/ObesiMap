"""Graph-based Leiden and Louvain clustering."""

from __future__ import annotations

from typing import TYPE_CHECKING

import scanpy as sc
from anndata import AnnData

if TYPE_CHECKING:
    from arc_me_sc.config import ClusteringParams


def run_clustering(
    adata: AnnData,
    params: "ClusteringParams",
    leiden: bool = True,
    louvain: bool = True,
    copy: bool = False,
) -> AnnData | None:
    """Leiden and/or Louvain on the precomputed neighbor graph."""
    if copy:
        adata = adata.copy()

    if leiden:
        sc.tl.leiden(
            adata,
            resolution=params.resolution,
            key_added=params.leiden_key,
            random_state=0,
        )
    if louvain:
        try:
            sc.tl.louvain(
                adata,
                resolution=params.resolution,
                key_added=params.louvain_key,
                random_state=0,
            )
        except ImportError as e:
            import warnings

            warnings.warn(
                "Louvain skipped (install optional dependency python-louvain). "
                f"Import error: {e}",
                stacklevel=2,
            )

    return adata if copy else None
