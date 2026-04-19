"""Quality control filters and optional doublet removal."""

from __future__ import annotations

from typing import TYPE_CHECKING

import scanpy as sc
from anndata import AnnData

if TYPE_CHECKING:
    from arc_me_sc.config import QCParams


def annotate_mitochondrial_fraction(
    adata: AnnData,
    mitochondrial_prefix: str = "mt-",
    colname: str = "pct_counts_mt",
) -> AnnData:
    """Compute percentage of counts from mitochondrial genes (mouse: mt-*, Mt-*)."""
    adata.var["mt"] = (
        adata.var_names.str.startswith(mitochondrial_prefix)
        | adata.var_names.str.startswith(mitochondrial_prefix.upper())
        | adata.var_names.str.startswith("Mt-")
    )
    sc.pp.calculate_qc_metrics(
        adata,
        qc_vars=["mt"],
        percent_top=None,
        log1p=False,
        inplace=True,
    )
    if colname != "pct_counts_mt" and "pct_counts_mt" in adata.obs.columns:
        adata.obs[colname] = adata.obs["pct_counts_mt"]
    return adata


def filter_cells(
    adata: AnnData,
    params: "QCParams",
) -> AnnData:
    """Apply gene-count and optional mitochondrial filters."""
    sc.pp.filter_cells(adata, min_genes=params.min_genes)
    if params.max_genes is not None:
        if "n_genes_by_counts" not in adata.obs.columns:
            sc.pp.calculate_qc_metrics(adata, inplace=True)
        adata = adata[adata.obs["n_genes_by_counts"] <= params.max_genes].copy()

    if params.max_pct_mitochondrial is not None:
        if "pct_counts_mt" not in adata.obs.columns:
            annotate_mitochondrial_fraction(
                adata,
                mitochondrial_prefix=params.mitochondrial_prefix,
            )
        adata = adata[adata.obs["pct_counts_mt"] <= params.max_pct_mitochondrial].copy()

    return adata


def drop_mitochondrial_genes(
    adata: AnnData,
    mitochondrial_prefix: str = "mt-",
) -> AnnData:
    """Remove mitochondrial genes from ``adata.var`` (optional downstream step)."""
    keep = ~(
        adata.var_names.str.startswith(mitochondrial_prefix)
        | adata.var_names.str.startswith(mitochondrial_prefix.upper())
        | adata.var_names.str.startswith("Mt-")
    )
    return adata[:, keep].copy()


def remove_doublets_scrublet(
    adata: AnnData,
    expected_doublet_rate: float = 0.06,
    random_state: int = 0,
) -> AnnData:
    """Run Scrublet and filter predicted doublets (log-normalized data)."""
    import scanpy.external as sce

    sce.pp.scrublet(
        adata,
        expected_doublet_rate=expected_doublet_rate,
        random_state=random_state,
    )
    if "predicted_doublet" not in adata.obs.columns:
        raise RuntimeError("Scrublet did not add `predicted_doublet` to obs.")
    return adata[~adata.obs["predicted_doublet"].astype(bool)].copy()


def run_qc(
    adata: AnnData,
    params: "QCParams",
    random_state: int = 0,
) -> AnnData:
    """
    QC: optional MT annotation, cell/gene filters, optional doublet removal.

    For preprocessed log-normalized matrices, doublet removal is off by default.
    """
    if params.max_pct_mitochondrial is not None or params.filter_mitochondrial_genes:
        annotate_mitochondrial_fraction(
            adata,
            mitochondrial_prefix=params.mitochondrial_prefix,
        )

    adata = filter_cells(adata, params)

    if params.remove_doublets:
        if params.doublet_method.lower() == "scrublet":
            adata = remove_doublets_scrublet(
                adata,
                expected_doublet_rate=params.scrublet_expected_doublet_rate,
                random_state=random_state,
            )
        else:
            raise ValueError(f"Unknown doublet_method: {params.doublet_method!r}")

    if params.filter_mitochondrial_genes:
        adata = drop_mitochondrial_genes(
            adata,
            mitochondrial_prefix=params.mitochondrial_prefix,
        )

    return adata
