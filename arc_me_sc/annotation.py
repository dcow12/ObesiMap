"""Marker-gene scoring and canonical Arc-ME population labels."""

from __future__ import annotations

import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData


def score_marker_sets(
    adata: AnnData,
    marker_sets: dict[str, list[str]],
    score_prefix: str = "score_",
) -> AnnData:
    """``sc.tl.score_genes`` for each named population; missing genes are skipped."""
    for name, genes in marker_sets.items():
        present = [g for g in genes if g in adata.var_names]
        if not present:
            adata.obs[f"{score_prefix}{name}"] = np.nan
            continue
        sc.tl.score_genes(
            adata,
            gene_list=present,
            score_name=f"{score_prefix}{name}",
            use_raw=False,
        )
    return adata


def assign_labels_by_max_score(
    adata: AnnData,
    marker_sets: dict[str, list[str]],
    score_prefix: str = "score_",
    out_key: str = "canonical_label",
    min_score: float | None = None,
) -> AnnData:
    """
    Assign each cell the population with highest marker score (ties -> first column order).
    """
    cols = [f"{score_prefix}{k}" for k in marker_sets if f"{score_prefix}{k}" in adata.obs.columns]
    if not cols:
        adata.obs[out_key] = "unknown"
        return adata

    mat = adata.obs[cols].astype(float)
    if min_score is not None:
        maxv = mat.max(axis=1)
        labels = mat.idxmax(axis=1).where(maxv >= min_score)
        adata.obs[out_key] = labels.map(
            lambda x: x.replace(score_prefix, "") if isinstance(x, str) else "low_confidence"
        )
    else:
        best = mat.idxmax(axis=1)
        adata.obs[out_key] = best.map(lambda x: x.replace(score_prefix, ""))

    return adata


def cluster_marker_summary(
    adata: AnnData,
    groupby: str,
    score_prefix: str = "score_",
) -> pd.DataFrame:
    """Mean marker scores per cluster (for heatmaps / dot plots)."""
    cols = [c for c in adata.obs.columns if c.startswith(score_prefix)]
    if not cols:
        return pd.DataFrame()
    return (
        adata.obs.groupby(groupby, observed=True)[cols]
        .mean()
        .sort_index()
    )


def plot_canonical_markers(
    adata: AnnData,
    marker_sets: dict[str, list[str]],
    *,
    leiden_key: str = "leiden",
    save: str | None = None,
    show: bool = True,
):
    """UMAP colored by Leiden and dot plot of key markers."""
    flat_markers: list[str] = []
    for genes in marker_sets.values():
        flat_markers.extend([g for g in genes if g in adata.var_names])
    flat_markers = list(dict.fromkeys(flat_markers))

    sc.pl.umap(adata, color=[leiden_key], show=show, save=save)
    if flat_markers:
        sc.pl.umap(
            adata,
            color=flat_markers[: min(len(flat_markers), 12)],
            ncols=4,
            show=show,
            save=False if save is None else "_markers",
        )
        sc.pl.dotplot(
            adata,
            var_names=flat_markers,
            groupby=leiden_key,
            show=show,
            save=False if save is None else "_dotplot",
        )
