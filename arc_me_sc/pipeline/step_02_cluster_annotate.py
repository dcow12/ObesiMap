# arc_me_sc/pipeline/step_02_cluster_annotate.py
#
# STEP 2 — Unsupervised clustering and canonical cell-type annotation
# =====================================================================
# Prerequisites: Step 1 completed (``adata.obsm['X_umap']``, neighbor graph present).
#
# This step:
#   1. Runs Leiden (and optionally Louvain) on the kNN graph from Step 1.
#   2. Scores predefined marker gene sets (AgRP, POMC, glutamatergic, GABAergic, tanycytes/glia).
#   3. Assigns a coarse ``canonical_label`` per cell from the highest marker module score.
#
# Annotation is heuristic: use with dot plots and literature markers; refine clusters as needed.
# -------------------------------------------------------------------------------------------

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from anndata import AnnData

from arc_me_sc.annotation import assign_labels_by_max_score, score_marker_sets
from arc_me_sc.clustering import run_clustering
from arc_me_sc.config import MARKER_SETS, ClusteringParams

if TYPE_CHECKING:
    pass


def run_step_02(
    adata: AnnData,
    *,
    clustering: ClusteringParams | None = None,
    marker_sets: dict[str, list[str]] | None = None,
    run_louvain: bool = True,
) -> AnnData:
    """
    Execute **Step 2**: graph clustering + marker scoring + canonical labels.

    Parameters
    ----------
    adata
        Object after Step 1 (must contain the neighbor graph and PCA/UMAP).
    clustering
        Resolution and obs keys for Leiden/Louvain. Defaults from :class:`ClusteringParams`.
    marker_sets
        Gene-symbol dictionaries per population; defaults to :data:`arc_me_sc.config.MARKER_SETS`.
    run_louvain
        If ``True``, runs Louvain when the optional ``python-louvain`` package is available.

    Returns
    -------
    AnnData
        In-place updates: ``adata.obs['leiden']``, optional ``louvain``, ``score_*`` columns,
        and ``canonical_label``.

    Notes
    -----
    - Leiden resolution ~0.4–0.8 is typical for Arc data; tune ``ClusteringParams.resolution``.
    - Missing genes in a module are skipped silently in :func:`score_marker_sets`.
    """
    params = clustering or ClusteringParams()
    markers = marker_sets if marker_sets is not None else MARKER_SETS

    # Unsupervised clustering on the precomputed neighbor graph (Step 1).
    run_clustering(adata, params, leiden=True, louvain=run_louvain)

    # Module scores then winner-take-all label for visualization and downstream masks.
    score_marker_sets(adata, markers)
    assign_labels_by_max_score(adata, markers)

    return adata


def run_step_02_and_save(
    adata: AnnData,
    out_h5ad: str | Path,
    **kwargs,
) -> AnnData:
    """Run :func:`run_step_02` and write ``out_h5ad``."""
    adata = run_step_02(adata, **kwargs)
    from arc_me_sc.io import save_h5ad

    save_h5ad(adata, Path(out_h5ad))
    return adata
