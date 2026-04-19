# arc_me_sc/pipeline/step_01_ingest_qc_dimred.py
#
# STEP 1 — Data ingestion, quality control, and dimensionality reduction
# =============================================================================
# Campbell et al. (2017) adult mouse Arcuate–ME (GSE93374 / SCP97) processed data:
#   - Expects log-normalized expression in expression.txt.gz (or .txt)
#   - Expects per-cell metadata in meta.txt (barcodes aligned with matrix)
#
# This module does NOT re-invent algorithms; it orchestrates:
#   1. Load matrix + metadata into AnnData (cells × genes).
#   2. Optional QC: min/max genes, mitochondrial %, optional MT-gene drop, Scrublet.
#   3. Embedding: highly variable genes → scale → PCA → kNN graph → UMAP.
#
# Downstream steps (clustering, DE, pathways) assume ``adata.obsm['X_umap']`` exists.
# -----------------------------------------------------------------------------

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from anndata import AnnData

from arc_me_sc.config import DataPaths, EmbeddingParams, QCParams, build_pipeline_config
from arc_me_sc.embedding import run_embedding
from arc_me_sc.io import load_arc_me, save_h5ad
from arc_me_sc.qc import run_qc

if TYPE_CHECKING:
    pass


def run_step_01(
    data_dir: str | Path,
    *,
    qc: QCParams | None = None,
    emb: EmbeddingParams | None = None,
    paths: DataPaths | None = None,
    random_state: int = 0,
) -> AnnData:
    """
    Execute **Step 1**: load data → QC → PCA + UMAP (+ neighbors).

    Parameters
    ----------
    data_dir
        Directory containing ``expression.txt.gz`` (or configured name) and ``meta.txt``.
    qc, emb, paths
        If ``None``, defaults are taken from :func:`arc_me_sc.config.build_pipeline_config`.
        Pass custom :class:`~arc_me_sc.config.QCParams` to tighten or relax filters on
        already-processed matrices (e.g. disable MT filtering for batch-corrected data).
    random_state
        Passed to doublet removal (Scrublet) when enabled.

    Returns
    -------
    AnnData
        ``adata.X`` = expression used for embedding; ``adata.obsm['X_pca']``,
        ``adata.obsm['X_umap']``, ``adata.uns['neighbors']`` populated.

    Notes
    -----
    - Mitochondrial filtering uses mouse-style prefixes ``mt-``, ``Mt-`` (see ``qc.py``).
    - For purely exploratory QC on published matrices, keep ``max_pct_mitochondrial=None``.
    """
    # Resolve paths: either caller-supplied DataPaths or default filenames under data_dir.
    if paths is None:
        paths, default_qc, default_emb, _ = build_pipeline_config(data_dir=data_dir)
        if qc is None:
            qc = default_qc
        if emb is None:
            emb = default_emb
    else:
        if qc is None or emb is None:
            _, default_qc, default_emb, _ = build_pipeline_config()
            qc = qc or default_qc
            emb = emb or default_emb

    # --- Load: inner-join cells between expression matrix and metadata (see io.load_arc_me).
    adata: AnnData = load_arc_me(paths.expression_path, paths.meta_path)

    # --- QC: parameterized filters; for GSE93374 processed data, defaults are permissive.
    adata = run_qc(adata, qc, random_state=random_state)

    # --- Embedding: HVG / scale / PCA / neighbors / UMAP (log-normalized input assumed).
    run_embedding(adata, emb)

    return adata


def run_step_01_and_save(
    data_dir: str | Path,
    out_h5ad: str | Path,
    **kwargs,
) -> AnnData:
    """
    Run :func:`run_step_01` and atomically persist the AnnData for checkpointing.

    ``out_h5ad`` parent directories are created if missing.
    """
    adata = run_step_01(data_dir, **kwargs)
    save_h5ad(adata, Path(out_h5ad))
    return adata
