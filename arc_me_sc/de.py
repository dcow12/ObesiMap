"""Differential expression within Lepr+ cells (Wilcoxon or pseudobulk DESeq2)."""

from __future__ import annotations

from typing import Literal, Sequence

import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData
from scipy import sparse

from arc_me_sc.config import LEPTIN_SIGNALING_DE_GENES
from arc_me_sc.lepr import annotate_lepr_status


def _ensure_lepr_labels(
    adata: AnnData,
    lepr_obs_key: str,
    lepr_kw: dict | None,
) -> None:
    if lepr_obs_key in adata.obs.columns and not lepr_kw:
        return
    annotate_lepr_status(adata, obs_key=lepr_obs_key, **(lepr_kw or {}))


def _subset_lepr_condition(
    adata: AnnData,
    *,
    condition_key: str,
    group_pos: str,
    group_neg: str,
    lepr_obs_key: str,
    stratify_value: str | None = None,
    stratify_by: str | None = None,
) -> AnnData:
    obs = adata.obs
    cond = obs[condition_key]
    if not pd.api.types.is_string_dtype(cond):
        cond = cond.astype(str)
    m = (
        obs[lepr_obs_key].astype(bool)
        & cond.isin([group_pos, group_neg])
    )
    if stratify_by is not None and stratify_value is not None:
        m &= obs[stratify_by].astype(str) == str(stratify_value)
    sub = adata[m].copy()
    if sub.n_obs == 0:
        return sub
    sub.obs["_de_group"] = cond.loc[sub.obs_names].astype(str)
    return sub


def _mean_expression(
    adata: AnnData,
    gene: str,
    mask: np.ndarray,
    layer: str | None,
) -> float:
    if gene not in adata.var_names:
        return float("nan")
    idx = int(adata.var_names.get_loc(gene))
    X = adata.layers[layer] if layer is not None else adata.X
    sub = X[mask, idx]
    if sparse.issparse(sub):
        return float(np.asarray(sub.mean()).ravel()[0])
    return float(np.mean(sub))


def _log2_fold_change_means(
    adata: AnnData,
    gene: str,
    *,
    group_pos: str,
    group_neg: str,
    layer: str | None,
    pseudocount: float = 1e-6,
) -> float:
    g = adata.obs["_de_group"].astype(str)
    m_pos = (g == group_pos).to_numpy()
    m_neg = (g == group_neg).to_numpy()
    a = _mean_expression(adata, gene, m_pos, layer)
    b = _mean_expression(adata, gene, m_neg, layer)
    if not np.isfinite(a) or not np.isfinite(b):
        return float("nan")
    return float(np.log2((a + pseudocount) / (b + pseudocount)))


def de_wilcoxon_lepr_stratum(
    adata: AnnData,
    *,
    condition_key: str,
    group_pos: str,
    group_neg: str,
    lepr_obs_key: str = "lepr_positive",
    layer: str | None = None,
    target_genes: Sequence[str],
    stratify_by: str | None = None,
    stratify_value: str | None = None,
    min_cells_per_group: int = 5,
) -> pd.DataFrame:
    """
    Wilcoxon rank-sum (Scanpy) for ``group_pos`` vs ``group_neg`` on one stratum.

    Log2 fold change is mean(group_pos) / mean(group_neg) with a small pseudocount.
    Adjusted p-values come from Scanpy's Benjamini–Hochberg correction over all genes
    tested; rows are filtered to ``target_genes``.
    """
    sub = _subset_lepr_condition(
        adata,
        condition_key=condition_key,
        group_pos=group_pos,
        group_neg=group_neg,
        lepr_obs_key=lepr_obs_key,
        stratify_by=stratify_by,
        stratify_value=stratify_value,
    )
    rows: list[dict] = []
    if sub.n_obs == 0:
        for g in target_genes:
            rows.append(
                {
                    "gene": g,
                    "log2_fold_change": np.nan,
                    "padj": np.nan,
                    "pval": np.nan,
                    "n_cells_pos": 0,
                    "n_cells_neg": 0,
                    "method": "wilcoxon",
                    "stratum": stratify_value if stratify_by else "all_Lepr_plus",
                }
            )
        return pd.DataFrame(rows)

    g = sub.obs["_de_group"].astype(str)
    n_pos = int((g == group_pos).sum())
    n_neg = int((g == group_neg).sum())
    if g.nunique() < 2:
        for gene in target_genes:
            rows.append(
                {
                    "gene": gene,
                    "log2_fold_change": _log2_fold_change_means(
                        sub, gene, group_pos=group_pos, group_neg=group_neg, layer=layer
                    ),
                    "padj": np.nan,
                    "pval": np.nan,
                    "n_cells_pos": n_pos,
                    "n_cells_neg": n_neg,
                    "method": "wilcoxon",
                    "stratum": stratify_value if stratify_by else "all_Lepr_plus",
                    "note": "single_condition_in_stratum",
                }
            )
        return pd.DataFrame(rows)

    if n_pos < min_cells_per_group or n_neg < min_cells_per_group:
        for gene in target_genes:
            rows.append(
                {
                    "gene": gene,
                    "log2_fold_change": _log2_fold_change_means(
                        sub, gene, group_pos=group_pos, group_neg=group_neg, layer=layer
                    ),
                    "padj": np.nan,
                    "pval": np.nan,
                    "n_cells_pos": n_pos,
                    "n_cells_neg": n_neg,
                    "method": "wilcoxon",
                    "stratum": stratify_value if stratify_by else "all_Lepr_plus",
                    "note": "insufficient_cells_per_group",
                }
            )
        return pd.DataFrame(rows)

    # rank_genes_groups uses adata.X; optionally test on a normalized layer
    if layer is not None:
        sub = sub.copy()
        L = sub.layers[layer]
        sub.X = L.copy() if sparse.issparse(L) else np.asarray(L, dtype=np.float64)

    sc.tl.rank_genes_groups(
        sub,
        groupby="_de_group",
        groups=[group_pos],
        reference=group_neg,
        method="wilcoxon",
        use_raw=False,
    )
    df = sc.get.rank_genes_groups_df(sub, group=group_pos)
    df = df.set_index("names", drop=False)

    stratum_label = stratify_value if stratify_by else "all_Lepr_plus"
    for gene in target_genes:
        log2_fc = _log2_fold_change_means(
            sub, gene, group_pos=group_pos, group_neg=group_neg, layer=None
        )
        if gene in df.index:
            r = df.loc[gene]
            rows.append(
                {
                    "gene": gene,
                    "log2_fold_change": log2_fc,
                    "padj": float(r["pvals_adj"]),
                    "pval": float(r["pvals"]),
                    "n_cells_pos": n_pos,
                    "n_cells_neg": n_neg,
                    "method": "wilcoxon",
                    "stratum": stratum_label,
                }
            )
        else:
            rows.append(
                {
                    "gene": gene,
                    "log2_fold_change": log2_fc,
                    "padj": np.nan,
                    "pval": np.nan,
                    "n_cells_pos": n_pos,
                    "n_cells_neg": n_neg,
                    "method": "wilcoxon",
                    "stratum": stratum_label,
                    "note": "gene_not_in_rank_genes_result",
                }
            )
    return pd.DataFrame(rows)


def _pseudobulk_sum_counts(
    adata: AnnData,
    *,
    sample_col: str,
    counts_layer: str,
) -> pd.DataFrame:
    """Pseudobulk matrix: rows = samples (``sample_col``), columns = genes."""
    if counts_layer not in adata.layers:
        raise KeyError(
            f"Layer {counts_layer!r} not found. Pseudobulk DESeq2 requires integer counts."
        )
    X = adata.layers[counts_layer]
    if sparse.issparse(X):
        X = X.toarray()
    else:
        X = np.asarray(X)
    genes = list(adata.var_names.astype(str))
    df = pd.DataFrame(X, columns=genes)
    df["_sample"] = adata.obs[sample_col].astype(str).values
    return df.groupby("_sample", observed=True, sort=False).sum(numeric_only=True)


def de_pseudobulk_deseq2_lepr(
    adata: AnnData,
    *,
    condition_key: str,
    group_pos: str,
    group_neg: str,
    lepr_obs_key: str = "lepr_positive",
    sample_col: str,
    counts_layer: str,
    target_genes: Sequence[str],
    min_total_counts_gene: int = 10,
    n_cpus: int | None = None,
) -> pd.DataFrame:
    """
    Pseudobulk DESeq2 (PyDESeq2) on summed raw counts per ``sample_col``, Lepr+ only.

    Contrast is ``group_pos`` vs ``group_neg`` (log2 fold change is DESeq2's
    ``log2FoldChange``). Requires at least two pseudobulk samples per group for stable
    dispersion fitting.
    """
    try:
        from pydeseq2.dds import DeseqDataSet
        from pydeseq2.default_inference import DefaultInference
        from pydeseq2.ds import DeseqStats
    except ImportError as e:
        raise ImportError(
            "Pseudobulk DESeq2 requires pydeseq2. Install with: pip install pydeseq2"
        ) from e

    sub = _subset_lepr_condition(
        adata,
        condition_key=condition_key,
        group_pos=group_pos,
        group_neg=group_neg,
        lepr_obs_key=lepr_obs_key,
    )
    if sub.n_obs == 0:
        return pd.DataFrame()

    pb = _pseudobulk_sum_counts(sub, sample_col=sample_col, counts_layer=counts_layer)
    meta = (
        sub.obs.groupby(sample_col, observed=True)[condition_key]
        .agg(lambda s: s.iloc[0])
        .loc[pb.index]
        .to_frame(name=condition_key)
    )
    meta[condition_key] = meta[condition_key].astype(str)

    keep_samples = meta[condition_key].isin([group_pos, group_neg])
    pb = pb.loc[keep_samples]
    meta = meta.loc[keep_samples]

    genes_keep = pb.columns[pb.sum(axis=0) >= min_total_counts_gene]
    pb = pb[genes_keep].astype(np.int64)

    n_pos = int((meta[condition_key] == group_pos).sum())
    n_neg = int((meta[condition_key] == group_neg).sum())
    if n_pos < 2 or n_neg < 2:
        raise ValueError(
            f"DESeq2 needs ≥2 pseudobulk samples per arm; got {group_pos}={n_pos}, "
            f"{group_neg}={n_neg}. Check {sample_col!r} replicates."
        )

    inference = DefaultInference(n_cpus=n_cpus)
    dds = DeseqDataSet(
        counts=pb,
        metadata=meta,
        design=f"~{condition_key}",
        refit_cooks=True,
        inference=inference,
    )
    dds.deseq2()

    ds = DeseqStats(
        dds,
        contrast=[condition_key, group_pos, group_neg],
        inference=inference,
    )
    ds.summary()
    res = ds.results_df
    if res is None or res.empty:
        return pd.DataFrame()

    rows = []
    stratum_label = "all_Lepr_plus"
    for gene in target_genes:
        if gene not in res.index:
            rows.append(
                {
                    "gene": gene,
                    "log2_fold_change": np.nan,
                    "padj": np.nan,
                    "pval": np.nan,
                    "n_pseudobulk_pos": n_pos,
                    "n_pseudobulk_neg": n_neg,
                    "method": "deseq2_pseudobulk",
                    "stratum": stratum_label,
                    "note": "gene_missing_from_deseq2_result",
                }
            )
            continue
        r = res.loc[gene]
        rows.append(
            {
                "gene": gene,
                "log2_fold_change": float(r.get("log2FoldChange", np.nan)),
                "padj": float(r.get("padj", np.nan)),
                "pval": float(r.get("pvalue", np.nan)),
                "n_pseudobulk_pos": n_pos,
                "n_pseudobulk_neg": n_neg,
                "method": "deseq2_pseudobulk",
                "stratum": stratum_label,
            }
        )
    return pd.DataFrame(rows)


def differential_expression_lepr_conditions(
    adata: AnnData,
    *,
    condition_key: str,
    group_pos: str,
    group_neg: str,
    method: Literal["wilcoxon", "deseq2"] = "wilcoxon",
    lepr_obs_key: str = "lepr_positive",
    lepr_annotate_kw: dict | None = None,
    expression_layer: str | None = None,
    counts_layer: str = "counts",
    sample_col: str | None = None,
    target_genes: Sequence[str] | None = None,
    stratify_by: str | None = None,
    min_cells_per_group: int = 5,
    deseq2_n_cpus: int | None = None,
) -> pd.DataFrame:
    """
    Compare ``group_pos`` vs ``group_neg`` in ``condition_key`` using only Lepr+ cells.

    If ``stratify_by`` is set (e.g. ``\"leiden\"``), runs one test per cluster value
    (Wilcoxon only). DESeq2 is only supported for unstratified analysis (global
    Lepr+ pseudobulk).

    Returns a table with ``gene``, ``log2_fold_change``, ``padj``, ``method``, and
    ``stratum``.
    """
    genes = list(target_genes) if target_genes is not None else list(LEPTIN_SIGNALING_DE_GENES)
    _ensure_lepr_labels(adata, lepr_obs_key, lepr_annotate_kw)

    if method == "deseq2":
        if stratify_by is not None:
            raise ValueError("stratify_by is not supported with method='deseq2'.")
        if not sample_col:
            raise ValueError("sample_col is required for method='deseq2'.")
        return de_pseudobulk_deseq2_lepr(
            adata,
            condition_key=condition_key,
            group_pos=group_pos,
            group_neg=group_neg,
            lepr_obs_key=lepr_obs_key,
            sample_col=sample_col,
            counts_layer=counts_layer,
            target_genes=genes,
            n_cpus=deseq2_n_cpus,
        )

    if stratify_by is None:
        return de_wilcoxon_lepr_stratum(
            adata,
            condition_key=condition_key,
            group_pos=group_pos,
            group_neg=group_neg,
            lepr_obs_key=lepr_obs_key,
            layer=expression_layer,
            target_genes=genes,
            stratify_by=None,
            stratify_value=None,
            min_cells_per_group=min_cells_per_group,
        )

    parts = []
    for val in sorted(adata.obs[stratify_by].astype(str).unique()):
        part = de_wilcoxon_lepr_stratum(
            adata,
            condition_key=condition_key,
            group_pos=group_pos,
            group_neg=group_neg,
            lepr_obs_key=lepr_obs_key,
            layer=expression_layer,
            target_genes=genes,
            stratify_by=stratify_by,
            stratify_value=val,
            min_cells_per_group=min_cells_per_group,
        )
        parts.append(part)
    return pd.concat(parts, ignore_index=True)
