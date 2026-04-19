"""Lepr+ cell subsetting, co-expression, and lineage characterization (Campbell et al. 2017 context)."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData
from scipy import sparse


def gene_expression_vector(
    adata: AnnData,
    gene: str,
    *,
    layer: str | None = None,
) -> np.ndarray:
    """1D expression vector for ``gene`` in ``adata`` observation order."""
    if gene not in adata.var_names:
        raise KeyError(f"Gene {gene!r} not found in adata.var_names")
    idx = int(adata.var_names.get_loc(gene))
    X = adata.layers[layer] if layer is not None else adata.X
    col = X[:, idx]
    if sparse.issparse(col):
        return np.asarray(col.todense()).ravel()
    return np.asarray(col).ravel()


def annotate_lepr_status(
    adata: AnnData,
    *,
    gene: str = "Agrp",
    layer: str | None = None,
    obs_key: str = "lepr_positive",
    expr_obs_key: str = "lepr_expression",
    mode: Literal["detected", "threshold", "quantile"] = "detected",
    min_expr: float = 0.0,
    quantile: float = 0.9,
    quantile_over: Literal["positive", "all"] = "positive",
) -> AnnData:
    """
    Flag Lepr-expressing cells in ``adata.obs``.

    For log-normalized matrices, ``mode='detected'`` with ``min_expr=0`` treats any
    expression above zero as detected. Use ``mode='quantile'`` to take the top
    fraction of cells (among those with Lepr > 0, or all cells).
    """
    x = gene_expression_vector(adata, gene, layer=layer)
    adata.obs[expr_obs_key] = x

    if mode == "detected":
        adata.obs[obs_key] = x > min_expr
    elif mode == "threshold":
        adata.obs[obs_key] = x >= min_expr
    elif mode == "quantile":
        if quantile_over == "positive":
            mask = x > min_expr
            if not np.any(mask):
                adata.obs[obs_key] = False
            else:
                thr = float(np.quantile(x[mask], quantile))
                adata.obs[obs_key] = x >= thr
        else:
            thr = float(np.quantile(x, quantile))
            adata.obs[obs_key] = x >= thr
    else:
        raise ValueError(f"Unknown mode: {mode!r}")

    adata.obs[obs_key] = adata.obs[obs_key].fillna(False).astype(bool)
    return adata


def subset_lepr_positive(
    adata: AnnData,
    *,
    gene: str = "Agrp",
    layer: str | None = None,
    obs_key: str = "lepr_positive",
    copy: bool = True,
    **annotate_kw,
) -> AnnData:
    """
    Return cells with Lepr expression according to :func:`annotate_lepr_status`.

    Any keyword arguments are forwarded to :func:`annotate_lepr_status` (e.g.
    ``mode``, ``min_expr``, ``quantile``). If ``obs_key`` is already present
    and ``annotate_kw`` is empty, existing labels are reused.
    """
    if obs_key not in adata.obs.columns or annotate_kw:
        annotate_lepr_status(
            adata,
            gene=gene,
            layer=layer,
            obs_key=obs_key,
            **annotate_kw,
        )
    mask = adata.obs[obs_key].astype(bool).to_numpy()
    out = adata[mask]
    return out.copy() if copy else out


def coexpression_matrix(
    adata: AnnData,
    genes: list[str],
    *,
    layer: str | None = None,
    cells: Literal["lepr", "all"] = "lepr",
    lepr_obs_key: str = "lepr_positive",
    method: Literal["pearson", "spearman"] = "pearson",
) -> pd.DataFrame:
    """
    Pairwise correlation of ``genes`` across cells (Lepr+ only or full object).

    Uses dense columns for the selected genes; intended for modest panel sizes
    (dozens of genes), not whole-transcriptome matrices.
    """
    present = [g for g in genes if g in adata.var_names]
    if len(present) < 2:
        return pd.DataFrame()

    sub = adata
    if cells == "lepr":
        if lepr_obs_key not in adata.obs.columns:
            raise KeyError(
                f"{lepr_obs_key!r} not in obs; run annotate_lepr_status first."
            )
        sub = adata[adata.obs[lepr_obs_key].astype(bool)]

    if sub.n_obs < 3:
        return pd.DataFrame(index=present, columns=present, dtype=float)

    X = sub[:, present].X
    if sparse.issparse(X):
        mat = X.toarray()
    else:
        mat = np.asarray(X)

    df = pd.DataFrame(mat, columns=present)
    if method == "pearson":
        return df.corr(method="pearson")
    if method == "spearman":
        return df.corr(method="spearman")
    raise ValueError(f"Unknown method: {method!r}")


def mean_expression_lepr_vs_rest(
    adata: AnnData,
    genes: list[str],
    *,
    layer: str | None = None,
    lepr_obs_key: str = "lepr_positive",
) -> pd.DataFrame:
    """Mean expression of ``genes`` in Lepr+ vs Lepr− cells."""
    if lepr_obs_key not in adata.obs.columns:
        raise KeyError(f"{lepr_obs_key!r} not in obs; run annotate_lepr_status first.")
    present = [g for g in genes if g in adata.var_names]
    if not present:
        return pd.DataFrame()

    pos = adata[adata.obs[lepr_obs_key].astype(bool)]
    neg = adata[~adata.obs[lepr_obs_key].astype(bool)]

    def _mean(a: AnnData) -> pd.Series:
        if a.n_obs == 0:
            return pd.Series(np.nan, index=present, dtype=float)
        X = a[:, present].X
        if sparse.issparse(X):
            m = np.asarray(X.mean(axis=0)).ravel()
        else:
            m = np.mean(np.asarray(X), axis=0)
        return pd.Series(m, index=present)

    return pd.DataFrame({"Lepr_positive": _mean(pos), "Lepr_negative": _mean(neg)})


def lepr_canonical_composition(
    adata: AnnData,
    *,
    label_key: str = "canonical_label",
    lepr_obs_key: str = "lepr_positive",
) -> pd.Series:
    """Fraction of Lepr+ cells in each ``label_key`` category (of all Lepr+ cells)."""
    if lepr_obs_key not in adata.obs.columns:
        raise KeyError(f"{lepr_obs_key!r} not in obs")
    if label_key not in adata.obs.columns:
        raise KeyError(f"{label_key!r} not in obs")

    sub = adata.obs.loc[adata.obs[lepr_obs_key].astype(bool)]
    if sub.empty:
        return pd.Series(dtype=float)
    counts = sub[label_key].value_counts(normalize=True)
    return counts.sort_values(ascending=False)


def plot_lepr_feature_umap(
    adata: AnnData,
    genes: list[str],
    *,
    extra_obs: list[str] | None = None,
    ncols: int = 4,
    wspace: float = 0.35,
    hspace: float = 0.4,
    save: str | None = None,
    show: bool = True,
    vmin: float | None = None,
    vmax: float | None = None,
):
    """
    Scanpy UMAP “FeaturePlots”: expression of ``genes`` (+ optional ``obs`` columns).

    ``vmin`` / ``vmax`` apply to gene colors only when passed to Scanpy
    (per-gene control is available via repeated calls if needed).
    """
    present_genes = [g for g in genes if g in adata.var_names]
    extra = list(extra_obs or [])
    color = present_genes + extra
    if not color:
        return
    color = [c for c in color if c in adata.var_names or c in adata.obs.columns]
    if not color:
        return
    kw: dict = dict(ncols=ncols, wspace=wspace, hspace=hspace, show=show)
    if save is not None:
        kw["save"] = save
    if vmin is not None:
        kw["vmin"] = vmin
    if vmax is not None:
        kw["vmax"] = vmax
    sc.pl.umap(adata, color=color, **kw)


def plot_lepr_coexpression_heatmap(
    corr: pd.DataFrame,
    *,
    title: str = "Co-expression (Lepr+ cells)",
    save: str | None = None,
    show: bool = True,
):
    """Heatmap of a correlation matrix from :func:`coexpression_matrix`."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(max(4, 0.35 * len(corr.columns)), max(3.5, 0.35 * len(corr.index))))
    im = ax.imshow(corr.values, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(corr.index)))
    ax.set_yticklabels(corr.index)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=150, bbox_inches="tight")
    if not show:
        plt.close(fig)


def run_lepr_characterization(
    adata: AnnData,
    lineage_genes: list[str],
    *,
    label_key: str = "canonical_label",
    leiden_key: str = "leiden",
    layer: str | None = None,
    figdir: str | Path | None = None,
    save_prefix: str = "lepr",
    show_plots: bool = False,
    **annotate_kw,
) -> dict[str, object]:
    """
    Annotate Lepr status, compute co-expression / contrasts, save tables and figures.

    Returns a dict with DataFrames and paths for downstream use.
    """
    annotate_lepr_status(adata, layer=layer, **annotate_kw)

    panel = [g for g in dict.fromkeys(lineage_genes) if g in adata.var_names]
    if "Lepr" in adata.var_names:
        panel = ["Lepr"] + [g for g in panel if g != "Lepr"]

    corr = coexpression_matrix(
        adata, panel, layer=layer, cells="lepr", method="pearson"
    )
    means = mean_expression_lepr_vs_rest(adata, panel, layer=layer)
    try:
        comp = lepr_canonical_composition(adata, label_key=label_key)
    except KeyError:
        comp = pd.Series(dtype=float)

    out: dict[str, object] = {
        "coexpression_pearson_lepr_plus": corr,
        "mean_expr_lepr_vs_rest": means,
        "lepr_plus_canonical_fractions": comp,
    }

    if figdir is not None:
        p = Path(figdir)
        p.mkdir(parents=True, exist_ok=True)
        sc.settings.figdir = str(p)

        extra_obs = [
            x
            for x in (
                "lepr_positive",
                "lepr_expression",
                label_key,
                leiden_key,
            )
            if x in adata.obs.columns
        ]
        genes_for_umap = panel[:16]
        plot_lepr_feature_umap(
            adata,
            genes_for_umap,
            extra_obs=extra_obs,
            save=f"_{save_prefix}_feature_umap",
            show=show_plots,
        )
        if not corr.empty:
            plot_lepr_coexpression_heatmap(
                corr,
                title="Pearson correlation (Lepr+ cells)",
                save=str(p / f"{save_prefix}_coexpression_pearson.png"),
                show=show_plots,
            )
        if "lepr_positive" in adata.obs.columns and panel:
            pass
            #sc.pl.dotplot(
            #    adata,
            #    var_names=panel[:20],
             #   groupby="lepr_positive",
             #   show=show_plots,
             #   save=f"_{save_prefix}_dotplot_lepr_flag",
           #)

    return out
