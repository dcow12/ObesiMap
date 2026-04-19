# arc_me_sc/viz/publication.py
#
# Publication-oriented figures for single-cell workflows (matplotlib + Scanpy).
# ================================================================================
# Typical exports: 300 DPI PDF/PNG for manuscripts; large fonts; colorblind-friendly palettes.
#
# Functions:
#   - apply_publication_matplotlib_style  — global rcParams for all following plots
#   - plot_umap_publication             — annotated UMAP (categorical obs column)
#   - plot_dotplot_publication          — marker gene dot plot (grouped by cluster / label)
#   - plot_volcano_de                   — DE table → volcano (use after Wilcoxon/DESeq2)
#
# ggplot2 users: see viz/ggplot2_templates.R for equivalent recipes in R.
# ------------------------------------------------------------------------------

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
from anndata import AnnData
from matplotlib.figure import Figure


def apply_publication_matplotlib_style(
    *,
    dpi: int = 300,
    font_size: int = 9,
    font_family: str = "sans-serif",
) -> None:
    """
    Set matplotlib **rcParams** suitable for multi-panel figure assembly.

    Call once at the start of a plotting session (or per script). Does not affect Scanpy's
    internal defaults unless ``scanpy.settings.set_figure_params`` is also set.
    """
    plt.rcParams.update(
        {
            "figure.dpi": dpi,
            "savefig.dpi": dpi,
            "font.size": font_size,
            "axes.titlesize": font_size + 1,
            "axes.labelsize": font_size,
            "xtick.labelsize": font_size - 1,
            "ytick.labelsize": font_size - 1,
            "legend.fontsize": font_size - 1,
            "font.family": font_family,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "lines.linewidth": 0.8,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "pdf.fonttype": 42,  # TrueType fonts in PDF (editable in Illustrator)
            "ps.fonttype": 42,
        }
    )


def save_figure(fig: Figure, path: str | Path, *, dpi: int | None = None) -> None:
    """Save figure to ``path`` (``.pdf`` or ``.png``); creates parent directories."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    kwargs = {"bbox_inches": "tight"}
    if dpi is not None:
        kwargs["dpi"] = dpi
    fig.savefig(path, **kwargs)


def plot_umap_publication(
    adata: AnnData,
    color_key: str,
    *,
    title: str | None = None,
    basis: str = "umap",
    palette: str = "husl",
    point_size: float = 8.0,
    figsize: tuple[float, float] = (6.2, 5.0),
    legend_outside: bool = True,
    frameon: bool = False,
) -> Figure:
    """
    **Annotated UMAP**: scatter colored by a categorical ``adata.obs[color_key]``.

    Uses ``adata.obsm[f'X_{basis}']`` (e.g. ``X_umap``). Categories are colored with a
    discrete seaborn palette; legend is placed outside the axes when ``legend_outside``.

    Parameters
    ----------
    color_key
        Column in ``adata.obs`` (e.g. ``\"leiden\"``, ``\"canonical_label\"``, ``condition``).
    title
        Panel title; defaults to ``color_key``.
    """
    emb_key = f"X_{basis}"
    if emb_key not in adata.obsm:
        raise KeyError(f"{emb_key} not in adata.obsm — run UMAP (Step 1) first.")

    coords = np.asarray(adata.obsm[emb_key])
    labels = adata.obs[color_key].astype(str)
    cats = sorted(labels.unique())
    n = len(cats)
    colors = sns.color_palette(palette, n_colors=max(n, 3))
    cat_to_color = {c: colors[i % len(colors)] for i, c in enumerate(cats)}

    fig, ax = plt.subplots(figsize=figsize)
    for i, cat in enumerate(cats):
        m = (labels == cat).values
        ax.scatter(
            coords[m, 0],
            coords[m, 1],
            c=[cat_to_color[cat]],
            s=point_size,
            alpha=0.85,
            linewidths=0,
            rasterized=True,
            label=cat,
        )

    ax.set_xlabel(f"{basis.upper()}1")
    ax.set_ylabel(f"{basis.upper()}2")
    ax.set_title(title if title is not None else color_key)
    ax.set_aspect("equal", adjustable="box")
    if not frameon:
        sns.despine(ax=ax, offset=2, trim=True)
    if legend_outside:
        ax.legend(
            bbox_to_anchor=(1.02, 1),
            loc="upper left",
            frameon=False,
            fontsize=plt.rcParams["legend.fontsize"],
        )
    else:
        ax.legend(frameon=False)
    fig.tight_layout()
    return fig


def plot_dotplot_publication(
    adata: AnnData,
    var_names: Sequence[str],
    groupby: str,
    *,
    figsize: tuple[float, float] | None = None,
    return_fig: bool = True,
    **dotplot_kw,
):
    """
    **Marker DotPlot** (Scanpy): mean expression + fraction expressing per ``groupby``.

    Wraps :func:`scanpy.pl.dotplot` with publication-friendly defaults. Extra keyword
    arguments are forwarded to Scanpy (e.g. ``dendrogram=True``, ``standard_scale='var'``).

    Returns
    -------
    If ``return_fig`` and Scanpy provides a figure object, returns that figure; else None.
    """
    genes = [g for g in var_names if g in adata.var_names]
    if not genes:
        raise ValueError("No requested genes are present in adata.var_names.")

    kw = dict(
        var_names=genes,
        groupby=groupby,
        return_fig=return_fig,
        show=False,
    )
    kw.update(dotplot_kw)
    if figsize is not None:
        kw["figsize"] = figsize

    dp = sc.pl.dotplot(adata, **kw)
    # Scanpy ≥1.8: DotPlot with .get_figure(); older may return dict of axes.
    fig = getattr(dp, "fig", None)
    if fig is None and hasattr(dp, "get_figure"):
        try:
            fig = dp.get_figure()
        except Exception:
            fig = None
    if fig is None and hasattr(dp, "fig"):
        fig = dp.fig
    return fig if fig is not None else dp


def plot_volcano_de(
    df: pd.DataFrame,
    *,
    logfc_col: str = "log2_fold_change",
    padj_col: str = "padj",
    gene_col: str | None = "gene",
    padj_threshold: float = 0.05,
    logfc_threshold: float = 0.5,
    title: str = "Differential expression",
    figsize: tuple[float, float] = (5.5, 5.0),
    label_top_n: int = 12,
) -> Figure:
    """
    **Volcano plot** from a DE results table (e.g. Lepr+ Fed vs Fasted).

    Points are colored by significance: |log2FC| and FDR thresholds. Optionally labels
    the top genes by |log2FC| among significant rows (requires ``gene_col``).

    Expected columns (rename if needed):
      - ``logfc_col``: log2 fold change (positive = up in “numerator” condition).
      - ``padj_col``: adjusted p-values (BH).
    """
    need = [logfc_col, padj_col]
    for c in need:
        if c not in df.columns:
            raise KeyError(f"Column {c!r} not in dataframe; have: {list(df.columns)}")

    d = df.copy()
    d["_log10p"] = -np.log10(np.clip(d[padj_col].astype(float), 1e-300, None))
    sig = (d[padj_col].astype(float) < padj_threshold) & (
        d[logfc_col].abs() > logfc_threshold
    )

    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(
        d.loc[~sig, logfc_col],
        d.loc[~sig, "_log10p"],
        c="#bdbdbd",
        s=10,
        alpha=0.6,
        linewidths=0,
        rasterized=True,
    )
    ax.scatter(
        d.loc[sig, logfc_col],
        d.loc[sig, "_log10p"],
        c="#e41a1c",
        s=14,
        alpha=0.85,
        linewidths=0,
        rasterized=True,
    )

    ax.axvline(-logfc_threshold, color="0.5", ls="--", lw=0.6)
    ax.axvline(logfc_threshold, color="0.5", ls="--", lw=0.6)
    ax.axhline(-np.log10(padj_threshold), color="0.5", ls="--", lw=0.6)

    ax.set_xlabel("log2 fold change")
    ax.set_ylabel(r"$-\log_{10}$ FDR")
    ax.set_title(title)
    sns.despine(ax=ax, offset=2, trim=True)

    if gene_col and gene_col in d.columns and label_top_n > 0:
        top = d.loc[sig].reindex(d.loc[sig][logfc_col].abs().sort_values(ascending=False).index).head(
            label_top_n
        )
        for _, row in top.iterrows():
            ax.annotate(
                str(row[gene_col]),
                (row[logfc_col], row["_log10p"]),
                fontsize=6,
                alpha=0.9,
                xytext=(3, 3),
                textcoords="offset points",
            )

    fig.tight_layout()
    return fig
