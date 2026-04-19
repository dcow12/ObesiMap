"""Pathway enrichment (gseapy Enrichr / prerank GSEA) — GO, KEGG, Reactome; highlight themes."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData
from scipy import sparse


def ranked_genes_condition_contrast(
    adata: AnnData,
    *,
    condition_key: str,
    group_pos: str,
    group_neg: str,
    obs_mask: np.ndarray | None = None,
    layer: str | None = None,
) -> pd.DataFrame:
    """
    Wilcoxon rank-sum for ``group_pos`` vs ``group_neg``; returns full gene table.

    Used to build gene lists or ranked matrices for enrichment. Optional ``obs_mask``
    restricts cells (e.g. Lepr+ only).
    """
    sub = adata[obs_mask] if obs_mask is not None else adata
    sub = sub.copy()
    g = sub.obs[condition_key].astype(str)
    sub = sub[g.isin([group_pos, group_neg])].copy()
    if sub.n_obs == 0 or sub.obs[condition_key].astype(str).nunique() < 2:
        return pd.DataFrame()

    sub.obs["_pathway_group"] = sub.obs[condition_key].astype(str)
    if layer is not None:
        L = sub.layers[layer]
        sub.X = L.copy() if sparse.issparse(L) else np.asarray(L, dtype=np.float64)

    sc.tl.rank_genes_groups(
        sub,
        groupby="_pathway_group",
        groups=[group_pos],
        reference=group_neg,
        method="wilcoxon",
        use_raw=False,
    )
    return sc.get.rank_genes_groups_df(sub, group=group_pos)


def gene_lists_for_ora(
    ranked: pd.DataFrame,
    *,
    padj_max: float = 0.05,
    logfc_min: float = 0.25,
) -> tuple[list[str], list[str]]:
    """Split significant genes into up (``group_pos``) and down vs reference."""
    if ranked.empty:
        return [], []
    up = ranked[
        (ranked["pvals_adj"] < padj_max) & (ranked["logfoldchanges"] > logfc_min)
    ]["names"].astype(str).tolist()
    down = ranked[
        (ranked["pvals_adj"] < padj_max) & (ranked["logfoldchanges"] < -logfc_min)
    ]["names"].astype(str).tolist()
    return up, down


def enrichr_mouse_pathways(
    gene_list: Sequence[str],
    *,
    gene_sets: Sequence[str] | None = None,
    organism: str = "mouse",
    outdir: str | Path | None = None,
    no_plot: bool = True,
) -> pd.DataFrame:
    """
    Run gseapy Enrichr on GO / KEGG / Reactome-style libraries (mouse).

    ``gene_sets`` defaults to :data:`arc_me_sc.config.PATHWAY_ENRICH_LIBRARIES_MOUSE`.
    Returns concatenated ``Enrichr.results`` tables with a ``library`` column.
    """
    import gseapy as gp

    from arc_me_sc.config import PATHWAY_ENRICH_LIBRARIES_MOUSE

    libs = list(gene_sets) if gene_sets is not None else list(PATHWAY_ENRICH_LIBRARIES_MOUSE)
    genes = [str(g) for g in gene_list if str(g).strip()]
    if len(genes) < 3:
        return pd.DataFrame()

    frames: list[pd.DataFrame] = []
    for lib in libs:
        try:
            enr = gp.enrichr(
                gene_list=genes,
                gene_sets=[lib],
                organism=organism,
                outdir=str(outdir) if outdir is not None else None,
                no_plot=no_plot,
            )
        except Exception as e:
            frames.append(
                pd.DataFrame(
                    {
                        "library": [lib],
                        "Error": [str(e)],
                    }
                )
            )
            continue
        res = enr.results
        if res is None or res.empty:
            continue
        r = res.copy()
        r["library"] = lib
        frames.append(r)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def prerank_gsea_mouse(
    ranked: pd.DataFrame,
    *,
    gene_sets: Sequence[str] | None = None,
    organism: str = "mouse",
    outdir: Path,
    permutation_num: int = 100,
    seed: int = 0,
    min_size: int = 15,
    max_size: int = 500,
) -> pd.DataFrame:
    """
    Prerank GSEA using Scanpy ``scores`` as ranking metric (higher = more in ``group_pos``).

    Writes results under ``outdir`` and returns the combined ``res2d`` DataFrame if available.
    """
    import gseapy as gp

    from arc_me_sc.config import PATHWAY_ENRICH_LIBRARIES_MOUSE

    if ranked.empty or "names" not in ranked.columns or "scores" not in ranked.columns:
        return pd.DataFrame()

    libs = list(gene_sets) if gene_sets is not None else list(PATHWAY_ENRICH_LIBRARIES_MOUSE)
    rnk = ranked[["names", "scores"]].dropna().drop_duplicates(subset="names")
    rnk = rnk.sort_values("scores", ascending=False)
    rnk.columns = ["gene", "scores"]

    outdir.mkdir(parents=True, exist_ok=True)
    frames: list[pd.DataFrame] = []
    for lib in libs:
        try:
            res = gp.prerank(
                rnk=rnk,
                gene_sets=lib,
                organism=organism,
                outdir=str(outdir / re.sub(r"[^\w\-.]+", "_", lib)),
                permutation_num=permutation_num,
                seed=seed,
                min_size=min_size,
                max_size=max_size,
                verbose=False,
            )
        except Exception:
            continue
        if hasattr(res, "res2d") and res.res2d is not None and not res.res2d.empty:
            df = res.res2d.copy()
            df["library"] = lib
            frames.append(df)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def highlight_pathway_terms(
    enrich_df: pd.DataFrame,
    patterns: Mapping[str, tuple[str, ...]] | None = None,
    term_col: str = "Term",
) -> pd.DataFrame:
    """
    Tag rows whose ``Term`` matches inflammation, ER stress, or JAK–STAT-related patterns.

    Patterns default to :data:`arc_me_sc.config.PATHWAY_HIGHLIGHT_PATTERNS`.
    Adds columns ``highlight_theme`` (semicolon-separated) and ``is_highlight``.
    """
    from arc_me_sc.config import PATHWAY_HIGHLIGHT_PATTERNS

    patmap = patterns or PATHWAY_HIGHLIGHT_PATTERNS
    if enrich_df.empty or term_col not in enrich_df.columns:
        enrich_df = enrich_df.copy()
        enrich_df["highlight_theme"] = ""
        enrich_df["is_highlight"] = False
        return enrich_df

    terms = enrich_df[term_col].astype(str)
    themes: list[str] = []
    flags: list[bool] = []
    for t in terms:
        hit: list[str] = []
        for theme, pats in patmap.items():
            for p in pats:
                if re.search(p, t, flags=re.IGNORECASE):
                    hit.append(theme)
                    break
        hit_u = sorted(set(hit))
        themes.append(";".join(hit_u))
        flags.append(len(hit_u) > 0)

    out = enrich_df.copy()
    out["highlight_theme"] = themes
    out["is_highlight"] = flags
    return out


def run_pathway_enrichment_pipeline(
    adata: AnnData,
    *,
    condition_key: str,
    group_pos: str,
    group_neg: str,
    out_dir: Path,
    obs_mask: np.ndarray | None = None,
    layer: str | None = None,
    padj_max: float = 0.05,
    logfc_min: float = 0.25,
    run_prerank: bool = False,
    prerank_permutations: int = 100,
) -> dict[str, pd.DataFrame]:
    """
    ORA (Enrichr) on up/down genes + optional prerank GSEA; saves tables and highlight slices.

    Returns keys: ``ranked_genes``, ``enrichr_up``, ``enrichr_down``, ``enrichr_up_highlight``,
    ``enrichr_down_highlight``, ``prerank`` (optional).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ranked = ranked_genes_condition_contrast(
        adata,
        condition_key=condition_key,
        group_pos=group_pos,
        group_neg=group_neg,
        obs_mask=obs_mask,
        layer=layer,
    )
    ranked.to_csv(out_dir / "ranked_genes_for_pathways.csv", index=False)

    up, down = gene_lists_for_ora(ranked, padj_max=padj_max, logfc_min=logfc_min)
    pd.Series(up, name="gene").to_csv(out_dir / "ora_genes_up.csv", index=False)
    pd.Series(down, name="gene").to_csv(out_dir / "ora_genes_down.csv", index=False)

    enrich_up = enrichr_mouse_pathways(up, outdir=out_dir / "enrichr_up")
    enrich_down = enrichr_mouse_pathways(down, outdir=out_dir / "enrichr_down")
    enrich_up.to_csv(out_dir / "enrichr_up_full.csv", index=False)
    enrich_down.to_csv(out_dir / "enrichr_down_full.csv", index=False)

    hu = highlight_pathway_terms(enrich_up)
    hd = highlight_pathway_terms(enrich_down)
    hu[hu["is_highlight"]].to_csv(out_dir / "enrichr_up_inflammation_er_jakstat.csv", index=False)
    hd[hd["is_highlight"]].to_csv(out_dir / "enrichr_down_inflammation_er_jakstat.csv", index=False)

    out: dict[str, pd.DataFrame] = {
        "ranked_genes": ranked,
        "enrichr_up": enrich_up,
        "enrichr_down": enrich_down,
        "enrichr_up_highlight": hu[hu["is_highlight"]],
        "enrichr_down_highlight": hd[hd["is_highlight"]],
    }

    if run_prerank and not ranked.empty:
        pre = prerank_gsea_mouse(
            ranked,
            outdir=out_dir / "gsea_prerank",
            permutation_num=prerank_permutations,
        )
        pre.to_csv(out_dir / "gsea_prerank_combined.csv", index=False)
        if not pre.empty:
            term_c = next(
                (c for c in pre.columns if str(c).lower() in ("term", "name")),
                pre.columns[0],
            )
            ph = highlight_pathway_terms(pre, term_col=term_c)
            out["prerank"] = pre
            out["prerank_highlight"] = ph[ph["is_highlight"]]
            out["prerank_highlight"].to_csv(
                out_dir / "gsea_prerank_inflammation_er_jakstat.csv", index=False
            )

    return out
