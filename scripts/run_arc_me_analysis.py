#!/usr/bin/env python3
"""
Campbell et al. (2017) adult mouse Arc-ME pipeline — steps 1–2.

Loads ``expression.txt.gz`` + ``meta.txt``, QC, PCA/UMAP, Leiden/Louvain,
marker scoring for canonical populations.

Example:
  python scripts/run_arc_me_analysis.py --data-dir ./data --out-dir ./results
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running without installing the package
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import matplotlib.pyplot as plt
import scanpy as sc

from arc_me_sc.annotation import (
    assign_labels_by_max_score,
    cluster_marker_summary,
    plot_canonical_markers,
    score_marker_sets,
)
from arc_me_sc.clustering import run_clustering
from arc_me_sc.config import LEPR_LINEAGE_PANEL, MARKER_SETS, build_pipeline_config
from arc_me_sc.ccc import (
    barrier_focus_cell_mask,
    compare_cellphonedb_means,
    run_cellphonedb_by_condition,
)
from arc_me_sc.de import differential_expression_lepr_conditions
from arc_me_sc.lepr import annotate_lepr_status, run_lepr_characterization
from arc_me_sc.pathways import run_pathway_enrichment_pipeline
from arc_me_sc.embedding import run_embedding
from arc_me_sc.io import load_arc_me, save_h5ad
from arc_me_sc.qc import run_qc


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Arc-ME scRNA-seq (GSE93374) Scanpy pipeline")
    p.add_argument("--data-dir", type=Path, default=Path("data"), help="Directory with expression + meta")
    p.add_argument("--out-dir", type=Path, default=Path("results"), help="Figures and h5ad output")
    p.add_argument("--min-genes", type=int, default=0)
    p.add_argument("--max-genes", type=int, default=None)
    p.add_argument("--max-pct-mt", type=float, default=None, help="Max %% mitochondrial counts")
    p.add_argument("--filter-mt-genes", action="store_true", help="Drop MT genes from matrix")
    p.add_argument("--remove-doublets", action="store_true")
    p.add_argument("--doublet-rate", type=float, default=0.06)
    p.add_argument("--resolution", type=float, default=0.6)
    p.add_argument("--n-neighbors", type=int, default=15)
    p.add_argument("--n-pcs", type=int, default=50)
    p.add_argument("--random-state", type=int, default=0)
    p.add_argument("--skip-plots", action="store_true")
    p.add_argument(
        "--lepr",
        action="store_true",
        help="Lepr+ characterization: subset flags, co-expression, FeaturePlots (UMAP)",
    )
    p.add_argument(
        "--lepr-mode",
        choices=("detected", "threshold", "quantile"),
        default="detected",
        help="How to define Lepr+ cells (see arc_me_sc.lepr.annotate_lepr_status)",
    )
    p.add_argument("--lepr-min-expr", type=float, default=0.0, help="For detected/threshold modes")
    p.add_argument(
        "--lepr-quantile",
        type=float,
        default=0.9,
        help="For quantile mode: expression >= this quantile",
    )
    p.add_argument(
        "--de",
        action="store_true",
        help="DE within Lepr+ (Wilcoxon or pseudobulk DESeq2); see --de-* options",
    )
    p.add_argument("--de-method", choices=("wilcoxon", "deseq2"), default="wilcoxon")
    p.add_argument(
        "--de-condition-key",
        type=str,
        default="condition",
        help="obs column for physiological state (e.g. Fed / Fasted)",
    )
    p.add_argument("--de-group-pos", type=str, default="Fasted", help="Numerator of log2 FC")
    p.add_argument("--de-group-neg", type=str, default="Fed", help="Denominator of log2 FC")
    p.add_argument(
        "--de-stratify",
        type=str,
        default=None,
        help="Optional obs key (e.g. leiden) to run Wilcoxon per cluster within Lepr+",
    )
    p.add_argument(
        "--de-expression-layer",
        type=str,
        default=None,
        help="Layer for Wilcoxon (default: adata.X)",
    )
    p.add_argument(
        "--de-counts-layer",
        type=str,
        default="counts",
        help="Integer count layer for pseudobulk DESeq2",
    )
    p.add_argument(
        "--de-sample-col",
        type=str,
        default=None,
        help="obs column with biological replicate / library ID (required for deseq2)",
    )
    p.add_argument("--de-min-cells", type=int, default=5, help="Min cells per arm (Wilcoxon)")
    p.add_argument(
        "--pathways",
        action="store_true",
        help="GO/KEGG/Reactome enrichment (gseapy Enrichr) + optional prerank GSEA",
    )
    p.add_argument("--pathways-condition-key", type=str, default="condition")
    p.add_argument("--pathways-group-pos", type=str, default="Fasted")
    p.add_argument("--pathways-group-neg", type=str, default="Fed")
    p.add_argument(
        "--pathways-lepr-only",
        action="store_true",
        help="Restrict pathway DE ranking to Lepr+ cells",
    )
    p.add_argument("--pathways-layer", type=str, default=None)
    p.add_argument("--pathways-padj", type=float, default=0.05)
    p.add_argument("--pathways-logfc", type=float, default=0.25)
    p.add_argument(
        "--pathways-prerank",
        action="store_true",
        help="Also run prerank GSEA (slower; uses Wilcoxon scores as ranking metric)",
    )
    p.add_argument(
        "--ccc",
        action="store_true",
        help="Export + run CellPhoneDB per dietary condition; merge significant_means",
    )
    p.add_argument("--ccc-cell-type-key", type=str, default="canonical_label")
    p.add_argument("--ccc-condition-key", type=str, default="condition")
    p.add_argument(
        "--ccc-conditions",
        nargs=2,
        metavar=("A", "B"),
        default=["Fed", "Fasted"],
        help="Two physiological states to compare (separate CellPhoneDB runs)",
    )
    p.add_argument("--ccc-layer", type=str, default=None, help="adata layer for counts (else X)")
    p.add_argument("--ccc-threads", type=int, default=4)
    p.add_argument(
        "--ccc-skip-run",
        action="store_true",
        help="Only write CellPhoneDB input files; do not invoke CLI",
    )
    p.add_argument(
        "--ccc-lepr-only",
        action="store_true",
        help="Restrict CCC inputs to Lepr+ cells",
    )
    p.add_argument(
        "--ccc-barrier-focus",
        action="store_true",
        help="Keep tanycyte/glia + ARC neuron labels (substring filter on cell type)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    sc.settings.verbosity = 2
    sc.settings.set_figure_params(dpi=120, facecolor="white")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    sc.settings.figdir = str(args.out_dir / "figures")

    paths, qc, emb, clust = build_pipeline_config(
        data_dir=args.data_dir,
        qc_overrides={
            "min_genes": args.min_genes,
            "max_genes": args.max_genes,
            "max_pct_mitochondrial": args.max_pct_mt,
            "filter_mitochondrial_genes": args.filter_mt_genes,
            "remove_doublets": args.remove_doublets,
            "scrublet_expected_doublet_rate": args.doublet_rate,
        },
        embedding_overrides={
            "n_neighbors": args.n_neighbors,
            "n_pcs": args.n_pcs,
            "random_state": args.random_state,
        },
        clustering_overrides={"resolution": args.resolution},
    )

    adata = load_arc_me(paths.expression_path, paths.meta_path)
    adata = run_qc(adata, qc, random_state=args.random_state)
    run_embedding(adata, emb)
    run_clustering(adata, clust, leiden=True, louvain=True)

    score_marker_sets(adata, MARKER_SETS)
    assign_labels_by_max_score(adata, MARKER_SETS)

    summary = cluster_marker_summary(adata, groupby=clust.leiden_key)
    summary_path = args.out_dir / "cluster_marker_means.csv"
    summary.to_csv(summary_path)

    if not args.skip_plots:
        plot_canonical_markers(
            adata,
            MARKER_SETS,
            leiden_key=clust.leiden_key,
            save="_arc_me",
            show=False,
        )
        plt.close("all")

    if args.lepr:
        lepr_dir = args.out_dir / "lepr"
        lepr_dir.mkdir(parents=True, exist_ok=True)
        annotate_kw = {
            "mode": args.lepr_mode,
            "min_expr": args.lepr_min_expr,
            "quantile": args.lepr_quantile,
        }
        lepr_out = run_lepr_characterization(
            adata,
            LEPR_LINEAGE_PANEL,
            label_key="canonical_label",
            leiden_key=clust.leiden_key,
            figdir=None if args.skip_plots else lepr_dir,
            save_prefix="lepr",
            show_plots=False,
            **annotate_kw,
        )
        corr = lepr_out["coexpression_pearson_lepr_plus"]
        if not corr.empty:
            corr.to_csv(lepr_dir / "lepr_plus_coexpression_pearson.csv")
        means = lepr_out["mean_expr_lepr_vs_rest"]
        if not means.empty:
            means.to_csv(lepr_dir / "lepr_vs_rest_mean_expression.csv")
        comp = lepr_out["lepr_plus_canonical_fractions"]
        if len(comp):
            comp.to_frame("fraction").to_csv(lepr_dir / "lepr_plus_canonical_composition.csv")
        n_pos = int(adata.obs["lepr_positive"].sum())
        print(f"Lepr+ cells: {n_pos} / {adata.n_obs}")

    if args.de:
        lepr_annotate_kw = {
            "mode": args.lepr_mode,
            "min_expr": args.lepr_min_expr,
            "quantile": args.lepr_quantile,
        }
        de_dir = args.out_dir / "de_lepr"
        de_dir.mkdir(parents=True, exist_ok=True)
        de_df = differential_expression_lepr_conditions(
            adata,
            condition_key=args.de_condition_key,
            group_pos=args.de_group_pos,
            group_neg=args.de_group_neg,
            method=args.de_method,
            lepr_annotate_kw=lepr_annotate_kw,
            expression_layer=args.de_expression_layer,
            counts_layer=args.de_counts_layer,
            sample_col=args.de_sample_col,
            stratify_by=args.de_stratify,
            min_cells_per_group=args.de_min_cells,
        )
        de_path = de_dir / f"lepr_plus_de_{args.de_method}.csv"
        de_df.to_csv(de_path, index=False)
        print(f"Wrote {de_path}")

    lepr_kw = {
        "mode": args.lepr_mode,
        "min_expr": args.lepr_min_expr,
        "quantile": args.lepr_quantile,
    }

    if args.pathways:
        pdir = args.out_dir / "pathways"
        obs_mask = None
        if args.pathways_lepr_only:
            if "lepr_positive" not in adata.obs.columns:
                annotate_lepr_status(adata, **lepr_kw)
            obs_mask = adata.obs["lepr_positive"].astype(bool).to_numpy()
        run_pathway_enrichment_pipeline(
            adata,
            condition_key=args.pathways_condition_key,
            group_pos=args.pathways_group_pos,
            group_neg=args.pathways_group_neg,
            out_dir=pdir,
            obs_mask=obs_mask,
            layer=args.pathways_layer,
            padj_max=args.pathways_padj,
            logfc_min=args.pathways_logfc,
            run_prerank=args.pathways_prerank,
        )
        print(f"Wrote pathway tables under {pdir}")

    if args.ccc:
        cdir = args.out_dir / "ccc"
        cdir.mkdir(parents=True, exist_ok=True)
        mask = None
        if args.ccc_lepr_only:
            if "lepr_positive" not in adata.obs.columns:
                annotate_lepr_status(adata, **lepr_kw)
            mask = adata.obs["lepr_positive"].astype(bool).to_numpy()
        if args.ccc_barrier_focus:
            bf = barrier_focus_cell_mask(adata, args.ccc_cell_type_key)
            mask = bf if mask is None else (mask & bf)
        paths = run_cellphonedb_by_condition(
            adata,
            cdir,
            cell_type_key=args.ccc_cell_type_key,
            condition_key=args.ccc_condition_key,
            conditions=args.ccc_conditions,
            layer=args.ccc_layer,
            lepr_mask=mask,
            threads=args.ccc_threads,
            run=not args.ccc_skip_run,
        )
        c0, c1 = args.ccc_conditions
        if not args.ccc_skip_run:
            comp = compare_cellphonedb_means(
                paths[c0],
                paths[c1],
                suffix_a=str(c0),
                suffix_b=str(c1),
            )
            if not comp.empty:
                comp.to_csv(cdir / "cellphonedb_significant_means_merged.csv", index=False)
        print(f"CellPhoneDB outputs under {cdir} (conditions: {c0}, {c1})")

    out_h5ad = args.out_dir / "arc_me_processed.h5ad"
    save_h5ad(adata, out_h5ad)

    print(f"Wrote {out_h5ad}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
