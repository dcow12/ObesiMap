#!/usr/bin/env python3
"""
Steps 1–2 only: ingest + QC + UMAP, then cluster + annotate + publication figures.

  Step 1 — Load ``data/expression.txt.gz`` and ``data/meta.txt``, QC, PCA, UMAP.
  Step 2 — Leiden/Louvain, marker scores, canonical labels.
  Figures — High-DPI UMAP (Leiden, canonical_label), marker dot plot (optional volcano if DE CSV exists).

Usage:
  python scripts/run_step01_step02.py --data-dir ./data --out-dir ./results
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from arc_me_sc.config import MARKER_SETS, build_pipeline_config
from arc_me_sc.io import save_h5ad
from arc_me_sc.pipeline.step_01_ingest_qc_dimred import run_step_01
from arc_me_sc.pipeline.step_02_cluster_annotate import run_step_02
import matplotlib.pyplot as plt

from arc_me_sc.viz.publication import (
    apply_publication_matplotlib_style,
    plot_dotplot_publication,
    plot_umap_publication,
    plot_volcano_de,
    save_figure,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Arc-ME pipeline: Step 1 + Step 2 + figures")
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--out-dir", type=Path, default=Path("results"))
    p.add_argument("--resolution", type=float, default=0.6)
    p.add_argument("--random-state", type=int, default=0)
    p.add_argument("--skip-plots", action="store_true")
    p.add_argument(
        "--de-csv",
        type=Path,
        default=None,
        help="Optional DE CSV for volcano plot (columns: gene, log2_fold_change, padj)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    pub_dir = args.out_dir / "publication"
    pub_dir.mkdir(parents=True, exist_ok=True)

    # Tunable pipeline config (QC, embedding, clustering) from a single entry point.
    paths, qc, emb, clust = build_pipeline_config(
        data_dir=args.data_dir,
        clustering_overrides={"resolution": args.resolution},
    )

    # --- Step 1
    adata = run_step_01(
        args.data_dir,
        paths=paths,
        qc=qc,
        emb=emb,
        random_state=args.random_state,
    )

    # --- Step 2
    run_step_02(adata, clustering=clust, marker_sets=MARKER_SETS, run_louvain=True)

    # Checkpoint
    h5ad_path = args.out_dir / "arc_me_step02.h5ad"
    save_h5ad(adata, h5ad_path)

    if args.skip_plots:
        print(f"Wrote {h5ad_path} (figures skipped)")
        return

    apply_publication_matplotlib_style(dpi=300, font_size=9)

    # Annotated UMAPs: graph clusters + biological labels
    fig_l = plot_umap_publication(adata, clust.leiden_key, title="Leiden clusters")
    save_figure(fig_l, pub_dir / "umap_leiden.pdf")
    plt.close(fig_l)

    fig_c = plot_umap_publication(adata, "canonical_label", title="Canonical labels (markers)")
    save_figure(fig_c, pub_dir / "umap_canonical_label.pdf")
    plt.close(fig_c)

    cond_cols = [c for c in ("condition", "diet", "physiological_state") if c in adata.obs.columns]
    if cond_cols:
        ck = cond_cols[0]
        fig_d = plot_umap_publication(adata, ck, title=f"Condition ({ck})")
        save_figure(fig_d, pub_dir / f"umap_{ck}.pdf")
        plt.close(fig_d)

    # Marker dot plot (Leiden on x)
    flat: list[str] = []
    for genes in MARKER_SETS.values():
        flat.extend([g for g in genes if g in adata.var_names])
    flat = list(dict.fromkeys(flat))
    if flat:
        fig_dp = plot_dotplot_publication(
            adata,
            flat,
            groupby=clust.leiden_key,
            figsize=(10, 4),
        )
        if fig_dp is not None and hasattr(fig_dp, "savefig"):
            save_figure(fig_dp, pub_dir / "dotplot_markers_leiden.pdf")
            plt.close(fig_dp)

    # Optional volcano from existing DE export (e.g. after running full pipeline with --de)
    if args.de_csv is not None and args.de_csv.is_file():
        import pandas as pd

        de = pd.read_csv(args.de_csv)
        gene_col = "gene" if "gene" in de.columns else ("names" if "names" in de.columns else None)
        fig_v = plot_volcano_de(
            de,
            gene_col=gene_col,
            title="DE (optional)",
        )
        save_figure(fig_v, pub_dir / "volcano_de.pdf")
        plt.close(fig_v)

    print(f"Wrote {h5ad_path}")
    print(f"Publication figures under {pub_dir}")


if __name__ == "__main__":
    main()

```
