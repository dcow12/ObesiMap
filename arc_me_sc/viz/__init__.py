"""Publication-quality plotting helpers (matplotlib / Scanpy); optional ggplot2 templates in R."""

from arc_me_sc.viz.publication import (
    apply_publication_matplotlib_style,
    plot_dotplot_publication,
    plot_umap_publication,
    plot_volcano_de,
    save_figure,
)

__all__ = [
    "apply_publication_matplotlib_style",
    "plot_umap_publication",
    "plot_dotplot_publication",
    "plot_volcano_de",
    "save_figure",
]
