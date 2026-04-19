# ggplot2 templates — publication-style panels (optional; use alongside Python pipeline)
# =====================================================================================
# Requires: ggplot2, dplyr, ggrepel (for volcano labels)
#   install.packages(c("ggplot2", "dplyr", "ggrepel"))
#
# These snippets mirror arc_me_sc.viz.publication (matplotlib). Replace paths and column names.

suppressPackageStartupMessages({
  library(ggplot2)
  library(dplyr)
})

# --- 1) UMAP from exported coordinates + metadata (CSV from Python) ------------
# umap_df <- read.csv("results/umap_coords.csv")  # columns: UMAP1, UMAP2, leiden, ...
# ggplot(umap_df, aes(x = UMAP1, y = UMAP2, color = leiden)) +
#   geom_point(size = 0.3, alpha = 0.85) +
#   coord_equal() +
#   theme_bw(base_size = 9) +
#   theme(
#     legend.position = "right",
#     panel.grid.minor = element_blank(),
#     axis.title = element_text(face = "plain")
#   )

# --- 2) Dot plot: use aggregated mean/pct from Scanpy export or pseudobulk --------

# --- 3) Volcano: DE table with log2FC and padj columns ---------------------------
# de <- read.csv("results/de_lepr/lepr_plus_de_wilcoxon.csv")
# de$neglog10padj <- -log10(pmax(de$padj, 1e-300))
# ggplot(de, aes(x = log2_fold_change, y = neglog10padj)) +
#   geom_point(aes(color = padj < 0.05 & abs(log2_fold_change) > 0.5), size = 0.8, alpha = 0.7) +
#   scale_color_manual(values = c("TRUE" = "#E41A1C", "FALSE" = "#BDBDBD")) +
#   geom_vline(xintercept = c(-0.5, 0.5), linetype = "dashed", linewidth = 0.3) +
#   geom_hline(yintercept = -log10(0.05), linetype = "dashed", linewidth = 0.3) +
#   theme_bw(base_size = 9) +
#   labs(x = "log2 fold change", y = expression(-log[10] ~ FDR), title = "DE") +
#   theme(legend.position = "none")

message("ggplot2_templates.R: edit and source() after exporting tables from Python.")
