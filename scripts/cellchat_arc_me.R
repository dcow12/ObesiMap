# CellChat ligand–receptor analysis (mouse Arc-ME style workflow)
# Install (once):
#   if (!requireNamespace("BiocManager", quietly = TRUE)) install.packages("BiocManager")
#   BiocManager::install(c("CellChat", "patchwork"))
#   install.packages(c("Seurat", "Matrix", "data.table"))
#
# Prepare inputs from Python:
#   Export counts (genes x cells) and meta with columns: Cell, cell_type, condition
#   Then load below and split by `condition` to compare Fed vs Fasted signaling.
#
# This template assumes a Seurat object `seu` with Idents = cell types.
# Adapt `data.input` and `meta` from your exported tables.

suppressPackageStartupMessages({
  library(CellChat)
  library(Matrix)
  library(patchwork)
})

# --- User: set paths exported from arc_me_sc.ccc.write_cellphonedb_inputs or Seurat ---
# counts <- readRDS("counts_genes_by_cells.rds")  # sparse matrix genes x cells
# meta <- read.csv("meta_celltype_condition.csv", row.names = 1)

# Example skeleton (replace with real data):
# cellchat <- createCellChat(object = counts, meta = meta, group.by = "cell_type")
# CellChatDB <- CellChatDB.mouse
# cellchat@DB <- CellChatDB
# cellchat <- subsetData(cellchat)
# cellchat <- identifyOverExpressedGenes(cellchat)
# cellchat <- identifyOverExpressedInteractions(cellchat)
# cellchat <- computeCommunProb(cellchat, type = "triMean", trim = 0.1)
# cellchat <- filterCommunication(cellchat, min.cells = 10)
# cellchat <- computeCommunProbPathway(cellchat)
# cellchat <- aggregateNet(cellchat)
#
# Compare conditions: build two CellChat objects (meta$condition == "Fed" vs "Fasted"),
# then use liftCellChat / compareInteractions as in CellChat vignette.

message("Template only: edit scripts/cellchat_arc_me.R with your counts + meta paths.")
message("See CellChat vignette: https://github.com/sqjin/CellChat")
