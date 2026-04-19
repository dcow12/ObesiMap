# clusterProfiler pathway enrichment (mouse) — GO, KEGG, Reactome
# Install (once):
#   if (!requireNamespace("BiocManager", quietly = TRUE)) install.packages("BiocManager")
#   BiocManager::install(c("clusterProfiler", "org.Mm.eg.db", "enrichplot", "ReactomePA"))
#
# Usage:
#   Rscript scripts/clusterprofiler_arc_me.R path/to/de_or_ranked_genes.csv path/to/out_dir
#
# Input CSV must contain a column of gene symbols (default column name: "gene" or "names").

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) {
  stop("Usage: Rscript clusterprofiler_arc_me.R <genes.csv> <out_dir>")
}
gene_csv <- args[[1]]
out_dir <- args[[2]]
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

suppressPackageStartupMessages({
  library(clusterProfiler)
  library(org.Mm.eg.db)
  library(ReactomePA)
})

tab <- read.csv(gene_csv, stringsAsFactors = FALSE)
gene_col <- if ("names" %in% names(tab)) "names" else if ("gene" %in% names(tab)) "gene" else names(tab)[[1]]
genes <- unique(na.omit(tab[[gene_col]]))
genes <- genes[nzchar(genes)]

ego <- enrichGO(
  gene = genes,
  OrgDb = org.Mm.eg.db,
  keyType = "SYMBOL",
  ont = "BP",
  pAdjustMethod = "BH",
  pvalueCutoff = 0.05,
  qvalueCutoff = 0.2,
  readable = TRUE
)
write.csv(as.data.frame(ego), file.path(out_dir, "clusterprofiler_GO_BP.csv"), row.names = FALSE)

ekegg <- enrichKEGG(
  gene = bitr(genes, fromType = "SYMBOL", toType = "ENTREZID", OrgDb = org.Mm.eg.db)$ENTREZID,
  organism = "mmu",
  pAdjustMethod = "BH",
  pvalueCutoff = 0.05
)
write.csv(as.data.frame(ekegg), file.path(out_dir, "clusterProfiler_KEGG.csv"), row.names = FALSE)

e_react <- tryCatch(
  enrichPathway(gene = bitr(genes, fromType = "SYMBOL", toType = "ENTREZID", OrgDb = org.Mm.eg.db)$ENTREZID,
                organism = "mouse", pAdjustMethod = "BH", pvalueCutoff = 0.05, readable = TRUE),
  error = function(e) NULL
)
if (!is.null(e_react)) {
  write.csv(as.data.frame(e_react), file.path(out_dir, "clusterProfiler_Reactome.csv"), row.names = FALSE)
}

message("Wrote clusterProfiler tables to ", normalizePath(out_dir))
