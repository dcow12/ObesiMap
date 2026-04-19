"""Load expression matrix and cell metadata into AnnData."""

from __future__ import annotations

import gzip
import re
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd


def _open_text(path: Path):
    if str(path).endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return open(path, "r", encoding="utf-8", errors="replace")


def read_expression_table(path: Path, sep: str | None = None) -> pd.DataFrame:
    """Read a tabular expression matrix (tab or comma separated)."""
    if sep is None:
        with _open_text(path) as fh:
            first = fh.readline()
        sep = "\t" if first.count("\t") >= first.count(",") else ","
    return pd.read_csv(path, sep=sep, index_col=0, compression="infer")


def read_meta_table(path: Path, sep: str | None = None) -> pd.DataFrame:
    """Read per-cell metadata; index should identify cells (barcodes)."""
    if sep is None:
        with _open_text(path) as fh:
            first = fh.readline()
        sep = "\t" if first.count("\t") >= first.count(",") else ","
    meta = pd.read_csv(path, sep=sep, compression="infer")
    if meta.columns[0].lower() in ("cell", "barcode", "cell_id", "index"):
        meta = meta.set_index(meta.columns[0])
    elif meta.shape[1] > 0 and not isinstance(meta.index, pd.RangeIndex):
        pass
    else:
        if meta.shape[1] > 0:
            meta = meta.set_index(meta.columns[0])
    meta.index = meta.index.astype(str)
    return meta


def _looks_like_gene_symbols(names: list[str], sample: int = 200) -> float:
    pat = re.compile(r"^[A-Za-z][A-Za-z0-9._-]{1,24}$")
    subset = names[: min(len(names), sample)]
    return sum(1 for n in subset if pat.match(n)) / max(len(subset), 1)


def infer_genes_in_rows(expr: pd.DataFrame) -> bool:
    """Infer whether rows are genes (vs cells) using label heuristics."""
    idx = [str(x) for x in expr.index]
    cols = [str(x) for x in expr.columns]
    return _looks_like_gene_symbols(idx) >= _looks_like_gene_symbols(cols)


def expression_to_anndata(
    expr: pd.DataFrame,
    assume_genes_in_rows: bool | None = None,
) -> ad.AnnData:
    """
    Build AnnData with obs = cells, var = genes.

    If ``assume_genes_in_rows`` is None, infer orientation with
    :func:`infer_genes_in_rows`.
    """
    if assume_genes_in_rows is None:
        assume_genes_in_rows = infer_genes_in_rows(expr)

    if assume_genes_in_rows:
        X = expr.values.T
        obs_names = [str(c) for c in expr.columns]
        var_names = [str(r) for r in expr.index]
    else:
        X = expr.values
        obs_names = [str(r) for r in expr.index]
        var_names = [str(c) for c in expr.columns]

    return ad.AnnData(
        X=np.asarray(X, dtype=np.float32),
        obs=pd.DataFrame(index=obs_names),
        var=pd.DataFrame(index=var_names),
    )


def load_arc_me(
    expression_path: Path,
    meta_path: Path,
    join: str = "inner",
    assume_genes_in_rows: bool | None = None,
) -> ad.AnnData:
    """
    Load Campbell-style ``expression.txt.gz`` and ``meta.txt`` into AnnData.

    Cells are aligned to ``meta`` row order (inner join on cell IDs).
    """
    expr = read_expression_table(expression_path)
    meta = read_meta_table(meta_path)

    if assume_genes_in_rows is None:
        assume_genes_in_rows = infer_genes_in_rows(expr)

    if assume_genes_in_rows:
        cell_ids = [str(c) for c in expr.columns]
    else:
        cell_ids = [str(r) for r in expr.index]

    meta_ids = [str(i) for i in meta.index]
    expr_cells = set(cell_ids)
    if join != "inner":
        raise ValueError("Only join='inner' is supported.")
    common = [c for c in meta_ids if c in expr_cells]

    if not common:
        raise ValueError(
            "No overlapping cell IDs between expression and metadata. "
            f"Example expr cells: {cell_ids[:3]!r}; meta: {meta_ids[:3]!r}"
        )

    if assume_genes_in_rows:
        expr = expr.loc[:, common]
    else:
        expr = expr.loc[common, :]

    meta = meta.loc[common]

    adata = expression_to_anndata(expr, assume_genes_in_rows=assume_genes_in_rows)
    adata.obs = meta.copy()
    adata.obs.index = adata.obs.index.astype(str)
    adata.obs_names_make_unique()
    adata.var_names_make_unique()
    return adata


def save_h5ad(adata: ad.AnnData, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(path)


def load_h5ad(path: Path) -> ad.AnnData:
    return ad.read_h5ad(path)
