"""Cell–cell communication: CellPhoneDB export/run helpers and dietary comparison."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from anndata import AnnData
from scipy import sparse


def _counts_genes_by_cells(adata: AnnData, layer: str | None = None) -> pd.DataFrame:
    """Genes × cells matrix (CellPhoneDB convention: rows = genes, columns = cells)."""
    X = adata.layers[layer] if layer is not None else adata.X
    if sparse.issparse(X):
        X = X.toarray()
    else:
        X = np.asarray(X, dtype=float)
    mat = X.T
    return pd.DataFrame(
        mat,
        index=adata.var_names.astype(str),
        columns=adata.obs_names.astype(str),
    )


def write_cellphonedb_inputs(
    adata: AnnData,
    out_dir: Path,
    *,
    cell_type_key: str,
    layer: str | None = None,
    obs_mask: np.ndarray | None = None,
    meta_cell_col: str = "Cell",
    meta_type_col: str = "cell_type",
) -> tuple[Path, Path]:
    """
    Write tab-separated ``counts.txt`` (genes × cells) and ``meta.txt`` for CellPhoneDB.

    Barcodes in ``meta`` match column names in ``counts``. Uses normalized or
    log-normalized expression if that is what ``adata.X`` / ``layer`` holds (as in
    Campbell processed matrices); for count-based runs, pass ``layer='counts'``.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sub = adata[obs_mask] if obs_mask is not None else adata
    sub = sub.copy()

    counts = _counts_genes_by_cells(sub, layer=layer)
    counts_path = out_dir / "counts.txt"
    counts.to_csv(counts_path, sep="\t")

    meta = pd.DataFrame(
        {
            meta_cell_col: sub.obs_names.astype(str),
            meta_type_col: sub.obs[cell_type_key].astype(str).values,
        }
    )
    meta_path = out_dir / "meta.txt"
    meta.to_csv(meta_path, sep="\t", index=False)

    return meta_path, counts_path


def run_cellphonedb_statistical(
    meta_path: Path,
    counts_path: Path,
    output_path: Path,
    *,
    threads: int = 4,
    cellphonedb_executable: str | None = None,
) -> None:
    """Invoke ``cellphonedb method statistical_analysis`` (CellPhoneDB ≥ 3)."""
    exe = cellphonedb_executable or shutil.which("cellphonedb")
    if not exe:
        raise RuntimeError(
            "cellphonedb CLI not found. Install: pip install cellphonedb "
            "and ensure the executable is on PATH."
        )
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    cmd = [
        exe,
        "method",
        "statistical_analysis",
        str(meta_path),
        str(counts_path),
        "--output-path",
        str(output_path),
        "--threads",
        str(threads),
    ]
    subprocess.run(cmd, check=True)


def run_cellphonedb_by_condition(
    adata: AnnData,
    base_out: Path,
    *,
    cell_type_key: str,
    condition_key: str,
    conditions: Sequence[str],
    layer: str | None = None,
    lepr_mask: np.ndarray | None = None,
    threads: int = 4,
    run: bool = True,
) -> dict[str, Path]:
    """
    Export and optionally run CellPhoneDB **separately** per dietary / physiological state.

    Comparing outputs highlights condition-specific ligand–receptor usage (e.g. across the
    Arc–ME barrier when ``cell_type_key`` encodes tanycyte vs neuronal populations).
    """
    base_out = Path(base_out)
    base_out.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for cond in conditions:
        cmask = adata.obs[condition_key].astype(str) == str(cond)
        if lepr_mask is not None:
            cmask = cmask & lepr_mask
        sub_dir = base_out / f"cellphonedb_{cond}"
        meta_p, cnt_p = write_cellphonedb_inputs(
            adata,
            sub_dir / "input",
            cell_type_key=cell_type_key,
            layer=layer,
            obs_mask=np.asarray(cmask),
        )
        out_run = sub_dir / "out"
        paths[str(cond)] = out_run
        if run:
            run_cellphonedb_statistical(meta_p, cnt_p, out_run, threads=threads)
    return paths


def _read_cellphonedb_table(path: Path, name: str) -> pd.DataFrame:
    p = path / name
    if not p.is_file():
        return pd.DataFrame()
    return pd.read_csv(p, sep="\t")


def compare_cellphonedb_means(
    result_dir_a: Path,
    result_dir_b: Path,
    *,
    means_file: str = "significant_means.txt",
    key_column: str = "interacting_pair",
    suffix_a: str = "cond_a",
    suffix_b: str = "cond_b",
) -> pd.DataFrame:
    """Outer-merge CellPhoneDB ``significant_means`` (or ``means``) on ``interacting_pair``."""
    a = _read_cellphonedb_table(Path(result_dir_a), means_file)
    b = _read_cellphonedb_table(Path(result_dir_b), means_file)
    if a.empty or b.empty or key_column not in a.columns or key_column not in b.columns:
        return pd.DataFrame()
    return a.merge(b, on=key_column, how="outer", suffixes=(f"_{suffix_a}", f"_{suffix_b}"))


def barrier_focus_cell_mask(
    adata: AnnData,
    cell_type_key: str,
    *,
    include_substrings: Sequence[str] = (
        "tanycyte",
        "glia",
        "AgRP",
        "POMC",
        "pomc",
        "glutamatergic",
        "GABAergic",
    ),
) -> np.ndarray:
    """Keep cell types whose label matches barrier-relevant substrings (case-insensitive)."""
    labels = adata.obs[cell_type_key].astype(str).str.lower()
    m = np.zeros(adata.n_obs, dtype=bool)
    for s in include_substrings:
        m |= labels.str.contains(s.lower(), regex=False)
    return m
