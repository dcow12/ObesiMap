"""Default paths and tunable parameters for the Arc-ME analysis pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


@dataclass
class DataPaths:
    """Locations of input matrices and metadata."""

    data_dir: Path = Path("data")
    expression_file: str = "expression.txt.gz"
    meta_file: str = "meta.txt"

    @property
    def expression_path(self) -> Path:
        return self.data_dir / self.expression_file

    @property
    def meta_path(self) -> Path:
        return self.data_dir / self.meta_file


@dataclass
class QCParams:
    """QC filters. For already processed data, keep these permissive."""

    min_genes: int = 200
    max_genes: int | None = None
    max_pct_mitochondrial: float | None = None
    filter_mitochondrial_genes: bool = False
    mitochondrial_prefix: str = "mt-"
    remove_doublets: bool = False
    doublet_method: str = "scrublet"
    scrublet_expected_doublet_rate: float = 0.06


@dataclass
class EmbeddingParams:
    n_top_genes: int = 2000
    n_pcs: int = 50
    n_neighbors: int = 15
    random_state: int = 0


@dataclass
class ClusteringParams:
    resolution: float = 0.6
    leiden_key: str = "leiden"
    louvain_key: str = "louvain"
    graph_key: str = "neighbors"


# Canonical Arc-ME / hypothalamic populations (Campbell et al. 2017; mouse symbols)
MARKER_SETS: dict[str, list[str]] = {
    "AgRP_neurons": ["Agrp", "Npy"],
    "POMC_neurons": ["Pomc", "Cartpt"],
    "glutamatergic_neurons": ["Slc17a6"],  # Vglut2
    "GABAergic_neurons": ["Slc32a1"],  # VGAT
    "tanycytes_glia": ["Gfap", "Slc1a2", "Aqp4"],
}

# Panel for Lepr+ lineage context: ARC neurons, broad neurotransmitter markers, glia/tanycytes.
# Campbell et al. (2017) highlight Lepr+ cells beyond canonical AgRP/POMC (a “novel” sensing pool);
# compare Lepr co-expression with AgRP/POMC vs glial markers on UMAP and in correlation matrices.
LEPR_LINEAGE_PANEL: list[str] = [
    "Lepr",
    "Agrp",
    "Npy",
    "Pomc",
    "Cartpt",
    "Slc17a6",
    "Slc32a1",
    "Gfap",
    "Slc1a2",
    "Aqp4",
]

# Leptin signaling / ARC effectors for Fed vs. Fasted DE within Lepr+ cells.
LEPTIN_SIGNALING_DE_GENES: list[str] = ["Stat3", "Socs3", "Pomc", "Agrp", "Npy"]

# gseapy Enrichr library names (mouse). Adjust if Enrichr updates catalog.
PATHWAY_ENRICH_LIBRARIES_MOUSE: tuple[str, ...] = (
    "GO_Biological_Process_2023",
    "KEGG_2021_Mouse",
    "Reactome_2022",
)

# Case-insensitive regex fragments matched against pathway ``Term`` strings.
PATHWAY_HIGHLIGHT_PATTERNS: dict[str, tuple[str, ...]] = {
    "inflammation": (
        r"inflam",
        r"cytokine",
        r"interleukin",
        r"TNF",
        r"NF-\s*kappa",
        r"immune response",
        r"innate immune",
        r"adaptive immune",
    ),
    "er_stress": (
        r"unfolded protein",
        r"ER stress",
        r"endoplasmic reticulum stress",
        r"response to topologically",
        r"PERK",
        r"ATF4",
        r"IRE1",
        r"XBP1",
    ),
    "jak_stat": (
        r"JAK",
        r"STAT",
        r"interferon",
        r"cytokine-mediated",
        r"Jak-STAT",
    ),
}


def build_pipeline_config(
    data_dir: str | Path | None = None,
    qc_overrides: Mapping[str, Any] | None = None,
    embedding_overrides: Mapping[str, Any] | None = None,
    clustering_overrides: Mapping[str, Any] | None = None,
) -> tuple[DataPaths, QCParams, EmbeddingParams, ClusteringParams]:
    """Merge optional overrides into default parameter objects."""

    paths = DataPaths()
    if data_dir is not None:
        paths.data_dir = Path(data_dir)

    qc = QCParams()
    if qc_overrides:
        for k, v in qc_overrides.items():
            if hasattr(qc, k):
                setattr(qc, k, v)

    emb = EmbeddingParams()
    if embedding_overrides:
        for k, v in embedding_overrides.items():
            if hasattr(emb, k):
                setattr(emb, k, v)

    clust = ClusteringParams()
    if clustering_overrides:
        for k, v in clustering_overrides.items():
            if hasattr(clust, k):
                setattr(clust, k, v)

    return paths, qc, emb, clust
