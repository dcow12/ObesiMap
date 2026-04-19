"""
High-level pipeline steps (Step 1: ingest + QC + embedding; Step 2: cluster + annotate).

Prefer importing :mod:`arc_me_sc.pipeline.step_01_ingest_qc_dimred` and
:mod:`arc_me_sc.pipeline.step_02_cluster_annotate` from scripts or notebooks.
"""

from arc_me_sc.pipeline.step_01_ingest_qc_dimred import run_step_01
from arc_me_sc.pipeline.step_02_cluster_annotate import run_step_02

__all__ = ["run_step_01", "run_step_02"]
