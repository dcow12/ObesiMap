# ObesiMap
**Targeting obesity at the cellular source.**

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](#) *(Link your live app here)*
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Scanpy](https://img.shields.io/badge/Scanpy-Single%20Cell-green.svg)](https://scanpy.readthedocs.io/en/stable/)

## Overview
Obesity is both a metabolic disorder and a breakdown in brain connectivity. Specifically, leptin resistance occurs when the brain's control center the **Arcuate Nucleus and Median Eminence (Arc-ME)** stops responding to leptin (the satiety hormone).

Our project ObesiMap provides an interactive, end-to-end bioinformatics dashboard to analyze single-cell RNA sequencing (scRNA-seq) data from the Arc-ME (based on *Campbell et al., 2017*). By algorithmically isolating **Lepr+ (Leptin Receptor)** expressing cells, this tool decodes the specific genetic pathways that drive hunger and satiety during metabolic stress (Fed vs. Fasted states).

## Key Features
* **Interactive UMAP Projections:** Visualize complex, high-dimensional genetic data in a 2D spatial map.
* **Targeted Cell Isolation:** Algorithmic thresholding to isolate and flag `Lepr+` cells from the broader hypothalamic atlas.
* **Gene Expression Search:** Real-time querying for canonical metabolic markers (e.g., *Agrp*, *Pomc*, *Npy*).
* **Condition Analysis:** Differential expression tracking between Fed and Fasted states to identify active signaling pathways.

## Languages and Libraries Used
* **Data Processing Pipeline:** Python, `Scanpy`, `AnnData`
* **Frontend Dashboard:** `Streamlit`, `Plotly`
* **Data Handling:** `Pandas`, `NumPy`

## 🚀 Live Demo
You can interact with the live dashboard here: **[INSERT_YOUR_STREAMLIT_CLOUD_LINK_HERE]**

## 💻 Local Installation & Usage
If you wish to run the biological pipeline and dashboard locally:

**1. Clone the repository**
```bash
git clone [https://github.com/yourusername/arc-me-explorer.git](https://github.com/yourusername/arc-me-explorer.git)
cd arc-me-explorer
2. Install dependencies

Bash
pip install -r requirements.txt
3. Run the application

Bash
python -m streamlit run app.py
📂 Repository Structure
app.py — The main Streamlit frontend dashboard.
requirements.txt — Python package dependencies.
Biological Deliverables Addressed
Annotated UMAPs: Visual identification of major Arc-ME clusters (AgRP, POMC, etc.).
Lepr+ Characterization: Isolation of exact metabolic sensors.
Pathway Enrichment: Identification of intracellular signaling shifts in response to starvation vs. satiety.
