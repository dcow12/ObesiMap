import os
import tempfile

import numpy as np
import pandas as pd
import plotly.express as px
import scanpy as sc
import streamlit as st

st.set_page_config(
    page_title="Arc-ME Cell Explorer",
    page_icon="🧬",
    layout="wide",
)

_H5AD_CANDIDATES = (
    "tmp_test_out/arc_me_processed.h5ad",
    "arc_me_processed.h5ad",
)


def inject_styles():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Pixelify+Sans:wght@400;500;700&family=Space+Grotesk:wght@400;500;700&family=IBM+Plex+Mono:wght@400;500;700&display=swap');

        :root {
            --bg: #070812;
            --panel: rgba(15, 18, 32, 0.72);
            --panel-strong: rgba(9, 12, 22, 0.92);
            --stroke: rgba(255, 255, 255, 0.12);
            --text: #fff7ee;
            --muted: #d8c8b5;
            --accent: #ff8a3d;
            --accent-2: #ffd166;
            --accent-3: #4df0ff;
            --accent-4: #8d7dff;
        }

        html, body, [class*="css"] {
            font-family: 'Space Grotesk', sans-serif;
        }

        .stApp {
            color: var(--text);
            background:
                radial-gradient(circle at 12% 18%, rgba(255, 138, 61, 0.18), transparent 25%),
                radial-gradient(circle at 84% 14%, rgba(77, 240, 255, 0.14), transparent 22%),
                radial-gradient(circle at 50% 78%, rgba(141, 125, 255, 0.16), transparent 26%),
                linear-gradient(145deg, #05060d 0%, #090b16 45%, #0d1021 100%);
        }

        .main .block-container {
            max-width: 1420px;
            padding-top: 1.2rem;
            padding-bottom: 3rem;
        }

        #MainMenu, footer, header {
            visibility: hidden;
        }

        [data-testid="stSidebar"] {
            background:
                linear-gradient(180deg, rgba(16, 18, 34, 0.98) 0%, rgba(10, 12, 22, 0.98) 100%);
            border-right: 1px solid rgba(255,255,255,0.08);
        }

        [data-testid="stSidebar"] * {
            color: var(--text) !important;
            font-family: 'Space Grotesk', sans-serif !important;
        }

        [data-testid="stSidebar"] .stFileUploader,
        [data-testid="stSidebar"] .stTextInput,
        [data-testid="stSidebar"] .stSelectbox,
        [data-testid="stSidebar"] .stRadio {
            background: rgba(255,255,255,0.03);
            border: 1px solid rgba(255,255,255,0.06);
            border-radius: 18px;
            padding: 0.35rem 0.6rem 0.65rem 0.6rem;
        }

        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] p {
            color: #f4e8dc !important;
        }

        .hero-shell {
            position: relative;
            overflow: hidden;
            border: 1px solid rgba(255,255,255,0.14);
            border-radius: 30px;
            padding: 2.4rem 2.4rem 2rem 2.4rem;
            margin-bottom: 1.2rem;
            background:
                linear-gradient(135deg, rgba(255,138,61,0.15), rgba(141,125,255,0.08) 42%, rgba(77,240,255,0.08));
            box-shadow:
                0 30px 80px rgba(0,0,0,0.38),
                inset 0 1px 0 rgba(255,255,255,0.08);
            animation: riseUp 0.9s ease;
        }

        .hero-shell::before {
            content: "";
            position: absolute;
            inset: -30%;
            background:
                radial-gradient(circle, rgba(255, 209, 102, 0.18) 0%, transparent 24%),
                radial-gradient(circle, rgba(77, 240, 255, 0.12) 0%, transparent 22%);
            animation: drift 16s linear infinite;
            pointer-events: none;
        }

        .hero-shell > * {
            position: relative;
            z-index: 1;
        }

        .hero-badge {
            display: inline-block;
            padding: 0.45rem 0.8rem;
            border-radius: 999px;
            font-family: 'IBM Plex Mono', monospace;
            font-size: 0.72rem;
            text-transform: uppercase;
            letter-spacing: 0.16em;
            color: #ffe1bf;
            border: 1px solid rgba(255,255,255,0.12);
            background: rgba(255,255,255,0.06);
            backdrop-filter: blur(14px);
        }

        .hero-title {
            margin: 0.9rem 0 0.7rem 0;
            font-family: 'Pixelify Sans', cursive;
            font-size: clamp(3rem, 6vw, 5.8rem);
            line-height: 0.92;
            letter-spacing: 0.03em;
            color: #fff7ec;
            text-shadow: 0 0 20px rgba(255, 138, 61, 0.18);
        }

        .hero-text {
            max-width: 780px;
            margin: 0;
            font-size: 1.06rem;
            line-height: 1.7;
            color: #efe0cf;
        }

        .hero-marquee {
            margin-top: 1rem;
            display: inline-flex;
            gap: 0.6rem;
            flex-wrap: wrap;
        }

        .pill {
            display: inline-flex;
            align-items: center;
            border: 1px solid rgba(255,255,255,0.1);
            border-radius: 999px;
            padding: 0.48rem 0.8rem;
            color: #ffe6c5;
            background: rgba(255,255,255,0.045);
            font-family: 'IBM Plex Mono', monospace;
            font-size: 0.74rem;
            letter-spacing: 0.08em;
            text-transform: uppercase;
        }

        .stat-card {
            border-radius: 24px;
            padding: 1rem 1rem 0.9rem 1rem;
            border: 1px solid rgba(255,255,255,0.1);
            background:
                linear-gradient(180deg, rgba(255,255,255,0.08), rgba(255,255,255,0.03));
            box-shadow:
                0 20px 40px rgba(0,0,0,0.24),
                inset 0 1px 0 rgba(255,255,255,0.08);
            backdrop-filter: blur(18px);
            animation: floatCard 7s ease-in-out infinite;
        }

        .stat-label {
            font-family: 'IBM Plex Mono', monospace;
            font-size: 0.73rem;
            text-transform: uppercase;
            letter-spacing: 0.16em;
            color: #ffc786;
            margin-bottom: 0.45rem;
        }

        .stat-value {
            font-family: 'Pixelify Sans', cursive;
            font-size: 2rem;
            line-height: 1;
            color: #fff8f0;
            margin-bottom: 0.25rem;
        }

        .stat-note {
            color: #d8c8b5;
            font-size: 0.88rem;
        }

        .section-kicker {
            font-family: 'IBM Plex Mono', monospace;
            text-transform: uppercase;
            letter-spacing: 0.18em;
            color: #ffb56d;
            font-size: 0.74rem;
            margin-bottom: 0.35rem;
        }

        .section-title {
            font-size: 1.3rem;
            font-weight: 700;
            color: #fff7ec;
            margin-bottom: 0.2rem;
        }

        .section-copy {
            color: #d8c8b5;
            margin-bottom: 0.9rem;
        }

        [data-testid="stPlotlyChart"] {
            border-radius: 26px;
            overflow: hidden;
            border: 1px solid rgba(255,255,255,0.1);
            background:
                linear-gradient(180deg, rgba(255,255,255,0.04), rgba(255,255,255,0.02));
            box-shadow: 0 28px 60px rgba(0,0,0,0.3);
            padding: 0.5rem;
        }

        [data-testid="stMetric"] {
            background: rgba(255,255,255,0.04);
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 18px;
            padding: 0.9rem 1rem;
        }

        [data-testid="stMetricLabel"] {
            color: #ffc786;
            font-family: 'IBM Plex Mono', monospace;
            text-transform: uppercase;
            letter-spacing: 0.1em;
        }

        [data-testid="stMetricValue"] {
            font-family: 'Pixelify Sans', cursive;
            color: #fff7ed;
        }

        div[data-testid="stDataFrame"] {
            border-radius: 20px;
            overflow: hidden;
            border: 1px solid rgba(255,255,255,0.08);
        }

        .info-panel {
            border-radius: 24px;
            padding: 1.1rem 1.15rem;
            border: 1px solid rgba(255,255,255,0.1);
            background: var(--panel);
            box-shadow: 0 18px 50px rgba(0,0,0,0.24);
            margin-bottom: 1rem;
        }

        .empty-state {
            border-radius: 28px;
            padding: 2.2rem;
            border: 1px dashed rgba(255,255,255,0.18);
            background: rgba(255,255,255,0.03);
            text-align: center;
            color: #f6e7d7;
        }

        .sidebar-head {
            font-family: 'Pixelify Sans', cursive;
            font-size: 2rem;
            margin: 0.2rem 0 0.5rem 0;
            color: #fff6e9;
        }

        .sidebar-copy {
            color: #d8c8b5;
            font-size: 0.95rem;
            line-height: 1.55;
            margin-bottom: 0.8rem;
        }

        @keyframes riseUp {
            from {
                opacity: 0;
                transform: translateY(12px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }

        @keyframes drift {
            0% {
                transform: rotate(0deg) translateX(0px);
            }
            50% {
                transform: rotate(180deg) translateX(30px);
            }
            100% {
                transform: rotate(360deg) translateX(0px);
            }
        }

        @keyframes floatCard {
            0%, 100% {
                transform: translateY(0px);
            }
            50% {
                transform: translateY(-4px);
            }
        }

        @media (max-width: 900px) {
            .hero-shell {
                padding: 1.6rem 1.2rem 1.4rem 1.2rem;
            }

            .hero-title {
                font-size: 3rem;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource(show_spinner=False)
def load_adata(path: str):
    adata = sc.read_h5ad(path)
    if "X_umap" not in adata.obsm:
        st.error("This file is missing `X_umap`. Run the pipeline first or load a processed `.h5ad`.")
        return None
    return adata


def persist_uploaded_file(uploaded_file):
    suffix = os.path.splitext(uploaded_file.name)[1] or ".h5ad"
    temp_path = os.path.join(tempfile.gettempdir(), f"arc_me_upload{suffix}")
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return temp_path


def resolve_gene_name(query, var_names):
    if not query:
        return None
    if query in var_names:
        return query
    lookup = {str(g).lower(): g for g in var_names}
    return lookup.get(query.lower())


def extract_gene_values(adata, gene_name):
    # Handles both dense and sparse AnnData matrices.
    values = adata[:, gene_name].X
    if hasattr(values, "toarray"):
        values = values.toarray().ravel()
    else:
        values = np.asarray(values).ravel()
    return values


def build_plot_frame(adata):
    return pd.DataFrame(
        adata.obsm["X_umap"],
        columns=["UMAP1", "UMAP2"],
        index=adata.obs_names,
    )


def is_category_like(series):
    return (
        pd.api.types.is_object_dtype(series)
        or pd.api.types.is_categorical_dtype(series)
        or pd.api.types.is_bool_dtype(series)
    )


def stat_card(label, value, note):
    return f"""
    <div class="stat-card">
        <div class="stat-label">{label}</div>
        <div class="stat-value">{value}</div>
        <div class="stat-note">{note}</div>
    </div>
    """


def create_umap_figure(df_plot, color_key, plot_mode):
     common_layout = dict(
         height=720,
         paper_bgcolor="rgba(0,0,0,0)",
         plot_bgcolor="rgba(0,0,0,0)",
         margin=dict(l=10, r=10, t=30, b=10),
         font=dict(family="Space Grotesk, sans-serif", color="#fff4e8"),
         legend=dict(
             bgcolor="rgba(9, 12, 22, 0.76)",
             bordercolor="rgba(255,255,255,0.08)",
             borderwidth=1,
             font=dict(size=11),
         ),
     )

     marker_size = 5 if len(df_plot) > 4000 else 7

     if plot_mode == "Gene Expression":
         low = float(np.percentile(df_plot[color_key], 1))
         high = float(np.percentile(df_plot[color_key], 99))
         if low == high:
             high = low + 1e-6

         fig = px.scatter(
             df_plot,
             x="UMAP1",
             y="UMAP2",
             color=color_key,
             hover_name=df_plot.index.astype(str),
             color_continuous_scale=[
                 [0.0, "#140f2d"],
                 [0.15, "#2659ff"],
                 [0.35, "#00d9ff"],
                 [0.62, "#ffe066"],
                 [1.0, "#ff6b35"],
             ],
             range_color=(low, high),
         )
         fig.update_traces(
             marker=dict(size=marker_size, opacity=0.92, line=dict(width=0)),
             selector=dict(mode="markers"),
         )
         fig.update_layout(
             **common_layout,
             coloraxis_colorbar=dict(
                 title=dict(
                     text="Expression",
                     font=dict(color="#fff4e8"),
                 ),
                 tickfont=dict(color="#fff4e8"),
             ),
         )
     else:
         palette = (
             px.colors.qualitative.Bold
             + px.colors.qualitative.Vivid
             + px.colors.qualitative.Safe
             + px.colors.qualitative.Dark24
         )
         fig = px.scatter(
             df_plot,
             x="UMAP1",
             y="UMAP2",
             color=color_key,
             hover_name=df_plot.index.astype(str),
             color_discrete_sequence=palette,
         )
         fig.update_traces(
             marker=dict(size=marker_size, opacity=0.9, line=dict(width=0)),
             selector=dict(mode="markers"),
         )
         fig.update_layout(**common_layout)

     fig.update_xaxes(visible=False, showgrid=False, zeroline=False)
     fig.update_yaxes(visible=False, showgrid=False, zeroline=False)
     return fig



inject_styles()

st.sidebar.markdown('<div class="sidebar-head">Control Deck</div>', unsafe_allow_html=True)
st.sidebar.markdown(
    '<div class="sidebar-copy">Load a processed Arc-ME `.h5ad`, flip between metadata and genes, and explore the atlas like a neon control room instead of a default dashboard.</div>',
    unsafe_allow_html=True,
)

found_path = next((p for p in _H5AD_CANDIDATES if os.path.exists(p)), None)
uploaded_file = st.sidebar.file_uploader("Upload a processed `.h5ad`", type=["h5ad"])

source_path = None
source_label = None

if uploaded_file is not None:
    source_path = persist_uploaded_file(uploaded_file)
    source_label = uploaded_file.name
elif found_path:
    source_path = found_path
    source_label = found_path

st.markdown(
    """
    <div class="hero-shell">
        <div class="hero-badge">Arc-ME Neural Atlas</div>
        <div class="hero-title">Single-Cell Explorer</div>
        <p class="hero-text">
            A visually aggressive, animation-heavy frontend for UMAP space, metadata landscapes,
            and gene-expression hotspots. Same biology, much stronger presentation.
        </p>
        <div class="hero-marquee">
            <span class="pill">Pixel UI</span>
            <span class="pill">Animated Glass Panels</span>
            <span class="pill">Gene + Metadata Modes</span>
            <span class="pill">Plotly UMAP Theater</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

if not source_path:
    st.markdown(
        """
        <div class="empty-state">
            <h2 style="font-family:'Pixelify Sans', cursive; font-size:2.2rem; margin:0 0 0.6rem 0;">No Dataset Loaded</h2>
            <p style="max-width:680px; margin:0 auto; line-height:1.7;">
                Upload a processed `.h5ad` file in the sidebar, or generate
                `tmp_test_out/arc_me_processed.h5ad` and refresh.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()

adata = load_adata(source_path)
if adata is None:
    st.stop()

categorical_cols = [col for col in adata.obs.columns if is_category_like(adata.obs[col])]
metadata_options = categorical_cols if categorical_cols else list(adata.obs.columns)

st.sidebar.markdown("---")
st.sidebar.subheader("Display Mode")
plot_mode = st.sidebar.radio(
    "Color cells by",
    ["Metadata / Clusters", "Gene Expression"],
)

default_metadata_index = 0
if "lepr_positive" in metadata_options:
    default_metadata_index = metadata_options.index("lepr_positive")

color_key = None
resolved_gene = None

if plot_mode == "Metadata / Clusters":
    color_key = st.sidebar.selectbox(
        "Metadata field",
        metadata_options,
        index=default_metadata_index,
    )
else:
    gene_query = st.sidebar.text_input(
        "Search gene",
        value="Agrp",
        placeholder="Agrp, Pomc, Lepr...",
    )
    resolved_gene = resolve_gene_name(gene_query, adata.var_names)
    if resolved_gene:
        color_key = resolved_gene
        st.sidebar.success(f"Using gene: {resolved_gene}")
    else:
        st.sidebar.warning(f"Gene '{gene_query}' not found.")
        color_key = None

df_plot = build_plot_frame(adata)

if color_key and plot_mode == "Metadata / Clusters":
    df_plot[color_key] = adata.obs[color_key].astype(str).fillna("NA").values
elif color_key and plot_mode == "Gene Expression":
    df_plot[color_key] = extract_gene_values(adata, color_key)

metric_cols = st.columns(4)
with metric_cols[0]:
    st.markdown(
        stat_card("Cells", f"{adata.n_obs:,}", "Total single cells in the loaded atlas"),
        unsafe_allow_html=True,
    )
with metric_cols[1]:
    st.markdown(
        stat_card("Genes", f"{adata.n_vars:,}", "Feature count available for query"),
        unsafe_allow_html=True,
    )
with metric_cols[2]:
    st.markdown(
        stat_card("Metadata", f"{adata.obs.shape[1]:,}", "Observation fields in `.obs`"),
        unsafe_allow_html=True,
    )
with metric_cols[3]:
    active_label = color_key if color_key else "No active layer"
    st.markdown(
        stat_card("Active Layer", active_label, f"Source: {source_label}"),
        unsafe_allow_html=True,
    )

left, right = st.columns([2.25, 1], gap="large")

with left:
    st.markdown(
        """
        <div class="section-kicker">Visualization</div>
        <div class="section-title">UMAP Projection</div>
        <div class="section-copy">Explore the embedding with a stronger visual system, cleaner hover detail, and much better presentation than the default Streamlit look.</div>
        """,
        unsafe_allow_html=True,
    )

    if color_key:
        fig = create_umap_figure(df_plot, color_key, plot_mode)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.markdown(
            """
            <div class="info-panel">
                Enter a valid gene in the sidebar to render the expression map.
            </div>
            """,
            unsafe_allow_html=True,
        )

with right:
    st.markdown(
        """
        <div class="section-kicker">Signal Board</div>
        <div class="section-title">Summary Stats</div>
        <div class="section-copy">A compact readout of what the active coloring layer is showing.</div>
        """,
        unsafe_allow_html=True,
    )

    if color_key and plot_mode == "Metadata / Clusters":
        counts = (
            adata.obs[color_key]
            .astype(str)
            .fillna("NA")
            .value_counts()
            .reset_index()
        )
        counts.columns = [color_key, "Cell Count"]

        dominant_group = counts.iloc[0, 0]
        dominant_count = int(counts.iloc[0, 1])
        group_count = counts.shape[0]

        a, b = st.columns(2)
        with a:
            st.metric("Top Group", dominant_group)
        with b:
            st.metric("Groups", f"{group_count:,}")

        st.metric("Largest Group Size", f"{dominant_count:,}")

        bar_fig = px.bar(
            counts.head(10).sort_values("Cell Count", ascending=True),
            x="Cell Count",
            y=color_key,
            orientation="h",
            color="Cell Count",
            color_continuous_scale=["#1f1147", "#4df0ff", "#ffd166", "#ff6b35"],
        )
        bar_fig.update_layout(
            height=360,
            margin=dict(l=10, r=10, t=20, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Space Grotesk, sans-serif", color="#fff4e8"),
            coloraxis_showscale=False,
            xaxis_title=None,
            yaxis_title=None,
        )
        st.plotly_chart(bar_fig, use_container_width=True)

        st.dataframe(counts.head(12), use_container_width=True, hide_index=True)

    elif color_key and plot_mode == "Gene Expression":
        values = df_plot[color_key].astype(float)
        nonzero = int((values > 0).sum())
        pct_nonzero = (nonzero / len(values)) * 100 if len(values) else 0

        a, b = st.columns(2)
        with a:
            st.metric("Max", f"{values.max():.2f}")
        with b:
            st.metric("Mean", f"{values.mean():.2f}")

        a, b = st.columns(2)
        with a:
            st.metric("Expressing Cells", f"{nonzero:,}")
        with b:
            st.metric("Pct Expressing", f"{pct_nonzero:.1f}%")

        hist_df = pd.DataFrame({color_key: values})
        hist_fig = px.histogram(
            hist_df,
            x=color_key,
            nbins=42,
            color_discrete_sequence=["#ff8a3d"],
        )
        hist_fig.update_layout(
            height=360,
            margin=dict(l=10, r=10, t=20, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Space Grotesk, sans-serif", color="#fff4e8"),
            xaxis_title="Expression",
            yaxis_title="Cells",
        )
        st.plotly_chart(hist_fig, use_container_width=True)

        quantiles = pd.DataFrame(
            {
                "Statistic": ["25th pct", "Median", "75th pct", "95th pct"],
                "Value": [
                    f"{np.percentile(values, 25):.2f}",
                    f"{np.percentile(values, 50):.2f}",
                    f"{np.percentile(values, 75):.2f}",
                    f"{np.percentile(values, 95):.2f}",
                ],
            }
        )
        st.dataframe(quantiles, use_container_width=True, hide_index=True)

tabs = st.tabs(["Metadata Explorer", "UMAP Data Preview"])

with tabs[0]:
    st.markdown(
        """
        <div class="section-kicker">Inspect</div>
        <div class="section-title">Raw Metadata Explorer</div>
        <div class="section-copy">Pick the columns you care about and scan the first slice of the atlas metadata directly from `adata.obs`.</div>
        """,
        unsafe_allow_html=True,
    )

    default_cols = list(adata.obs.columns[: min(6, len(adata.obs.columns))])
    chosen_cols = st.multiselect(
        "Columns to show",
        options=list(adata.obs.columns),
        default=default_cols,
    )
    row_count = st.slider("Rows", min_value=10, max_value=200, value=50, step=10)

    preview_df = adata.obs[chosen_cols].head(row_count) if chosen_cols else adata.obs.head(row_count)
    st.dataframe(preview_df, use_container_width=True)

with tabs[1]:
    st.markdown(
        """
        <div class="section-kicker">Preview</div>
        <div class="section-title">UMAP Data Slice</div>
        <div class="section-copy">A quick look at the actual frame feeding the interactive Plotly scatter.</div>
        """,
        unsafe_allow_html=True,
    )

    preview_cols = ["UMAP1", "UMAP2"]
    if color_key:
        preview_cols.append(color_key)
    st.dataframe(df_plot[preview_cols].head(100), use_container_width=True)

