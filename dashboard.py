"""
===============================================================
CAMSF Interactive Results & Visualization Dashboard
Confidence-Driven Adaptive Multi-Sensor Fusion Framework

Author : Vishwa
Run using: streamlit run dashboard.py
===============================================================
"""

from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np

# Page Config
st.set_page_config(
    page_title="CAMSF - Multi-Sensor Fusion Dashboard",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

PROJECT_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_ROOT / "results"
MERGED_DIR = RESULTS_DIR / "merged"
FIGURES_DIR = PROJECT_ROOT / "figures"
TABLES_DIR = PROJECT_ROOT / "tables"

# Custom Styling
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E88E5;
        margin-bottom: 0px;
    }
    .sub-title {
        font-size: 1.1rem;
        color: #555555;
        margin-bottom: 25px;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 15px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<p class="main-title">🚗 CAMSF Framework Dashboard</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-title">Confidence-Driven Adaptive Multi-Sensor Fusion for Autonomous Vehicle Localization</p>',
    unsafe_allow_html=True,
)

# Sidebar Selection
st.sidebar.header("📌 Navigation & Settings")

section = st.sidebar.radio(
    "Select View:",
    [
        "📊 System Summary & Key Metrics",
        "🖼️ All Figures & Diagnostic Plots Gallery",
        "🤖 ML Model Performance & Error Regressor",
        "🗺️ Trajectories & Fusion Performance",
        "📋 Data Tables & Exported CSVs",
    ],
)

# Helper function to display figure safely
def display_fig(fig_name: str, caption: str = ""):
    fig_path = FIGURES_DIR / fig_name
    if fig_path.exists():
        st.image(str(fig_path), caption=caption or fig_name, use_container_width=True)
    else:
        st.warning(f"Figure file `{fig_name}` not found in `{FIGURES_DIR}`.")

# ==============================================================================
# SECTION 1: System Summary & Key Metrics
# ==============================================================================
if section == "📊 System Summary & Key Metrics":
    st.header("📊 Benchmark Overview")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Processed Drives", "9 KITTI Drives")
    with col2:
        st.metric("Total Synchronized Frames", "2,472 Frames")
    with col3:
        st.metric("Best ML Regressor R²", "0.9411 (Random Forest)")
    with col4:
        st.metric("Best ML Test RMSE", "0.0207 m")

    st.markdown("---")

    st.subheader("Key System Figures")
    c1, c2 = st.columns(2)
    with c1:
        display_fig("Figure_01_Dataset_Summary.png", "Figure 1: Dataset Summary & Frame Distribution")
    with c2:
        display_fig("Figure_16_Performance_Comparison.png", "Figure 16: Regressor Model Comparison (RMSE)")

    st.markdown("---")

    st.subheader("Sensor Confidence Index (SCI) & Weight Distributions")
    c3, c4 = st.columns(2)
    with c3:
        display_fig("Figure_04_SCI.png", "Figure 4: Sensor Confidence Index (SCI)")
    with c4:
        display_fig("Figure_07_Adaptive_Weights.png", "Figure 7: Adaptive Sensor Weights (CADWG)")

# ==============================================================================
# SECTION 2: All Figures & Diagnostic Plots Gallery
# ==============================================================================
elif section == "🖼️ All Figures & Diagnostic Plots Gallery":
    st.header("🖼️ Comprehensive Diagnostic Figure Gallery")
    st.write("Displaying all 16 paper figures generated across the pipeline stages:")

    # Find all main figures
    fig_files = sorted([f.name for f in FIGURES_DIR.glob("Figure_*.png")])

    if not fig_files:
        st.info("No `Figure_*.png` files found in `figures/`. Run `python src/paper_results.py` to generate them.")
    else:
        num_cols = 2
        cols = st.columns(num_cols)
        for idx, fig_file in enumerate(fig_files):
            col_idx = idx % num_cols
            with cols[col_idx]:
                st.image(str(FIGURES_DIR / fig_file), caption=fig_file, use_container_width=True)

# ==============================================================================
# SECTION 3: ML Model Performance & Error Regressor
# ==============================================================================
elif section == "🤖 ML Model Performance & Error Regressor":
    st.header("🤖 Machine Learning Model Evaluation")
    st.write("Supervised regressors predicting per-frame position estimation error from sensor quality & confidence features:")

    model_comp_csv = RESULTS_DIR / "model_comparison.csv"
    if model_comp_csv.exists():
        df_comp = pd.read_csv(model_comp_csv)
        st.subheader("Model Comparison Metrics")
        st.dataframe(df_comp.style.highlight_min(subset=["RMSE", "MAE"], color="#d4edda").highlight_max(subset=["R2"], color="#d4edda"))

    c1, c2 = st.columns(2)
    with c1:
        display_fig("feature_importance.png", "Feature Importance Analysis")
        display_fig("Figure_14_Actual_vs_Predicted.png", "Actual vs Predicted Position Error")
    with c2:
        display_fig("residual_plot.png", "Residual Distribution Analysis")
        display_fig("prediction_plot.png", "Per-Frame Prediction Tracking")

# ==============================================================================
# SECTION 4: Trajectories & Fusion Performance
# ==============================================================================
elif section == "🗺️ Trajectories & Fusion Performance":
    st.header("🗺️ CDA-EKF Trajectory & Localization Error")

    c1, c2 = st.columns(2)
    with c1:
        display_fig("Figure_13_Trajectory.png", "Figure 13: Ground Truth vs GPS vs CDA-EKF Trajectory")
        display_fig("gps_vs_cdaekf_trajectory.png", "GPS vs CDA-EKF ENU Local Trajectory")
    with c2:
        display_fig("Figure_12_EKF_Position_Error.png", "Figure 12: Position Error Comparison Over Time")
        display_fig("Figure_10_Localization_Reliability.png", "Figure 10: Localization Reliability Levels")

# ==============================================================================
# SECTION 5: Data Tables & Exported CSVs
# ==============================================================================
elif section == "📋 Data Tables & Exported CSVs":
    st.header("📋 Exported Dataset & Metric Tables")

    tab1, tab2, tab3 = st.tabs(["Fusion Summary Table", "Sensor Confidence Table", "ML Regression Dataset"])

    with tab1:
        f_table = RESULTS_DIR / "paper" / "fusion_error_summary_table.csv"
        if f_table.exists():
            st.dataframe(pd.read_csv(f_table), use_container_width=True)
        else:
            st.warning(f"File `{f_table}` not found.")

    with tab2:
        s_table = RESULTS_DIR / "paper" / "sensor_confidence_summary_table.csv"
        if s_table.exists():
            st.dataframe(pd.read_csv(s_table), use_container_width=True)
        else:
            st.warning(f"File `{s_table}` not found.")

    with tab3:
        ml_csv = MERGED_DIR / "merged_localization_ml_dataset.csv"
        if ml_csv.exists():
            df_ml = pd.read_csv(ml_csv)
            st.write(f"Dataset Shape: `{df_ml.shape[0]}` rows, `{df_ml.shape[1]}` columns")
            st.dataframe(df_ml.head(100), use_container_width=True)
        else:
            st.warning(f"File `{ml_csv}` not found.")
