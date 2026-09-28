"""
Tactical Reporting & Export View for VahniX Dashboard.
SpaceX-inspired dark glassmorphism treatment.
Provides one-click generation and downloading of PDF briefs, HTML dossiers, and GIS bundles.
Authoritative Specifications: ORIGINAL_REQUEST.md § R3, PROJECT.md § 3
"""

from __future__ import annotations

import os
from typing import Sequence
import streamlit as st

from src.data_pipeline.schemas import FIRMSDetection
from src.clustering.schemas import ThermalCluster
from src.classification.ensemble_classifier import PredictionOutput
from src.reporting.pdf_exporter import PDFReportExporter
from src.reporting.html_exporter import HTMLReportExporter
from src.reporting.gis_exporter import GISDataExporter
from src.dashboard.state_manager import DashboardStateManager


def _render_section_header(title: str, subtitle: str = "") -> None:
    """Render a SpaceX-style section header."""
    html = f'<h2 class="section-header">{title}</h2>'
    if subtitle:
        html += f'<p class="section-subtitle">{subtitle}</p>'
    st.markdown(html, unsafe_allow_html=True)


def render_reporting_view(
    state_manager: DashboardStateManager,
    detections: Sequence[FIRMSDetection],
    predictions: Sequence[PredictionOutput],
    clusters: Sequence[ThermalCluster]
) -> None:
    """Render Reporting & Export View with SpaceX styling."""

    _render_section_header(
        "TACTICAL REPORTING",
        "Multi-format intelligence export & incident briefing generation"
    )

    if not predictions:
        st.markdown("""
        <div class="glass-card" style="text-align:center; padding:3rem;">
            <div class="kpi-label">NO DATA AVAILABLE</div>
            <p style="color: var(--text-muted); font-size: 0.85rem; margin-top: 0.5rem;">
                No incident predictions available for export.
            </p>
        </div>
        """, unsafe_allow_html=True)
        return

    # ── SECTION 1: INCIDENT BRIEFING GENERATOR ──
    _render_section_header(
        "INCIDENT INTELLIGENCE BRIEFING",
        "Generate tactical PDF & HTML dossiers for individual incidents"
    )

    selected_id = st.selectbox(
        "Select Incident for Export",
        [p.detection_id for p in predictions],
        label_visibility="collapsed"
    )

    target_det = next((d for d in detections if d.detection_id == selected_id), None)
    target_pred = next((p for p in predictions if p.detection_id == selected_id), None)
    target_exp = state_manager.get_or_create_explanation(selected_id) if target_pred else None

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        <div class="glass-card" style="margin-bottom: 1rem;">
            <div class="kpi-label">PDF INTELLIGENCE REPORT</div>
            <p style="color: var(--text-muted); font-size: 0.75rem; margin-top: 0.5rem; margin-bottom: 1rem;">
                Generate a comprehensive tactical PDF briefing with full incident analysis,
                classification rationale, and geospatial context.
            </p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("GENERATE TACTICAL PDF", key="btn_pdf"):
            if target_det and target_pred:
                pdf_path = f"data/reports/incident_{selected_id}.pdf"
                PDFReportExporter.generate_incident_pdf_report(
                    target_det, target_pred, target_exp, output_path=pdf_path
                )
                st.success(f"PDF Intelligence Report created: `{pdf_path}`")
                if os.path.exists(pdf_path):
                    with open(pdf_path, "rb") as f:
                        st.download_button(
                            label="DOWNLOAD PDF REPORT",
                            data=f.read(),
                            file_name=f"VahniX_Report_{selected_id}.pdf",
                            mime="application/pdf"
                        )

    with c2:
        st.markdown("""
        <div class="glass-card" style="margin-bottom: 1rem;">
            <div class="kpi-label">HTML STANDALONE DOSSIER</div>
            <p style="color: var(--text-muted); font-size: 0.75rem; margin-top: 0.5rem; margin-bottom: 1rem;">
                Generate a self-contained HTML dossier with interactive elements,
                suitable for secure offline distribution and archival.
            </p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("GENERATE HTML DOSSIER", key="btn_html"):
            if target_det and target_pred:
                html_path = f"data/reports/incident_{selected_id}.html"
                HTMLReportExporter.generate_incident_html_dossier(
                    target_det, target_pred, target_exp, output_path=html_path
                )
                st.success(f"HTML Dossier created: `{html_path}`")
                if os.path.exists(html_path):
                    with open(html_path, "r", encoding="utf-8") as f:
                        st.download_button(
                            label="DOWNLOAD HTML DOSSIER",
                            data=f.read(),
                            file_name=f"VahniX_Dossier_{selected_id}.html",
                            mime="text/html"
                        )

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # ── SECTION 2: BULK GIS EXPORT ──
    _render_section_header(
        "BULK GEOSPATIAL DATA EXPORT",
        "GeoJSON & CSV for GIS integration and downstream analysis"
    )

    c3, c4 = st.columns(2)
    with c3:
        st.markdown("""
        <div class="glass-card" style="margin-bottom: 1rem;">
            <div class="kpi-label">GEOJSON EXPORT</div>
            <p style="color: var(--text-muted); font-size: 0.75rem; margin-top: 0.5rem; margin-bottom: 1rem;">
                Export all classified incidents as a standards-compliant GeoJSON feature collection
                for integration with QGIS, ArcGIS, or custom geospatial pipelines.
            </p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("EXPORT GEOJSON", key="btn_geojson"):
            geojson_path = "data/processed/classified_incidents.geojson"
            GISDataExporter.export_detections_geojson(detections, predictions, output_path=geojson_path)
            st.success(f"GeoJSON exported to `{geojson_path}`")
            if os.path.exists(geojson_path):
                with open(geojson_path, "r", encoding="utf-8") as f:
                    st.download_button(
                        label="DOWNLOAD GEOJSON",
                        data=f.read(),
                        file_name="classified_incidents.geojson",
                        mime="application/geo+json"
                    )

    with c4:
        st.markdown("""
        <div class="glass-card" style="margin-bottom: 1rem;">
            <div class="kpi-label">FLAT CSV EXPORT</div>
            <p style="color: var(--text-muted); font-size: 0.75rem; margin-top: 0.5rem; margin-bottom: 1rem;">
                Export classified incidents as a flat CSV table for spreadsheet analysis,
                data science workflows, or integration with tabular databases.
            </p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("EXPORT CSV TABLE", key="btn_csv"):
            csv_path = "data/processed/classified_incidents.csv"
            GISDataExporter.export_detections_csv(detections, predictions, output_path=csv_path)
            st.success(f"CSV exported to `{csv_path}`")
            if os.path.exists(csv_path):
                with open(csv_path, "r", encoding="utf-8") as f:
                    st.download_button(
                        label="DOWNLOAD CSV TABLE",
                        data=f.read(),
                        file_name="classified_incidents.csv",
                        mime="text/csv"
                    )
