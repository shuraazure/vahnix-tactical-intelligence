"""
Milestone 4 Test Suite: Offline Tactical Dashboard & Multi-Format Reporting.
Validates DashboardStateManager, OfflineMapBuilder, PDFReportExporter, HTMLReportExporter, and GISDataExporter.
Authoritative Specifications: ORIGINAL_REQUEST.md § R3, PROJECT.md § 3, TEST_INFRA.md
"""

from __future__ import annotations

import datetime
import os
import json
import tempfile
import pandas as pd
import pytest

from src.data_pipeline.schemas import FIRMSDetection
from src.clustering.schemas import ThermalCluster, ClusterType
from src.classification.ensemble_classifier import PredictionOutput
from src.classification.heuristic_baseline import ThermalAnomalyClass
from src.dashboard.state_manager import DashboardStateManager, DashboardFilterState
from src.dashboard.offline_map import OfflineMapBuilder, CLASS_COLOR_RGBA
from src.reporting.pdf_exporter import PDFReportExporter
from src.reporting.html_exporter import HTMLReportExporter
from src.reporting.gis_exporter import GISDataExporter
from src.explainability.shap_engine import SHAPExplanation


# =========================================================================
# 1. DASHBOARD STATE MANAGER TESTS
# =========================================================================

def test_dashboard_state_manager_initialization_and_kpis():
    """Verify DashboardStateManager loads offline reference data, clusters, and predictions."""
    mgr = DashboardStateManager()
    assert mgr.is_initialized is True
    assert len(mgr.all_detections) > 0
    assert len(mgr.all_predictions) == len(mgr.all_detections)
    assert len(mgr.all_clusters) > 0

    kpis = mgr.get_kpi_summary(mgr.all_predictions)
    assert kpis["total_incidents"] == len(mgr.all_predictions)
    assert kpis["max_frp"] > 0.0
    assert "industrial_emergencies" in kpis


def test_dashboard_filter_state():
    """Verify DashboardFilterState correctly subsets detections and predictions."""
    mgr = DashboardStateManager()
    
    # Filter for only critical emergencies
    filters_crit = DashboardFilterState(only_critical=True)
    dets, preds, _ = mgr.get_filtered_data(filters_crit)
    assert all(p.is_critical_alert for p in preds)

    # Filter by confidence threshold
    filters_conf = DashboardFilterState(min_confidence=0.90)
    dets, preds, _ = mgr.get_filtered_data(filters_conf)
    assert all(d.confidence >= 0.90 for d in dets)


# =========================================================================
# 2. OFFLINE MAP BUILDER TESTS
# =========================================================================

def test_offline_map_builder_dataframe_and_spec():
    """Verify OfflineMapBuilder dataframe colors and PyDeck spec generation."""
    now = datetime.datetime.utcnow()
    dets = [
        FIRMSDetection(
            detection_id="MAP_DET_001",
            latitude=22.3582,
            longitude=69.8681,
            acq_date="2026-03-15",
            acq_time="1200",
            timestamp=now,
            sensor="VIIRS_SNPP",
            satellite="SNPP",
            frp=50.0,
            brightness_temp_t4=340.0,
            brightness_temp_t11=295.0,
            bright_delta=45.0,
            confidence=0.95,
            daynight="D"
        )
    ]
    preds = [
        PredictionOutput(
            detection_id="MAP_DET_001",
            predicted_class=ThermalAnomalyClass.CONTROLLED_INDUSTRIAL_FLARE,
            predicted_label="Controlled Industrial Flare",
            confidence_score=0.94,
            class_probabilities={"FLARE": 0.94},
            risk_severity_index=25.0,
            is_critical_alert=False
        )
    ]

    df = OfflineMapBuilder.prepare_detections_dataframe(dets, preds)
    assert len(df) == 1
    assert df.iloc[0]["color_r"] == CLASS_COLOR_RGBA[0][0]

    spec = OfflineMapBuilder.build_pydeck_spec(dets, preds)
    assert "initialViewState" in spec
    assert "layers" in spec
    assert len(spec["layers"]) >= 1


# =========================================================================
# 3. REPORTING EXPORTERS TESTS (PDF, HTML, GIS)
# =========================================================================

def test_pdf_report_exporter():
    """Verify PDF intelligence report exporter creates valid file on disk."""
    now = datetime.datetime.utcnow()
    det = FIRMSDetection(
        detection_id="PDF_DET_001",
        latitude=22.3582,
        longitude=69.8681,
        acq_date="2026-03-15",
        acq_time="1200",
        timestamp=now,
        sensor="VIIRS_SNPP",
        satellite="SNPP",
        frp=150.0,
        brightness_temp_t4=360.0,
        brightness_temp_t11=290.0,
        bright_delta=70.0,
        confidence=0.98,
        daynight="N"
    )
    pred = PredictionOutput(
        detection_id="PDF_DET_001",
        predicted_class=ThermalAnomalyClass.INDUSTRIAL_FIRE_DISASTER,
        predicted_label="Industrial Fire Emergency",
        confidence_score=0.98,
        class_probabilities={"INDUSTRIAL_EMERGENCY": 0.98},
        risk_severity_index=85.0,
        is_critical_alert=True
    )
    exp = SHAPExplanation(
        detection_id="PDF_DET_001",
        target_class=ThermalAnomalyClass.INDUSTRIAL_FIRE_DISASTER,
        base_value=0.25,
        shap_values={"frp": 0.45},
        top_positive_features=[("frp", 0.45, 150.0)],
        top_negative_features=[],
        natural_language_summary="Critical industrial emergency detected."
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        pdf_path = os.path.join(tmpdir, "test_report.pdf")
        out = PDFReportExporter.generate_incident_pdf_report(det, pred, exp, output_path=pdf_path)
        assert os.path.exists(out)
        assert os.path.getsize(out) > 0


def test_html_report_exporter():
    """Verify HTML dossier exporter creates valid self-contained HTML file."""
    now = datetime.datetime.utcnow()
    det = FIRMSDetection(
        detection_id="HTML_DET_001",
        latitude=22.3582,
        longitude=69.8681,
        acq_date="2026-03-15",
        acq_time="1200",
        timestamp=now,
        sensor="VIIRS_SNPP",
        satellite="SNPP",
        frp=35.0,
        brightness_temp_t4=335.0,
        brightness_temp_t11=295.0,
        bright_delta=40.0,
        confidence=0.90,
        daynight="D"
    )
    pred = PredictionOutput(
        detection_id="HTML_DET_001",
        predicted_class=ThermalAnomalyClass.CONTROLLED_INDUSTRIAL_FLARE,
        predicted_label="Controlled Industrial Flare",
        confidence_score=0.92,
        class_probabilities={"FLARE": 0.92},
        risk_severity_index=20.0,
        is_critical_alert=False
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        html_path = os.path.join(tmpdir, "test_dossier.html")
        out = HTMLReportExporter.generate_incident_html_dossier(det, pred, output_path=html_path)
        assert os.path.exists(out)
        with open(out, "r", encoding="utf-8") as f:
            content = f.read()
            assert "NATIONAL TECHNICAL RESEARCH ORGANISATION" in content
            assert "HTML_DET_001" in content


def test_gis_data_exporter():
    """Verify GeoJSON and CSV data exports."""
    now = datetime.datetime.utcnow()
    dets = [
        FIRMSDetection(
            detection_id="GIS_DET_001",
            latitude=22.3582,
            longitude=69.8681,
            acq_date="2026-03-15",
            acq_time="1200",
            timestamp=now,
            sensor="VIIRS_SNPP",
            satellite="SNPP",
            frp=35.0,
            brightness_temp_t4=335.0,
            brightness_temp_t11=295.0,
            bright_delta=40.0,
            confidence=0.90,
            daynight="D"
        )
    ]
    preds = [
        PredictionOutput(
            detection_id="GIS_DET_001",
            predicted_class=ThermalAnomalyClass.CONTROLLED_INDUSTRIAL_FLARE,
            predicted_label="Controlled Industrial Flare",
            confidence_score=0.92,
            class_probabilities={"FLARE": 0.92},
            risk_severity_index=20.0,
            is_critical_alert=False
        )
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        geojson_path = os.path.join(tmpdir, "test_incidents.geojson")
        csv_path = os.path.join(tmpdir, "test_incidents.csv")

        GISDataExporter.export_detections_geojson(dets, preds, output_path=geojson_path)
        assert os.path.exists(geojson_path)
        with open(geojson_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            assert data["type"] == "FeatureCollection"
            assert len(data["features"]) == 1

        GISDataExporter.export_detections_csv(dets, preds, output_path=csv_path)
        assert os.path.exists(csv_path)
        df = pd.read_csv(csv_path)
        assert len(df) == 1
        assert df.iloc[0]["detection_id"] == "GIS_DET_001"
