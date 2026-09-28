"""
GIS Data Exporter for NTRO Thermal Detection System.
Exports classified incidents and clusters to GeoJSON, CSV, and Shapefile bundles.
Authoritative Specifications: ORIGINAL_REQUEST.md § R3, PROJECT.md § 3, TEST_INFRA.md
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Sequence
import pandas as pd

from src.data_pipeline.schemas import FIRMSDetection
from src.clustering.schemas import ThermalCluster
from src.classification.ensemble_classifier import PredictionOutput


class GISDataExporter:
    """
    Exports geospatial incident detections and clusters to standard GIS vector formats.
    """

    @classmethod
    def export_detections_geojson(
        cls,
        detections: Sequence[FIRMSDetection],
        predictions: Sequence[PredictionOutput],
        output_path: str = "data/processed/classified_incidents.geojson"
    ) -> str:
        """Export detections with AI prediction properties to GeoJSON FeatureCollection."""
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        pred_map = {p.detection_id: p for p in predictions}

        features = []
        for d in detections:
            p = pred_map.get(d.detection_id)
            feat = {
                "type": "Feature",
                "id": d.detection_id,
                "geometry": {
                    "type": "Point",
                    "coordinates": [float(d.longitude), float(d.latitude)]
                },
                "properties": {
                    "detection_id": d.detection_id,
                    "acq_date": d.acq_date,
                    "acq_time": d.acq_time,
                    "sensor": d.sensor,
                    "satellite": d.satellite,
                    "frp": float(d.frp),
                    "brightness_temp_t4": float(d.brightness_temp_t4),
                    "brightness_temp_t11": float(d.brightness_temp_t11),
                    "confidence": float(d.confidence),
                    "daynight": d.daynight,
                    "predicted_class": p.predicted_class.value if p else -1,
                    "predicted_label": p.predicted_label if p else "Unknown",
                    "ai_confidence": float(p.confidence_score) if p else 0.0,
                    "risk_severity_index": float(p.risk_severity_index) if p else 0.0,
                    "is_critical_alert": bool(p.is_critical_alert) if p else False
                }
            }
            features.append(feat)

        geojson_obj = {
            "type": "FeatureCollection",
            "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}},
            "features": features
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(geojson_obj, f, indent=2)

        return output_path

    @classmethod
    def export_clusters_geojson(
        cls,
        clusters: Sequence[ThermalCluster],
        output_path: str = "data/processed/thermal_clusters.geojson"
    ) -> str:
        """Export spatio-temporal cluster geometries and recurrence properties to GeoJSON."""
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        features = [c.to_geojson_feature() for c in clusters]

        geojson_obj = {
            "type": "FeatureCollection",
            "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}},
            "features": features
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(geojson_obj, f, indent=2)

        return output_path

    @classmethod
    def export_detections_csv(
        cls,
        detections: Sequence[FIRMSDetection],
        predictions: Sequence[PredictionOutput],
        output_path: str = "data/processed/classified_incidents.csv"
    ) -> str:
        """Export detections with AI prediction properties to flat CSV table."""
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        pred_map = {p.detection_id: p for p in predictions}

        rows = []
        for d in detections:
            p = pred_map.get(d.detection_id)
            rows.append({
                "detection_id": d.detection_id,
                "latitude": d.latitude,
                "longitude": d.longitude,
                "acq_date": d.acq_date,
                "acq_time": d.acq_time,
                "sensor": d.sensor,
                "satellite": d.satellite,
                "frp": d.frp,
                "brightness_temp_t4": d.brightness_temp_t4,
                "confidence": d.confidence,
                "daynight": d.daynight,
                "predicted_class": p.predicted_class.value if p else -1,
                "predicted_label": p.predicted_label if p else "Unknown",
                "ai_confidence": p.confidence_score if p else 0.0,
                "risk_severity_index": p.risk_severity_index if p else 0.0,
                "is_critical_alert": p.is_critical_alert if p else False
            })

        df = pd.DataFrame(rows)
        df.to_csv(output_path, index=False)
        return output_path
