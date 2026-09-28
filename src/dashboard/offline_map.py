"""
Offline Geospatial Map Builder for VahniX Tactical Dashboard.
Generates PyDeck & GeoJSON layer specifications with offline fallback rendering.
Authoritative Specifications: ORIGINAL_REQUEST.md § R3, PROJECT.md § 3, TEST_INFRA.md
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np
import pandas as pd

from src.data_pipeline.schemas import FIRMSDetection, IndustrialFacility
from src.clustering.schemas import ThermalCluster
from src.classification.ensemble_classifier import PredictionOutput
from src.classification.heuristic_baseline import CLASS_SHORT_LABELS


# Tactical Class Color Palette [R, G, B, A]
CLASS_COLOR_RGBA: dict[int, list[int]] = {
    0: [41, 128, 185, 200],   # Controlled Industrial Flare: Royal Blue
    1: [231, 76, 60, 240],    # Industrial Fire Emergency: Crimson Red
    2: [46, 204, 113, 200],   # Agricultural Stubble: Emerald Green
    3: [230, 126, 34, 200],   # Wildfire / Vegetation: Vivid Orange
    4: [176, 102, 255, 200],  # Volcanic Activity: Purple
    5: [148, 163, 184, 180],  # Noise / Glint: Slate Gray
}

FACILITY_COLOR_RGBA: list[int] = [142, 68, 173, 140]  # Translucent Purple
CLUSTER_HULL_COLOR: list[int] = [241, 196, 15, 80]     # Translucent Amber


class OfflineMapBuilder:
    """
    Constructs tactical multi-layer geospatial visualizations configured for 100% offline usage.
    """

    @staticmethod
    def prepare_detections_dataframe(
        detections: Sequence[FIRMSDetection],
        predictions: Sequence[PredictionOutput]
    ) -> pd.DataFrame:
        """Prepare dataframe with RGB colors, radius, and tooltips for Deck.gl / PyDeck."""
        pred_map = {p.detection_id: p for p in predictions}
        
        rows = []
        for d in detections:
            p = pred_map.get(d.detection_id)
            c_val = p.predicted_class.value if p else 3
            c_name = p.predicted_label if p else "Unclassified"
            rsi = p.risk_severity_index if p else 0.0
            is_crit = p.is_critical_alert if p else False
            conf = p.confidence_score if p else d.confidence

            rgba = list(CLASS_COLOR_RGBA.get(c_val, [128, 128, 128, 180]))
            # Enlarge critical emergencies
            radius = 350.0 if c_val == 1 or is_crit else max(80.0, float(d.frp) * 2.5)

            rows.append({
                "detection_id": d.detection_id,
                "latitude": float(d.latitude),
                "longitude": float(d.longitude),
                "frp": float(d.frp),
                "brightness_temp": float(d.brightness_temp_t4),
                "sensor": d.sensor,
                "acq_date": d.acq_date,
                "acq_time": d.acq_time,
                "daynight": d.daynight,
                "class_val": c_val,
                "class_label": c_name,
                "confidence": float(conf),
                "risk_severity": float(rsi),
                "is_critical": bool(is_crit),
                "color_r": rgba[0],
                "color_g": rgba[1],
                "color_b": rgba[2],
                "color_a": rgba[3],
                "radius": float(radius)
            })

        return pd.DataFrame(rows)

    @staticmethod
    def prepare_facility_polygons(
        facilities: Sequence[IndustrialFacility]
    ) -> list[dict[str, Any]]:
        """Prepare GeoJSON polygon features for industrial plant boundaries."""
        features = []
        for fac in facilities:
            # Extract coordinates from bounding box or geometry
            bbox = fac.bounding_box
            coords = [
                [bbox[0], bbox[1]],
                [bbox[2], bbox[1]],
                [bbox[2], bbox[3]],
                [bbox[0], bbox[3]],
                [bbox[0], bbox[1]]
            ]
            features.append({
                "facility_id": fac.facility_id,
                "name": fac.name,
                "facility_type": fac.facility_type,
                "polygon": coords,
                "area_sq_km": fac.area_sq_km
            })
        return features

    @staticmethod
    def build_pydeck_spec(
        detections: Sequence[FIRMSDetection],
        predictions: Sequence[PredictionOutput],
        facilities: Optional[Sequence[IndustrialFacility]] = None,
        clusters: Optional[Sequence[ThermalCluster]] = None,
        view_lat: Optional[float] = None,
        view_lon: Optional[float] = None,
        zoom: int = 7
    ) -> dict[str, Any]:
        """
        Build PyDeck / Deck.gl dictionary spec with scatterplot, polygon, and heatmap layers.
        """
        df_dets = OfflineMapBuilder.prepare_detections_dataframe(detections, predictions)
        
        # Calculate centroid if not provided
        if view_lat is None or view_lon is None:
            if not df_dets.empty:
                view_lat = float(df_dets["latitude"].mean())
                view_lon = float(df_dets["longitude"].mean())
            else:
                view_lat, view_lon = 22.3582, 69.8681  # Default Jamnagar

        spec = {
            "initialViewState": {
                "latitude": float(view_lat),
                "longitude": float(view_lon),
                "zoom": int(zoom),
                "pitch": 25,
                "bearing": 0
            },
            "mapStyle": "https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json",
            "layers": [
                {
                    "type": "ScatterplotLayer",
                    "id": "detections-scatter",
                    "data": df_dets.to_dict(orient="records"),
                    "getPosition": "@@=[longitude, latitude]",
                    "getRadius": "@@=radius",
                    "getFillColor": "@@=[color_r, color_g, color_b, color_a]",
                    "pickable": True,
                    "autoHighlight": True
                }
            ]
        }
        return spec
