"""
Schemas and Data Contracts for NTRO Thermal Clustering & Recurrence Analysis.
Defines immutable, type-safe data structures for thermal clusters, H3 spatial bins, and cluster evaluation metrics.
Authoritative Specifications: ORIGINAL_REQUEST.md, PROJECT.md § Interface Contracts, m2_blueprint.md
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional, Sequence, Union
import numpy as np
import pandas as pd


class ClusterType(str, Enum):
    """Canonical classification types for spatio-temporal thermal clusters."""
    STATIONARY_PERSISTENT = "stationary_persistent"
    DYNAMIC_FRONT = "dynamic_front"
    TRANSIENT_NOISE = "transient_noise"


@dataclass
class ThermalCluster:
    """
    Spatio-temporal cluster of thermal anomaly detections with persistence indicators.
    Authoritative specification: PROJECT.md § Interface Contracts, m2_blueprint.md
    """
    cluster_id: str
    cluster_type: str                  # 'stationary_persistent' | 'dynamic_front' | 'transient_noise'
    detection_ids: list[str]           # List of constituent FIRMSDetection detection_id strings
    centroid_lat: float                # Centroid latitude in WGS84 decimal degrees
    centroid_lon: float                # Centroid longitude in WGS84 decimal degrees
    bounding_box: tuple[float, float, float, float]  # (min_lon, min_lat, max_lon, max_lat)
    convex_hull_geojson: dict[str, Any]# RFC 7946 GeoJSON Polygon or LineString/Point
    total_detections: int              # Total detection count in cluster
    first_seen: datetime               # Earliest acquisition UTC timestamp
    last_seen: datetime                # Latest acquisition UTC timestamp
    temporal_span_days: float          # Days between first and last detection (float)
    active_days_count: int             # Number of distinct calendar dates with detections
    recurrence_rate: float             # active_days / max(monitoring_window_days, 1.0)
    day_count: int                     # Number of daytime detections ('D')
    night_count: int                   # Number of nighttime detections ('N')
    day_night_balance_index: float     # DNBI = 2 * min(D, N) / (D + N) in [0.0, 1.0]
    spatial_jitter_m: float            # 1-sigma dispersion from centroid in meters
    mean_frp: float                    # Average Fire Radiative Power (MW)
    max_frp: float                     # Peak Fire Radiative Power (MW)
    frp_std: float                     # Standard deviation of FRP (MW)
    nearest_industrial_facility_id: Optional[str] = None # Nearest OSM facility ID
    distance_to_nearest_industrial_m: float = 999999.0   # Geodesic distance (meters)
    is_inside_industrial_boundary: bool = False          # True if centroid is within raw polygon
    properties: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Enforce domain invariants and range validations."""
        if not (-90.0 <= self.centroid_lat <= 90.0):
            raise ValueError(f"Invalid centroid_lat {self.centroid_lat}. Must be in [-90.0, 90.0].")
        if not (-180.0 <= self.centroid_lon <= 180.0):
            raise ValueError(f"Invalid centroid_lon {self.centroid_lon}. Must be in [-180.0, 180.0].")
        if not (-1e-5 <= self.day_night_balance_index <= 1.0 + 1e-5):
            raise ValueError(f"Invalid DNBI {self.day_night_balance_index}. Must be in [0.0, 1.0].")
        self.day_night_balance_index = max(0.0, min(1.0, float(self.day_night_balance_index)))
        if self.total_detections != len(self.detection_ids):
            self.total_detections = len(self.detection_ids)

    def to_dict(self) -> dict[str, Any]:
        """Convert to clean JSON-serializable dictionary."""
        d = asdict(self)
        d["first_seen"] = self.first_seen.isoformat()
        d["last_seen"] = self.last_seen.isoformat()
        d["bounding_box"] = list(self.bounding_box)
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ThermalCluster:
        """Instantiate ThermalCluster from dictionary."""
        d = dict(data)
        if isinstance(d.get("first_seen"), str):
            try:
                d["first_seen"] = datetime.fromisoformat(d["first_seen"])
            except ValueError:
                d["first_seen"] = pd.to_datetime(d["first_seen"]).to_pydatetime()
        elif isinstance(d.get("first_seen"), pd.Timestamp):
            d["first_seen"] = d["first_seen"].to_pydatetime()

        if isinstance(d.get("last_seen"), str):
            try:
                d["last_seen"] = datetime.fromisoformat(d["last_seen"])
            except ValueError:
                d["last_seen"] = pd.to_datetime(d["last_seen"]).to_pydatetime()
        elif isinstance(d.get("last_seen"), pd.Timestamp):
            d["last_seen"] = d["last_seen"].to_pydatetime()

        if isinstance(d.get("bounding_box"), (list, tuple)):
            d["bounding_box"] = tuple(float(x) for x in d["bounding_box"])
        else:
            d["bounding_box"] = (0.0, 0.0, 0.0, 0.0)

        d["centroid_lat"] = float(d.get("centroid_lat", 0.0))
        d["centroid_lon"] = float(d.get("centroid_lon", 0.0))
        d["total_detections"] = int(d.get("total_detections", len(d.get("detection_ids", []))))
        d["temporal_span_days"] = float(d.get("temporal_span_days", 0.0))
        d["active_days_count"] = int(d.get("active_days_count", 1))
        d["recurrence_rate"] = float(d.get("recurrence_rate", 0.0))
        d["day_count"] = int(d.get("day_count", 0))
        d["night_count"] = int(d.get("night_count", 0))
        d["day_night_balance_index"] = float(d.get("day_night_balance_index", 0.0))
        d["spatial_jitter_m"] = float(d.get("spatial_jitter_m", 0.0))
        d["mean_frp"] = float(d.get("mean_frp", 0.0))
        d["max_frp"] = float(d.get("max_frp", 0.0))
        d["frp_std"] = float(d.get("frp_std", 0.0))
        d["distance_to_nearest_industrial_m"] = float(d.get("distance_to_nearest_industrial_m", 999999.0))
        d["is_inside_industrial_boundary"] = bool(d.get("is_inside_industrial_boundary", False))

        if "properties" not in d or not isinstance(d["properties"], dict):
            d["properties"] = {}

        valid_keys = {
            "cluster_id", "cluster_type", "detection_ids", "centroid_lat", "centroid_lon",
            "bounding_box", "convex_hull_geojson", "total_detections", "first_seen", "last_seen",
            "temporal_span_days", "active_days_count", "recurrence_rate", "day_count", "night_count",
            "day_night_balance_index", "spatial_jitter_m", "mean_frp", "max_frp", "frp_std",
            "nearest_industrial_facility_id", "distance_to_nearest_industrial_m",
            "is_inside_industrial_boundary", "properties"
        }
        filtered_d = {k: v for k, v in d.items() if k in valid_keys}
        return cls(**filtered_d)

    def to_geojson_feature(self) -> dict[str, Any]:
        """Export as RFC 7946 GeoJSON Feature dictionary."""
        geom = self.convex_hull_geojson if self.convex_hull_geojson else {
            "type": "Point",
            "coordinates": [self.centroid_lon, self.centroid_lat]
        }
        return {
            "type": "Feature",
            "id": self.cluster_id,
            "geometry": geom,
            "properties": {
                "cluster_id": self.cluster_id,
                "cluster_type": self.cluster_type,
                "total_detections": self.total_detections,
                "centroid_lat": self.centroid_lat,
                "centroid_lon": self.centroid_lon,
                "bounding_box": list(self.bounding_box),
                "first_seen": self.first_seen.isoformat(),
                "last_seen": self.last_seen.isoformat(),
                "temporal_span_days": round(self.temporal_span_days, 2),
                "active_days_count": self.active_days_count,
                "recurrence_rate": round(self.recurrence_rate, 4),
                "day_count": self.day_count,
                "night_count": self.night_count,
                "day_night_balance_index": round(self.day_night_balance_index, 4),
                "spatial_jitter_m": round(self.spatial_jitter_m, 2),
                "mean_frp": round(self.mean_frp, 2),
                "max_frp": round(self.max_frp, 2),
                "frp_std": round(self.frp_std, 2),
                "nearest_industrial_facility_id": self.nearest_industrial_facility_id,
                "distance_to_nearest_industrial_m": round(self.distance_to_nearest_industrial_m, 2),
                "is_inside_industrial_boundary": self.is_inside_industrial_boundary,
                **self.properties
            }
        }

    @staticmethod
    def to_dataframe(clusters: Sequence[ThermalCluster]) -> pd.DataFrame:
        """Convert a sequence of ThermalCluster instances into a Pandas DataFrame."""
        if not clusters:
            return pd.DataFrame(columns=[
                "cluster_id", "cluster_type", "total_detections", "centroid_lat", "centroid_lon",
                "temporal_span_days", "active_days_count", "recurrence_rate", "day_count", "night_count",
                "day_night_balance_index", "spatial_jitter_m", "mean_frp", "max_frp", "frp_std",
                "nearest_industrial_facility_id", "distance_to_nearest_industrial_m", "is_inside_industrial_boundary"
            ])
        records = []
        for c in clusters:
            rec = c.to_dict()
            if "convex_hull_geojson" in rec:
                rec["convex_hull_geojson"] = json.dumps(rec["convex_hull_geojson"])
            if "detection_ids" in rec:
                rec["detection_ids"] = json.dumps(rec["detection_ids"])
            if "properties" in rec:
                rec["properties"] = json.dumps(rec["properties"])
            records.append(rec)
        df = pd.DataFrame(records)
        df["first_seen"] = pd.to_datetime(df["first_seen"])
        df["last_seen"] = pd.to_datetime(df["last_seen"])
        return df

    @staticmethod
    def from_dataframe(df: pd.DataFrame) -> list[ThermalCluster]:
        """Reconstruct a list of ThermalCluster objects from a Pandas DataFrame."""
        clusters: list[ThermalCluster] = []
        if df.empty:
            return clusters
        for row in df.to_dict(orient="records"):
            for k in ("convex_hull_geojson", "detection_ids", "properties"):
                if isinstance(row.get(k), str):
                    try:
                        row[k] = json.loads(row[k])
                    except Exception:
                        pass
            clusters.append(ThermalCluster.from_dict(row))
        return clusters


@dataclass
class H3SpatialBin:
    """Aggregated spatial metrics for an individual Uber H3 hexagonal cell."""
    h3_index: str
    resolution: int
    centroid_lat: float
    centroid_lon: float
    boundary_geojson: dict[str, Any]
    total_detections: int
    detection_ids: list[str]
    sum_frp: float
    mean_frp: float
    max_frp: float
    min_frp: float
    std_frp: float
    day_count: int
    night_count: int
    day_night_balance_index: float
    active_days_count: int
    first_seen: datetime
    last_seen: datetime
    temporal_span_days: float
    k_ring_neighbors: list[str] = field(default_factory=list)
    properties: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert H3SpatialBin to clean dictionary."""
        d = asdict(self)
        d["first_seen"] = self.first_seen.isoformat()
        d["last_seen"] = self.last_seen.isoformat()
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> H3SpatialBin:
        """Instantiate H3SpatialBin from dictionary."""
        d = dict(data)
        if isinstance(d.get("first_seen"), str):
            try:
                d["first_seen"] = datetime.fromisoformat(d["first_seen"])
            except ValueError:
                d["first_seen"] = pd.to_datetime(d["first_seen"]).to_pydatetime()
        if isinstance(d.get("last_seen"), str):
            try:
                d["last_seen"] = datetime.fromisoformat(d["last_seen"])
            except ValueError:
                d["last_seen"] = pd.to_datetime(d["last_seen"]).to_pydatetime()
        d["resolution"] = int(d.get("resolution", 8))
        d["centroid_lat"] = float(d.get("centroid_lat", 0.0))
        d["centroid_lon"] = float(d.get("centroid_lon", 0.0))
        d["total_detections"] = int(d.get("total_detections", len(d.get("detection_ids", []))))
        d["sum_frp"] = float(d.get("sum_frp", 0.0))
        d["mean_frp"] = float(d.get("mean_frp", 0.0))
        d["max_frp"] = float(d.get("max_frp", 0.0))
        d["min_frp"] = float(d.get("min_frp", 0.0))
        d["std_frp"] = float(d.get("std_frp", 0.0))
        d["day_count"] = int(d.get("day_count", 0))
        d["night_count"] = int(d.get("night_count", 0))
        d["day_night_balance_index"] = float(d.get("day_night_balance_index", 0.0))
        d["active_days_count"] = int(d.get("active_days_count", 1))
        d["temporal_span_days"] = float(d.get("temporal_span_days", 0.0))

        valid_keys = {
            "h3_index", "resolution", "centroid_lat", "centroid_lon", "boundary_geojson",
            "total_detections", "detection_ids", "sum_frp", "mean_frp", "max_frp", "min_frp",
            "std_frp", "day_count", "night_count", "day_night_balance_index", "active_days_count",
            "first_seen", "last_seen", "temporal_span_days", "k_ring_neighbors", "properties"
        }
        filtered_d = {k: v for k, v in d.items() if k in valid_keys}
        return cls(**filtered_d)

    def to_geojson_feature(self) -> dict[str, Any]:
        """Export as RFC 7946 GeoJSON Feature dictionary."""
        return {
            "type": "Feature",
            "id": self.h3_index,
            "geometry": self.boundary_geojson,
            "properties": {
                "h3_index": self.h3_index,
                "resolution": self.resolution,
                "centroid_lat": self.centroid_lat,
                "centroid_lon": self.centroid_lon,
                "total_detections": self.total_detections,
                "sum_frp": round(self.sum_frp, 2),
                "mean_frp": round(self.mean_frp, 2),
                "max_frp": round(self.max_frp, 2),
                "min_frp": round(self.min_frp, 2),
                "std_frp": round(self.std_frp, 2),
                "day_count": self.day_count,
                "night_count": self.night_count,
                "day_night_balance_index": round(self.day_night_balance_index, 4),
                "active_days_count": self.active_days_count,
                "first_seen": self.first_seen.isoformat(),
                "last_seen": self.last_seen.isoformat(),
                "temporal_span_days": round(self.temporal_span_days, 2),
                **self.properties
            }
        }

    @staticmethod
    def to_dataframe(bins: Sequence[H3SpatialBin]) -> pd.DataFrame:
        """Convert a sequence of H3SpatialBin instances into a Pandas DataFrame."""
        if not bins:
            return pd.DataFrame(columns=[
                "h3_index", "resolution", "centroid_lat", "centroid_lon", "total_detections",
                "sum_frp", "mean_frp", "max_frp", "min_frp", "std_frp", "day_count", "night_count",
                "day_night_balance_index", "active_days_count", "temporal_span_days"
            ])
        records = [b.to_dict() for b in bins]
        df = pd.DataFrame(records)
        df["first_seen"] = pd.to_datetime(df["first_seen"])
        df["last_seen"] = pd.to_datetime(df["last_seen"])
        return df


@dataclass
class ClusterEvaluationMetrics:
    """Master benchmark evaluation container for clustering performance."""
    total_samples: int
    total_clusters: int
    noise_count: int
    cluster_purity: float              # Purity score in [0.0, 1.0] (Target >= 0.90)
    homogeneity: float                 # Scikit-learn homogeneity score [0.0, 1.0]
    completeness: float                # Scikit-learn completeness score [0.0, 1.0]
    v_measure: float                   # Harmonic mean of homogeneity & completeness
    adjusted_rand_index: float         # ARI in [-1.0, 1.0]
    normalized_mutual_info: float      # NMI in [0.0, 1.0]
    silhouette_score: Optional[float] = None  # Unsupervised Haversine silhouette score
    davies_bouldin_index: Optional[float] = None

    def to_dict(self) -> dict[str, Any]:
        """Convert metrics to dictionary."""
        return asdict(self)
