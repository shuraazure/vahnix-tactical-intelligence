"""
Statistical Persistence and Flare Profile Analyzer for Thermal Anomaly Clusters.
Computes Day-Night Balance Index (DNBI), recurrence rates, spatial jitter dispersion,
FRP radiometric profiles, centroid drift velocity, and OSM infrastructure spatial association.
Authoritative Specifications: ORIGINAL_REQUEST.md § R1, PROJECT.md, m2_blueprint.md
"""

from __future__ import annotations

import math
from datetime import datetime
from typing import Any, Optional, Sequence, Union
import numpy as np
import shapely.geometry

from src.data_pipeline.schemas import FIRMSDetection
from src.clustering.schemas import ThermalCluster, ClusterType


EARTH_RADIUS_METERS = 6371000.0


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Computes exact great-circle geodesic distance between two WGS84 coordinates in meters.
    """
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lam = math.radians(lon2 - lon1)
    a = math.sin(d_phi / 2.0)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(d_lam / 2.0)**2
    a = min(1.0, max(0.0, a))
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return EARTH_RADIUS_METERS * c


def calculate_dnbi(day_count: int, night_count: int) -> float:
    """
    Compute Day-Night Balance Index (DNBI):
    DNBI = 2 * min(D, N) / (D + N)
    Continuous industrial flares typically exhibit DNBI >= 0.35 (often >= 0.70).
    Diurnal fires (stubble burning, vegetation) exhibit DNBI < 0.35.
    """
    total = day_count + night_count
    if total == 0:
        return 0.0
    return float(2.0 * min(day_count, night_count) / total)


def calculate_spatial_jitter(
    coords: Sequence[tuple[float, float]],
    centroid: tuple[float, float]
) -> float:
    """
    Calculate 1-sigma spatial dispersion distance from centroid in meters.
    Stationary industrial stacks exhibit jitter <= 350m (orbital parallax / sub-pixel noise).
    Dynamic wildfire fronts exhibit jitter > 1000m.
    """
    if len(coords) <= 1:
        return 0.0
    distances = [haversine_distance(lat, lon, centroid[0], centroid[1]) for lat, lon in coords]
    return float(np.std(distances))


def calculate_centroid_drift_velocity(
    detections: list[FIRMSDetection],
    span_days: float
) -> float:
    """
    Compute centroid progression velocity across observation chronology in meters/day.
    Useful for characterizing advancing wildfire and crop burn fronts.
    """
    if len(detections) < 4 or span_days <= 0.1:
        return 0.0

    sorted_dets = sorted(detections, key=lambda d: d.timestamp)
    mid = len(sorted_dets) // 2
    early_group = sorted_dets[:mid]
    late_group = sorted_dets[mid:]

    lat1 = float(np.mean([d.latitude for d in early_group]))
    lon1 = float(np.mean([d.longitude for d in early_group]))
    lat2 = float(np.mean([d.latitude for d in late_group]))
    lon2 = float(np.mean([d.longitude for d in late_group]))

    dist_m = haversine_distance(lat1, lon1, lat2, lon2)
    return float(dist_m / max(span_days, 0.5))


class PersistenceAnalyzer:
    """
    Synthesizes physical persistence indicators, radiometric profiles, and spatial morphology from cluster groups.
    """

    def __init__(self, spatial_index_engine: Optional[Any] = None):
        self.spatial_index = spatial_index_engine

    def analyze_cluster(
        self,
        cluster_id: str,
        detections: list[FIRMSDetection],
        monitoring_window_days: float = 30.0,
        forced_type: Optional[str] = None
    ) -> ThermalCluster:
        """
        Synthesize full ThermalCluster instance from constituent detections.
        """
        if not detections:
            raise ValueError(f"Cannot analyze empty detection cluster for {cluster_id}")

        coords = [(d.latitude, d.longitude) for d in detections]
        lats = [c[0] for c in coords]
        lons = [c[1] for c in coords]

        cent_lat = float(np.mean(lats))
        cent_lon = float(np.mean(lons))
        bbox = (float(np.min(lons)), float(np.min(lats)), float(np.max(lons)), float(np.max(lats)))

        # Convex hull geometry generation
        convex_hull_geojson = self._generate_convex_hull(coords, cent_lat, cent_lon)

        # Temporal spans and recurrence
        timestamps = [d.timestamp for d in detections]
        first_seen = min(timestamps)
        last_seen = max(timestamps)
        span_days = max(0.0, (last_seen - first_seen).total_seconds() / 86400.0)

        unique_dates = {d.acq_date for d in detections}
        active_days = len(unique_dates)
        recurrence_rate = active_days / max(monitoring_window_days, 1.0)

        # Diurnal balance
        day_count = sum(1 for d in detections if d.daynight == "D")
        night_count = sum(1 for d in detections if d.daynight == "N")
        dnbi = calculate_dnbi(day_count, night_count)

        # Spatial jitter
        jitter_m = calculate_spatial_jitter(coords, (cent_lat, cent_lon))

        # Radiometric FRP statistics
        frps = [d.frp for d in detections]
        mean_frp = float(np.mean(frps))
        max_frp = float(np.max(frps))
        frp_std = float(np.std(frps)) if len(frps) > 1 else 0.0

        # Centroid drift velocity
        drift_velocity = calculate_centroid_drift_velocity(detections, span_days)

        # OSM Industrial Spatial Enrichment
        facility_id: Optional[str] = None
        dist_to_ind = 999999.0
        is_inside = False

        if self.spatial_index is not None:
            fac, dist = self.spatial_index.query_nearest_facility(cent_lat, cent_lon)
            if fac is not None:
                facility_id = fac.facility_id
                dist_to_ind = dist
            is_in, _ = self.spatial_index.check_containment(cent_lat, cent_lon, buffer_mode="raw")
            is_inside = is_in

        # Cluster Morphology Classification
        if forced_type:
            c_type = forced_type
        else:
            c_type = self._classify_cluster_type(
                len(detections), span_days, jitter_m, dnbi, is_inside, dist_to_ind
            )

        extra_props = {
            "drift_velocity_m_per_day": round(drift_velocity, 2),
            "frp_cv": round(frp_std / max(mean_frp, 1e-4), 4),
            "monitoring_window_days": monitoring_window_days,
        }

        return ThermalCluster(
            cluster_id=cluster_id,
            cluster_type=c_type,
            detection_ids=[d.detection_id for d in detections],
            centroid_lat=cent_lat,
            centroid_lon=cent_lon,
            bounding_box=bbox,
            convex_hull_geojson=convex_hull_geojson,
            total_detections=len(detections),
            first_seen=first_seen,
            last_seen=last_seen,
            temporal_span_days=span_days,
            active_days_count=active_days,
            recurrence_rate=recurrence_rate,
            day_count=day_count,
            night_count=night_count,
            day_night_balance_index=dnbi,
            spatial_jitter_m=jitter_m,
            mean_frp=mean_frp,
            max_frp=max_frp,
            frp_std=frp_std,
            nearest_industrial_facility_id=facility_id,
            distance_to_nearest_industrial_m=dist_to_ind,
            is_inside_industrial_boundary=is_inside,
            properties=extra_props
        )

    def _generate_convex_hull(
        self,
        coords: list[tuple[float, float]],
        cent_lat: float,
        cent_lon: float
    ) -> dict[str, Any]:
        """
        Generate GeoJSON geometry for cluster convex hull with degenerate fallback for collinear/low-sample points.
        """
        if len(coords) == 1:
            return {"type": "Point", "coordinates": [coords[0][1], coords[0][0]]}
        elif len(coords) == 2:
            return {
                "type": "LineString",
                "coordinates": [[coords[0][1], coords[0][0]], [coords[1][1], coords[1][0]]]
            }

        try:
            points = [shapely.geometry.Point(lon, lat) for lat, lon in coords]
            mp = shapely.geometry.MultiPoint(points)
            hull = mp.convex_hull

            if hasattr(hull, "__geo_interface__"):
                return shapely.geometry.mapping(hull)
        except Exception:
            pass

        # Fallback to bounding box polygon
        min_lon = min(c[1] for c in coords)
        max_lon = max(c[1] for c in coords)
        min_lat = min(c[0] for c in coords)
        max_lat = max(c[0] for c in coords)

        if min_lon == max_lon and min_lat == max_lat:
            return {"type": "Point", "coordinates": [min_lon, min_lat]}
        elif min_lon == max_lon or min_lat == max_lat:
            return {
                "type": "LineString",
                "coordinates": [[min_lon, min_lat], [max_lon, max_lat]]
            }

        return {
            "type": "Polygon",
            "coordinates": [[
                [min_lon, min_lat],
                [max_lon, min_lat],
                [max_lon, max_lat],
                [min_lon, max_lat],
                [min_lon, min_lat]
            ]]
        }

    def _classify_cluster_type(
        self,
        n_pts: int,
        span_days: float,
        jitter_m: float,
        dnbi: float,
        is_inside: bool,
        dist_ind_m: float
    ) -> str:
        """
        Rule-based classifier for cluster morphology and persistence regime.
        """
        if n_pts < 3:
            return ClusterType.TRANSIENT_NOISE.value
        if (jitter_m <= 500.0 and (dnbi >= 0.35 or is_inside or dist_ind_m <= 750.0)) or (span_days >= 7.0 and jitter_m <= 400.0):
            return ClusterType.STATIONARY_PERSISTENT.value
        if jitter_m > 750.0 or span_days <= 5.0:
            return ClusterType.DYNAMIC_FRONT.value
        return ClusterType.STATIONARY_PERSISTENT.value
