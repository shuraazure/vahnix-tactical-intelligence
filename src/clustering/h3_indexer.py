"""
Uber H3 Discrete Global Grid Spatial Binning & Time-Window Aggregation Engine.
Supports multi-resolution H3 spatial binning (Res 7, 8, 9), dual h3-py API (v3/v4),
and deterministic mathematical hexagonal fallback for air-gapped environments.
Authoritative Specifications: ORIGINAL_REQUEST.md § R1, PROJECT.md, m2_blueprint.md
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta
from typing import Any, Optional, Sequence, Union
import numpy as np
import pandas as pd

from src.data_pipeline.schemas import FIRMSDetection
from src.clustering.schemas import H3SpatialBin

try:
    import h3
    _H3_AVAILABLE = True
except ImportError:
    h3 = None
    _H3_AVAILABLE = False


def compute_h3_index(lat: float, lon: float, resolution: int = 8) -> str:
    """
    Compute Uber H3 cell code for a (latitude, longitude) coordinate.
    Supports h3-py v4+ (latlng_to_cell), v3 (geo_to_h3), and deterministic offline fallback.
    """
    if _H3_AVAILABLE and h3 is not None:
        try:
            if hasattr(h3, "latlng_to_cell"):
                res_val = h3.latlng_to_cell(lat, lon, resolution)
                return str(res_val)
            elif hasattr(h3, "geo_to_h3"):
                res_val = h3.geo_to_h3(lat, lon, resolution)
                return str(res_val)
        except Exception:
            pass

    # Deterministic spatial hex fallback (15-character valid lowercase hex string)
    lat_clamped = max(-90.0, min(90.0, float(lat)))
    lon_clamped = max(-180.0, min(180.0, float(lon)))
    lat_q = int(math.floor((lat_clamped + 90.0) * 1000.0)) & 0xFFFF
    lon_q = int(math.floor((lon_clamped + 180.0) * 1000.0)) & 0xFFFF
    res_hex = f"{resolution:x}"
    return f"8{res_hex}61{lat_q:04x}{lon_q:04x}fff"[:15]


def get_h3_boundary(h3_index: str) -> list[tuple[float, float]]:
    """
    Return boundary polygon coordinates as a list of (lon, lat) tuples for GeoJSON.
    Guarantees a closed polygon loop.
    """
    if _H3_AVAILABLE and h3 is not None:
        try:
            if hasattr(h3, "cell_to_boundary"):
                coords = h3.cell_to_boundary(h3_index)  # returns tuple/list of (lat, lng)
                ring = [(float(c[1]), float(c[0])) for c in coords]
                if ring and (ring[0] != ring[-1]):
                    ring.append(ring[0])
                return ring
            elif hasattr(h3, "h3_to_geo_boundary"):
                coords = h3.h3_to_geo_boundary(h3_index)
                ring = [(float(c[1]), float(c[0])) for c in coords]
                if ring and (ring[0] != ring[-1]):
                    ring.append(ring[0])
                return ring
        except Exception:
            pass

    # Fallback synthetic regular hexagon centered on approximate coordinate
    cent_lat, cent_lon = get_h3_centroid(h3_index)
    res = 8
    if len(h3_index) > 1 and h3_index[1].isalnum():
        try:
            res = int(h3_index[1], 16)
        except ValueError:
            res = 8

    # Approximate radius in degrees per resolution
    radius_deg = 0.005 if res == 8 else (0.015 if res == 7 else 0.002)
    boundary: list[tuple[float, float]] = []
    for angle in [0, 60, 120, 180, 240, 300]:
        rad = math.radians(angle)
        boundary.append((
            cent_lon + radius_deg * math.cos(rad),
            cent_lat + radius_deg * math.sin(rad)
        ))
    boundary.append(boundary[0])  # Close polygon ring
    return boundary


def get_h3_centroid(h3_index: str) -> tuple[float, float]:
    """
    Return (latitude, longitude) center of an H3 cell.
    """
    if _H3_AVAILABLE and h3 is not None:
        try:
            if hasattr(h3, "cell_to_latlng"):
                lat, lon = h3.cell_to_latlng(h3_index)
                return float(lat), float(lon)
            elif hasattr(h3, "h3_to_geo"):
                lat, lon = h3.h3_to_geo(h3_index)
                return float(lat), float(lon)
        except Exception:
            pass

    # Approximate decoding from fallback hex string
    try:
        lat_q = int(h3_index[4:8], 16)
        lon_q = int(h3_index[8:12], 16)
        lat = (lat_q / 1000.0) - 90.0
        lon = (lon_q / 1000.0) - 180.0
        return max(-90.0, min(90.0, float(lat))), max(-180.0, min(180.0, float(lon)))
    except Exception:
        return 0.0, 0.0


def get_h3_k_ring(h3_index: str, k: int = 1) -> list[str]:
    """
    Return neighboring cell IDs within k rings (inclusive of center cell).
    """
    if _H3_AVAILABLE and h3 is not None:
        try:
            if hasattr(h3, "grid_disk"):
                cells = h3.grid_disk(h3_index, k)
                return [str(c) for c in cells]
            elif hasattr(h3, "k_ring"):
                cells = h3.k_ring(h3_index, k)
                return [str(c) for c in cells]
        except Exception:
            pass
    return [h3_index]


class H3SpatialIndexer:
    """
    High-performance discrete global grid spatial binning and time-window aggregation engine.
    """

    def __init__(self, default_resolution: int = 8):
        self.default_resolution = default_resolution

    def add_h3_indices(self, detections: list[FIRMSDetection]) -> list[FIRMSDetection]:
        """
        Enrich FIRMSDetection records with h3_res7, h3_res8, and h3_res9 indices.
        Returns newly constructed FIRMSDetection instances with populated fields.
        """
        enriched: list[FIRMSDetection] = []
        for d in detections:
            res7 = d.h3_res7 if d.h3_res7 else compute_h3_index(d.latitude, d.longitude, 7)
            res8 = d.h3_res8 if d.h3_res8 else compute_h3_index(d.latitude, d.longitude, 8)
            res9 = d.h3_res9 if d.h3_res9 else compute_h3_index(d.latitude, d.longitude, 9)
            enriched.append(FIRMSDetection(
                detection_id=d.detection_id,
                latitude=d.latitude,
                longitude=d.longitude,
                acq_date=d.acq_date,
                acq_time=d.acq_time,
                timestamp=d.timestamp,
                sensor=d.sensor,
                satellite=d.satellite,
                frp=d.frp,
                brightness_temp_t4=d.brightness_temp_t4,
                brightness_temp_t11=d.brightness_temp_t11,
                bright_delta=d.bright_delta,
                confidence=d.confidence,
                daynight=d.daynight,
                scan=d.scan,
                track=d.track,
                h3_res7=res7,
                h3_res8=res8,
                h3_res9=res9,
                raw_properties=d.raw_properties
            ))
        return enriched

    def aggregate_by_h3(
        self,
        detections: list[FIRMSDetection],
        resolution: Optional[int] = None,
        min_detections: int = 1
    ) -> list[H3SpatialBin]:
        """
        Aggregate detections into discrete H3 spatial bins with statistical persistence summaries.
        """
        if not detections:
            return []

        res = resolution or self.default_resolution
        bins: dict[str, list[FIRMSDetection]] = {}

        for d in detections:
            if res == 7 and d.h3_res7:
                cell = d.h3_res7
            elif res == 8 and d.h3_res8:
                cell = d.h3_res8
            elif res == 9 and d.h3_res9:
                cell = d.h3_res9
            else:
                cell = compute_h3_index(d.latitude, d.longitude, res)

            bins.setdefault(cell, []).append(d)

        results: list[H3SpatialBin] = []
        for cell_id, group in bins.items():
            if len(group) < min_detections:
                continue

            frps = [d.frp for d in group]
            timestamps = [d.timestamp for d in group]
            day_cnt = sum(1 for d in group if d.daynight == "D")
            night_cnt = sum(1 for d in group if d.daynight == "N")
            unique_dates = {d.acq_date for d in group}

            first_seen = min(timestamps)
            last_seen = max(timestamps)
            span_days = max(0.0, (last_seen - first_seen).total_seconds() / 86400.0)

            total_dn = day_cnt + night_cnt
            dnbi = float(2.0 * min(day_cnt, night_cnt) / total_dn) if total_dn > 0 else 0.0

            cent_lat, cent_lon = get_h3_centroid(cell_id)
            boundary_coords = get_h3_boundary(cell_id)

            boundary_geom = {
                "type": "Polygon",
                "coordinates": [boundary_coords]
            }

            neighbors = get_h3_k_ring(cell_id, k=1)

            results.append(H3SpatialBin(
                h3_index=cell_id,
                resolution=res,
                centroid_lat=cent_lat,
                centroid_lon=cent_lon,
                boundary_geojson=boundary_geom,
                total_detections=len(group),
                detection_ids=[d.detection_id for d in group],
                sum_frp=float(np.sum(frps)),
                mean_frp=float(np.mean(frps)),
                max_frp=float(np.max(frps)),
                min_frp=float(np.min(frps)),
                std_frp=float(np.std(frps)) if len(frps) > 1 else 0.0,
                day_count=day_cnt,
                night_count=night_cnt,
                day_night_balance_index=dnbi,
                active_days_count=len(unique_dates),
                first_seen=first_seen,
                last_seen=last_seen,
                temporal_span_days=span_days,
                k_ring_neighbors=neighbors
            ))

        return results

    def aggregate_time_windows(
        self,
        detections: list[FIRMSDetection],
        resolution: int = 8,
        window_days: int = 7
    ) -> dict[str, list[H3SpatialBin]]:
        """
        Partition detections into rolling or discrete temporal windows and compute H3 bins.
        Returns a mapping from window label ('YYYY-MM-DD_to_YYYY-MM-DD') to H3SpatialBin lists.
        """
        if not detections:
            return {}

        min_ts = min(d.timestamp for d in detections)
        max_ts = max(d.timestamp for d in detections)

        window_results: dict[str, list[H3SpatialBin]] = {}
        curr_start = min_ts

        while curr_start <= max_ts:
            curr_end = curr_start + timedelta(days=window_days)
            window_label = f"{curr_start.strftime('%Y-%m-%d')}_to_{curr_end.strftime('%Y-%m-%d')}"
            
            sub_dets = [d for d in detections if curr_start <= d.timestamp < curr_end]
            if sub_dets:
                bins = self.aggregate_by_h3(sub_dets, resolution=resolution, min_detections=1)
                window_results[window_label] = bins

            curr_start = curr_end

        return window_results

    def find_k_ring_candidates(
        self,
        center_lat: float,
        center_lon: float,
        k: int = 1,
        resolution: int = 8
    ) -> list[str]:
        """
        Return all H3 cell indices within k topological rings of a given coordinate.
        """
        center_h3 = compute_h3_index(center_lat, center_lon, resolution)
        return get_h3_k_ring(center_h3, k=k)
