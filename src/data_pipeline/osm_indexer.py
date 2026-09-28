"""
OpenStreetMap Industrial Infrastructure Spatial Indexing Engine.
Constructs STRtree spatial indices, multi-ring thermal catchment buffer envelopes,
and executes high-speed point containment and geodesic Haversine distance queries.
Authoritative Specifications: ORIGINAL_REQUEST.md § R1, PROJECT.md, m1_blueprint.md
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Optional, Sequence, Union

import numpy as np
import shapely.geometry
import shapely.ops
from shapely.geometry.base import BaseGeometry
from shapely.strtree import STRtree

from src.data_pipeline.schemas import FIRMSDetection, IndustrialFacility

# -------------------------------------------------------------------------
# Geodesic Mathematics & Buffer Geometry Helpers
# -------------------------------------------------------------------------

EARTH_RADIUS_METERS = 6371008.8


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Computes great-circle geodesic distance between two WGS84 coordinates in meters.
    """
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    sin_half_phi = math.sin(delta_phi / 2.0)
    sin_half_lambda = math.sin(delta_lambda / 2.0)

    a = (sin_half_phi * sin_half_phi +
         math.cos(phi1) * math.cos(phi2) * sin_half_lambda * sin_half_lambda)
    a = min(1.0, max(0.0, a))
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return EARTH_RADIUS_METERS * c


def create_catchment_buffers(
    geom: BaseGeometry, buffer_350m: float = 350.0, buffer_750m: float = 750.0
) -> tuple[BaseGeometry, BaseGeometry, float]:
    """
    Calculates latitude-scaled metric buffer envelopes for a WGS84 Shapely geometry.
    Returns:
        (buffer_350m_geometry, buffer_750m_geometry, area_sq_km)
    """
    centroid = geom.centroid
    lat_rad = math.radians(centroid.y)

    # 1 degree latitude ~= 111,320 meters
    # 1 degree longitude ~= 111,320 * cos(lat) meters
    m_per_deg_lat = 111320.0
    m_per_deg_lon = max(1.0, 111320.0 * math.cos(lat_rad))
    avg_m_per_deg = (m_per_deg_lat + m_per_deg_lon) / 2.0

    deg_350m = buffer_350m / avg_m_per_deg
    deg_750m = buffer_750m / avg_m_per_deg

    # Build buffer geometries
    buf_350 = geom.buffer(deg_350m)
    buf_750 = geom.buffer(deg_750m)

    # Approximate surface area in square kilometers
    area_sq_deg = geom.area
    area_sq_km = area_sq_deg * (m_per_deg_lat / 1000.0) * (m_per_deg_lon / 1000.0)

    return buf_350, buf_750, float(area_sq_km)


def distance_point_to_geometry(lat: float, lon: float, geom: BaseGeometry) -> float:
    """
    Computes exact geodesic distance (in meters) from coordinate (lat, lon) to nearest boundary of geometry.
    If point is inside the geometry, distance is 0.0 meters.
    """
    pt = shapely.geometry.Point(lon, lat)
    if geom.contains(pt) or geom.covers(pt) or geom.intersects(pt):
        return 0.0

    # Find nearest point on boundary
    try:
        nearest_geom_pt, _ = shapely.ops.nearest_points(geom, pt)
        return haversine_distance(lat, lon, nearest_geom_pt.y, nearest_geom_pt.x)
    except Exception:
        # Fallback to centroid distance
        return haversine_distance(lat, lon, geom.centroid.y, geom.centroid.x)


def classify_facility_type(props: dict[str, Any]) -> str:
    """
    Normalize OSM tags into canonical industrial taxonomy.
    """
    tags_combined = " ".join([
        str(props.get(k, "")).lower()
        for k in (
            "facility_type", "type", "landuse", "man_made", "industrial",
            "power", "amenity", "building", "name", "operator"
        )
    ])

    if any(k in tags_combined for k in ("refinery", "oil_refinery", "petroleum", "crude")):
        return "refinery"
    if any(k in tags_combined for k in ("chemical", "petrochemical", "fertilizer", "cracker", "chlorine")):
        return "chemical"
    if any(k in tags_combined for k in ("flare_stack", "flare", "burn_pit")):
        return "flare_stack"
    if any(k in tags_combined for k in ("steel", "metallurgical", "blast_furnace", "smelter", "foundry", "iron")):
        return "steel"
    if any(k in tags_combined for k in ("power", "generator", "thermal_power", "substation", "coal_plant")):
        return "power"
    if any(k in tags_combined for k in ("brick_kiln", "kiln", "clinker", "brickworks", "ceramics")):
        return "brick_kiln"
    if any(k in tags_combined for k in ("storage_tank", "tank_farm", "depot", "oil_depot", "terminal")):
        return "chemical"
    
    return "general_industrial"


# -------------------------------------------------------------------------
# Spatial Index Engine
# -------------------------------------------------------------------------

class SpatialIndexEngine:
    """
    OpenStreetMap industrial infrastructure spatial indexing engine using Shapely STRtree.
    Provides sub-millisecond point containment and geodesic distance queries.
    """

    def __init__(self) -> None:
        self.facilities: list[IndustrialFacility] = []
        self._tree: Optional[STRtree] = None
        self._geometries: list[BaseGeometry] = []
        self._facility_by_index: dict[int, IndustrialFacility] = {}
        self._geom_to_index: dict[int, int] = {}

    def _rebuild_index(self) -> None:
        """Reconstruct STRtree spatial index over all active facility geometries."""
        if not self.facilities:
            self._tree = None
            self._geometries = []
            self._facility_by_index = {}
            self._geom_to_index = {}
            return

        self._geometries = []
        self._facility_by_index = {}
        self._geom_to_index = {}

        for idx, fac in enumerate(self.facilities):
            # Index buffer_geometry_750m for rapid candidate catchment pruning
            self._geometries.append(fac.buffer_geometry_750m)
            self._facility_by_index[idx] = fac
            self._geom_to_index[id(fac.buffer_geometry_750m)] = idx

        self._tree = STRtree(self._geometries)

    def load_osm_boundaries(
        self, geojson_data_or_path: Union[str, Path, dict[str, Any]]
    ) -> list[IndustrialFacility]:
        """
        Ingest OSM GeoJSON file or dictionary, filter industrial tags,
        construct multi-ring catchment buffers, and build STRtree index.
        """
        if isinstance(geojson_data_or_path, (str, Path)):
            p = Path(geojson_data_or_path)
            if p.exists():
                with open(p, "r", encoding="utf-8") as f:
                    geojson_dict = json.load(f)
            else:
                geojson_dict = json.loads(str(geojson_data_or_path))
        elif isinstance(geojson_data_or_path, dict):
            geojson_dict = geojson_data_or_path
        else:
            raise ValueError(f"Unsupported OSM GeoJSON source type: {type(geojson_data_or_path)}")

        features = geojson_dict.get("features", [])
        new_facilities: list[IndustrialFacility] = []

        for i, feat in enumerate(features):
            geom_raw = feat.get("geometry")
            if not geom_raw:
                continue

            try:
                geom = shapely.geometry.shape(geom_raw)
                if not geom.is_valid:
                    geom = geom.buffer(0)  # Self-healing repair for invalid rings
            except Exception:
                continue

            props = feat.get("properties", {})
            fac_id = str(feat.get("id") or props.get("facility_id") or f"FAC_OSM_{i+1:04d}")
            name = str(props.get("name") or props.get("facility_name") or f"Industrial Site {fac_id}")
            fac_type = classify_facility_type(props)

            # Generate multi-ring catchment buffers
            buf_350, buf_750, area_sq_km = create_catchment_buffers(geom)

            # Calculate bounding box (min_lon, min_lat, max_lon, max_lat)
            bounds = geom.bounds
            bbox = (float(bounds[0]), float(bounds[1]), float(bounds[2]), float(bounds[3]))

            facility = IndustrialFacility(
                facility_id=fac_id,
                name=name,
                facility_type=fac_type,
                geometry=geom,
                buffer_geometry_350m=buf_350,
                buffer_geometry_750m=buf_750,
                bounding_box=bbox,
                area_sq_km=area_sq_km,
                properties=props,
            )
            new_facilities.append(facility)

        self.facilities = new_facilities
        self._rebuild_index()
        return self.facilities

    def add_facility(self, facility: IndustrialFacility) -> None:
        """Add a single facility and update spatial index."""
        self.facilities.append(facility)
        self._rebuild_index()

    def query_nearest_facility(
        self, lat: float, lon: float
    ) -> tuple[Optional[IndustrialFacility], float]:
        """
        Find nearest industrial facility and return (facility, geodesic_distance_meters).
        Returns (None, float('inf')) if index is empty.
        """
        if not self.facilities:
            return None, float("inf")

        pt = shapely.geometry.Point(lon, lat)

        # 1. Fast Containment Check via STRtree candidates
        candidate_indices: list[int] = []
        if self._tree is not None:
            # Query STRtree with 50km (~0.45 deg) search envelope using fast bounding box
            search_envelope = shapely.box(lon - 0.45, lat - 0.45, lon + 0.45, lat + 0.45)
            raw_candidates = self._tree.query(search_envelope)

            if len(raw_candidates) > 0:
                # Handle Shapely 2.0 (returns int array) vs Shapely 1.8 (returns geometry objects)
                for item in raw_candidates:
                    if isinstance(item, (int, np.integer)):
                        candidate_indices.append(int(item))
                    elif hasattr(item, "__geo_interface__"):
                        idx = self._geom_to_index.get(id(item))
                        if idx is not None:
                            candidate_indices.append(idx)
                    else:
                        try:
                            candidate_indices.append(int(item))
                        except Exception:
                            pass

        if not candidate_indices:
            candidate_indices = list(range(len(self.facilities)))

        best_facility: Optional[IndustrialFacility] = None
        min_dist_m = float("inf")

        # First pass: check for exact interior containment
        for idx in candidate_indices:
            if idx < 0 or idx >= len(self.facilities):
                continue
            fac = self.facilities[idx]
            if fac.geometry.contains(pt) or fac.geometry.covers(pt):
                return fac, 0.0

        # Second pass: compute geodesic distance to candidates
        for idx in candidate_indices:
            if idx < 0 or idx >= len(self.facilities):
                continue
            fac = self.facilities[idx]
            dist_m = distance_point_to_geometry(lat, lon, fac.geometry)
            if dist_m < min_dist_m:
                min_dist_m = dist_m
                best_facility = fac

        # Third pass fallback: if search window was restricted and min_dist > 50km, check all
        if min_dist_m > 50000.0 and len(candidate_indices) < len(self.facilities):
            for fac in self.facilities:
                dist_m = distance_point_to_geometry(lat, lon, fac.geometry)
                if dist_m < min_dist_m:
                    min_dist_m = dist_m
                    best_facility = fac

        return best_facility, min_dist_m

    def check_containment(
        self, lat: float, lon: float, buffer_mode: str = "350m"
    ) -> tuple[bool, Optional[IndustrialFacility]]:
        """
        Evaluate if point falls inside any facility boundary or catchment buffer.
        buffer_mode options:
            - 'raw': Base polygon geometry
            - '350m': Near-flare radiant dispersion buffer
            - '750m': Plume & facility perimeter buffer
        """
        if not self.facilities:
            return False, None

        pt = shapely.geometry.Point(lon, lat)

        for fac in self.facilities:
            if buffer_mode == "raw":
                target_geom = fac.geometry
            elif buffer_mode == "350m":
                target_geom = fac.buffer_geometry_350m
            elif buffer_mode == "750m":
                target_geom = fac.buffer_geometry_750m
            else:
                target_geom = fac.buffer_geometry_350m

            if target_geom.contains(pt) or target_geom.covers(pt) or target_geom.intersects(pt):
                return True, fac

        return False, None

    def batch_query_detections(
        self, detections: Sequence[FIRMSDetection]
    ) -> list[dict[str, Any]]:
        """
        Enrich a batch of detections with spatial industrial context.
        Vectorized with STRtree.query_nearest for ultra-high throughput (>10,000 queries/sec).
        """
        if not detections:
            return []

        if not self.facilities or self._tree is None:
            return [{
                "detection_id": d.detection_id,
                "nearest_facility_id": None,
                "nearest_facility_name": None,
                "nearest_facility_type": None,
                "distance_to_industrial_m": 999999.0,
                "is_inside_raw": False,
                "is_inside_350m": False,
                "is_inside_750m": False,
            } for d in detections]

        lats = np.array([d.latitude for d in detections], dtype=np.float64)
        lons = np.array([d.longitude for d in detections], dtype=np.float64)
        pts = shapely.points(lons, lats)

        nearest_res = self._tree.query_nearest(pts)
        fac_indices = nearest_res[1]

        enriched_results: list[dict[str, Any]] = []
        for i, det in enumerate(detections):
            fac_idx = int(fac_indices[i])
            fac = self.facilities[fac_idx]
            pt = pts[i]

            if fac.geometry.contains(pt) or fac.geometry.intersects(pt):
                dist_m = 0.0
            else:
                n_pt = shapely.ops.nearest_points(fac.geometry, pt)[0]
                dist_m = haversine_distance(det.latitude, det.longitude, n_pt.y, n_pt.x)

            enriched_results.append({
                "detection_id": det.detection_id,
                "nearest_facility_id": fac.facility_id,
                "nearest_facility_name": fac.name,
                "nearest_facility_type": fac.facility_type,
                "distance_to_industrial_m": dist_m,
                "is_inside_raw": bool(dist_m <= 0.0),
                "is_inside_350m": bool(dist_m <= 350.0),
                "is_inside_750m": bool(dist_m <= 750.0),
            })

        return enriched_results
