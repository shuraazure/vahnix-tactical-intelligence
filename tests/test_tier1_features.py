"""
Tier 1: Feature Coverage and Isolation Tests for NTRO Thermal Detection System
Authoritative Specifications: ORIGINAL_REQUEST.md, PROJECT.md, TEST_INFRA.md, m1_blueprint.md
"""

from __future__ import annotations

import io
import math
from datetime import datetime
from pathlib import Path
import pytest
import shapely.geometry
import numpy as np

from src.data_pipeline.schemas import (
    FIRMSDetection,
    IndustrialFacility,
    NormalizedDataset,
)
from src.data_pipeline.firms_ingestion import (
    FIRMSIngestionEngine,
    compute_h3_cell,
)
from src.data_pipeline.osm_indexer import (
    SpatialIndexEngine,
    haversine_distance,
    create_catchment_buffers,
    distance_point_to_geometry,
)
from src.data_pipeline.synthetic_generator import (
    SyntheticDataGenerator,
)


# =========================================================================
# Feature 1: FIRMS Ingestion Engine Tests
# =========================================================================

def test_modis_csv_ingestion(sample_modis_csv_content: str):
    """Verify standard MODIS CSV ingestion, schema extraction, and derived fields."""
    stream = io.StringIO(sample_modis_csv_content)
    detections = FIRMSIngestionEngine.ingest_csv(stream)

    assert len(detections) == 5
    d0 = detections[0]
    assert d0.sensor == "MODIS"
    assert d0.satellite == "Terra"
    assert math.isclose(d0.latitude, 22.3582, abs_tol=1e-4)
    assert math.isclose(d0.longitude, 69.8681, abs_tol=1e-4)
    assert d0.acq_date == "2026-03-15"
    assert d0.acq_time == "0630"
    assert d0.timestamp == datetime(2026, 3, 15, 6, 30)
    assert d0.daynight == "D"
    assert math.isclose(d0.confidence, 0.85, abs_tol=1e-3)
    assert math.isclose(d0.brightness_temp_t4, 335.4, abs_tol=1e-2)
    assert math.isclose(d0.brightness_temp_t11, 298.2, abs_tol=1e-2)
    assert math.isclose(d0.bright_delta, 335.4 - 298.2, abs_tol=1e-2)
    assert math.isclose(d0.frp, 34.5, abs_tol=1e-2)


def test_viirs_snpp_csv_ingestion(sample_viirs_csv_content: str):
    """Verify VIIRS CSV ingestion with categorical confidence ('h', 'n') mapping."""
    stream = io.StringIO(sample_viirs_csv_content)
    detections = FIRMSIngestionEngine.ingest_csv(stream)

    assert len(detections) == 5
    d0 = detections[0]
    assert d0.sensor == "VIIRS_SNPP"
    assert d0.satellite == "SNPP"
    assert math.isclose(d0.confidence, 0.95, abs_tol=1e-3)  # 'h' -> 0.95
    assert math.isclose(d0.brightness_temp_t4, 345.8, abs_tol=1e-2)
    assert math.isclose(d0.brightness_temp_t11, 294.5, abs_tol=1e-2)
    assert math.isclose(d0.bright_delta, 345.8 - 294.5, abs_tol=1e-2)
    assert math.isclose(d0.frp, 28.4, abs_tol=1e-2)
    assert d0.daynight == "D"

    # Row 3 has 'n' confidence
    d3 = detections[3]
    assert math.isclose(d3.confidence, 0.70, abs_tol=1e-3)  # 'n' -> 0.70
    assert d3.sensor == "VIIRS_NOAA20"
    assert d3.satellite == "NOAA-20"


def test_viirs_geojson_ingestion():
    """Verify GeoJSON format active fire ingestion."""
    geojson_data = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "id": "DET_GEOJSON_001",
                "geometry": {"type": "Point", "coordinates": [69.8680, 22.3585]},
                "properties": {
                    "bright_ti4": 348.0,
                    "bright_ti5": 292.0,
                    "frp": 32.0,
                    "confidence": "h",
                    "daynight": "N",
                    "acq_date": "2026-03-15",
                    "acq_time": "2030",
                    "satellite": "SNPP",
                    "instrument": "VIIRS"
                }
            }
        ]
    }
    detections = FIRMSIngestionEngine.ingest_geojson(geojson_data)
    assert len(detections) == 1
    d = detections[0]
    assert d.detection_id == "DET_GEOJSON_001"
    assert math.isclose(d.latitude, 22.3585, abs_tol=1e-4)
    assert math.isclose(d.longitude, 69.8680, abs_tol=1e-4)
    assert d.daynight == "N"
    assert math.isclose(d.confidence, 0.95, abs_tol=1e-3)


def test_confidence_normalization_rules():
    """Verify all confidence normalization mappings for continuous and categorical inputs."""
    norm = FIRMSIngestionEngine.normalize_confidence
    # Categorical VIIRS
    assert math.isclose(norm("h", "VIIRS_SNPP"), 0.95)
    assert math.isclose(norm("high", "VIIRS_SNPP"), 0.95)
    assert math.isclose(norm("n", "VIIRS_SNPP"), 0.70)
    assert math.isclose(norm("nominal", "VIIRS_SNPP"), 0.70)
    assert math.isclose(norm("l", "VIIRS_SNPP"), 0.30)
    assert math.isclose(norm("low", "VIIRS_SNPP"), 0.30)

    # MODIS Percentage (numeric and string)
    assert math.isclose(norm(85, "MODIS"), 0.85)
    assert math.isclose(norm(100, "MODIS"), 1.0)
    assert math.isclose(norm(0, "MODIS"), 0.0)
    assert math.isclose(norm("92%", "MODIS"), 0.92)
    assert math.isclose(norm("78", "MODIS"), 0.78)
    assert math.isclose(norm(0.85, "MODIS"), 0.85)

    # Missing / None fallback
    assert math.isclose(norm(None, "MODIS"), 0.50)
    assert math.isclose(norm(float("nan"), "VIIRS_SNPP"), 0.50)


def test_timestamp_and_time_parsing():
    """Verify acquisition time normalization and UTC timestamp combination."""
    parse_time = FIRMSIngestionEngine.parse_acq_time
    assert parse_time("630") == "0630"
    assert parse_time(630) == "0630"
    assert parse_time(630.0) == "0630"
    assert parse_time("0630") == "0630"
    assert parse_time("1845") == "1845"
    assert parse_time("06:30") == "0630"

    parse_ts = FIRMSIngestionEngine.parse_timestamp
    dt1 = parse_ts("2026-03-15", "0630")
    assert dt1 == datetime(2026, 3, 15, 6, 30)

    dt2 = parse_ts("2026/03/15", "1845")
    assert dt2 == datetime(2026, 3, 15, 18, 45)


# =========================================================================
# Feature 2 & 4: H3 Discrete Global Grid & Schemas
# =========================================================================

def test_h3_spatial_binning_resolutions():
    """Verify H3 cell index generation across resolutions 7, 8, and 9."""
    lat, lon = 22.3582, 69.8681
    h3_7 = compute_h3_cell(lat, lon, 7)
    h3_8 = compute_h3_cell(lat, lon, 8)
    h3_9 = compute_h3_cell(lat, lon, 9)

    assert isinstance(h3_7, str) and len(h3_7) >= 12
    assert isinstance(h3_8, str) and len(h3_8) >= 12
    assert isinstance(h3_9, str) and len(h3_9) >= 12
    assert h3_7.startswith("87") or h3_7.startswith("8")
    assert h3_8.startswith("88") or h3_8.startswith("8")
    assert h3_9.startswith("89") or h3_9.startswith("8")


def test_firms_detection_dataclass_invariants():
    """Verify FIRMSDetection dataclass range assertions and validations."""
    # Valid instance
    det = FIRMSDetection(
        detection_id="DET_TEST_001",
        latitude=22.35,
        longitude=69.85,
        acq_date="2026-03-15",
        acq_time="0630",
        timestamp=datetime(2026, 3, 15, 6, 30),
        sensor="MODIS",
        satellite="Terra",
        frp=35.0,
        brightness_temp_t4=335.0,
        brightness_temp_t11=295.0,
        bright_delta=40.0,
        confidence=0.85,
        daynight="D"
    )
    assert det.detection_id == "DET_TEST_001"

    # Invariant failure: invalid latitude
    with pytest.raises(ValueError, match="Invalid latitude"):
        FIRMSDetection(
            detection_id="BAD_LAT",
            latitude=95.0,
            longitude=69.85,
            acq_date="2026-03-15",
            acq_time="0630",
            timestamp=datetime(2026, 3, 15, 6, 30),
            sensor="MODIS",
            satellite="Terra",
            frp=35.0,
            brightness_temp_t4=335.0,
            brightness_temp_t11=295.0,
            bright_delta=40.0,
            confidence=0.85,
            daynight="D"
        )

    # Invariant failure: negative FRP
    with pytest.raises(ValueError, match="Invalid negative FRP"):
        FIRMSDetection(
            detection_id="BAD_FRP",
            latitude=22.35,
            longitude=69.85,
            acq_date="2026-03-15",
            acq_time="0630",
            timestamp=datetime(2026, 3, 15, 6, 30),
            sensor="MODIS",
            satellite="Terra",
            frp=-10.0,
            brightness_temp_t4=335.0,
            brightness_temp_t11=295.0,
            bright_delta=40.0,
            confidence=0.85,
            daynight="D"
        )


def test_firms_detection_dataframe_conversions(sample_firms_detections: list[FIRMSDetection]):
    """Verify DataFrame roundtrip conversion via to_dataframe and from_dataframe."""
    df = FIRMSDetection.to_dataframe(sample_firms_detections)
    assert len(df) == len(sample_firms_detections)
    assert "frp" in df.columns
    assert "bright_delta" in df.columns

    reconstructed = FIRMSDetection.from_dataframe(df)
    assert len(reconstructed) == len(sample_firms_detections)
    for orig, rec in zip(sample_firms_detections, reconstructed):
        assert orig.detection_id == rec.detection_id
        assert math.isclose(orig.latitude, rec.latitude, abs_tol=1e-5)
        assert math.isclose(orig.longitude, rec.longitude, abs_tol=1e-5)
        assert math.isclose(orig.frp, rec.frp, abs_tol=1e-3)
        assert orig.daynight == rec.daynight


# =========================================================================
# Feature 2 & 3: OSM Spatial Indexing & Geodesic Distance Tests
# =========================================================================

def test_haversine_geodesic_distance_accuracy():
    """Verify great-circle distance computation against known geographic coordinates."""
    # 1 degree latitude at equator ~= 111,195 m
    d_lat = haversine_distance(0.0, 0.0, 1.0, 0.0)
    assert math.isclose(d_lat, 111195.0, rel_tol=1e-2)

    # Identical points -> 0.0 meters
    d_zero = haversine_distance(22.3582, 69.8681, 22.3582, 69.8681)
    assert math.isclose(d_zero, 0.0, abs_tol=1e-6)

    # Jamnagar (22.3582, 69.8681) to Mumbai Trombay (18.9950, 72.8950) ~= 489 km
    d_jam_mum = haversine_distance(22.3582, 69.8681, 18.9950, 72.8950)
    assert 480000.0 <= d_jam_mum <= 500000.0


def test_osm_strtree_spatial_index_building(sample_jamnagar_geojson: dict):
    """Verify loading OSM GeoJSON into STRtree spatial index."""
    indexer = SpatialIndexEngine()
    facilities = indexer.load_osm_boundaries(sample_jamnagar_geojson)

    assert len(facilities) == 1
    fac = facilities[0]
    assert fac.facility_id == "IND_JAMNAGAR_001"
    assert fac.facility_type == "refinery"
    assert fac.area_sq_km > 20.0
    assert indexer._tree is not None


def test_point_in_polygon_exact_containment(sample_jamnagar_geojson: dict):
    """Verify point inside industrial polygon returns true containment and distance 0.0."""
    indexer = SpatialIndexEngine()
    indexer.load_osm_boundaries(sample_jamnagar_geojson)

    # Point strictly inside Jamnagar bounding box [69.840, 22.330] to [69.890, 22.380]
    lat_in, lon_in = 22.3582, 69.8681
    is_contained_raw, fac_raw = indexer.check_containment(lat_in, lon_in, buffer_mode="raw")
    assert is_contained_raw is True
    assert fac_raw is not None
    assert fac_raw.facility_id == "IND_JAMNAGAR_001"

    fac_nearest, dist_m = indexer.query_nearest_facility(lat_in, lon_in)
    assert fac_nearest is not None
    assert fac_nearest.facility_id == "IND_JAMNAGAR_001"
    assert math.isclose(dist_m, 0.0, abs_tol=1e-3)


def test_multi_ring_catchment_buffering(sample_jamnagar_geojson: dict):
    """
    Verify multi-ring buffering:
    - Point 200m outside is inside 350m and 750m buffers, but outside raw.
    - Point 500m outside is inside 750m buffer, but outside 350m buffer.
    - Point 2000m outside is outside all buffers.
    """
    indexer = SpatialIndexEngine()
    indexer.load_osm_boundaries(sample_jamnagar_geojson)

    # North boundary is at lat = 22.380.
    # 1 deg lat ~= 111,320m -> 200m ~= 0.0018 deg, 500m ~= 0.0045 deg, 2000m ~= 0.0180 deg
    lon_mid = 69.865

    # 1. Point ~200m North of boundary (lat = 22.3818)
    lat_200m = 22.380 + (200.0 / 111320.0)
    in_raw, _ = indexer.check_containment(lat_200m, lon_mid, buffer_mode="raw")
    in_350, _ = indexer.check_containment(lat_200m, lon_mid, buffer_mode="350m")
    in_750, _ = indexer.check_containment(lat_200m, lon_mid, buffer_mode="750m")
    assert in_raw is False
    assert in_350 is True
    assert in_750 is True

    # 2. Point ~500m North of boundary (lat = 22.3845)
    lat_500m = 22.380 + (500.0 / 111320.0)
    in_raw, _ = indexer.check_containment(lat_500m, lon_mid, buffer_mode="raw")
    in_350, _ = indexer.check_containment(lat_500m, lon_mid, buffer_mode="350m")
    in_750, _ = indexer.check_containment(lat_500m, lon_mid, buffer_mode="750m")
    assert in_raw is False
    assert in_350 is False
    assert in_750 is True

    # 3. Point ~2000m North of boundary (lat = 22.3980)
    lat_2000m = 22.380 + (2000.0 / 111320.0)
    in_raw, _ = indexer.check_containment(lat_2000m, lon_mid, buffer_mode="raw")
    in_350, _ = indexer.check_containment(lat_2000m, lon_mid, buffer_mode="350m")
    in_750, _ = indexer.check_containment(lat_2000m, lon_mid, buffer_mode="750m")
    assert in_raw is False
    assert in_350 is False
    assert in_750 is False


def test_batch_query_detections(sample_jamnagar_geojson: dict, sample_firms_detections: list[FIRMSDetection]):
    """Verify batch spatial enrichment adds nearest facility and multi-ring flags."""
    indexer = SpatialIndexEngine()
    indexer.load_osm_boundaries(sample_jamnagar_geojson)

    enriched = indexer.batch_query_detections(sample_firms_detections)
    assert len(enriched) == len(sample_firms_detections)

    # First detection is Jamnagar (inside)
    e0 = enriched[0]
    assert e0["nearest_facility_id"] == "IND_JAMNAGAR_001"
    assert e0["is_inside_raw"] is True
    assert math.isclose(e0["distance_to_industrial_m"], 0.0, abs_tol=1e-3)

    # Stubble burning detection (Punjab) should be far away
    stubble_entry = next(e for e in enriched if "STUBBLE" in e["detection_id"])
    assert stubble_entry["is_inside_raw"] is False
    assert stubble_entry["distance_to_industrial_m"] > 500000.0  # Punjab to Jamnagar > 500km


# =========================================================================
# Feature 5: Synthetic Data Generator Tests
# =========================================================================

def test_synthetic_jamnagar_corridor():
    """Verify Jamnagar refinery corridor: steady FRP, balanced day/night, inside industrial."""
    gen = SyntheticDataGenerator()
    ds = gen.generate_jamnagar_corridor(num_days=30, seed=42)

    assert ds.dataset_name == "jamnagar_refinery_corridor"
    assert len(ds.detections) > 50
    assert len(ds.facilities) == 1

    # Check Day/Night balance (DNBI)
    day_count = sum(1 for d in ds.detections if d.daynight == "D")
    night_count = sum(1 for d in ds.detections if d.daynight == "N")
    dnbi = 2.0 * min(day_count, night_count) / (day_count + night_count)
    assert dnbi >= 0.70  # Highly balanced

    # Check FRP profile
    frps = [d.frp for d in ds.detections]
    assert 20.0 <= np.mean(frps) <= 75.0


def test_synthetic_chemical_disaster_corridor():
    """Verify Chemical Depot disaster corridor: sudden surge in FRP and Delta-T."""
    gen = SyntheticDataGenerator()
    ds = gen.generate_chemical_disaster_corridor(seed=101)

    assert ds.dataset_name == "trombay_chemical_disaster_corridor"
    assert len(ds.detections) >= 10
    max_frp = max(d.frp for d in ds.detections)
    assert max_frp > 1000.0  # Catastrophic surge

    max_delta = max(d.bright_delta for d in ds.detections)
    assert max_delta > 60.0


def test_synthetic_punjab_stubble_corridor():
    """Verify Punjab stubble corridor: 100% Day, moderate FRP, rural."""
    gen = SyntheticDataGenerator()
    ds = gen.generate_punjab_stubble_corridor(num_detections=100, seed=202)

    assert ds.dataset_name == "punjab_stubble_burning_corridor"
    assert len(ds.detections) == 100
    # 100% Daytime
    assert all(d.daynight == "D" for d in ds.detections)


def test_synthetic_simlipal_wildfire_corridor():
    """Verify Simlipal forest wildfire: dynamic front propagation."""
    gen = SyntheticDataGenerator()
    ds = gen.generate_simlipal_wildfire_corridor(num_days=5, seed=303)

    assert ds.dataset_name == "simlipal_wildfire_corridor"
    assert len(ds.detections) >= 20
    # Both day and night detections present
    assert any(d.daynight == "D" for d in ds.detections)
    assert any(d.daynight == "N" for d in ds.detections)


def test_synthetic_jurong_corridor():
    """Verify Jurong Island corridor: multi-facility industrial isolation."""
    gen = SyntheticDataGenerator()
    ds = gen.generate_jurong_petrochemical_corridor(seed=404)

    assert ds.dataset_name == "jurong_petrochemical_corridor"
    assert len(ds.facilities) >= 4
    assert len(ds.detections) > 30


def test_master_benchmark_suite_generation():
    """Verify Master benchmark suite generation produces all 5 corridors."""
    gen = SyntheticDataGenerator()
    suite = gen.generate_master_benchmark_suite()

    assert "jamnagar" in suite
    assert "chemical_disaster" in suite
    assert "punjab_stubble" in suite
    assert "simlipal_wildfire" in suite
    assert "jurong" in suite


# =========================================================================
# Shared Mock & Parser Helpers for Tier 2, 3, 4 Tests
# =========================================================================

def parse_modis_csv_line(line: str) -> dict:
    if not line or not line.strip():
        return {}
    parts = [p.strip() for p in line.strip().split(",")]
    if parts[0] == "latitude" or len(parts) < 10:
        return {}
    try:
        conf_str = parts[9]
        if "%" in conf_str:
            conf = float(conf_str.replace("%", "")) / 100.0
        else:
            try:
                c_val = float(conf_str)
                conf = c_val / 100.0 if c_val > 1.0 else c_val
            except ValueError:
                conf = 0.50
        t4 = float(parts[2])
        t11 = float(parts[11]) if len(parts) > 11 else 280.0
        return {
            "latitude": float(parts[0]),
            "longitude": float(parts[1]),
            "brightness": t4,
            "brightness_temp_t4": t4,
            "scan": float(parts[3]),
            "track": float(parts[4]),
            "acq_date": parts[5],
            "acq_time": parts[6],
            "satellite": parts[7],
            "instrument": parts[8],
            "sensor": parts[8],
            "confidence": conf,
            "version": parts[10] if len(parts) > 10 else "6.1NRT",
            "bright_t31": t11,
            "brightness_temp_t11": t11,
            "bright_delta": t4 - t11,
            "frp": float(parts[12]) if len(parts) > 12 else 10.0,
            "daynight": parts[13] if len(parts) > 13 else "D",
        }
    except Exception:
        return {}


def parse_viirs_csv_line(line: str) -> dict:
    rec = parse_modis_csv_line(line)
    if not rec:
        return {}
    parts = [p.strip() for p in line.strip().split(",")]
    c = parts[9].lower()
    if c == "h":
        rec["confidence"] = 0.95
    elif c == "n":
        rec["confidence"] = 0.70
    elif c == "l":
        rec["confidence"] = 0.30
    elif c in ("x", "unknown") or not c.replace(".", "", 1).isdigit():
        rec["confidence"] = 0.50
    return rec


def point_in_polygon_bbox(point: tuple[float, float], bbox: tuple[float, float, float, float]) -> bool:
    min_lon, min_lat, max_lon, max_lat = bbox
    lon, lat = point
    return min_lon <= lon <= max_lon and min_lat <= lat <= max_lat


def mock_lat_lon_to_h3(lat: float, lon: float, res: int = 8) -> str:
    try:
        import h3
        return h3.latlng_to_cell(lat, lon, res)
    except Exception:
        h = abs(hash((round(lat, 3), round(lon, 3), res))) % (16 ** 13)
        return f"8{res}{h:013x}"[:15]


def mock_haversine_dbscan(points: list[tuple[float, float]], eps_m: float = 1000.0, min_samples: int = 3) -> list[int]:
    from tests.conftest import haversine_distance
    if not points:
        return []
    n = len(points)
    labels = [-1] * n
    cluster_id = 0
    visited = [False] * n

    for i in range(n):
        if visited[i]:
            continue
        visited[i] = True
        pt = points[i]
        neighbors = [j for j, other in enumerate(points) if haversine_distance(pt[0], pt[1], other[0], other[1]) <= eps_m]
        if len(neighbors) < min_samples:
            labels[i] = -1
        else:
            labels[i] = cluster_id
            queue = [j for j in neighbors if j != i]
            while queue:
                j = queue.pop(0)
                if not visited[j]:
                    visited[j] = True
                    other_pt = points[j]
                    j_neighbors = [k for k, p_k in enumerate(points) if haversine_distance(other_pt[0], other_pt[1], p_k[0], p_k[1]) <= eps_m]
                    if len(j_neighbors) >= min_samples:
                        for k in j_neighbors:
                            if not visited[k] and k not in queue:
                                queue.append(k)
                if labels[j] == -1:
                    labels[j] = cluster_id
            cluster_id += 1
    return labels


class MockEnsembleClassifier:
    def __init__(self, random_state: int = 42):
        self.random_state = random_state

    def predict(self, X: np.ndarray) -> np.ndarray:
        from tests.conftest import heuristic_baseline_classify
        preds = []
        for row in X:
            preds.append(heuristic_baseline_classify(row).value)
        return np.array(preds, dtype=int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        preds = self.predict(X)
        probs = np.full((len(X), 4), 0.05)
        for i, p in enumerate(preds):
            probs[i, p] = 0.85
        return probs / probs.sum(axis=1, keepdims=True)


def mock_shap_explain_instance(features: np.ndarray, pred_class: int = 0, detection_id: str = "DET_TEST_001") -> Any:
    from tests.conftest import SHAPExplanation, ThermalAnomalyClass, FEATURE_NAMES_28D
    shap_dict = {name: (1.2 if i == 0 else (0.8 if i == 3 else 0.0)) for i, name in enumerate(FEATURE_NAMES_28D)}
    label = ThermalAnomalyClass(pred_class).name
    return SHAPExplanation(
        detection_id=detection_id,
        target_class=ThermalAnomalyClass(pred_class),
        base_value=0.25,
        shap_values=shap_dict,
        top_positive_features=[("frp", 1.2, float(features[0]))],
        top_negative_features=[("dist_to_industrial_m", -0.5, float(features[17]))],
        waterfall_plot_base64="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==",
        natural_language_summary=f"Classified as {label} (+1.20 frp). High FRP indicates significant thermal emission."
    )


def build_pydeck_offline_spec(detections: list[Any]) -> dict:
    data = []
    for d in detections:
        data.append({
            "lat": getattr(d, "latitude", 0.0),
            "lon": getattr(d, "longitude", 0.0),
            "frp": getattr(d, "frp", 0.0),
        })
    return {
        "mapStyle": None,
        "views": [{"type": "MapView", "controller": True}],
        "layers": [{"type": "ScatterplotLayer", "data": data}],
    }

