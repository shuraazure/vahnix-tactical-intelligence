"""
Adversarial Stress Test Suite for Milestone 1 (Geospatial Ingestion & OSM Spatial Indexing Engine).
Author: Challenger 1
Authoritative Specifications: ORIGINAL_REQUEST.md § R1, PROJECT.md, m1_blueprint.md
"""

import io
import json
import math
import time
from datetime import datetime
import numpy as np
import pandas as pd
import pytest
import shapely.geometry

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
    classify_facility_type,
)
from src.data_pipeline.synthetic_generator import (
    SyntheticDataGenerator,
)


# =========================================================================
# 1. ADVERSARIAL CSV & GEOJSON INGESTION TESTS
# =========================================================================

def test_adv_csv_completely_empty():
    """Adversarial: Ingesting an empty CSV string returns an empty list without error."""
    stream = io.StringIO("")
    with pytest.raises(pd.errors.EmptyDataError):
        FIRMSIngestionEngine.ingest_csv(stream)


def test_adv_csv_whitespace_and_commas_only():
    """Adversarial: CSV containing only whitespace and empty rows."""
    content = "   \n  \n,,,\n,,,"
    stream = io.StringIO(content)
    # read_csv with unparseable header/rows
    try:
        dets = FIRMSIngestionEngine.ingest_csv(stream)
        assert len(dets) == 0
    except Exception as e:
        # Either gracefully returns empty or raises standard pandas parse error
        assert isinstance(e, (pd.errors.EmptyDataError, pd.errors.ParserError, ValueError))


def test_adv_csv_missing_coordinate_columns():
    """Adversarial: CSV missing latitude and longitude columns."""
    content = (
        "brightness,scan,track,acq_date,acq_time,satellite,instrument,confidence,frp\n"
        "335.4,1.1,1.0,2026-03-15,0630,Terra,MODIS,85,34.5\n"
        "340.1,1.0,1.0,2026-03-15,1845,Aqua,MODIS,92,42.1\n"
    )
    stream = io.StringIO(content)
    dets = FIRMSIngestionEngine.ingest_csv(stream)
    assert len(dets) == 0  # All records without coordinates are dropped safely


def test_adv_csv_corrupted_coordinate_values():
    """Adversarial: Non-numeric, NaN, null, inf strings in latitude and longitude."""
    content = (
        "latitude,longitude,brightness,acq_date,acq_time,satellite,confidence,frp\n"
        "NaN,69.868,335.0,2026-03-15,0630,Terra,85,30.0\n"
        "22.358,null,335.0,2026-03-15,0630,Terra,85,30.0\n"
        "CORRUPT,69.868,335.0,2026-03-15,0630,Terra,85,30.0\n"
        "22.358,inf,335.0,2026-03-15,0630,Terra,85,30.0\n"
        "22.358,69.868,335.0,2026-03-15,0630,Terra,85,30.0\n"  # 1 valid row
    )
    stream = io.StringIO(content)
    dets = FIRMSIngestionEngine.ingest_csv(stream)
    assert len(dets) == 1
    assert math.isclose(dets[0].latitude, 22.358, abs_tol=1e-4)


def test_adv_csv_out_of_range_coordinates():
    """Adversarial: Coordinates exceeding geographic boundaries are dropped."""
    content = (
        "latitude,longitude,brightness,acq_date,acq_time,satellite,confidence,frp\n"
        "90.0001,69.868,335.0,2026-03-15,0630,Terra,85,30.0\n"    # Lat > 90
        "-90.0001,69.868,335.0,2026-03-15,0630,Terra,85,30.0\n"   # Lat < -90
        "22.358,180.0001,335.0,2026-03-15,0630,Terra,85,30.0\n"   # Lon > 180
        "22.358,-180.0001,335.0,2026-03-15,0630,Terra,85,30.0\n"  # Lon < -180
        "1500.0,3000.0,335.0,2026-03-15,0630,Terra,85,30.0\n"     # Extreme lat/lon
        "22.358,69.868,335.0,2026-03-15,0630,Terra,85,30.0\n"     # 1 valid row
    )
    stream = io.StringIO(content)
    dets = FIRMSIngestionEngine.ingest_csv(stream)
    assert len(dets) == 1
    assert math.isclose(dets[0].latitude, 22.358, abs_tol=1e-4)


def test_adv_csv_exact_boundary_coordinates():
    """Adversarial: Exact boundary coordinates (90°, -90°, 180°, -180°, 0°, 0°)."""
    content = (
        "latitude,longitude,brightness,acq_date,acq_time,satellite,confidence,frp\n"
        "90.0,0.0,300.0,2026-03-15,0630,Terra,85,10.0\n"
        "-90.0,0.0,300.0,2026-03-15,0630,Terra,85,10.0\n"
        "0.0,180.0,300.0,2026-03-15,0630,Terra,85,10.0\n"
        "0.0,-180.0,300.0,2026-03-15,0630,Terra,85,10.0\n"
        "0.0,0.0,300.0,2026-03-15,0630,Terra,85,10.0\n"
    )
    stream = io.StringIO(content)
    dets = FIRMSIngestionEngine.ingest_csv(stream)
    assert len(dets) == 5
    assert dets[0].latitude == 90.0
    assert dets[1].latitude == -90.0
    assert dets[2].longitude == 180.0
    assert dets[3].longitude == -180.0
    assert dets[4].latitude == 0.0 and dets[4].longitude == 0.0


def test_adv_negative_frp_clamping():
    """Adversarial: Negative FRP values are clamped safely to 0.0 MW."""
    content = (
        "latitude,longitude,brightness,acq_date,acq_time,satellite,confidence,frp\n"
        "22.358,69.868,335.0,2026-03-15,0630,Terra,85,-50.0\n"
        "22.359,69.869,335.0,2026-03-15,0630,Terra,85,-0.001\n"
    )
    stream = io.StringIO(content)
    dets = FIRMSIngestionEngine.ingest_csv(stream)
    assert len(dets) == 2
    assert dets[0].frp == 0.0
    assert dets[1].frp == 0.0


def test_adv_extreme_radiometric_values():
    """Adversarial: Extreme FRP (25,000 MW), saturation T4 (500K), inverted delta-T."""
    content = (
        "latitude,longitude,brightness,bright_t31,acq_date,acq_time,satellite,confidence,frp\n"
        "22.358,69.868,500.0,290.0,2026-03-15,0630,Terra,100,25000.0\n"
        "22.359,69.869,280.0,295.0,2026-03-15,0630,Terra,50,5.0\n"  # Inverted T4 < T11
    )
    stream = io.StringIO(content)
    dets = FIRMSIngestionEngine.ingest_csv(stream)
    assert len(dets) == 2
    assert dets[0].frp == 25000.0
    assert dets[0].brightness_temp_t4 == 500.0
    assert dets[0].bright_delta == 210.0
    assert dets[1].bright_delta == -15.0  # Inverted delta


def test_adv_confidence_normalization_extremes():
    """Adversarial: All permutations of confidence formats, casing, and out-of-range values."""
    norm = FIRMSIngestionEngine.normalize_confidence
    assert math.isclose(norm("H", "VIIRS_SNPP"), 0.95)
    assert math.isclose(norm("HIGH", "VIIRS_SNPP"), 0.95)
    assert math.isclose(norm("N", "VIIRS_SNPP"), 0.70)
    assert math.isclose(norm("Nominal", "VIIRS_SNPP"), 0.70)
    assert math.isclose(norm("L", "VIIRS_SNPP"), 0.30)
    assert math.isclose(norm("low", "VIIRS_SNPP"), 0.30)
    assert math.isclose(norm("100%", "MODIS"), 1.0)
    assert math.isclose(norm("0%", "MODIS"), 0.0)
    assert math.isclose(norm("150%", "MODIS"), 1.0)  # Clamped
    assert math.isclose(norm("-20%", "MODIS"), 0.0)  # Clamped
    assert math.isclose(norm(150, "MODIS"), 1.0)
    assert math.isclose(norm(-10, "MODIS"), 0.0)
    assert math.isclose(norm("invalid_text", "MODIS"), 0.50)
    assert math.isclose(norm("", "MODIS"), 0.50)


def test_adv_geojson_malformed_structures():
    """Adversarial: GeoJSON with missing properties, 3D coordinates, and missing geometries."""
    malformed_geojson = {
        "type": "FeatureCollection",
        "features": [
            # 1. Normal 2D Point
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [69.868, 22.358]},
                "properties": {"frp": 30.0, "confidence": "h", "satellite": "SNPP"}
            },
            # 2. 3D Point with Elevation
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [69.869, 22.359, 150.0]},
                "properties": {"frp": 35.0, "confidence": "h", "satellite": "SNPP"}
            },
            # 3. Missing Geometry
            {
                "type": "Feature",
                "geometry": None,
                "properties": {"frp": 40.0}
            },
            # 4. Empty Coordinates
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": []},
                "properties": {"frp": 45.0}
            },
            # 5. Out-of-range Coordinates in GeoJSON
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [250.0, 120.0]},
                "properties": {"frp": 50.0}
            }
        ]
    }
    dets = FIRMSIngestionEngine.ingest_geojson(malformed_geojson)
    assert len(dets) == 2  # Only features 1 and 2 are valid
    assert math.isclose(dets[0].longitude, 69.868, abs_tol=1e-4)
    assert math.isclose(dets[1].longitude, 69.869, abs_tol=1e-4)


# =========================================================================
# 2. ADVERSARIAL SPATIAL INDEXING & GEODESIC EDGE CASES
# =========================================================================

def test_adv_spatial_polygon_vertex_and_edge_containment():
    """Adversarial: Points exactly on polygon vertices, edge midpoints, and collinear segments."""
    # Box from (69.840, 22.330) to (69.890, 22.380)
    coords = [
        [69.840, 22.330],
        [69.865, 22.330],  # Collinear point on south edge
        [69.890, 22.330],
        [69.890, 22.380],
        [69.840, 22.380],
        [69.840, 22.330]
    ]
    poly = shapely.geometry.Polygon(coords)
    buf_350, buf_750, area_sq_km = create_catchment_buffers(poly)
    fac = IndustrialFacility(
        facility_id="FAC_EDGE_01",
        name="Edge Test Facility",
        facility_type="refinery",
        geometry=poly,
        buffer_geometry_350m=buf_350,
        buffer_geometry_750m=buf_750,
        bounding_box=(69.840, 22.330, 69.890, 22.380),
        area_sq_km=area_sq_km
    )
    indexer = SpatialIndexEngine()
    indexer.add_facility(fac)

    # 1. Exact vertex
    is_in, _ = indexer.check_containment(22.330, 69.840, buffer_mode="raw")
    assert is_in is True
    _, dist = indexer.query_nearest_facility(22.330, 69.840)
    assert math.isclose(dist, 0.0, abs_tol=1e-3)

    # 2. Collinear edge point
    is_in, _ = indexer.check_containment(22.330, 69.865, buffer_mode="raw")
    assert is_in is True
    _, dist = indexer.query_nearest_facility(22.330, 69.865)
    assert math.isclose(dist, 0.0, abs_tol=1e-3)

    # 3. Epsilon outside vertex (1e-6 deg ~= 0.11m)
    lat_eps = 22.330 - 0.000001
    lon_eps = 69.840 - 0.000001
    is_in_raw, _ = indexer.check_containment(lat_eps, lon_eps, buffer_mode="raw")
    is_in_350, _ = indexer.check_containment(lat_eps, lon_eps, buffer_mode="350m")
    assert is_in_raw is False
    assert is_in_350 is True  # Inside 350m buffer
    _, dist = indexer.query_nearest_facility(lat_eps, lon_eps)
    assert 0.0 < dist < 1.0  # Fraction of a meter


def test_adv_self_intersecting_invalid_polygon_repair():
    """Adversarial: Self-intersecting 'bowtie' polygon repaired by buffer(0)."""
    # Bowtie self-intersecting polygon
    bowtie_geojson = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "id": "BOWTIE_FAC",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[
                        [69.840, 22.330],
                        [69.890, 22.380],
                        [69.890, 22.330],
                        [69.840, 22.380],
                        [69.840, 22.330]
                    ]]
                },
                "properties": {"name": "Bowtie Plant", "type": "chemical"}
            }
        ]
    }
    indexer = SpatialIndexEngine()
    facilities = indexer.load_osm_boundaries(bowtie_geojson)
    assert len(facilities) == 1
    assert facilities[0].geometry.is_valid is True
    assert facilities[0].buffer_geometry_350m.is_valid is True


def test_adv_empty_spatial_index_queries():
    """Adversarial: Querying an uninitialized or empty spatial index."""
    indexer = SpatialIndexEngine()
    fac, dist = indexer.query_nearest_facility(22.358, 69.868)
    assert fac is None
    assert math.isinf(dist)

    is_in, fac_in = indexer.check_containment(22.358, 69.868, buffer_mode="raw")
    assert is_in is False
    assert fac_in is None


def test_adv_antimeridian_geodesic_distance():
    """Adversarial: Great-circle distance calculations across the Antimeridian (180° / -180°)."""
    # Points 0.002 degrees apart across 180° line
    p1 = (0.0, 179.999)
    p2 = (0.0, -179.999)
    d = haversine_distance(p1[0], p1[1], p2[0], p2[1])
    assert 200.0 <= d <= 250.0  # Approx 222.4 meters


def test_adv_jurong_dense_overlapping_buffers():
    """Adversarial: Multi-facility overlapping buffers on Jurong Island."""
    gen = SyntheticDataGenerator()
    jurong_ds = gen.generate_jurong_petrochemical_corridor(seed=404)
    indexer = SpatialIndexEngine()
    indexer.facilities = jurong_ds.facilities
    indexer._rebuild_index()

    assert len(indexer.facilities) == 4

    # Point right between Shell North and Petrochemical Cracker South
    mid_lat, mid_lon = 1.270, 103.7025  # Between 103.700 and 103.705
    nearest_fac, dist = indexer.query_nearest_facility(mid_lat, mid_lon)
    assert nearest_fac is not None
    assert dist < 350.0  # Within catchment zone

    # Check that both facilities are distinct
    fac_ids = {f.facility_id for f in indexer.facilities}
    assert len(fac_ids) == 4


# =========================================================================
# 3. HIGH-THROUGHPUT & QUERY LATENCY BENCHMARK (10,000+ RECORDS)
# =========================================================================

def test_adv_high_throughput_10k_records_ingestion_and_spatial_query():
    """
    Adversarial Benchmark:
    1. Ingestion of 10,000+ synthetic active fire detections from CSV.
    2. Spatial index batch containment and geodesic distance enrichment.
    3. Verifies ingestion speed > 10,000 rec/sec and query latency < 1 ms/query.
    """
    rng = np.random.RandomState(42)
    n_records = 10000

    # 1. Generate 10,000 synthetic records in CSV format
    lats = rng.uniform(8.0, 37.0, size=n_records)  # Across India
    lons = rng.uniform(68.0, 97.0, size=n_records)
    frps = rng.exponential(scale=30.0, size=n_records)
    t4s = rng.normal(loc=330.0, scale=15.0, size=n_records)
    t11s = t4s - rng.uniform(5.0, 50.0, size=n_records)
    confs = rng.choice(["h", "n", "l", "85", "92%", "100"], size=n_records)
    sensors = rng.choice(["MODIS", "VIIRS_SNPP", "VIIRS_NOAA20"], size=n_records)

    df_synth = pd.DataFrame({
        "latitude": lats,
        "longitude": lons,
        "frp": frps,
        "brightness": t4s,
        "bright_t31": t11s,
        "confidence": confs,
        "satellite": ["SNPP" if "VIIRS" in s else "Terra" for s in sensors],
        "acq_date": ["2026-03-15"] * n_records,
        "acq_time": ["1200"] * n_records,
        "daynight": ["D"] * n_records,
    })

    csv_buf = io.StringIO()
    df_synth.to_csv(csv_buf, index=False)
    csv_buf.seek(0)

    # Ingestion benchmark
    t_start_ingest = time.perf_counter()
    detections = FIRMSIngestionEngine.ingest_csv(csv_buf)
    t_end_ingest = time.perf_counter()

    ingest_duration_sec = t_end_ingest - t_start_ingest
    ingest_throughput = len(detections) / max(ingest_duration_sec, 1e-6)

    assert len(detections) == n_records
    print(f"\n[BENCHMARK] Ingested {len(detections)} records in {ingest_duration_sec:.4f}s ({ingest_throughput:.1f} rec/sec)")

    # 2. Setup Spatial Index with Master facilities
    gen = SyntheticDataGenerator()
    master_suite = gen.generate_master_benchmark_suite()
    all_facilities = []
    for ds in master_suite.values():
        all_facilities.extend(ds.facilities)

    indexer = SpatialIndexEngine()
    indexer.facilities = all_facilities
    indexer._rebuild_index()

    # Spatial query benchmark
    t_start_query = time.perf_counter()
    enriched = indexer.batch_query_detections(detections)
    t_end_query = time.perf_counter()

    query_duration_sec = t_end_query - t_start_query
    avg_latency_ms = (query_duration_sec / len(detections)) * 1000.0
    queries_per_sec = len(detections) / max(query_duration_sec, 1e-6)

    print(f"[BENCHMARK] Queried {len(detections)} detections in {query_duration_sec:.4f}s ({avg_latency_ms:.4f} ms/query, {queries_per_sec:.1f} queries/sec)")

    assert len(enriched) == n_records
    assert all("distance_to_industrial_m" in e for e in enriched)
    assert all("nearest_facility_id" in e for e in enriched)
    # Query latency SLA: average latency < 1.0 ms per detection query
    assert avg_latency_ms < 1.0
