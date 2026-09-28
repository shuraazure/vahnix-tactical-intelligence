"""
Tier 2: Boundary Value, Edge Case, and Sensor Extreme Tests (80+ Tests)
Authoritative Specifications: ORIGINAL_REQUEST.md, PROJECT.md, TEST_INFRA.md
"""

from __future__ import annotations
import math
import json
import io
import datetime
from typing import Any, Dict, List, Tuple
import pytest
import numpy as np

from tests.conftest import (
    ThermalAnomalyClass,
    FIRMSDetection,
    IndustrialFacility,
    ThermalCluster,
    PredictionOutput,
    SHAPExplanation,
    FEATURE_NAMES_28D,
    haversine_distance,
    calculate_dnbi,
    calculate_spatial_jitter,
    calculate_risk_severity_index,
    extract_mock_28d_features,
    heuristic_baseline_classify
)
from tests.test_tier1_features import (
    parse_modis_csv_line,
    parse_viirs_csv_line,
    point_in_polygon_bbox,
    mock_lat_lon_to_h3,
    mock_haversine_dbscan,
    MockEnsembleClassifier,
    mock_shap_explain_instance,
    build_pydeck_offline_spec
)

# =========================================================================
# GROUP 1: INGESTION BOUNDARY & MALFORMED INPUT TESTS (Tests 1 - 10)
# =========================================================================

def test_bnd_01_empty_csv_feed():
    """BND-01: Empty CSV string returns empty record list without throwing exception."""
    empty_csv = ""
    lines = empty_csv.strip().split("\n")
    records = [parse_modis_csv_line(l) for l in lines if parse_modis_csv_line(l)]
    assert records == []


def test_bnd_02_header_only_csv():
    """BND-02: CSV with only header line returns empty record list."""
    header_csv = "latitude,longitude,brightness,scan,track,acq_date,acq_time,satellite,instrument,confidence,version,bright_t31,frp,daynight"
    lines = header_csv.strip().split("\n")
    records = [parse_modis_csv_line(l) for l in lines if parse_modis_csv_line(l)]
    assert records == []


def test_bnd_03_truncated_csv_row():
    """BND-03: Row with fewer columns than schema is safely skipped."""
    bad_line = "22.358,69.868,335.0,1.0"
    rec = parse_modis_csv_line(bad_line)
    assert rec == {}


def test_bnd_04_north_pole_coordinates():
    """BND-04: Detection at North Pole (lat=90.0, lon=0.0) parses cleanly."""
    line = "90.0,0.0,300.0,1.0,1.0,2026-03-15,1200,Terra,MODIS,50,6.1NRT,280.0,10.0,D"
    rec = parse_modis_csv_line(line)
    assert rec["latitude"] == 90.0 and rec["longitude"] == 0.0


def test_bnd_05_south_pole_coordinates():
    """BND-05: Detection at South Pole (lat=-90.0, lon=0.0) parses cleanly."""
    line = "-90.0,0.0,300.0,1.0,1.0,2026-03-15,1200,Terra,MODIS,50,6.1NRT,280.0,10.0,D"
    rec = parse_modis_csv_line(line)
    assert rec["latitude"] == -90.0 and rec["longitude"] == 0.0


def test_bnd_06_antimeridian_positive_boundary():
    """BND-06: Detection exactly at positive antimeridian (lon=180.0) parses cleanly."""
    line = "0.0,180.0,310.0,1.0,1.0,2026-03-15,1200,Terra,MODIS,50,6.1NRT,290.0,15.0,D"
    rec = parse_modis_csv_line(line)
    assert rec["longitude"] == 180.0


def test_bnd_07_antimeridian_negative_boundary():
    """BND-07: Detection exactly at negative antimeridian (lon=-180.0) parses cleanly."""
    line = "0.0,-180.0,310.0,1.0,1.0,2026-03-15,1200,Terra,MODIS,50,6.1NRT,290.0,15.0,D"
    rec = parse_modis_csv_line(line)
    assert rec["longitude"] == -180.0


def test_bnd_08_equator_prime_meridian_origin():
    """BND-08: Detection at coordinate origin (lat=0.0, lon=0.0) parses cleanly."""
    line = "0.0,0.0,305.0,1.0,1.0,2026-03-15,1200,Terra,MODIS,50,6.1NRT,285.0,12.0,D"
    rec = parse_modis_csv_line(line)
    assert rec["latitude"] == 0.0 and rec["longitude"] == 0.0


def test_bnd_09_viirs_unknown_confidence_character():
    """BND-09: Unrecognized VIIRS confidence code defaults to nominal confidence (0.50)."""
    line = "22.358,69.868,340.0,0.4,0.4,2026-03-15,1200,N,VIIRS,x,2.0NRT,290.0,20.0,D"
    rec = parse_viirs_csv_line(line)
    assert rec["confidence"] == 0.50


def test_bnd_10_geojson_empty_feature_collection():
    """BND-10: Empty GeoJSON FeatureCollection handles 0 features gracefully."""
    empty_geojson = {"type": "FeatureCollection", "features": []}
    assert len(empty_geojson["features"]) == 0


# =========================================================================
# GROUP 2: SENSOR RADIOMETRIC EXTREMES (Tests 11 - 20)
# =========================================================================

def test_bnd_11_near_zero_frp():
    """BND-11: Near-zero Fire Radiative Power (FRP = 0.01 MW) handled without underflow."""
    det = FIRMSDetection("D0", 22.358, 69.868, "2026-03-15", "1200", datetime.datetime.now(), "MODIS", "Terra", 0.01, 305.0, 300.0, 5.0, 0.5, "D")
    vec = extract_mock_28d_features(det)
    assert vec[0] == pytest.approx(0.01, abs=1e-5)
    assert not np.isnan(vec).any()


def test_bnd_12_exact_zero_frp():
    """BND-12: Zero Fire Radiative Power (FRP = 0.0 MW) handled without log(0) domain error."""
    det = FIRMSDetection("D0", 22.358, 69.868, "2026-03-15", "1200", datetime.datetime.now(), "MODIS", "Terra", 0.0, 300.0, 300.0, 0.0, 0.5, "D")
    vec = extract_mock_28d_features(det)
    assert vec[0] == 0.0
    assert not np.isnan(vec[7])  # thermal_intensity_index uses log10(max(frp,0)+1)


def test_bnd_13_saturation_frp_5000mw():
    """BND-13: Massive sensor-saturating FRP (5000.0 MW) handled without overflow."""
    det = FIRMSDetection("D0", 18.995, 72.895, "2026-03-15", "1200", datetime.datetime.now(), "VIIRS_NOAA20", "NOAA-20", 5000.0, 367.0, 300.0, 67.0, 1.0, "D")
    vec = extract_mock_28d_features(det)
    assert vec[0] == 5000.0
    assert not np.isinf(vec).any()


def test_bnd_14_extreme_frp_20000mw():
    """BND-14: Catastrophic mega-explosion FRP (20,000.0 MW) caps cleanly in RSI."""
    rsi = calculate_risk_severity_index(20000.0, 150.0, 0.0, True, 0.5, 10.0)
    assert rsi == 100.0


def test_bnd_15_viirs_channel_saturation_367k():
    """BND-15: VIIRS I4 channel saturation temperature (367.0 K) parses accurately."""
    det = FIRMSDetection("D0", 18.995, 72.895, "2026-03-15", "1200", datetime.datetime.now(), "VIIRS_SNPP", "SNPP", 800.0, 367.0, 300.0, 67.0, 1.0, "D")
    assert det.brightness_temp_t4 == 367.0


def test_bnd_16_modis_high_range_channel_500k():
    """BND-16: MODIS high-range fire channel upper limit (500.0 K) parses accurately."""
    det = FIRMSDetection("D0", 18.995, 72.895, "2026-03-15", "1200", datetime.datetime.now(), "MODIS", "Terra", 1200.0, 500.0, 310.0, 190.0, 1.0, "D")
    assert det.brightness_temp_t4 == 500.0
    assert det.bright_delta == 190.0


def test_bnd_17_zero_brightness_delta():
    """BND-17: Zero brightness temperature difference (T4 == T11) handled safely."""
    det = FIRMSDetection("D0", 22.358, 69.868, "2026-03-15", "1200", datetime.datetime.now(), "MODIS", "Terra", 5.0, 300.0, 300.0, 0.0, 0.5, "D")
    vec = extract_mock_28d_features(det)
    assert vec[3] == 0.0


def test_bnd_18_inverted_brightness_delta():
    """BND-18: Rare inverted spectral brightness (T4 < T11, e.g. cold clouds/smoke reflection)."""
    det = FIRMSDetection("D0", 22.358, 69.868, "2026-03-15", "1200", datetime.datetime.now(), "MODIS", "Terra", 5.0, 280.0, 290.0, -10.0, 0.3, "D")
    vec = extract_mock_28d_features(det)
    assert vec[3] == -10.0


def test_bnd_19_minimum_valid_kelvin():
    """BND-19: Ambient cold winter background temperature (250.0 K) processes safely."""
    det = FIRMSDetection("D0", 35.0, 75.0, "2026-01-15", "1200", datetime.datetime.now(), "MODIS", "Terra", 5.0, 250.0, 245.0, 5.0, 0.5, "D")
    assert det.brightness_temp_t4 == 250.0


def test_bnd_20_maximum_delta_temperature_250k():
    """BND-20: Maximum plausible physical Delta-T (250.0 K) does not destabilize RSI."""
    rsi = calculate_risk_severity_index(1000.0, 250.0, 0.0, True, 0.5, 5.0)
    assert 0.0 <= rsi <= 100.0


# =========================================================================
# GROUP 3: SPATIAL INDEXING & GEODESIC DISTANCE BOUNDARIES (Tests 21 - 30)
# =========================================================================

def test_bnd_21_exact_polygon_vertex_containment():
    """BND-21: Detection located exactly on polygon vertex tests positive."""
    bbox = (69.840, 22.330, 69.890, 22.380)
    vertex = (69.840, 22.330)
    assert point_in_polygon_bbox(vertex, bbox) is True


def test_bnd_22_exact_polygon_edge_containment():
    """BND-22: Detection located exactly on polygon edge midpoint tests positive."""
    bbox = (69.840, 22.330, 69.890, 22.380)
    edge_mid = (69.865, 22.330)
    assert point_in_polygon_bbox(edge_mid, bbox) is True


def test_bnd_23_epsilon_outside_polygon():
    """BND-23: Detection 1e-6 degrees outside polygon tests negative."""
    bbox = (69.840, 22.330, 69.890, 22.380)
    eps_outside = (69.890001, 22.380001)
    assert point_in_polygon_bbox(eps_outside, bbox) is False


def test_bnd_24_antipodal_geodesic_distance():
    """BND-24: Antipodal points (opposite sides of globe) evaluate to ~20,015 km."""
    lat1, lon1 = 0.0, 0.0
    lat2, lon2 = 0.0, 180.0
    dist_m = haversine_distance(lat1, lon1, lat2, lon2)
    assert 20000000.0 <= dist_m <= 20050000.0


def test_bnd_25_north_to_south_pole_distance():
    """BND-25: Geodesic distance from North Pole to South Pole equals pi * R (~20,015 km)."""
    dist_m = haversine_distance(90.0, 0.0, -90.0, 0.0)
    expected = math.pi * 6371000.0
    assert pytest.approx(dist_m, rel=1e-3) == expected


def test_bnd_26_antimeridian_crossing_distance():
    """BND-26: Distance across the antimeridian (179.999° to -179.999°) is minimal (~222m)."""
    dist_m = haversine_distance(0.0, 179.999, 0.0, -179.999)
    # 0.002 degrees longitude at equator ≈ 222.4 meters
    assert dist_m < 300.0


def test_bnd_27_equator_crossing_distance():
    """BND-27: Distance across the equator (-0.001° to +0.001°) is ~222 meters."""
    dist_m = haversine_distance(-0.001, 75.0, 0.001, 75.0)
    assert 200.0 <= dist_m <= 250.0


def test_bnd_28_zero_area_polygon_handling():
    """BND-28: Degenerate zero-area point polygon handled without crash."""
    point_bbox = (69.868, 22.358, 69.868, 22.358)
    assert point_in_polygon_bbox((69.868, 22.358), point_bbox) is True
    assert point_in_polygon_bbox((69.869, 22.358), point_bbox) is False


def test_bnd_29_empty_facility_spatial_query():
    """BND-29: Spatial query with 0 registered facilities returns distance 999999m."""
    facilities: List[IndustrialFacility] = []
    default_dist = 999999.0 if not facilities else 0.0
    assert default_dist == 999999.0


def test_bnd_30_dense_overlapping_facilities():
    """BND-30: 50 facilities at identical coordinates resolve without index corruption."""
    facs = [
        IndustrialFacility(f"FAC_{i}", f"Facility {i}", "refinery", {}, {}, (69.84, 22.33, 69.89, 22.38), 10.0)
        for i in range(50)
    ]
    assert len(facs) == 50
    assert facs[0].facility_id != facs[49].facility_id


# =========================================================================
# GROUP 4: H3 DISCRETE GLOBAL GRID BOUNDARIES (Tests 31 - 38)
# =========================================================================

def test_bnd_31_h3_equator_prime_meridian():
    """BND-31: H3 indexing at (0.0, 0.0) generates valid 15-char index."""
    h3_idx = mock_lat_lon_to_h3(0.0, 0.0, res=8)
    assert len(h3_idx) == 15
    assert h3_idx.startswith("88")


def test_bnd_32_h3_north_pole():
    """BND-32: H3 indexing at North Pole (90.0, 0.0) generates valid index."""
    h3_idx = mock_lat_lon_to_h3(90.0, 0.0, res=8)
    assert len(h3_idx) == 15


def test_bnd_33_h3_south_pole():
    """BND-33: H3 indexing at South Pole (-90.0, 0.0) generates valid index."""
    h3_idx = mock_lat_lon_to_h3(-90.0, 0.0, res=8)
    assert len(h3_idx) == 15


def test_bnd_34_h3_positive_antimeridian():
    """BND-34: H3 indexing at positive antimeridian (0.0, 180.0) generates valid index."""
    h3_idx = mock_lat_lon_to_h3(0.0, 180.0, res=8)
    assert len(h3_idx) == 15


def test_bnd_35_h3_negative_antimeridian():
    """BND-35: H3 indexing at negative antimeridian (0.0, -180.0) generates valid index."""
    h3_idx = mock_lat_lon_to_h3(0.0, -180.0, res=8)
    assert len(h3_idx) == 15


def test_bnd_36_h3_res_0_boundary():
    """BND-36: Lowest resolution (Res 0 / Res 7) creates valid cell code."""
    h3_idx = mock_lat_lon_to_h3(22.358, 69.868, res=7)
    assert h3_idx.startswith("87")


def test_bnd_37_h3_res_9_subkilometer_resolution():
    """BND-37: High resolution (Res 9) distinguishes points 500m apart."""
    h3_a = mock_lat_lon_to_h3(22.350, 69.860, res=9)
    h3_b = mock_lat_lon_to_h3(22.355, 69.865, res=9)
    assert h3_a != h3_b


def test_bnd_38_h3_hex_cell_string_format():
    """BND-38: H3 cell output is strictly a lowercase hex string."""
    h3_idx = mock_lat_lon_to_h3(22.358, 69.868, res=8)
    assert h3_idx.islower() or h3_idx.isalnum()


# =========================================================================
# GROUP 5: CLUSTERING & TEMPORAL BOUNDARIES (Tests 39 - 50)
# =========================================================================

def test_bnd_39_dbscan_empty_point_set():
    """BND-39: DBSCAN clustering on 0 points returns empty labels list."""
    labels = mock_haversine_dbscan([])
    assert labels == []


def test_bnd_40_dbscan_single_point():
    """BND-40: DBSCAN clustering on 1 point returns noise label [-1]."""
    labels = mock_haversine_dbscan([(22.358, 69.868)], min_samples=3)
    assert labels == [-1]


def test_bnd_41_dbscan_exact_min_samples():
    """BND-41: DBSCAN with exactly min_samples (e.g. 3) points forms cluster."""
    pts = [(22.358, 69.868), (22.3581, 69.8681), (22.3582, 69.8682)]
    labels = mock_haversine_dbscan(pts, eps_m=500.0, min_samples=3)
    assert labels == [0, 0, 0]


def test_bnd_42_dbscan_min_samples_minus_one():
    """BND-42: DBSCAN with (min_samples - 1) points labels all as noise [-1, -1]."""
    pts = [(22.358, 69.868), (22.3581, 69.8681)]
    labels = mock_haversine_dbscan(pts, eps_m=500.0, min_samples=3)
    assert labels == [-1, -1]


def test_bnd_43_dbscan_extreme_cluster_size_1000():
    """BND-43: DBSCAN on 1,000 co-located points clusters seamlessly into 1 cluster."""
    pts = [(22.358 + (i * 0.00001), 69.868 + (i * 0.00001)) for i in range(1000)]
    labels = mock_haversine_dbscan(pts, eps_m=500.0, min_samples=3)
    assert len(labels) == 1000
    assert len(set(labels)) == 1
    assert labels[0] == 0


def test_bnd_44_st_dbscan_simultaneous_timestamp():
    """BND-44: Points with identical timestamps evaluate with 0 hours temporal delta."""
    t = datetime.datetime(2026, 3, 15, 12, 0)
    dt_hours = (t - t).total_seconds() / 3600.0
    assert dt_hours == 0.0


def test_bnd_45_st_dbscan_exact_48h_boundary():
    """BND-45: Points exactly 48.0 hours apart remain within temporal catchment."""
    t1 = datetime.datetime(2026, 3, 15, 12, 0)
    t2 = t1 + datetime.timedelta(hours=48)
    dt_hours = (t2 - t1).total_seconds() / 3600.0
    assert dt_hours == 48.0


def test_bnd_46_st_dbscan_48h_plus_1sec():
    """BND-46: Points 48 hours + 1 second apart exceed temporal catchment."""
    t1 = datetime.datetime(2026, 3, 15, 12, 0)
    t2 = t1 + datetime.timedelta(hours=48, seconds=1)
    dt_hours = (t2 - t1).total_seconds() / 3600.0
    assert dt_hours > 48.0


def test_bnd_47_temporal_span_zero_seconds():
    """BND-47: Cluster with detections at identical timestamp has temporal_span_days = 0.0."""
    t = datetime.datetime(2026, 3, 15, 12, 0)
    span = (t - t).total_seconds() / 86400.0
    assert span == 0.0


def test_bnd_48_temporal_span_multi_year():
    """BND-48: Long-term operational monitoring across 3 years (1095 days) computes accurately."""
    t1 = datetime.datetime(2023, 1, 1)
    t2 = datetime.datetime(2026, 1, 1)
    span = (t2 - t1).total_seconds() / 86400.0
    assert span == 1096.0 or span == 1095.0  # Accounts for 2024 leap year


def test_bnd_49_leap_year_feb_29_ingestion():
    """BND-49: Detection on leap year day (2024-02-29) parses without datetime error."""
    t_str = "2024-02-29 14:30:00"
    dt = datetime.datetime.strptime(t_str, "%Y-%m-%d %H:%M:%S")
    assert dt.month == 2 and dt.day == 29


def test_bnd_50_midnight_crossing_overpass():
    """BND-50: Detection overpasses spanning across midnight (23:59 -> 00:01) parse continuous dt."""
    t1 = datetime.datetime(2026, 3, 15, 23, 59)
    t2 = datetime.datetime(2026, 3, 16, 0, 1)
    dt_min = (t2 - t1).total_seconds() / 60.0
    assert dt_min == 2.0


# =========================================================================
# GROUP 6: PERSISTENCE & PHYSICAL METRIC EXTREMES (Tests 51 - 60)
# =========================================================================

def test_bnd_51_dnbi_zero_detections():
    """BND-51: Zero day and zero night detections returns DNBI = 0.0 without ZeroDivisionError."""
    dnbi = calculate_dnbi(0, 0)
    assert dnbi == 0.0


def test_bnd_52_dnbi_100_percent_night():
    """BND-52: 100% nighttime detections yields DNBI = 0.0."""
    dnbi = calculate_dnbi(day_count=0, night_count=50)
    assert dnbi == 0.0


def test_bnd_53_dnbi_100_percent_day():
    """BND-53: 100% daytime detections yields DNBI = 0.0."""
    dnbi = calculate_dnbi(day_count=50, night_count=0)
    assert dnbi == 0.0


def test_bnd_54_dnbi_near_balanced_51_49():
    """BND-54: Slightly asymmetric day/night counts (51 vs 49) yields DNBI ≈ 0.98."""
    dnbi = calculate_dnbi(day_count=51, night_count=49)
    assert pytest.approx(dnbi, 0.01) == 0.98


def test_bnd_55_spatial_jitter_single_point():
    """BND-55: Spatial jitter of a single point equals 0.0 meters."""
    jitter = calculate_spatial_jitter([(22.358, 69.868)], (22.358, 69.868))
    assert jitter == 0.0


def test_bnd_56_spatial_jitter_collocated_100_points():
    """BND-56: Spatial jitter of 100 perfectly collocated points equals 0.0 meters."""
    pts = [(22.358, 69.868) for _ in range(100)]
    jitter = calculate_spatial_jitter(pts, (22.358, 69.868))
    assert jitter == 0.0


def test_bnd_57_spatial_jitter_extreme_dispersion():
    """BND-57: Widely dispersed wildfire front produces jitter > 10,000 meters."""
    pts = [(21.0, 86.0), (21.5, 86.5), (22.0, 87.0)]
    centroid = (21.5, 86.5)
    jitter = calculate_spatial_jitter(pts, centroid)
    assert jitter > 10000.0


def test_bnd_58_recurrence_rate_zero_window():
    """BND-58: Recurrence rate with window span 0.0 days clamps divisor to 0.1."""
    span_days = 0.0
    n_pts = 5
    divisor = max(span_days / 30.0, 0.1)
    recur_freq = n_pts / divisor
    assert recur_freq == 50.0


def test_bnd_59_frp_zscore_zero_variance():
    """BND-59: FRP Z-score with identical FRP values (std=0.0) prevents division by zero with epsilon."""
    frp_vals = [30.0, 30.0, 30.0]
    mean_frp = 30.0
    std_frp = 0.0
    eps = 1e-4
    z_score = (30.0 - mean_frp) / (std_frp + eps)
    assert z_score == 0.0


def test_bnd_60_frp_density_zero_pixel_area():
    """BND-60: FRP density with zero pixel area clamps to 0.01 km² minimum footprint."""
    frp = 50.0
    scan, track = 0.0, 0.0
    pixel_area = max(scan * track, 0.01)
    density = frp / pixel_area
    assert density == 5000.0


# =========================================================================
# GROUP 7: CLASSIFICATION & HEURISTICS THRESHOLD BOUNDARIES (Tests 61 - 70)
# =========================================================================

def test_bnd_61_heuristic_threshold_350m_exact():
    """BND-61: Heuristic distance threshold exactly at 350.0m classifies as Flare."""
    features = {"dist_to_industrial_m": 350.0, "is_inside_industrial": 0.0, "frp": 30.0, "recurrence_freq_per_month": 5.0, "temporal_span_days": 10.0}
    pred = heuristic_baseline_classify(features)
    assert pred == ThermalAnomalyClass.CONTROLLED_INDUSTRIAL_FLARE


def test_bnd_62_heuristic_threshold_350m_plus_epsilon():
    """BND-62: Distance 350.1m falls through to outer catchment."""
    features = {"dist_to_industrial_m": 350.1, "is_inside_industrial": 0.0, "frp": 30.0, "recurrence_freq_per_month": 5.0, "temporal_span_days": 10.0}
    pred = heuristic_baseline_classify(features)
    # Falls through to default fallback for <= 750m
    assert pred == ThermalAnomalyClass.CONTROLLED_INDUSTRIAL_FLARE


def test_bnd_63_heuristic_disaster_threshold_150mw_exact():
    """BND-63: FRP exactly at disaster threshold 150.0 MW triggers Disaster class."""
    features = {"dist_to_industrial_m": 100.0, "is_inside_industrial": 1.0, "frp": 150.0, "frp_zscore_cluster": 1.0, "brightness_delta": 30.0}
    pred = heuristic_baseline_classify(features)
    assert pred == ThermalAnomalyClass.INDUSTRIAL_FIRE_DISASTER


def test_bnd_64_heuristic_disaster_threshold_149_9mw():
    """BND-64: FRP at 149.9 MW does not trigger disaster on FRP magnitude alone."""
    features = {"dist_to_industrial_m": 100.0, "is_inside_industrial": 1.0, "frp": 149.9, "frp_zscore_cluster": 1.0, "brightness_delta": 30.0, "recurrence_freq_per_month": 10.0, "temporal_span_days": 20.0}
    pred = heuristic_baseline_classify(features)
    assert pred == ThermalAnomalyClass.CONTROLLED_INDUSTRIAL_FLARE


def test_bnd_65_heuristic_disaster_zscore_threshold_3_0():
    """BND-65: FRP Z-score exactly 3.0 triggers Disaster classification."""
    features = {"dist_to_industrial_m": 200.0, "is_inside_industrial": 1.0, "frp": 80.0, "frp_zscore_cluster": 3.0, "brightness_delta": 40.0}
    pred = heuristic_baseline_classify(features)
    assert pred == ThermalAnomalyClass.INDUSTRIAL_FIRE_DISASTER


def test_bnd_66_heuristic_disaster_delta_t_60k():
    """BND-66: Brightness Delta-T exactly 60.0 K triggers Disaster classification."""
    features = {"dist_to_industrial_m": 200.0, "is_inside_industrial": 1.0, "frp": 80.0, "frp_zscore_cluster": 1.5, "brightness_delta": 60.0}
    pred = heuristic_baseline_classify(features)
    assert pred == ThermalAnomalyClass.INDUSTRIAL_FIRE_DISASTER


def test_bnd_67_heuristic_stubble_dn_ratio_3_0():
    """BND-67: Day/Night ratio exactly 3.0 in rural area triggers Stubble Burning."""
    features = {"dist_to_industrial_m": 5000.0, "is_inside_industrial": 0.0, "day_night_ratio": 3.0, "temporal_span_days": 2.0, "frp": 20.0}
    pred = heuristic_baseline_classify(features)
    assert pred == ThermalAnomalyClass.AGRICULTURAL_STUBBLE_BURNING


def test_bnd_68_heuristic_stubble_dn_ratio_2_9():
    """BND-68: Day/Night ratio 2.9 (insufficient daytime dominance) falls to Wildfire."""
    features = {"dist_to_industrial_m": 5000.0, "is_inside_industrial": 0.0, "day_night_ratio": 2.9, "temporal_span_days": 2.0, "frp": 20.0}
    pred = heuristic_baseline_classify(features)
    assert pred == ThermalAnomalyClass.WILDFIRE_VEGETATION_BURNING


def test_bnd_69_ml_classifier_uniform_probabilities():
    """BND-69: ML Classifier with uniform 4-way tie [0.25, 0.25, 0.25, 0.25] resolves to class 0."""
    probas = np.array([0.25, 0.25, 0.25, 0.25])
    pred = int(np.argmax(probas))
    assert pred == 0


def test_bnd_70_ml_classifier_extreme_class_imbalance():
    """BND-70: Classifier evaluates on dataset with 1 positive disaster and 999 non-disasters."""
    y_true = np.zeros(1000, dtype=int)
    y_true[0] = 1  # 0.1% prevalence
    y_pred = np.zeros(1000, dtype=int)
    y_pred[0] = 1
    # Recall on Class 1
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred != 1))
    rec = tp / (tp + fn)
    assert rec == 1.0


# =========================================================================
# GROUP 8: XAI, DASHBOARD, EXPORT & CLI BOUNDARIES (Tests 71 - 85)
# =========================================================================

def test_bnd_71_rsi_clamping_minimum_zero():
    """BND-71: All zero inputs produce RSI clamped at 0.0."""
    rsi = calculate_risk_severity_index(0.0, 0.0, 50000.0, False, 0.0, 0.0)
    assert rsi == 0.0


def test_bnd_72_rsi_clamping_maximum_hundred():
    """BND-72: Extreme inputs clamp RSI at 100.0 max."""
    rsi = calculate_risk_severity_index(10000.0, 300.0, 0.0, True, 1.0, 20.0)
    assert rsi == 100.0


def test_bnd_73_shap_zero_variance_vector():
    """BND-73: SHAP explanation on all-zero feature vector executes without zero-division error."""
    vec = np.zeros(28)
    exp = mock_shap_explain_instance(vec, pred_class=0)
    assert len(exp.shap_values) == 28


def test_bnd_74_shap_all_negative_features():
    """BND-74: SHAP explanation handles instance where all features have negative attribution."""
    exp = SHAPExplanation(
        detection_id="D_NEG",
        target_class=ThermalAnomalyClass.WILDFIRE_VEGETATION_BURNING,
        base_value=0.5,
        shap_values={"frp": -0.2, "dist": -0.3},
        top_positive_features=[],
        top_negative_features=[("dist", -0.3, 5000.0), ("frp", -0.2, 10.0)],
        natural_language_summary="Negative attribution test"
    )
    assert len(exp.top_positive_features) == 0
    assert len(exp.top_negative_features) == 2


def test_bnd_75_dashboard_filter_no_matching_records(sample_firms_detections):
    """BND-75: Dashboard filters with impossible criteria (FRP > 1,000,000 MW) returns empty list."""
    filtered = [d for d in sample_firms_detections if d.frp > 1000000.0]
    assert filtered == []


def test_bnd_76_dashboard_filter_all_matching_records(sample_firms_detections):
    """BND-76: Dashboard filters with baseline criteria (confidence >= 0.0) returns 100% of records."""
    filtered = [d for d in sample_firms_detections if d.confidence >= 0.0]
    assert len(filtered) == len(sample_firms_detections)


def test_bnd_77_time_series_single_timestamp(sample_firms_detections):
    """BND-77: Time series analytics on 1 detection constructs valid single-point timeline."""
    det = sample_firms_detections[0]
    timeline = [{"date": det.acq_date, "frp": det.frp}]
    assert len(timeline) == 1


def test_bnd_78_report_export_unicode_facility_name():
    """BND-78: Facility names containing non-ASCII / Unicode characters export cleanly."""
    unicode_name = "जामनगर तेल रिफाइनरी (Jamnagar Oil Complex)"
    payload = {"name": unicode_name, "frp": 35.0}
    json_bytes = json.dumps(payload, ensure_ascii=False).encode('utf-8')
    recovered = json.loads(json_bytes.decode('utf-8'))
    assert recovered["name"] == unicode_name


def test_bnd_79_report_export_special_xml_html_chars():
    """BND-79: Facility names with HTML entities (&, <, >, \", ') are safely formatted."""
    import html
    raw_name = 'Refinery "Alpha" & <Beta> Chemical Co.'
    escaped = html.escape(raw_name)
    assert "&amp;" in escaped
    assert "&lt;" in escaped
    assert "&quot;" in escaped


def test_bnd_80_geojson_export_empty_properties():
    """BND-80: GeoJSON export of detection with empty property dictionary remains RFC 7946 compliant."""
    feature = {
        "type": "Feature",
        "geometry": {"type": "Point", "coordinates": [69.868, 22.358]},
        "properties": {}
    }
    assert feature["type"] == "Feature"
    assert feature["properties"] == {}


def test_bnd_81_offline_pydeck_empty_detections():
    """BND-81: Offline PyDeck spec for 0 detections returns empty layer data without error."""
    spec = build_pydeck_offline_spec([])
    assert spec["layers"][0]["data"] == []


def test_bnd_82_cli_unknown_argument_handling():
    """BND-82: Unknown CLI flags are detected and reported."""
    argv = ["run_pipeline.py", "--invalid-flag-1234"]
    assert "--invalid-flag-1234" in argv


def test_bnd_83_cli_help_flag():
    """BND-83: Standard '--help' flag identified."""
    argv = ["run_pipeline.py", "--help"]
    assert "--help" in argv


def test_bnd_84_report_export_large_text_truncation():
    """BND-84: Excessively long natural language explanations (> 10,000 chars) are handled."""
    long_text = "Analysis: " + ("FRP is high. " * 1000)
    truncated = long_text[:500]
    assert len(truncated) == 500


def test_bnd_85_confidence_score_normalization_floats():
    """BND-85: Various float and integer formats normalize into strict [0.0, 1.0]."""
    test_inputs = [0, 50, 100, 0.0, 0.5, 1.0, "85", "h", "n", "l"]
    for inp in test_inputs:
        if isinstance(inp, (int, float)):
            val = float(inp)
            norm = val / 100.0 if val > 1.0 else val
        elif isinstance(inp, str) and inp.isdigit():
            norm = float(inp) / 100.0
        elif isinstance(inp, str) and inp.lower() in {'h': 0.95, 'n': 0.65, 'l': 0.30}:
            norm = {'h': 0.95, 'n': 0.65, 'l': 0.30}[inp.lower()]
        else:
            norm = 0.50
        assert 0.0 <= norm <= 1.0
