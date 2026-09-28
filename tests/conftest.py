"""
Pytest Fixtures and Mock Data Generators for NTRO Thermal Detection System
Authoritative Specifications: ORIGINAL_REQUEST.md, PROJECT.md, TEST_INFRA.md
"""

from __future__ import annotations
import math
import json
import datetime
from enum import Enum
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple, Union
import pytest
import numpy as np

# -------------------------------------------------------------------------
# Interface Contract Dataclasses & Enums (PROJECT.md § Interface Contracts)
# -------------------------------------------------------------------------

class ThermalAnomalyClass(Enum):
    CONTROLLED_INDUSTRIAL_FLARE = 0
    INDUSTRIAL_FIRE_DISASTER = 1
    AGRICULTURAL_STUBBLE_BURNING = 2
    WILDFIRE_VEGETATION_BURNING = 3


from src.data_pipeline.schemas import FIRMSDetection


@dataclass
class IndustrialFacility:
    facility_id: str
    name: str
    facility_type: str  # 'refinery', 'chemical', 'steel', 'power', 'brick_kiln', 'flare_stack'
    geometry: Any       # shapely geometry or GeoJSON dict
    buffer_geometry_350m: Any = None
    buffer_geometry_750m: Any = None
    bounding_box: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    area_sq_km: float = 0.0


@dataclass
class ThermalCluster:
    cluster_id: str
    cluster_type: str  # 'stationary_persistent', 'dynamic_front', 'transient_noise'
    detection_ids: List[str]
    centroid_lat: float
    centroid_lon: float
    bounding_box: Tuple[float, float, float, float]
    convex_hull_geojson: Dict[str, Any]
    total_detections: int
    first_seen: datetime.datetime
    last_seen: datetime.datetime
    temporal_span_days: float
    active_days_count: int
    recurrence_rate: float  # active_days / total_days_in_window
    day_count: int
    night_count: int
    day_night_balance_index: float  # 2 * min(D, N) / (D + N)
    spatial_jitter_m: float         # 1-sigma dispersion from centroid
    mean_frp: float
    max_frp: float
    frp_std: float
    nearest_industrial_facility_id: Optional[str] = None
    distance_to_nearest_industrial_m: float = 0.0
    is_inside_industrial_boundary: bool = False


@dataclass
class PredictionOutput:
    detection_id: str
    predicted_class: ThermalAnomalyClass
    predicted_label: str
    confidence_score: float
    class_probabilities: Dict[str, float]
    risk_severity_index: float  # [0.0, 100.0]
    is_critical_alert: bool
    feature_values: Dict[str, float] = field(default_factory=dict)


@dataclass
class SHAPExplanation:
    detection_id: str
    target_class: ThermalAnomalyClass
    base_value: float
    shap_values: Dict[str, float]
    top_positive_features: List[Tuple[str, float, float]]  # (feature_name, shap_value, raw_value)
    top_negative_features: List[Tuple[str, float, float]]
    waterfall_plot_base64: Optional[str] = None
    natural_language_summary: str = ""


# -------------------------------------------------------------------------
# Exact Mathematical & Physical Oracle Functions
# -------------------------------------------------------------------------

EARTH_RADIUS_METERS = 6371000.0

def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Computes exact great-circle distance between two coordinates in meters."""
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    a = (math.sin(delta_phi / 2.0) ** 2 +
         math.cos(phi1) * math.cos(phi2) * (math.sin(delta_lambda / 2.0) ** 2))
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return EARTH_RADIUS_METERS * c


def calculate_dnbi(day_count: int, night_count: int) -> float:
    """Calculates Day-Night Balance Index: 2 * min(D, N) / (D + N)."""
    total = day_count + night_count
    if total == 0:
        return 0.0
    return float(2.0 * min(day_count, night_count) / total)


def calculate_spatial_jitter(coords: List[Tuple[float, float]], centroid: Tuple[float, float]) -> float:
    """Calculates 1-sigma dispersion from centroid in meters."""
    if not coords:
        return 0.0
    distances = [haversine_distance(lat, lon, centroid[0], centroid[1]) for lat, lon in coords]
    return float(np.std(distances))


def calculate_risk_severity_index(
    frp: float,
    bright_delta: float,
    dist_to_industrial_m: float,
    is_inside_industrial: bool,
    night_fraction: float,
    frp_zscore: float
) -> float:
    """
    Computes composite Risk Severity Index (RSI) [0.0 - 100.0].
    Weighted multi-factor index combining FRP, Delta-T, Industrial proximity, and Surge Z-score.
    """
    w_frp = min(frp / 500.0, 1.0) * 35.0
    w_delta = min(bright_delta / 100.0, 1.0) * 25.0
    w_prox = 20.0 if (is_inside_industrial or dist_to_industrial_m <= 350.0) else (
        10.0 if dist_to_industrial_m <= 750.0 else max(0.0, 20.0 - (dist_to_industrial_m / 100.0))
    )
    w_surge = min(max(frp_zscore, 0.0) / 4.0, 1.0) * 20.0
    total = w_frp + w_delta + w_prox + w_surge
    return float(np.clip(total, 0.0, 100.0))


FEATURE_NAMES_28D = [
    # Radiometric (8)
    "frp", "brightness", "bright_t31", "brightness_delta",
    "frp_zscore_cluster", "frp_density_km2", "frp_relative_regional", "thermal_intensity_index",
    # Spatio-temporal (9)
    "cluster_size", "temporal_span_days", "recurrence_freq_per_month",
    "day_night_ratio", "night_fraction", "avg_nn_distance_m",
    "cluster_radius_m", "spatial_dispersion_rate", "centroid_drift_velocity",
    # Geospatial (7)
    "dist_to_industrial_m", "is_inside_industrial", "industrial_facility_type",
    "facility_count_5km", "landcover_type", "elevation_m", "slope_degrees",
    # Sensor Quality (4)
    "confidence_score_norm", "pixel_area_km2", "satellite_platform_encoded", "daynight_binary"
]


def extract_mock_28d_features(
    detection: FIRMSDetection,
    cluster: Optional[ThermalCluster] = None,
    facility: Optional[IndustrialFacility] = None,
    distance_m: float = 5000.0
) -> np.ndarray:
    """Extracts exactly 28 physical and spatial features for a detection."""
    # Group A: Radiometric
    frp = float(detection.frp)
    t_mir = float(detection.brightness_temp_t4)
    t_tir = float(detection.brightness_temp_t11)
    delta_t = float(detection.bright_delta)
    mean_cluster_frp = cluster.mean_frp if cluster else frp
    std_cluster_frp = cluster.frp_std if cluster and cluster.frp_std > 0 else 1.0
    frp_zscore = (frp - mean_cluster_frp) / (std_cluster_frp + 1e-4)
    pixel_area = float(max(detection.scan * detection.track, 0.01))
    frp_density = frp / pixel_area
    frp_rel_regional = frp / (mean_cluster_frp + 1.0)
    thermal_intensity = ((t_mir - 300.0) / 50.0) * math.log10(max(frp, 0.0) + 1.0)

    # Group B: Spatio-Temporal
    c_size = float(cluster.total_detections if cluster else 1)
    t_span = float(cluster.temporal_span_days if cluster else 0.0)
    recur_freq = float(cluster.recurrence_rate * 30.0 if cluster else 1.0)
    day_cnt = cluster.day_count if cluster else (1 if detection.daynight == 'D' else 0)
    night_cnt = cluster.night_count if cluster else (1 if detection.daynight == 'N' else 0)
    dn_ratio = float((day_cnt + 1.0) / (night_cnt + 1.0))
    night_frac = float(night_cnt / max(day_cnt + night_cnt, 1))
    avg_nn_dist = float(cluster.spatial_jitter_m * 0.8 if cluster else 0.0)
    c_radius = float(cluster.spatial_jitter_m * 2.0 if cluster else 50.0)
    disp_rate = float(c_radius / max(t_span, 1.0))
    drift_vel = 0.0 if (cluster and cluster.cluster_type == 'stationary_persistent') else 450.0

    # Group C: Geospatial
    is_inside = 1.0 if (facility and distance_m <= 0.0) or (cluster and cluster.is_inside_industrial_boundary) else 0.0
    fac_type_map = {'none': 0, 'refinery': 1, 'chemical': 2, 'flare_stack': 3, 'steel': 4, 'power': 5, 'brick_kiln': 6}
    fac_type = float(fac_type_map.get(facility.facility_type if facility else 'none', 0))
    fac_count_5k = 3.0 if is_inside else (1.0 if distance_m <= 5000.0 else 0.0)
    landcover = 5.0 if is_inside else (4.0 if distance_m > 2000.0 and detection.latitude > 25.0 else 1.0)
    elevation = 25.0
    slope = 1.5

    # Group D: Sensor Quality
    conf_norm = float(detection.confidence)
    sat_map = {'Terra': 0, 'Aqua': 1, 'SNPP': 2, 'NOAA-20': 3, 'NOAA-21': 4}
    sat_enc = float(sat_map.get(detection.satellite, 0))
    daynight_bin = 1.0 if detection.daynight == 'D' else 0.0

    vector = np.array([
        frp, t_mir, t_tir, delta_t, frp_zscore, frp_density, frp_rel_regional, thermal_intensity,
        c_size, t_span, recur_freq, dn_ratio, night_frac, avg_nn_dist, c_radius, disp_rate, drift_vel,
        distance_m, is_inside, fac_type, fac_count_5k, landcover, elevation, slope,
        conf_norm, pixel_area, sat_enc, daynight_bin
    ], dtype=np.float64)

    return vector


def heuristic_baseline_classify(features: Union[np.ndarray, Dict[str, float]]) -> ThermalAnomalyClass:
    """Deterministic Rule-Based Baseline Classifier (PROJECT.md § 2.2)."""
    if isinstance(features, np.ndarray):
        dist_ind = features[17]
        is_inside = features[18]
        frp = features[0]
        frp_zscore = features[4]
        delta_t = features[3]
        recur_freq = features[10]
        t_span = features[9]
        dn_ratio = features[11]
        c_radius = features[14]
    else:
        dist_ind = features.get("dist_to_industrial_m", 5000.0)
        is_inside = features.get("is_inside_industrial", 0.0)
        frp = features.get("frp", 20.0)
        frp_zscore = features.get("frp_zscore_cluster", 0.0)
        delta_t = features.get("brightness_delta", 30.0)
        recur_freq = features.get("recurrence_freq_per_month", 1.0)
        t_span = features.get("temporal_span_days", 1.0)
        dn_ratio = features.get("day_night_ratio", 1.0)
        c_radius = features.get("cluster_radius_m", 100.0)

    # Rule 1: Industrial Fire Emergency
    if (dist_ind <= 500.0 or is_inside == 1.0) and (frp >= 150.0 or frp_zscore >= 3.0 or delta_t >= 60.0):
        return ThermalAnomalyClass.INDUSTRIAL_FIRE_DISASTER

    # Rule 2: Controlled Industrial Flare
    if (dist_ind <= 350.0 or is_inside == 1.0) and (recur_freq >= 3.0 or t_span >= 5.0):
        return ThermalAnomalyClass.CONTROLLED_INDUSTRIAL_FLARE

    # Rule 3: Agricultural Stubble Burning
    if dist_ind > 1000.0 and dn_ratio >= 3.0 and t_span <= 4.0:
        return ThermalAnomalyClass.AGRICULTURAL_STUBBLE_BURNING

    # Rule 4: Wildfire / Vegetation
    if dist_ind > 1000.0 and (c_radius >= 1000.0 or t_span > 4.0 or frp >= 50.0):
        return ThermalAnomalyClass.WILDFIRE_VEGETATION_BURNING

    # Default fallback
    if dist_ind <= 750.0:
        return ThermalAnomalyClass.CONTROLLED_INDUSTRIAL_FLARE
    return ThermalAnomalyClass.WILDFIRE_VEGETATION_BURNING


# -------------------------------------------------------------------------
# Pytest Fixtures
# -------------------------------------------------------------------------

@pytest.fixture
def sample_jamnagar_geojson() -> Dict[str, Any]:
    """Reliance Jamnagar Mega-Refinery boundary polygon GeoJSON."""
    return {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {
                    "facility_id": "IND_JAMNAGAR_001",
                    "name": "Reliance Jamnagar Petroleum Refinery Complex",
                    "facility_type": "refinery",
                    "area_sq_km": 30.5
                },
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[
                        [69.840, 22.330],
                        [69.890, 22.330],
                        [69.890, 22.380],
                        [69.840, 22.380],
                        [69.840, 22.330]
                    ]]
                }
            }
        ]
    }


@pytest.fixture
def sample_chemical_plant_geojson() -> Dict[str, Any]:
    """Petrochemical storage depot boundary polygon GeoJSON."""
    return {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {
                    "facility_id": "IND_CHEM_DEPOT_002",
                    "name": "Trombay Petrochemical Terminal Depot",
                    "facility_type": "chemical",
                    "area_sq_km": 4.2
                },
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[
                        [72.880, 18.980],
                        [72.910, 18.980],
                        [72.910, 19.010],
                        [72.880, 19.010],
                        [72.880, 18.980]
                    ]]
                }
            }
        ]
    }


@pytest.fixture
def sample_stubble_fields_geojson() -> Dict[str, Any]:
    """Punjab rural agricultural zone GeoJSON."""
    return {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {
                    "facility_id": "AGRI_LUDHIANA_001",
                    "name": "Ludhiana Agricultural District Block",
                    "facility_type": "cropland",
                    "area_sq_km": 150.0
                },
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[
                        [75.750, 30.850],
                        [75.950, 30.850],
                        [75.950, 31.050],
                        [75.750, 31.050],
                        [75.750, 30.850]
                    ]]
                }
            }
        ]
    }


@pytest.fixture
def sample_forest_reserve_geojson() -> Dict[str, Any]:
    """Simlipal Biosphere Forest Reserve GeoJSON."""
    return {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {
                    "facility_id": "FOR_SIMLIPAL_001",
                    "name": "Simlipal National Park Wildland Reserve",
                    "facility_type": "forest",
                    "area_sq_km": 845.0
                },
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[
                        [86.200, 21.500],
                        [86.600, 21.500],
                        [86.600, 21.900],
                        [86.200, 21.900],
                        [86.200, 21.500]
                    ]]
                }
            }
        ]
    }


@pytest.fixture
def sample_jurong_geojson() -> Dict[str, Any]:
    """Jurong Island multi-facility industrial cluster GeoJSON."""
    return {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {
                    "facility_id": "JURONG_REF_01",
                    "name": "Jurong Shell Refinery North",
                    "facility_type": "refinery",
                    "area_sq_km": 5.0
                },
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[
                        [103.680, 1.260],
                        [103.700, 1.260],
                        [103.700, 1.280],
                        [103.680, 1.280],
                        [103.680, 1.260]
                    ]]
                }
            },
            {
                "type": "Feature",
                "properties": {
                    "facility_id": "JURONG_CHEM_02",
                    "name": "Jurong Petrochemical Cracker South",
                    "facility_type": "chemical",
                    "area_sq_km": 4.5
                },
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[
                        [103.705, 1.260],
                        [103.725, 1.260],
                        [103.725, 1.280],
                        [103.705, 1.280],
                        [103.705, 1.260]
                    ]]
                }
            }
        ]
    }


@pytest.fixture
def sample_modis_csv_content() -> str:
    """Standard NASA FIRMS MODIS CSV text content."""
    return (
        "latitude,longitude,brightness,scan,track,acq_date,acq_time,satellite,instrument,confidence,version,bright_t31,frp,daynight\n"
        "22.3582,69.8681,335.4,1.1,1.0,2026-03-15,0630,Terra,MODIS,85,6.1NRT,298.2,34.5,D\n"
        "22.3590,69.8675,340.1,1.0,1.0,2026-03-15,1845,Aqua,MODIS,92,6.1NRT,290.0,42.1,N\n"
        "18.9950,72.8950,385.0,1.2,1.1,2026-03-16,0700,Terra,MODIS,100,6.1NRT,301.5,820.0,D\n"
        "30.9100,75.8600,328.0,1.0,1.0,2026-03-16,0815,Terra,MODIS,78,6.1NRT,299.1,18.3,D\n"
        "21.6800,86.3700,352.0,1.4,1.2,2026-03-16,1930,Aqua,MODIS,88,6.1NRT,295.4,125.0,N\n"
    )


@pytest.fixture
def sample_viirs_csv_content() -> str:
    """Standard NASA FIRMS VIIRS (S-NPP / NOAA-20 / NOAA-21) CSV text content."""
    return (
        "latitude,longitude,bright_ti4,scan,track,acq_date,acq_time,satellite,instrument,confidence,version,bright_ti5,frp,daynight\n"
        "22.3585,69.8680,345.8,0.38,0.37,2026-03-15,0812,N,VIIRS,h,2.0NRT,294.5,28.4,D\n"
        "22.3581,69.8683,348.2,0.40,0.38,2026-03-15,2030,N,VIIRS,h,2.0NRT,289.0,31.2,N\n"
        "18.9945,72.8960,367.0,0.39,0.38,2026-03-16,0850,20,VIIRS,h,2.0NRT,305.0,910.0,D\n"
        "30.9150,75.8620,330.4,0.41,0.39,2026-03-16,0855,20,VIIRS,n,2.0NRT,298.0,15.6,D\n"
        "21.6850,86.3750,358.9,0.45,0.42,2026-03-16,2110,21,VIIRS,h,2.0NRT,292.3,160.0,N\n"
    )


@pytest.fixture
def sample_firms_detections() -> List[FIRMSDetection]:
    """Diverse list of calibrated FIRMS detections across 4 operational classes."""
    detections = []
    base_time = datetime.datetime(2026, 3, 1, 6, 30)

    # 1. Jamnagar Flare Detections (Class 0: stationary, persistent, day+night)
    for i in range(20):
        t = base_time + datetime.timedelta(days=i, hours=(12 if i % 2 == 1 else 0))
        dn = 'N' if i % 2 == 1 else 'D'
        lat_jitter = 22.3582 + (0.0005 * (i % 3 - 1))
        lon_jitter = 69.8681 + (0.0004 * (i % 2 - 0.5))
        frp_val = 30.0 + (i % 5) * 2.0
        detections.append(FIRMSDetection(
            detection_id=f"JAMNAGAR_DET_{i:03d}",
            latitude=lat_jitter,
            longitude=lon_jitter,
            acq_date=t.strftime("%Y-%m-%d"),
            acq_time=t.strftime("%H%M"),
            timestamp=t,
            sensor="VIIRS_SNPP",
            satellite="SNPP",
            frp=frp_val,
            brightness_temp_t4=342.0 + (i % 3),
            brightness_temp_t11=294.0,
            bright_delta=48.0 + (i % 3),
            confidence=0.95,
            daynight=dn,
            scan=0.38,
            track=0.37,
            h3_res8="8861214041fffff"
        ))

    # 2. Chemical Depot Disaster (Class 1: massive surge, sudden onset, inside industrial)
    for i in range(5):
        t = datetime.datetime(2026, 3, 10, 14, 0) + datetime.timedelta(hours=i*2)
        detections.append(FIRMSDetection(
            detection_id=f"DISASTER_DET_{i:03d}",
            latitude=18.9950 + (i * 0.0002),
            longitude=72.8950 + (i * 0.0001),
            acq_date=t.strftime("%Y-%m-%d"),
            acq_time=t.strftime("%H%M"),
            timestamp=t,
            sensor="VIIRS_NOAA20",
            satellite="NOAA-20",
            frp=750.0 + (i * 120.0),
            brightness_temp_t4=367.0,
            brightness_temp_t11=302.0,
            bright_delta=65.0,
            confidence=1.0,
            daynight='D' if i < 3 else 'N',
            scan=0.40,
            track=0.38,
            h3_res8="8861294025fffff"
        ))

    # 3. Punjab Stubble Burning (Class 2: daytime, rural, transient)
    for i in range(15):
        t = datetime.datetime(2026, 3, 12, 11, 30) + datetime.timedelta(hours=i*0.5)
        detections.append(FIRMSDetection(
            detection_id=f"STUBBLE_DET_{i:03d}",
            latitude=30.8500 + (i * 0.015),
            longitude=75.8000 + (i * 0.012),
            acq_date=t.strftime("%Y-%m-%d"),
            acq_time=t.strftime("%H%M"),
            timestamp=t,
            sensor="MODIS",
            satellite="Terra",
            frp=22.0 + (i % 6) * 3.0,
            brightness_temp_t4=328.0,
            brightness_temp_t11=298.0,
            bright_delta=30.0,
            confidence=0.80,
            daynight='D',
            scan=1.0,
            track=1.0,
            h3_res8="8861314067fffff"
        ))

    # 4. Simlipal Wildfire (Class 3: expanding front, wildland, high FRP)
    for i in range(12):
        t = datetime.datetime(2026, 3, 14, 9, 0) + datetime.timedelta(days=i//3, hours=(i%3)*4)
        dn = 'N' if (i % 3) == 2 else 'D'
        detections.append(FIRMSDetection(
            detection_id=f"WILDFIRE_DET_{i:03d}",
            latitude=21.6500 + (i * 0.018),
            longitude=86.3500 + (i * 0.015),
            acq_date=t.strftime("%Y-%m-%d"),
            acq_time=t.strftime("%H%M"),
            timestamp=t,
            sensor="VIIRS_NOAA21",
            satellite="NOAA-21",
            frp=140.0 + (i * 25.0),
            brightness_temp_t4=355.0 + (i % 4),
            brightness_temp_t11=292.0,
            bright_delta=63.0 + (i % 4),
            confidence=0.92,
            daynight=dn,
            scan=0.42,
            track=0.40,
            h3_res8="8861454089fffff"
        ))

    return detections


@pytest.fixture
def sample_industrial_facilities(sample_jamnagar_geojson, sample_chemical_plant_geojson) -> List[IndustrialFacility]:
    """List of parsed IndustrialFacility objects."""
    facilities = []
    # Jamnagar
    jam_props = sample_jamnagar_geojson["features"][0]["properties"]
    facilities.append(IndustrialFacility(
        facility_id=jam_props["facility_id"],
        name=jam_props["name"],
        facility_type=jam_props["facility_type"],
        geometry=sample_jamnagar_geojson["features"][0]["geometry"],
        buffer_geometry_350m=sample_jamnagar_geojson["features"][0]["geometry"],
        buffer_geometry_750m=sample_jamnagar_geojson["features"][0]["geometry"],
        bounding_box=(69.840, 22.330, 69.890, 22.380),
        area_sq_km=jam_props["area_sq_km"]
    ))
    # Chemical Depot
    chem_props = sample_chemical_plant_geojson["features"][0]["properties"]
    facilities.append(IndustrialFacility(
        facility_id=chem_props["facility_id"],
        name=chem_props["name"],
        facility_type=chem_props["facility_type"],
        geometry=sample_chemical_plant_geojson["features"][0]["geometry"],
        buffer_geometry_350m=sample_chemical_plant_geojson["features"][0]["geometry"],
        buffer_geometry_750m=sample_chemical_plant_geojson["features"][0]["geometry"],
        bounding_box=(72.880, 18.980, 72.910, 19.010),
        area_sq_km=chem_props["area_sq_km"]
    ))
    return facilities


@pytest.fixture
def sample_thermal_clusters(sample_firms_detections) -> List[ThermalCluster]:
    """Pre-built ThermalCluster objects representing distinct anomaly archetypes."""
    # Jamnagar Flare Cluster
    jam_dets = [d for d in sample_firms_detections if d.detection_id.startswith("JAMNAGAR")]
    c_jam = ThermalCluster(
        cluster_id="CLUSTER_JAMNAGAR_001",
        cluster_type="stationary_persistent",
        detection_ids=[d.detection_id for d in jam_dets],
        centroid_lat=22.3582,
        centroid_lon=69.8681,
        bounding_box=(69.8670, 22.3575, 69.8690, 22.3590),
        convex_hull_geojson={"type": "Polygon", "coordinates": [[[69.867, 22.357], [69.869, 22.357], [69.869, 22.359], [69.867, 22.359], [69.867, 22.357]]]},
        total_detections=len(jam_dets),
        first_seen=jam_dets[0].timestamp,
        last_seen=jam_dets[-1].timestamp,
        temporal_span_days=20.0,
        active_days_count=20,
        recurrence_rate=1.0,
        day_count=10,
        night_count=10,
        day_night_balance_index=calculate_dnbi(10, 10),
        spatial_jitter_m=65.4,
        mean_frp=34.2,
        max_frp=42.1,
        frp_std=3.8,
        nearest_industrial_facility_id="IND_JAMNAGAR_001",
        distance_to_nearest_industrial_m=0.0,
        is_inside_industrial_boundary=True
    )

    # Chemical Disaster Cluster
    dis_dets = [d for d in sample_firms_detections if d.detection_id.startswith("DISASTER")]
    c_dis = ThermalCluster(
        cluster_id="CLUSTER_DISASTER_002",
        cluster_type="stationary_persistent",
        detection_ids=[d.detection_id for d in dis_dets],
        centroid_lat=18.9952,
        centroid_lon=72.8952,
        bounding_box=(72.8940, 18.9940, 72.8965, 18.9960),
        convex_hull_geojson={"type": "Polygon", "coordinates": [[[72.894, 18.994], [72.8965, 18.994], [72.8965, 18.996], [72.894, 18.996], [72.894, 18.994]]]},
        total_detections=len(dis_dets),
        first_seen=dis_dets[0].timestamp,
        last_seen=dis_dets[-1].timestamp,
        temporal_span_days=0.4,
        active_days_count=1,
        recurrence_rate=1.0,
        day_count=3,
        night_count=2,
        day_night_balance_index=calculate_dnbi(3, 2),
        spatial_jitter_m=42.1,
        mean_frp=950.0,
        max_frp=1230.0,
        frp_std=185.0,
        nearest_industrial_facility_id="IND_CHEM_DEPOT_002",
        distance_to_nearest_industrial_m=0.0,
        is_inside_industrial_boundary=True
    )

    # Stubble Burning Cluster
    stub_dets = [d for d in sample_firms_detections if d.detection_id.startswith("STUBBLE")]
    c_stub = ThermalCluster(
        cluster_id="CLUSTER_STUBBLE_003",
        cluster_type="transient_noise",
        detection_ids=[d.detection_id for d in stub_dets],
        centroid_lat=30.9500,
        centroid_lon=75.8800,
        bounding_box=(75.8000, 30.8500, 76.0100, 31.0500),
        convex_hull_geojson={"type": "Polygon", "coordinates": [[[75.8, 30.85], [76.01, 30.85], [76.01, 31.05], [75.8, 31.05], [75.8, 30.85]]]},
        total_detections=len(stub_dets),
        first_seen=stub_dets[0].timestamp,
        last_seen=stub_dets[-1].timestamp,
        temporal_span_days=1.0,
        active_days_count=1,
        recurrence_rate=1.0,
        day_count=15,
        night_count=0,
        day_night_balance_index=calculate_dnbi(15, 0),
        spatial_jitter_m=12400.0,
        mean_frp=28.5,
        max_frp=45.0,
        frp_std=6.2,
        nearest_industrial_facility_id=None,
        distance_to_nearest_industrial_m=18500.0,
        is_inside_industrial_boundary=False
    )

    # Wildfire Cluster
    wild_dets = [d for d in sample_firms_detections if d.detection_id.startswith("WILDFIRE")]
    c_wild = ThermalCluster(
        cluster_id="CLUSTER_WILDFIRE_004",
        cluster_type="dynamic_front",
        detection_ids=[d.detection_id for d in wild_dets],
        centroid_lat=21.7500,
        centroid_lon=86.4300,
        bounding_box=(86.3500, 21.6500, 86.5500, 21.8500),
        convex_hull_geojson={"type": "Polygon", "coordinates": [[[86.35, 21.65], [86.55, 21.65], [86.55, 21.85], [86.35, 21.85], [86.35, 21.65]]]},
        total_detections=len(wild_dets),
        first_seen=wild_dets[0].timestamp,
        last_seen=wild_dets[-1].timestamp,
        temporal_span_days=4.0,
        active_days_count=4,
        recurrence_rate=1.0,
        day_count=8,
        night_count=4,
        day_night_balance_index=calculate_dnbi(8, 4),
        spatial_jitter_m=8500.0,
        mean_frp=265.0,
        max_frp=415.0,
        frp_std=82.0,
        nearest_industrial_facility_id=None,
        distance_to_nearest_industrial_m=35000.0,
        is_inside_industrial_boundary=False
    )

    return [c_jam, c_dis, c_stub, c_wild]
