"""
Schemas and Data Contracts for NTRO Thermal Detection System.
Defines immutable, type-safe data structures for FIRMS detections, OSM industrial facilities, and normalized datasets.
Authoritative Specifications: ORIGINAL_REQUEST.md, PROJECT.md, m1_blueprint.md
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Optional, Sequence, Union
import pandas as pd
import shapely.geometry
from shapely.geometry.base import BaseGeometry


@dataclass(frozen=True)
class FIRMSDetection:
    """
    Normalized NASA FIRMS active fire detection record.
    Harmonizes MODIS (1km) and VIIRS (375m) satellite observations into a unified schema.
    """
    detection_id: str
    latitude: float
    longitude: float
    acq_date: str                 # YYYY-MM-DD
    acq_time: str                 # HHMM (e.g. '0430', '1345')
    timestamp: datetime           # Combined UTC datetime
    sensor: str                   # 'MODIS' | 'VIIRS_SNPP' | 'VIIRS_NOAA20' | 'VIIRS_NOAA21'
    satellite: str                # 'Terra' | 'Aqua' | 'SNPP' | 'NOAA-20' | 'NOAA-21'
    frp: float                    # Fire Radiative Power (MW)
    brightness_temp_t4: float     # Mid-Infrared brightness temperature (Kelvin, ~4µm / I4 channel)
    brightness_temp_t11: float    # Thermal Infrared brightness temperature (Kelvin, ~11µm / I5 / T31 channel)
    bright_delta: float           # T4 - T11 (Kelvin)
    confidence: float             # Normalized continuous confidence in [0.0, 1.0]
    daynight: str                 # 'D' (Day) | 'N' (Night)
    scan: float = 1.0             # Along-scan pixel dimension (km)
    track: float = 1.0            # Along-track pixel dimension (km)
    h3_res7: str = ""             # Uber H3 Hexagon index at Resolution 7 (~1.22 km edge)
    h3_res8: str = ""             # Uber H3 Hexagon index at Resolution 8 (~461 m edge)
    h3_res9: str = ""             # Uber H3 Hexagon index at Resolution 9 (~174 m edge)
    raw_properties: dict[str, Any] = field(default_factory=dict, hash=False, compare=False)

    def __post_init__(self) -> None:
        """Enforce domain invariants and coordinate/range validations."""
        if not (-90.0 <= self.latitude <= 90.0):
            raise ValueError(f"Invalid latitude {self.latitude}. Must be in [-90.0, 90.0].")
        if not (-180.0 <= self.longitude <= 180.0):
            raise ValueError(f"Invalid longitude {self.longitude}. Must be in [-180.0, 180.0].")
        # Floating point epsilon tolerance for confidence bounds
        if not (-1e-5 <= self.confidence <= 1.0 + 1e-5):
            raise ValueError(f"Invalid confidence {self.confidence}. Must be in [0.0, 1.0].")
        if self.daynight not in ("D", "N"):
            raise ValueError(f"Invalid daynight '{self.daynight}'. Must be 'D' or 'N'.")
        if self.frp < 0.0:
            raise ValueError(f"Invalid negative FRP {self.frp} MW.")

    def to_dict(self) -> dict[str, Any]:
        """Convert detection instance to a clean JSON-serializable dictionary."""
        d = asdict(self)
        d["timestamp"] = self.timestamp.isoformat()
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FIRMSDetection:
        """Instantiate FIRMSDetection from dictionary, handling timestamp conversions and optional defaults."""
        d = dict(data)
        ts = d.get("timestamp")
        if isinstance(ts, str):
            try:
                d["timestamp"] = datetime.fromisoformat(ts)
            except ValueError:
                d["timestamp"] = pd.to_datetime(ts).to_pydatetime()
        elif isinstance(ts, pd.Timestamp):
            d["timestamp"] = ts.to_pydatetime()
        elif ts is None:
            # Fallback construct from acq_date and acq_time
            acq_date_val = str(d.get("acq_date", "2026-01-01"))
            acq_time_val = str(d.get("acq_time", "0000")).zfill(4)
            d["timestamp"] = datetime.strptime(f"{acq_date_val} {acq_time_val}", "%Y-%m-%d %H%M")

        # Type casts for safety
        d["latitude"] = float(d["latitude"])
        d["longitude"] = float(d["longitude"])
        d["frp"] = float(d.get("frp", 0.0))
        d["brightness_temp_t4"] = float(d.get("brightness_temp_t4", 300.0))
        d["brightness_temp_t11"] = float(d.get("brightness_temp_t11", 290.0))
        d["bright_delta"] = float(d.get("bright_delta", d["brightness_temp_t4"] - d["brightness_temp_t11"]))
        
        # Clamp confidence to [0.0, 1.0] if slight float deviation
        conf = float(d.get("confidence", 0.5))
        d["confidence"] = max(0.0, min(1.0, conf))
        d["daynight"] = str(d.get("daynight", "D")).upper()
        if d["daynight"] not in ("D", "N"):
            d["daynight"] = "D"

        d["scan"] = float(d.get("scan", 1.0))
        d["track"] = float(d.get("track", 1.0))
        d["h3_res7"] = str(d.get("h3_res7", ""))
        d["h3_res8"] = str(d.get("h3_res8", ""))
        d["h3_res9"] = str(d.get("h3_res9", ""))
        
        if "raw_properties" not in d or not isinstance(d["raw_properties"], dict):
            d["raw_properties"] = {}

        # Remove extra keys not in dataclass
        valid_keys = {
            "detection_id", "latitude", "longitude", "acq_date", "acq_time",
            "timestamp", "sensor", "satellite", "frp", "brightness_temp_t4",
            "brightness_temp_t11", "bright_delta", "confidence", "daynight",
            "scan", "track", "h3_res7", "h3_res8", "h3_res9", "raw_properties"
        }
        filtered_d = {k: v for k, v in d.items() if k in valid_keys}
        return cls(**filtered_d)

    @staticmethod
    def to_dataframe(detections: Sequence[FIRMSDetection]) -> pd.DataFrame:
        """Convert a sequence of FIRMSDetection records into a Pandas DataFrame."""
        if not detections:
            return pd.DataFrame(columns=[
                "detection_id", "latitude", "longitude", "acq_date", "acq_time",
                "timestamp", "sensor", "satellite", "frp", "brightness_temp_t4",
                "brightness_temp_t11", "bright_delta", "confidence", "daynight",
                "scan", "track", "h3_res7", "h3_res8", "h3_res9"
            ])
        records = []
        for d in detections:
            rec = d.to_dict()
            if "raw_properties" in rec:
                rec["raw_properties"] = json.dumps(rec["raw_properties"])
            records.append(rec)
        df = pd.DataFrame(records)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df

    @staticmethod
    def from_dataframe(df: pd.DataFrame) -> list[FIRMSDetection]:
        """Reconstruct a list of FIRMSDetection objects from a Pandas DataFrame."""
        detections: list[FIRMSDetection] = []
        if df.empty:
            return detections
        for row in df.to_dict(orient="records"):
            if isinstance(row.get("timestamp"), (pd.Timestamp, str)):
                row["timestamp"] = pd.to_datetime(row["timestamp"]).to_pydatetime()
            raw_props = row.get("raw_properties", {})
            if isinstance(raw_props, str):
                try:
                    raw_props = json.loads(raw_props)
                except Exception:
                    raw_props = {}
            row["raw_properties"] = raw_props if isinstance(raw_props, dict) else {}
            detections.append(FIRMSDetection.from_dict(row))
        return detections


@dataclass
class IndustrialFacility:
    """
    OpenStreetMap industrial facility with multi-ring thermal catchment buffers.
    """
    facility_id: str
    name: str
    facility_type: str            # 'refinery' | 'chemical' | 'steel' | 'power' | 'brick_kiln' | 'flare_stack' | 'general_industrial'
    geometry: Any                 # Base geometry (Shapely BaseGeometry or dict) in WGS84
    buffer_geometry_350m: Any     # 350m thermal dispersion buffer
    buffer_geometry_750m: Any     # 750m thermal plume catchment buffer
    bounding_box: tuple[float, float, float, float]  # (min_lon, min_lat, max_lon, max_lat)
    area_sq_km: float
    properties: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert facility metadata to dictionary (excluding raw Shapely geometries for serialization)."""
        wkt = self.geometry.wkt if hasattr(self.geometry, "wkt") else str(self.geometry)
        return {
            "facility_id": self.facility_id,
            "name": self.name,
            "facility_type": self.facility_type,
            "bounding_box": list(self.bounding_box),
            "area_sq_km": self.area_sq_km,
            "properties": self.properties,
            "geometry_wkt": wkt,
        }

    def to_geojson_feature(self) -> dict[str, Any]:
        """Export as RFC 7946 GeoJSON Feature dictionary."""
        if hasattr(self.geometry, "__geo_interface__"):
            geom_dict = shapely.geometry.mapping(self.geometry)
        elif isinstance(self.geometry, dict):
            geom_dict = self.geometry
        else:
            geom_dict = {"type": "Point", "coordinates": [self.bounding_box[0], self.bounding_box[1]]}

        return {
            "type": "Feature",
            "id": self.facility_id,
            "geometry": geom_dict,
            "properties": {
                "facility_id": self.facility_id,
                "name": self.name,
                "facility_type": self.facility_type,
                "area_sq_km": self.area_sq_km,
                **self.properties
            }
        }


@dataclass
class NormalizedDataset:
    """
    Encapsulates an ingested and indexed dataset batch, pairing detections with spatial facilities.
    """
    dataset_name: str
    detections: list[FIRMSDetection]
    facilities: list[IndustrialFacility]
    bounding_box: tuple[float, float, float, float]  # (min_lon, min_lat, max_lon, max_lat)
    start_date: str
    end_date: str
    total_detections: int
    sensor_counts: dict[str, int]
    ingestion_timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_summary_dict(self) -> dict[str, Any]:
        """Generate high-level dataset summary metrics."""
        return {
            "dataset_name": self.dataset_name,
            "total_detections": self.total_detections,
            "total_facilities": len(self.facilities),
            "bounding_box": list(self.bounding_box),
            "start_date": self.start_date,
            "end_date": self.end_date,
            "sensor_counts": self.sensor_counts,
            "ingestion_timestamp": self.ingestion_timestamp.isoformat(),
        }
