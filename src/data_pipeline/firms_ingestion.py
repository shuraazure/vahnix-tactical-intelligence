"""
NASA FIRMS Satellite Telemetry Ingestion and Harmonization Engine.
Processes MODIS (1km) and VIIRS (375m S-NPP / NOAA-20 / NOAA-21) active fire feeds (CSV & GeoJSON).
Authoritative Specifications: ORIGINAL_REQUEST.md § R1, PROJECT.md, m1_blueprint.md
"""

from __future__ import annotations

import io
import json
import math
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Sequence, Union

import pandas as pd

from src.data_pipeline.schemas import FIRMSDetection

# -------------------------------------------------------------------------
# H3 Spatial Indexing Wrapper with Version & Offline Fallback Support
# -------------------------------------------------------------------------

try:
    import h3
    _H3_AVAILABLE = True
except ImportError:
    h3 = None
    _H3_AVAILABLE = False


def compute_h3_cell(lat: float, lon: float, resolution: int) -> str:
    """
    Compute Uber H3 discrete global grid cell index for a coordinate.
    Supports h3-py v3 and v4 APIs, with deterministic hex fallback.
    """
    if _H3_AVAILABLE and h3 is not None:
        try:
            # h3-py v4+ API
            if hasattr(h3, "latlng_to_cell"):
                return str(h3.latlng_to_cell(lat, lon, resolution))
            # h3-py v3 API
            elif hasattr(h3, "geo_to_h3"):
                return str(h3.geo_to_h3(lat, lon, resolution))
        except Exception:
            pass

    # Deterministic spatial hex fallback (15-char valid lowercase hex string)
    lat_q = int(math.floor((lat + 90.0) * 1000.0)) & 0xFFFF
    lon_q = int(math.floor((lon + 180.0) * 1000.0)) & 0xFFFF
    return f"8{resolution:x}61{lat_q:04x}{lon_q:04x}fff"[:15]


# -------------------------------------------------------------------------
# FIRMS Ingestion Engine
# -------------------------------------------------------------------------

class FIRMSIngestionEngine:
    """
    High-throughput parser, normalizer, and validator for NASA FIRMS telemetry feeds.
    Supports MODIS (Terra/Aqua) and VIIRS (S-NPP, NOAA-20, NOAA-21) in CSV, GeoJSON, and DataFrame formats.
    """

    @staticmethod
    def parse_acq_time(raw_time: Any) -> str:
        """Normalize acquisition time to 4-character 'HHMM' string."""
        if raw_time is None or (isinstance(raw_time, float) and math.isnan(raw_time)):
            return "0000"
        s = str(raw_time).strip()
        # Handle decimal floats like '630.0'
        if "." in s:
            s = s.split(".")[0]
        # Remove colons like '06:30'
        s = s.replace(":", "")
        s = s.zfill(4)
        if len(s) > 4:
            s = s[:4]
        return s

    @staticmethod
    def parse_timestamp(acq_date_str: str, acq_time_str: str) -> datetime:
        """Combine acquisition date and time into a UTC datetime object."""
        clean_date = str(acq_date_str).strip().replace("/", "-")
        # Extract YYYY-MM-DD if ISO format with time is passed
        if "T" in clean_date:
            clean_date = clean_date.split("T")[0]
        elif " " in clean_date:
            clean_date = clean_date.split(" ")[0]

        time_clean = acq_time_str.zfill(4)
        try:
            hh = int(time_clean[:2])
            mm = int(time_clean[2:4])
        except ValueError:
            hh, mm = 0, 0
        hh = max(0, min(23, hh))
        mm = max(0, min(59, mm))

        try:
            dt = datetime.strptime(clean_date, "%Y-%m-%d")
            return dt.replace(hour=hh, minute=mm, second=0, microsecond=0)
        except ValueError:
            try:
                dt = pd.to_datetime(clean_date).to_pydatetime()
                return dt.replace(hour=hh, minute=mm, second=0, microsecond=0)
            except Exception:
                return datetime(2026, 1, 1, hh, mm, 0)

    @classmethod
    def resolve_sensor_and_satellite(
        cls, sat_raw: str, inst_raw: str, row: dict[str, Any]
    ) -> tuple[str, str]:
        """
        Harmonize raw satellite and instrument identifiers into standard taxonomy.
        Sensors: 'MODIS', 'VIIRS_SNPP', 'VIIRS_NOAA20', 'VIIRS_NOAA21'
        Satellites: 'Terra', 'Aqua', 'SNPP', 'NOAA-20', 'NOAA-21'
        """
        sat_clean = sat_raw.strip().upper()
        inst_clean = inst_raw.strip().upper()

        # Check explicit sensor in row
        row_sensor = str(row.get("sensor", "")).strip().upper()
        if "MODIS" in row_sensor:
            inst_clean = "MODIS"
        elif "VIIRS" in row_sensor:
            inst_clean = "VIIRS"

        # Check instrument / column signatures
        has_viirs_cols = ("bright_ti4" in row or "BRIGHT_TI4" in row or "bright_ti5" in row)
        has_modis_cols = ("bright_t31" in row or "BRIGHT_T31" in row)

        if "VIIRS" in inst_clean or has_viirs_cols:
            # Determine VIIRS satellite platform
            if sat_clean in ("N", "SNPP", "NPP", "SUOMI-NPP", "VIIRS_SNPP"):
                return "VIIRS_SNPP", "SNPP"
            elif sat_clean in ("20", "NOAA-20", "NOAA20", "N20", "JPSS-1", "JPSS1", "VIIRS_NOAA20"):
                return "VIIRS_NOAA20", "NOAA-20"
            elif sat_clean in ("21", "NOAA-21", "NOAA21", "N21", "JPSS-2", "JPSS2", "VIIRS_NOAA21"):
                return "VIIRS_NOAA21", "NOAA-21"
            else:
                # Default to SNPP if unspecified
                return "VIIRS_SNPP", "SNPP"

        elif "MODIS" in inst_clean or has_modis_cols or sat_clean in ("TERRA", "AQUA", "T", "A"):
            if sat_clean in ("A", "AQUA"):
                return "MODIS", "Aqua"
            else:
                return "MODIS", "Terra"

        # Fallback heuristic
        if sat_clean in ("20", "NOAA-20"):
            return "VIIRS_NOAA20", "NOAA-20"
        elif sat_clean in ("21", "NOAA-21"):
            return "VIIRS_NOAA21", "NOAA-21"
        elif sat_clean in ("N", "SNPP"):
            return "VIIRS_SNPP", "SNPP"
        elif sat_clean in ("AQUA", "A"):
            return "MODIS", "Aqua"
        else:
            return "MODIS", "Terra"

    @classmethod
    def normalize_confidence(cls, conf_raw: Any, sensor: str) -> float:
        """
        Normalize confidence score to continuous interval [0.0, 1.0].
        Handles MODIS (0-100%) and VIIRS categorical ('l' -> 0.30, 'n' -> 0.70, 'h' -> 0.95).
        """
        if conf_raw is None or (isinstance(conf_raw, float) and math.isnan(conf_raw)):
            return 0.50

        # String parsing
        if isinstance(conf_raw, str):
            s = conf_raw.strip().lower()
            if s in ("l", "low"):
                return 0.30
            elif s in ("n", "nominal", "med", "medium"):
                return 0.70
            elif s in ("h", "high"):
                return 0.95
            
            # Numeric string
            s_clean = s.replace("%", "").strip()
            try:
                num = float(s_clean)
                if num > 1.0:
                    return max(0.0, min(1.0, num / 100.0))
                return max(0.0, min(1.0, num))
            except ValueError:
                return 0.50

        # Numeric float/int
        if isinstance(conf_raw, (int, float)):
            val = float(conf_raw)
            if val > 1.0:
                return max(0.0, min(1.0, val / 100.0))
            return max(0.0, min(1.0, val))

        return 0.50

    @classmethod
    def normalize_record(cls, row: dict[str, Any], index: int = 0) -> Optional[FIRMSDetection]:
        """
        Transform and validate a raw dictionary into an immutable FIRMSDetection instance.
        Returns None if coordinates are missing, corrupted, or out of range.
        """
        # 1. Coordinates validation
        lat_val = (
            row.get("latitude") if row.get("latitude") is not None
            else (row.get("LATITUDE") if row.get("LATITUDE") is not None
            else (row.get("lat") if row.get("lat") is not None
            else row.get("LAT")))
        )
        lon_val = (
            row.get("longitude") if row.get("longitude") is not None
            else (row.get("LONGITUDE") if row.get("LONGITUDE") is not None
            else (row.get("lon") if row.get("lon") is not None
            else (row.get("LON") if row.get("LON") is not None
            else row.get("long"))))
        )

        try:
            lat = float(lat_val)
            lon = float(lon_val)
            if math.isnan(lat) or math.isnan(lon):
                return None
        except (TypeError, ValueError):
            return None

        if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
            return None

        # 2. Date and Time
        acq_date_val = (
            row.get("acq_date") or row.get("ACQ_DATE") or
            row.get("date") or row.get("DATE") or
            datetime.utcnow().strftime("%Y-%m-%d")
        )
        acq_date = str(acq_date_val).strip()

        acq_time_val = (
            row.get("acq_time") or row.get("ACQ_TIME") or
            row.get("time") or row.get("TIME") or "0000"
        )
        acq_time = cls.parse_acq_time(acq_time_val)
        ts = cls.parse_timestamp(acq_date, acq_time)

        # 3. Sensor & Satellite Identification
        sat_raw = str(row.get("satellite") or row.get("SATELLITE") or "")
        inst_raw = str(row.get("instrument") or row.get("INSTRUMENT") or "")
        sensor, satellite = cls.resolve_sensor_and_satellite(sat_raw, inst_raw, row)

        # 4. Radiometric Channels
        # T4 (MIR)
        t4_val = (
            row.get("brightness") or row.get("bright_ti4") or
            row.get("BRIGHTNESS") or row.get("BRIGHT_TI4") or
            row.get("brightness_temp_t4")
        )
        try:
            t4 = float(t4_val) if t4_val is not None and not (isinstance(t4_val, float) and math.isnan(t4_val)) else 300.0
        except (ValueError, TypeError):
            t4 = 300.0

        # T11 (TIR)
        t11_val = (
            row.get("bright_t31") or row.get("bright_ti5") or
            row.get("BRIGHT_T31") or row.get("BRIGHT_TI5") or
            row.get("brightness_temp_t11")
        )
        try:
            t11 = float(t11_val) if t11_val is not None and not (isinstance(t11_val, float) and math.isnan(t11_val)) else (t4 - 10.0)
        except (ValueError, TypeError):
            t11 = t4 - 10.0

        # Derived Delta-T
        bright_delta_val = row.get("bright_delta")
        if bright_delta_val is not None:
            try:
                bright_delta = float(bright_delta_val)
            except (ValueError, TypeError):
                bright_delta = t4 - t11
        else:
            bright_delta = t4 - t11

        # 5. FRP (Fire Radiative Power)
        frp_val = row.get("frp") or row.get("FRP") or row.get("power")
        try:
            frp = max(0.0, float(frp_val)) if frp_val is not None and not (isinstance(frp_val, float) and math.isnan(frp_val)) else 0.0
        except (ValueError, TypeError):
            frp = 0.0

        # 6. Confidence Normalization
        conf_val = row.get("confidence") or row.get("CONFIDENCE")
        confidence = cls.normalize_confidence(conf_val, sensor)

        # 7. Day / Night
        dn_val = str(row.get("daynight") or row.get("DAYNIGHT") or "D").strip().upper()
        daynight = "N" if dn_val.startswith("N") else "D"

        # 8. Scan & Track
        try:
            scan = float(row.get("scan") or row.get("SCAN") or 1.0)
        except (ValueError, TypeError):
            scan = 1.0

        try:
            track = float(row.get("track") or row.get("TRACK") or 1.0)
        except (ValueError, TypeError):
            track = 1.0

        # 9. Spatial H3 Binning
        h3_7 = compute_h3_cell(lat, lon, 7)
        h3_8 = compute_h3_cell(lat, lon, 8)
        h3_9 = compute_h3_cell(lat, lon, 9)

        # 10. Detection ID
        det_id_raw = row.get("detection_id") or row.get("id")
        if det_id_raw is not None:
            det_id = str(det_id_raw)
        else:
            det_id = f"DET_{sensor}_{ts.strftime('%Y%m%d%H%M')}_{index:06d}"

        # 11. Raw Properties Capture
        raw_props = {k: v for k, v in row.items() if k not in (
            "latitude", "longitude", "LATITUDE", "LONGITUDE", "lat", "lon"
        )}

        return FIRMSDetection(
            detection_id=det_id,
            latitude=lat,
            longitude=lon,
            acq_date=acq_date,
            acq_time=acq_time,
            timestamp=ts,
            sensor=sensor,
            satellite=satellite,
            frp=frp,
            brightness_temp_t4=t4,
            brightness_temp_t11=t11,
            bright_delta=bright_delta,
            confidence=confidence,
            daynight=daynight,
            scan=scan,
            track=track,
            h3_res7=h3_7,
            h3_res8=h3_8,
            h3_res9=h3_9,
            raw_properties=raw_props,
        )

    @classmethod
    def ingest_dataframe(cls, df: pd.DataFrame) -> list[FIRMSDetection]:
        """Ingest a pandas DataFrame of active fire records."""
        detections: list[FIRMSDetection] = []
        if df.empty:
            return detections
        records = df.to_dict(orient="records")
        for i, row in enumerate(records):
            det = cls.normalize_record(row, index=i)
            if det is not None:
                detections.append(det)
        return detections

    @classmethod
    def ingest_csv(cls, file_path_or_buffer: Union[str, Path, io.StringIO, io.BytesIO]) -> list[FIRMSDetection]:
        """Ingest NASA FIRMS CSV telemetry from a file path or in-memory stream."""
        if isinstance(file_path_or_buffer, (str, Path)):
            p = Path(file_path_or_buffer)
            if not p.exists():
                raise FileNotFoundError(f"FIRMS CSV file not found: {file_path_or_buffer}")
            df = pd.read_csv(p)
        else:
            df = pd.read_csv(file_path_or_buffer)
        return cls.ingest_dataframe(df)

    @classmethod
    def ingest_geojson(cls, file_path_or_data: Union[str, Path, dict[str, Any]]) -> list[FIRMSDetection]:
        """Ingest NASA FIRMS GeoJSON feature collection."""
        if isinstance(file_path_or_data, dict):
            geojson_dict = file_path_or_data
        elif isinstance(file_path_or_data, (str, Path)):
            p = Path(file_path_or_data)
            if p.exists():
                with open(p, "r", encoding="utf-8") as f:
                    geojson_dict = json.load(f)
            else:
                # Attempt to parse string as JSON
                geojson_dict = json.loads(str(file_path_or_data))
        else:
            raise ValueError(f"Unsupported GeoJSON input type: {type(file_path_or_data)}")

        features = geojson_dict.get("features", [])
        records: list[dict[str, Any]] = []
        for feat in features:
            geom = feat.get("geometry") or {}
            props = feat.get("properties") or {}
            coords = geom.get("coordinates", [])
            row = dict(props)
            if len(coords) >= 2:
                row["longitude"] = coords[0]
                row["latitude"] = coords[1]
            if "id" in feat and "detection_id" not in row:
                row["detection_id"] = str(feat["id"])
            records.append(row)

        df = pd.DataFrame(records)
        return cls.ingest_dataframe(df)

    @classmethod
    def ingest_auto(cls, source: Any) -> list[FIRMSDetection]:
        """Auto-detect format (CSV, GeoJSON, DataFrame, Dict) and parse records."""
        if isinstance(source, pd.DataFrame):
            return cls.ingest_dataframe(source)
        if isinstance(source, dict):
            return cls.ingest_geojson(source)
        if isinstance(source, (str, Path)):
            s_str = str(source).strip()
            if s_str.endswith(".csv"):
                return cls.ingest_csv(source)
            elif s_str.endswith(".geojson") or s_str.endswith(".json"):
                return cls.ingest_geojson(source)
            elif s_str.startswith("{"):
                return cls.ingest_geojson(source)
            else:
                # Default attempt CSV
                return cls.ingest_csv(source)
        raise ValueError(f"Unable to auto-ingest source of type {type(source)}")
