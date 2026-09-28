"""
Synthetic Reference and Benchmark Dataset Generator.
Produces physics-grounded, spatio-temporally calibrated datasets for operational corridors:
1. Jamnagar Mega-Refinery (Controlled Industrial Flare)
2. Trombay Chemical Depot Disaster (Industrial Fire Emergency)
3. Punjab Post-Monsoon Stubble Burning (Agricultural Fire)
4. Simlipal National Park Wildfire Front (Forest Wildfire)
5. Jurong Island Petrochemical Cluster (Dense Multi-Facility Cluster)
Authoritative Specifications: ORIGINAL_REQUEST.md, PROJECT.md, m1_blueprint.md
"""

from __future__ import annotations

import json
import math
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import shapely.geometry

from src.data_pipeline.firms_ingestion import compute_h3_cell
from src.data_pipeline.osm_indexer import SpatialIndexEngine, create_catchment_buffers
from src.data_pipeline.schemas import FIRMSDetection, IndustrialFacility, NormalizedDataset


class SyntheticDataGenerator:
    """
    Calibrated generator for synthetic NASA FIRMS telemetry and OpenStreetMap infrastructure.
    """

    def __init__(
        self,
        spatial_engine: Optional[SpatialIndexEngine] = None,
        seed: Optional[int] = None,
    ) -> None:
        self.spatial_engine = spatial_engine or SpatialIndexEngine()
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)

    @staticmethod
    def _create_facility_from_coords(
        fac_id: str,
        name: str,
        fac_type: str,
        coordinates: list[list[float]],
        properties: Optional[dict[str, Any]] = None,
    ) -> IndustrialFacility:
        """Helper to construct an IndustrialFacility with multi-ring buffers from polygon coords."""
        geom = shapely.geometry.Polygon(coordinates)
        buf_350, buf_750, area_sq_km = create_catchment_buffers(geom)
        bounds = geom.bounds
        bbox = (float(bounds[0]), float(bounds[1]), float(bounds[2]), float(bounds[3]))
        props = properties or {}
        props["facility_id"] = fac_id
        props["name"] = name
        props["facility_type"] = fac_type

        return IndustrialFacility(
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

    # -------------------------------------------------------------------------
    # Corridor 1: Jamnagar Mega-Refinery (Controlled Flare)
    # -------------------------------------------------------------------------
    def generate_jamnagar_corridor(
        self, num_days: int = 60, seed: int = 42
    ) -> NormalizedDataset:
        """
        Generate continuous persistent industrial flaring at Reliance Jamnagar Refinery.
        Characteristics: Stationary, Day+Night balanced (DNBI ~ 0.90), steady FRP (25-85 MW).
        """
        rng = np.random.RandomState(seed)
        
        # Jamnagar Refinery boundary polygon (~30 sq km)
        coords = [
            [69.840, 22.330],
            [69.890, 22.330],
            [69.890, 22.380],
            [69.840, 22.380],
            [69.840, 22.330]
        ]
        facility = self._create_facility_from_coords(
            fac_id="IND_JAMNAGAR_001",
            name="Reliance Jamnagar Petroleum Refinery Complex",
            fac_type="refinery",
            coordinates=coords,
            properties={"country": "India", "state": "Gujarat", "capacity_bpd": 1240000}
        )

        # 6 discrete flare stack coordinate anchors inside boundary
        flare_anchors = [
            (22.3582, 69.8681),
            (22.3550, 69.8620),
            (22.3610, 69.8710),
            (22.3520, 69.8750),
            (22.3650, 69.8590),
            (22.3480, 69.8660),
        ]

        detections: list[FIRMSDetection] = []
        base_date = datetime(2026, 1, 1, 0, 0, 0)
        det_idx = 1
        sensor_counts: dict[str, int] = {"MODIS": 0, "VIIRS_SNPP": 0, "VIIRS_NOAA20": 0, "VIIRS_NOAA21": 0}

        # Conversion: 65 meters in degrees lat/lon at 22.35 deg N
        m_to_deg_lat = 1.0 / 111320.0
        m_to_deg_lon = 1.0 / (111320.0 * math.cos(math.radians(22.35)))

        for day in range(num_days):
            current_day = base_date + timedelta(days=day)
            acq_date_str = current_day.strftime("%Y-%m-%d")

            # Satellite pass schedule:
            # 1. Terra daytime (~05:30 UTC) - MODIS
            # 2. Aqua daytime (~08:00 UTC) - MODIS
            # 3. SNPP nighttime (~19:45 UTC) - VIIRS
            # 4. NOAA-20 nighttime (~20:30 UTC) - VIIRS
            passes = [
                ("0530", "MODIS", "Terra", "D", 1.0, 1.0),
                ("0800", "MODIS", "Aqua", "D", 1.0, 1.0),
                ("1945", "VIIRS_SNPP", "SNPP", "N", 0.38, 0.37),
                ("2030", "VIIRS_NOAA20", "NOAA-20", "N", 0.39, 0.38),
            ]

            for acq_time, sensor, sat, dn, scan, track in passes:
                # 3 to 5 active flares detected per pass
                active_flares = rng.choice(len(flare_anchors), size=rng.randint(3, 6), replace=False)
                ts = datetime.strptime(f"{acq_date_str} {acq_time}", "%Y-%m-%d %H%M")

                for anchor_idx in active_flares:
                    anchor_lat, anchor_lon = flare_anchors[anchor_idx]
                    # Gaussian spatial jitter (~65m sigma)
                    lat_jitter = anchor_lat + rng.normal(0, 65.0 * m_to_deg_lat)
                    lon_jitter = anchor_lon + rng.normal(0, 65.0 * m_to_deg_lon)

                    frp = float(np.clip(rng.normal(45.0, 10.0), 22.0, 95.0))
                    t4 = float(np.clip(rng.normal(342.0, 6.0), 325.0, 360.0))
                    t11 = float(np.clip(rng.normal(294.0, 3.0), 288.0, 302.0))
                    delta_t = t4 - t11
                    conf = float(np.clip(rng.uniform(0.88, 0.99), 0.0, 1.0))

                    h3_7 = compute_h3_cell(lat_jitter, lon_jitter, 7)
                    h3_8 = compute_h3_cell(lat_jitter, lon_jitter, 8)
                    h3_9 = compute_h3_cell(lat_jitter, lon_jitter, 9)

                    det = FIRMSDetection(
                        detection_id=f"JAMNAGAR_DET_{det_idx:06d}",
                        latitude=lat_jitter,
                        longitude=lon_jitter,
                        acq_date=acq_date_str,
                        acq_time=acq_time,
                        timestamp=ts,
                        sensor=sensor,
                        satellite=sat,
                        frp=frp,
                        brightness_temp_t4=t4,
                        brightness_temp_t11=t11,
                        bright_delta=delta_t,
                        confidence=conf,
                        daynight=dn,
                        scan=scan,
                        track=track,
                        h3_res7=h3_7,
                        h3_res8=h3_8,
                        h3_res9=h3_9,
                        raw_properties={"ground_truth_class": 0, "corridor": "jamnagar_refinery"}
                    )
                    detections.append(det)
                    sensor_counts[sensor] = sensor_counts.get(sensor, 0) + 1
                    det_idx += 1

        bbox = (69.840, 22.330, 69.890, 22.380)
        return NormalizedDataset(
            dataset_name="jamnagar_refinery_corridor",
            detections=detections,
            facilities=[facility],
            bounding_box=bbox,
            start_date=detections[0].acq_date,
            end_date=detections[-1].acq_date,
            total_detections=len(detections),
            sensor_counts=sensor_counts,
        )

    # -------------------------------------------------------------------------
    # Corridor 2: Petrochemical Depot Disaster (Industrial Disaster)
    # -------------------------------------------------------------------------
    def generate_chemical_disaster_corridor(self, seed: int = 101) -> NormalizedDataset:
        """
        Generate rapid high-intensity explosion and fire disaster at a chemical depot.
        Characteristics: Sudden onset, extreme FRP (450-2800 MW), high Delta-T (>65 K).
        """
        rng = np.random.RandomState(seed)

        coords = [
            [72.880, 18.980],
            [72.910, 18.980],
            [72.910, 19.010],
            [72.880, 19.010],
            [72.880, 18.980]
        ]
        facility = self._create_facility_from_coords(
            fac_id="IND_CHEM_DEPOT_002",
            name="Trombay Petrochemical Terminal Depot",
            fac_type="chemical",
            coordinates=coords,
            properties={"hazard_class": "Hydrocarbon Terminal", "tier": "Seveso III"}
        )

        detections: list[FIRMSDetection] = []
        sensor_counts: dict[str, int] = {}
        det_idx = 1

        # Center of explosion
        center_lat, center_lon = 18.9950, 72.8950
        m_to_deg_lat = 1.0 / 111320.0
        m_to_deg_lon = 1.0 / (111320.0 * math.cos(math.radians(18.995)))

        # 1. Pre-disaster baseline (Days 1 to 10): minor background process heat (0-1 det/week)
        for day in [2, 6, 9]:
            ts = datetime(2026, 2, day, 8, 30)
            det = FIRMSDetection(
                detection_id=f"CHEM_DISASTER_{det_idx:06d}",
                latitude=center_lat + rng.normal(0, 30.0 * m_to_deg_lat),
                longitude=center_lon + rng.normal(0, 30.0 * m_to_deg_lon),
                acq_date=ts.strftime("%Y-%m-%d"),
                acq_time=ts.strftime("%H%M"),
                timestamp=ts,
                sensor="MODIS",
                satellite="Terra",
                frp=14.5,
                brightness_temp_t4=315.0,
                brightness_temp_t11=298.0,
                bright_delta=17.0,
                confidence=0.65,
                daynight="D",
                scan=1.0,
                track=1.0,
                h3_res7=compute_h3_cell(center_lat, center_lon, 7),
                h3_res8=compute_h3_cell(center_lat, center_lon, 8),
                h3_res9=compute_h3_cell(center_lat, center_lon, 9),
                raw_properties={"ground_truth_class": 1, "phase": "pre_disaster"}
            )
            detections.append(det)
            sensor_counts["MODIS"] = sensor_counts.get("MODIS", 0) + 1
            det_idx += 1

        # 2. Catastrophic Surge Phase (Days 11 to 12, Feb 11-12)
        # Multiple satellite passes observing extreme inferno
        disaster_passes = [
            (datetime(2026, 2, 11, 2, 15), "VIIRS_SNPP", "SNPP", "N", 0.38, 0.37, 1450.0, 385.0),
            (datetime(2026, 2, 11, 6, 45), "MODIS", "Terra", "D", 1.2, 1.1, 1920.0, 395.0),
            (datetime(2026, 2, 11, 8, 30), "MODIS", "Aqua", "D", 1.1, 1.0, 2450.0, 410.0),
            (datetime(2026, 2, 11, 14, 20), "VIIRS_NOAA20", "NOAA-20", "D", 0.39, 0.38, 2200.0, 405.0),
            (datetime(2026, 2, 11, 20, 10), "VIIRS_NOAA21", "NOAA-21", "N", 0.42, 0.40, 1850.0, 390.0),
            (datetime(2026, 2, 12, 1, 50), "VIIRS_SNPP", "SNPP", "N", 0.38, 0.37, 1600.0, 380.0),
            (datetime(2026, 2, 12, 7, 0), "MODIS", "Terra", "D", 1.0, 1.0, 1100.0, 370.0),
            (datetime(2026, 2, 12, 19, 30), "VIIRS_NOAA20", "NOAA-20", "N", 0.40, 0.38, 750.0, 360.0),
        ]

        for ts, sensor, sat, dn, scan, track, base_frp, base_t4 in disaster_passes:
            # 2 to 4 detections per pass due to multi-tank propagation
            num_pts = rng.randint(2, 5)
            for _ in range(num_pts):
                lat_j = center_lat + rng.normal(0, 80.0 * m_to_deg_lat)
                lon_j = center_lon + rng.normal(0, 80.0 * m_to_deg_lon)
                frp = float(np.clip(rng.normal(base_frp, 150.0), 450.0, 2900.0))
                t4 = float(np.clip(rng.normal(base_t4, 8.0), 365.0, 420.0))
                t11 = float(np.clip(rng.normal(305.0, 4.0), 298.0, 315.0))
                delta_t = t4 - t11
                conf = float(np.clip(rng.uniform(0.96, 1.00), 0.0, 1.0))

                det = FIRMSDetection(
                    detection_id=f"CHEM_DISASTER_{det_idx:06d}",
                    latitude=lat_j,
                    longitude=lon_j,
                    acq_date=ts.strftime("%Y-%m-%d"),
                    acq_time=ts.strftime("%H%M"),
                    timestamp=ts,
                    sensor=sensor,
                    satellite=sat,
                    frp=frp,
                    brightness_temp_t4=t4,
                    brightness_temp_t11=t11,
                    bright_delta=delta_t,
                    confidence=conf,
                    daynight=dn,
                    scan=scan,
                    track=track,
                    h3_res7=compute_h3_cell(lat_j, lon_j, 7),
                    h3_res8=compute_h3_cell(lat_j, lon_j, 8),
                    h3_res9=compute_h3_cell(lat_j, lon_j, 9),
                    raw_properties={"ground_truth_class": 1, "phase": "disaster_surge"}
                )
                detections.append(det)
                sensor_counts[sensor] = sensor_counts.get(sensor, 0) + 1
                det_idx += 1

        # 3. Post-disaster smoldering (Days 13 to 14)
        for day in [13, 14]:
            ts = datetime(2026, 2, day, 7, 30)
            det = FIRMSDetection(
                detection_id=f"CHEM_DISASTER_{det_idx:06d}",
                latitude=center_lat + rng.normal(0, 40.0 * m_to_deg_lat),
                longitude=center_lon + rng.normal(0, 40.0 * m_to_deg_lon),
                acq_date=ts.strftime("%Y-%m-%d"),
                acq_time=ts.strftime("%H%M"),
                timestamp=ts,
                sensor="MODIS",
                satellite="Aqua",
                frp=38.0,
                brightness_temp_t4=330.0,
                brightness_temp_t11=300.0,
                bright_delta=30.0,
                confidence=0.82,
                daynight="D",
                scan=1.0,
                track=1.0,
                h3_res7=compute_h3_cell(center_lat, center_lon, 7),
                h3_res8=compute_h3_cell(center_lat, center_lon, 8),
                h3_res9=compute_h3_cell(center_lat, center_lon, 9),
                raw_properties={"ground_truth_class": 1, "phase": "post_smolder"}
            )
            detections.append(det)
            sensor_counts["MODIS"] = sensor_counts.get("MODIS", 0) + 1
            det_idx += 1

        bbox = (72.880, 18.980, 72.910, 19.010)
        return NormalizedDataset(
            dataset_name="trombay_chemical_disaster_corridor",
            detections=detections,
            facilities=[facility],
            bounding_box=bbox,
            start_date=detections[0].acq_date,
            end_date=detections[-1].acq_date,
            total_detections=len(detections),
            sensor_counts=sensor_counts,
        )

    # -------------------------------------------------------------------------
    # Corridor 3: Punjab Post-Monsoon Stubble Burning (Agricultural)
    # -------------------------------------------------------------------------
    def generate_punjab_stubble_corridor(
        self, num_detections: int = 500, seed: int = 202
    ) -> NormalizedDataset:
        """
        Generate dispersed transient agricultural stubble burn detections across Punjab/Haryana.
        Characteristics: 100% Daytime (DNBI = 0.0), far from industry (>6km), moderate FRP (12-55 MW).
        """
        rng = np.random.RandomState(seed)

        # Representative rural agricultural district polygon
        coords = [
            [75.750, 30.850],
            [75.950, 30.850],
            [75.950, 31.050],
            [75.750, 31.050],
            [75.750, 30.850]
        ]
        facility = self._create_facility_from_coords(
            fac_id="AGRI_LUDHIANA_DIST_001",
            name="Ludhiana Agricultural District Block",
            fac_type="cropland",
            coordinates=coords,
            properties={"land_type": "Paddy Cropland", "state": "Punjab"}
        )

        detections: list[FIRMSDetection] = []
        sensor_counts: dict[str, int] = {}
        base_date = datetime(2026, 10, 15, 6, 30)

        # Region bounding box: 30.2 to 31.5 N, 74.8 to 76.5 E
        min_lat, max_lat = 30.20, 31.50
        min_lon, max_lon = 74.80, 76.50

        for i in range(num_detections):
            # Scattered across fields
            lat = float(rng.uniform(min_lat, max_lat))
            lon = float(rng.uniform(min_lon, max_lon))

            # Dispersed over 30 days, exclusively afternoon agricultural burn window (06:00 to 09:30 UTC)
            day_offset = rng.randint(0, 30)
            hour_val = rng.randint(6, 10)
            min_val = rng.randint(0, 60)
            ts = base_date + timedelta(days=day_offset, hours=hour_val - 6, minutes=min_val)

            # Sensors: MODIS Terra/Aqua & VIIRS S-NPP/NOAA-20 daytime
            sensor_choice = rng.choice(["MODIS", "VIIRS_SNPP", "VIIRS_NOAA20"])
            if sensor_choice == "MODIS":
                sat = "Terra" if rng.rand() > 0.5 else "Aqua"
                scan, track = 1.0, 1.0
            elif sensor_choice == "VIIRS_SNPP":
                sat = "SNPP"
                scan, track = 0.38, 0.37
            else:
                sat = "NOAA-20"
                scan, track = 0.39, 0.38

            # Agricultural Gamma distribution FRP
            frp = float(np.clip(rng.gamma(shape=3.0, scale=8.0) + 10.0, 12.0, 65.0))
            t4 = float(np.clip(rng.normal(328.0, 5.0), 312.0, 345.0))
            t11 = float(np.clip(rng.normal(299.0, 3.0), 292.0, 308.0))
            delta_t = t4 - t11
            conf = float(np.clip(rng.uniform(0.70, 0.92), 0.0, 1.0))

            det = FIRMSDetection(
                detection_id=f"STUBBLE_PUNJAB_{i+1:06d}",
                latitude=lat,
                longitude=lon,
                acq_date=ts.strftime("%Y-%m-%d"),
                acq_time=ts.strftime("%H%M"),
                timestamp=ts,
                sensor=sensor_choice,
                satellite=sat,
                frp=frp,
                brightness_temp_t4=t4,
                brightness_temp_t11=t11,
                bright_delta=delta_t,
                confidence=conf,
                daynight="D",  # 100% Daytime
                scan=scan,
                track=track,
                h3_res7=compute_h3_cell(lat, lon, 7),
                h3_res8=compute_h3_cell(lat, lon, 8),
                h3_res9=compute_h3_cell(lat, lon, 9),
                raw_properties={"ground_truth_class": 2, "crop": "paddy_straw"}
            )
            detections.append(det)
            sensor_counts[sensor_choice] = sensor_counts.get(sensor_choice, 0) + 1

        # Sort chronologically
        detections.sort(key=lambda d: d.timestamp)

        bbox = (min_lon, min_lat, max_lon, max_lat)
        return NormalizedDataset(
            dataset_name="punjab_stubble_burning_corridor",
            detections=detections,
            facilities=[facility],
            bounding_box=bbox,
            start_date=detections[0].acq_date,
            end_date=detections[-1].acq_date,
            total_detections=len(detections),
            sensor_counts=sensor_counts,
        )

    # -------------------------------------------------------------------------
    # Corridor 4: Simlipal Wildfire Front (Forest Wildfire)
    # -------------------------------------------------------------------------
    def generate_simlipal_wildfire_corridor(
        self, num_days: int = 7, seed: int = 303
    ) -> NormalizedDataset:
        """
        Generate dynamic propagating forest wildfire front across Simlipal National Park.
        Characteristics: Spatial propagation vector, high FRP (35-320 MW), day+night fire front.
        """
        rng = np.random.RandomState(seed)

        # Forest reserve boundary
        coords = [
            [86.200, 21.500],
            [86.600, 21.500],
            [86.600, 21.900],
            [86.200, 21.900],
            [86.200, 21.500]
        ]
        facility = self._create_facility_from_coords(
            fac_id="FOR_SIMLIPAL_001",
            name="Simlipal National Park Biosphere Reserve",
            fac_type="forest",
            coordinates=coords,
            properties={"biome": "Tropical Moist Deciduous Forest", "state": "Odisha"}
        )

        detections: list[FIRMSDetection] = []
        sensor_counts: dict[str, int] = {}
        base_date = datetime(2026, 3, 10, 0, 0, 0)
        det_idx = 1

        # Propagation origin and velocity vector (degrees per day)
        origin_lat, origin_lon = 21.620, 86.320
        vel_lat, vel_lon = 0.022, 0.018  # Progresses North-Eastward

        for day in range(num_days):
            current_day = base_date + timedelta(days=day)
            front_center_lat = origin_lat + (day * vel_lat)
            front_center_lon = origin_lon + (day * vel_lon)
            front_radius_deg = 0.015 + (day * 0.005)  # Expanding curvilinear perimeter

            # 4 passes per day (2 Day, 2 Night)
            passes = [
                ("0615", "MODIS", "Terra", "D", 1.1, 1.0, 1.0),
                ("0845", "VIIRS_SNPP", "SNPP", "D", 0.38, 0.37, 1.2),
                ("1830", "MODIS", "Aqua", "N", 1.2, 1.1, 0.7),
                ("2115", "VIIRS_NOAA21", "NOAA-21", "N", 0.42, 0.40, 0.8),
            ]

            for acq_time, sensor, sat, dn, scan, track, intensity_scale in passes:
                ts = datetime.strptime(f"{current_day.strftime('%Y-%m-%d')} {acq_time}", "%Y-%m-%d %H%M")
                # 6 to 12 detections along active flaming arc
                num_points = rng.randint(6, 13)
                for _ in range(num_points):
                    angle = rng.uniform(-0.5 * math.pi, 0.5 * math.pi)  # Forward-facing arc
                    r = front_radius_deg + rng.normal(0, 0.004)
                    lat = front_center_lat + (r * math.cos(angle))
                    lon = front_center_lon + (r * math.sin(angle))

                    base_frp = 120.0 * intensity_scale
                    frp = float(np.clip(rng.normal(base_frp, 35.0), 35.0, 360.0))
                    t4 = float(np.clip(rng.normal(355.0, 8.0), 335.0, 385.0))
                    t11 = float(np.clip(rng.normal(293.0, 4.0), 288.0, 302.0))
                    delta_t = t4 - t11
                    conf = float(np.clip(rng.uniform(0.85, 0.98), 0.0, 1.0))

                    det = FIRMSDetection(
                        detection_id=f"WILDFIRE_SIMLIPAL_{det_idx:06d}",
                        latitude=lat,
                        longitude=lon,
                        acq_date=current_day.strftime("%Y-%m-%d"),
                        acq_time=acq_time,
                        timestamp=ts,
                        sensor=sensor,
                        satellite=sat,
                        frp=frp,
                        brightness_temp_t4=t4,
                        brightness_temp_t11=t11,
                        bright_delta=delta_t,
                        confidence=conf,
                        daynight=dn,
                        scan=scan,
                        track=track,
                        h3_res7=compute_h3_cell(lat, lon, 7),
                        h3_res8=compute_h3_cell(lat, lon, 8),
                        h3_res9=compute_h3_cell(lat, lon, 9),
                        raw_properties={"ground_truth_class": 3, "propagation_day": day}
                    )
                    detections.append(det)
                    sensor_counts[sensor] = sensor_counts.get(sensor, 0) + 1
                    det_idx += 1

        bbox = (86.200, 21.500, 86.600, 21.900)
        return NormalizedDataset(
            dataset_name="simlipal_wildfire_corridor",
            detections=detections,
            facilities=[facility],
            bounding_box=bbox,
            start_date=detections[0].acq_date,
            end_date=detections[-1].acq_date,
            total_detections=len(detections),
            sensor_counts=sensor_counts,
        )

    # -------------------------------------------------------------------------
    # Corridor 5: Jurong Island Petrochemical Cluster (Multi-Facility Isolation)
    # -------------------------------------------------------------------------
    def generate_jurong_petrochemical_corridor(self, seed: int = 404) -> NormalizedDataset:
        """
        Generate contiguous multi-facility industrial park on Jurong Island, Singapore.
        Validates multi-ring buffering and cluster purity >= 90% in dense proximity.
        """
        rng = np.random.RandomState(seed)

        # 4 distinct industrial facilities on Jurong Island
        facility_specs = [
            ("JURONG_REF_01", "Jurong Shell Refinery North", "refinery", [
                [103.680, 1.260], [103.700, 1.260], [103.700, 1.280], [103.680, 1.280], [103.680, 1.260]
            ], [(1.270, 103.690), (1.275, 103.685)]),
            ("JURONG_CHEM_02", "Jurong Petrochemical Cracker South", "chemical", [
                [103.705, 1.260], [103.725, 1.260], [103.725, 1.280], [103.705, 1.280], [103.705, 1.260]
            ], [(1.270, 103.715), (1.265, 103.710)]),
            ("JURONG_STORAGE_03", "Jurong Aromatics & Chemical Terminal", "chemical", [
                [103.680, 1.240], [103.700, 1.240], [103.700, 1.255], [103.680, 1.255], [103.680, 1.240]
            ], [(1.248, 103.690)]),
            ("JURONG_POWER_04", "Tuas & Jurong Thermal Power Station", "power", [
                [103.705, 1.240], [103.725, 1.240], [103.725, 1.255], [103.705, 1.255], [103.705, 1.240]
            ], [(1.248, 103.715)]),
        ]

        facilities: list[IndustrialFacility] = []
        all_anchors: list[tuple[str, float, float]] = []

        for fac_id, name, fac_type, poly_coords, anchors in facility_specs:
            fac = self._create_facility_from_coords(
                fac_id=fac_id, name=name, fac_type=fac_type, coordinates=poly_coords
            )
            facilities.append(fac)
            for lat, lon in anchors:
                all_anchors.append((fac_id, lat, lon))

        detections: list[FIRMSDetection] = []
        sensor_counts: dict[str, int] = {}
        base_date = datetime(2026, 1, 1, 0, 0, 0)
        det_idx = 1

        m_to_deg_lat = 1.0 / 111320.0
        m_to_deg_lon = 1.0 / (111320.0 * math.cos(math.radians(1.26)))

        # 45 days of observation
        for day in range(45):
            current_day = base_date + timedelta(days=day)
            acq_date_str = current_day.strftime("%Y-%m-%d")

            passes = [
                ("0600", "VIIRS_SNPP", "SNPP", "D", 0.38, 0.37),
                ("1830", "VIIRS_NOAA20", "NOAA-20", "N", 0.39, 0.38),
            ]

            for acq_time, sensor, sat, dn, scan, track in passes:
                ts = datetime.strptime(f"{acq_date_str} {acq_time}", "%Y-%m-%d %H%M")
                for fac_id, a_lat, a_lon in all_anchors:
                    if rng.rand() < 0.85:  # 85% flare availability
                        lat_j = a_lat + rng.normal(0, 50.0 * m_to_deg_lat)
                        lon_j = a_lon + rng.normal(0, 50.0 * m_to_deg_lon)

                        frp = float(np.clip(rng.normal(38.0, 8.0), 18.0, 75.0))
                        t4 = float(np.clip(rng.normal(340.0, 5.0), 325.0, 355.0))
                        t11 = float(np.clip(rng.normal(296.0, 2.0), 292.0, 300.0))
                        delta_t = t4 - t11
                        conf = float(np.clip(rng.uniform(0.90, 0.99), 0.0, 1.0))

                        det = FIRMSDetection(
                            detection_id=f"JURONG_DET_{det_idx:06d}",
                            latitude=lat_j,
                            longitude=lon_j,
                            acq_date=acq_date_str,
                            acq_time=acq_time,
                            timestamp=ts,
                            sensor=sensor,
                            satellite=sat,
                            frp=frp,
                            brightness_temp_t4=t4,
                            brightness_temp_t11=t11,
                            bright_delta=delta_t,
                            confidence=conf,
                            daynight=dn,
                            scan=scan,
                            track=track,
                            h3_res7=compute_h3_cell(lat_j, lon_j, 7),
                            h3_res8=compute_h3_cell(lat_j, lon_j, 8),
                            h3_res9=compute_h3_cell(lat_j, lon_j, 9),
                            raw_properties={"ground_truth_class": 0, "parent_facility_id": fac_id}
                        )
                        detections.append(det)
                        sensor_counts[sensor] = sensor_counts.get(sensor, 0) + 1
                        det_idx += 1

        bbox = (103.680, 1.240, 103.725, 1.280)
        return NormalizedDataset(
            dataset_name="jurong_petrochemical_corridor",
            detections=detections,
            facilities=facilities,
            bounding_box=bbox,
            start_date=detections[0].acq_date,
            end_date=detections[-1].acq_date,
            total_detections=len(detections),
            sensor_counts=sensor_counts,
        )

    # -------------------------------------------------------------------------
    # Master Benchmark Suite Generator
    # -------------------------------------------------------------------------
    def generate_master_benchmark_suite(
        self, output_dir: Optional[Union[str, Path]] = None
    ) -> dict[str, NormalizedDataset]:
        """
        Generate full suite of operational corridors and optionally persist sample datasets
        to data/raw/, data/osm/, and data/processed/.
        """
        jamnagar_ds = self.generate_jamnagar_corridor(num_days=30, seed=42)
        chemical_ds = self.generate_chemical_disaster_corridor(seed=101)
        punjab_ds = self.generate_punjab_stubble_corridor(num_detections=300, seed=202)
        simlipal_ds = self.generate_simlipal_wildfire_corridor(num_days=5, seed=303)
        jurong_ds = self.generate_jurong_petrochemical_corridor(seed=404)

        suite = {
            "jamnagar": jamnagar_ds,
            "chemical_disaster": chemical_ds,
            "punjab_stubble": punjab_ds,
            "simlipal_wildfire": simlipal_ds,
            "jurong": jurong_ds,
        }

        if output_dir is not None:
            out_path = Path(output_dir)
            raw_dir = out_path / "raw"
            osm_dir = out_path / "osm"
            proc_dir = out_path / "processed"

            raw_dir.mkdir(parents=True, exist_ok=True)
            osm_dir.mkdir(parents=True, exist_ok=True)
            proc_dir.mkdir(parents=True, exist_ok=True)

            # 1. Export sample raw feeds
            all_dets: list[FIRMSDetection] = []
            for ds in suite.values():
                all_dets.extend(ds.detections)

            # Master detections dataframe
            df_master = FIRMSDetection.to_dataframe(all_dets)
            df_master.to_csv(proc_dir / "master_benchmark_detections.csv", index=False)

            # Sample MODIS CSV
            modis_dets = [d for d in all_dets if d.sensor == "MODIS"]
            if modis_dets:
                df_modis = FIRMSDetection.to_dataframe(modis_dets)
                df_modis.to_csv(raw_dir / "sample_modis_india.csv", index=False)

            # Sample VIIRS SNPP CSV
            viirs_dets = [d for d in all_dets if "VIIRS" in d.sensor]
            if viirs_dets:
                df_viirs = FIRMSDetection.to_dataframe(viirs_dets)
                df_viirs.to_csv(raw_dir / "sample_viirs_snpp_india.csv", index=False)

            # Sample GeoJSON
            sample_features = []
            for d in all_dets[:200]:
                sample_features.append({
                    "type": "Feature",
                    "id": d.detection_id,
                    "geometry": {
                        "type": "Point",
                        "coordinates": [d.longitude, d.latitude]
                    },
                    "properties": d.to_dict()
                })
            with open(raw_dir / "sample_viirs_noaa20_india.geojson", "w", encoding="utf-8") as f:
                json.dump({"type": "FeatureCollection", "features": sample_features}, f, indent=2)

            # 2. Export OSM GeoJSON bundles
            all_facilities: list[IndustrialFacility] = []
            for ds in suite.values():
                all_facilities.extend(ds.facilities)

            # Master OSM GeoJSON
            master_osm_features = [fac.to_geojson_feature() for fac in all_facilities]
            with open(osm_dir / "industrial_polygons_master.geojson", "w", encoding="utf-8") as f:
                json.dump({"type": "FeatureCollection", "features": master_osm_features}, f, indent=2)

            # Jamnagar OSM GeoJSON
            jam_osm = [fac.to_geojson_feature() for fac in jamnagar_ds.facilities]
            with open(osm_dir / "industrial_polygons_jamnagar.geojson", "w", encoding="utf-8") as f:
                json.dump({"type": "FeatureCollection", "features": jam_osm}, f, indent=2)

            # Jurong OSM GeoJSON
            jurong_osm = [fac.to_geojson_feature() for fac in jurong_ds.facilities]
            with open(osm_dir / "industrial_polygons_jurong.geojson", "w", encoding="utf-8") as f:
                json.dump({"type": "FeatureCollection", "features": jurong_osm}, f, indent=2)

        return suite
