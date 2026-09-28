"""
Geospatial Data Pipeline Package
Handles NASA FIRMS ingestion, OSM spatial indexing, schema definitions, and synthetic data generation.
"""

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
)
from src.data_pipeline.synthetic_generator import (
    SyntheticDataGenerator,
)

__all__ = [
    "FIRMSDetection",
    "IndustrialFacility",
    "NormalizedDataset",
    "FIRMSIngestionEngine",
    "compute_h3_cell",
    "SpatialIndexEngine",
    "haversine_distance",
    "create_catchment_buffers",
    "SyntheticDataGenerator",
]
