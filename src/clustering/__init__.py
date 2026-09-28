"""
NTRO Thermal Clustering & Recurrence Analysis Package.
Exposes spatio-temporal clustering engines, H3 discrete global grid indexing,
persistence and flare profile analytics, and cluster purity benchmark evaluation.
Authoritative Specifications: ORIGINAL_REQUEST.md § R1, PROJECT.md, m2_blueprint.md
"""

from __future__ import annotations

from src.clustering.schemas import (
    ThermalCluster,
    ClusterType,
    H3SpatialBin,
    ClusterEvaluationMetrics,
)
from src.clustering.h3_indexer import (
    H3SpatialIndexer,
    compute_h3_index,
    get_h3_boundary,
    get_h3_centroid,
    get_h3_k_ring,
)
from src.clustering.dbscan_engine import (
    SpatioTemporalClusteringEngine,
    run_hdbscan,
)
from src.clustering.persistence_analyzer import (
    PersistenceAnalyzer,
    haversine_distance,
    calculate_dnbi,
    calculate_spatial_jitter,
    calculate_centroid_drift_velocity,
    EARTH_RADIUS_METERS,
)
from src.clustering.cluster_metrics import (
    ClusterMetricsEvaluator,
)

__all__ = [
    "ThermalCluster",
    "ClusterType",
    "H3SpatialBin",
    "ClusterEvaluationMetrics",
    "H3SpatialIndexer",
    "compute_h3_index",
    "get_h3_boundary",
    "get_h3_centroid",
    "get_h3_k_ring",
    "SpatioTemporalClusteringEngine",
    "run_hdbscan",
    "PersistenceAnalyzer",
    "haversine_distance",
    "calculate_dnbi",
    "calculate_spatial_jitter",
    "calculate_centroid_drift_velocity",
    "EARTH_RADIUS_METERS",
    "ClusterMetricsEvaluator",
]
