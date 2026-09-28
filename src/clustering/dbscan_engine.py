"""
Spatio-Temporal Clustering Engine.
Implements Haversine DBSCAN (500m core stationary discovery), ST-DBSCAN (1000m, 48h temporal window),
HDBSCAN multi-density clustering, and rapid Uber H3 discrete global grid clustering.
Authoritative Specifications: ORIGINAL_REQUEST.md § R1, PROJECT.md, m2_blueprint.md
"""

from __future__ import annotations

from collections import deque
from datetime import datetime
from typing import Any, Optional, Sequence, Union
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.neighbors import BallTree

from src.data_pipeline.schemas import FIRMSDetection
from src.clustering.schemas import ThermalCluster, ClusterType
from src.clustering.h3_indexer import H3SpatialIndexer
from src.clustering.persistence_analyzer import PersistenceAnalyzer, EARTH_RADIUS_METERS


def run_hdbscan(
    coords_rad: np.ndarray,
    min_cluster_size: int = 3,
    min_samples: Optional[int] = None
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run HDBSCAN on radian coordinates.
    Returns (labels, probabilities).
    """
    n = len(coords_rad)
    if n < min_cluster_size:
        return np.full(n, -1, dtype=int), np.zeros(n, dtype=float)

    # 1. Try sklearn.cluster.HDBSCAN (scikit-learn >= 1.3)
    try:
        import sklearn.cluster
        if hasattr(sklearn.cluster, "HDBSCAN"):
            hdb = sklearn.cluster.HDBSCAN(
                min_cluster_size=min_cluster_size,
                min_samples=min_samples or min_cluster_size,
                metric="haversine"
            )
            labels = hdb.fit_predict(coords_rad)
            probs = getattr(hdb, "probabilities_", np.ones(n, dtype=float))
            return labels, probs
    except Exception:
        pass

    # 2. Try standalone hdbscan package
    try:
        import hdbscan
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples or min_cluster_size,
            metric="haversine"
        )
        labels = clusterer.fit_predict(coords_rad)
        probs = clusterer.probabilities_
        return labels, probs
    except Exception:
        pass

    # 3. Fallback to Haversine DBSCAN
    try:
        eps_rad = 500.0 / EARTH_RADIUS_METERS
        db = DBSCAN(eps=eps_rad, min_samples=min_cluster_size, metric="haversine", algorithm="ball_tree")
        labels = db.fit_predict(coords_rad)
        probs = np.where(labels >= 0, 1.0, 0.0)
        return labels, probs
    except Exception:
        return np.full(n, -1, dtype=int), np.zeros(n, dtype=float)


class SpatioTemporalClusteringEngine:
    """
    Unified spatio-temporal clustering orchestrator implementing H3, Haversine DBSCAN,
    ST-DBSCAN, HDBSCAN, and cluster-to-facility enrichment.
    """

    def __init__(self, spatial_index_engine: Optional[Any] = None):
        self.spatial_index = spatial_index_engine
        self.h3_indexer = H3SpatialIndexer(default_resolution=8)
        self.persistence_analyzer = PersistenceAnalyzer(spatial_index_engine=spatial_index_engine)

    def run_dbscan_clustering(
        self,
        detections: list[FIRMSDetection],
        eps_m: float = 500.0,
        min_samples: int = 3
    ) -> list[ThermalCluster]:
        """
        Run Haversine DBSCAN clustering to isolate stationary, persistent thermal cores.
        Converts eps_m into radians: eps_rad = eps_m / 6,371,000.0
        """
        if not detections:
            return []

        coords = np.array([[d.latitude, d.longitude] for d in detections], dtype=float)
        coords_rad = np.radians(coords)

        eps_rad = eps_m / EARTH_RADIUS_METERS
        db = DBSCAN(eps=eps_rad, min_samples=min_samples, metric="haversine", algorithm="ball_tree")
        labels = db.fit_predict(coords_rad)

        return self._build_clusters_from_labels(detections, labels, algorithm_name="dbscan")

    def run_st_dbscan(
        self,
        detections: list[FIRMSDetection],
        spatial_eps_m: float = 1000.0,
        temporal_eps_hours: float = 48.0,
        min_samples: int = 4
    ) -> list[ThermalCluster]:
        """
        Run Spatio-Temporal DBSCAN (ST-DBSCAN) combining spatial radius and temporal delta.
        Isolates advancing wildfire fronts and seasonal crop burning progression.
        """
        if not detections:
            return []

        labels = self._execute_st_dbscan_algorithm(
            detections,
            spatial_eps_m=spatial_eps_m,
            temporal_eps_sec=temporal_eps_hours * 3600.0,
            min_samples=min_samples
        )

        return self._build_clusters_from_labels(detections, labels, algorithm_name="stdbscan")

    def run_hdbscan_clustering(
        self,
        detections: list[FIRMSDetection],
        min_cluster_size: int = 3
    ) -> list[ThermalCluster]:
        """
        Run HDBSCAN multi-density hierarchical clustering on detections.
        """
        if not detections:
            return []

        coords = np.array([[d.latitude, d.longitude] for d in detections], dtype=float)
        coords_rad = np.radians(coords)
        labels, _ = run_hdbscan(coords_rad, min_cluster_size=min_cluster_size)

        return self._build_clusters_from_labels(detections, labels, algorithm_name="hdbscan")

    def run_h3_clustering(
        self,
        detections: list[FIRMSDetection],
        res: int = 8,
        min_samples: int = 3
    ) -> list[ThermalCluster]:
        """
        Run rapid O(N) Uber H3 discrete global grid clustering.
        """
        if not detections:
            return []

        bins = self.h3_indexer.aggregate_by_h3(detections, resolution=res, min_detections=min_samples)
        clusters: list[ThermalCluster] = []
        det_map = {d.detection_id: d for d in detections}

        for idx, b in enumerate(bins):
            sub_dets = [det_map[did] for did in b.detection_ids if did in det_map]
            if not sub_dets:
                continue
            cluster = self.persistence_analyzer.analyze_cluster(
                cluster_id=f"CLUS_H3_{res}_{idx+1:04d}",
                detections=sub_dets,
                forced_type=ClusterType.STATIONARY_PERSISTENT.value
            )
            clusters.append(cluster)

        return clusters

    def _execute_st_dbscan_algorithm(
        self,
        detections: list[FIRMSDetection],
        spatial_eps_m: float,
        temporal_eps_sec: float,
        min_samples: int
    ) -> np.ndarray:
        """
        Optimized ST-DBSCAN using BallTree spatial neighborhood querying and temporal delta filtering.
        """
        n = len(detections)
        if n < min_samples:
            return np.full(n, -1, dtype=int)

        coords = np.array([[d.latitude, d.longitude] for d in detections], dtype=float)
        coords_rad = np.radians(coords)
        timestamps = np.array([d.timestamp.timestamp() for d in detections], dtype=float)

        eps_rad = spatial_eps_m / EARTH_RADIUS_METERS
        tree = BallTree(coords_rad, metric="haversine")
        spatial_neighbors = tree.query_radius(coords_rad, r=eps_rad)

        labels = np.full(n, -1, dtype=int)
        visited = np.zeros(n, dtype=bool)
        cluster_id = 0

        for i in range(n):
            if visited[i]:
                continue
            visited[i] = True

            cand = spatial_neighbors[i]
            st_nbrs = [j for j in cand if abs(timestamps[i] - timestamps[j]) <= temporal_eps_sec]

            if len(st_nbrs) < min_samples:
                labels[i] = -1
            else:
                labels[i] = cluster_id
                queue = deque([j for j in st_nbrs if j != i])

                while queue:
                    curr = queue.popleft()
                    if not visited[curr]:
                        visited[curr] = True
                        sub_cand = spatial_neighbors[curr]
                        sub_st = [k for k in sub_cand if abs(timestamps[curr] - timestamps[k]) <= temporal_eps_sec]

                        if len(sub_st) >= min_samples:
                            for k in sub_st:
                                if k not in queue and labels[k] == -1:
                                    queue.append(k)

                    if labels[curr] == -1:
                        labels[curr] = cluster_id

                cluster_id += 1

        return labels

    def _build_clusters_from_labels(
        self,
        detections: list[FIRMSDetection],
        labels: np.ndarray,
        algorithm_name: str
    ) -> list[ThermalCluster]:
        """
        Group detections by cluster label and construct ThermalCluster instances.
        """
        cluster_groups: dict[int, list[FIRMSDetection]] = {}
        for label, det in zip(labels, detections):
            if label >= 0:
                cluster_groups.setdefault(int(label), []).append(det)

        clusters: list[ThermalCluster] = []
        for label in sorted(cluster_groups.keys()):
            group = cluster_groups[label]
            cluster_id = f"CLUS_{algorithm_name.upper()[:4]}_{label+1:04d}"
            cluster = self.persistence_analyzer.analyze_cluster(cluster_id, group)
            clusters.append(cluster)

        return clusters

    def evaluate_cluster_purity(
        self,
        clusters: list[ThermalCluster],
        ground_truth_labels: list[int] | list[str] | Sequence[Any],
        detections: Optional[Sequence[FIRMSDetection]] = None
    ) -> dict[str, float]:
        """
        Evaluate cluster purity and clustering benchmark metrics against ground truth.
        Supports passing detections list to map detection_id strings to ground truth indices.
        """
        from src.clustering.cluster_metrics import ClusterMetricsEvaluator

        n = len(ground_truth_labels)
        pred_labels = np.full(n, -1, dtype=int)
        
        # Build index mapping from detection_id string to array index
        det_id_to_idx: dict[str, int] = {}
        if detections is not None and len(detections) == n:
            for i, d in enumerate(detections):
                det_id_to_idx[d.detection_id] = i
        else:
            for i, item in enumerate(ground_truth_labels):
                if hasattr(item, "detection_id"):
                    det_id_to_idx[item.detection_id] = i
                elif isinstance(item, dict) and "detection_id" in item:
                    det_id_to_idx[item["detection_id"]] = i
                elif isinstance(item, str):
                    det_id_to_idx[item] = i
                else:
                    det_id_to_idx[f"DET_{i}"] = i

        for c_idx, c in enumerate(clusters):
            for did in c.detection_ids:
                if did in det_id_to_idx:
                    pred_labels[det_id_to_idx[did]] = c_idx

        # If ground_truth_labels is a sequence of objects with .label or .class_id
        target_truths: list[Union[int, str]] = []
        for item in ground_truth_labels:
            if hasattr(item, "ground_truth_class"):
                target_truths.append(item.ground_truth_class)
            elif hasattr(item, "class_id"):
                target_truths.append(item.class_id)
            elif hasattr(item, "label"):
                target_truths.append(item.label)
            elif isinstance(item, dict) and "ground_truth_class" in item:
                target_truths.append(item["ground_truth_class"])
            else:
                target_truths.append(item)

        metrics = ClusterMetricsEvaluator.evaluate_benchmark(pred_labels, target_truths)
        return {
            "cluster_purity": metrics.cluster_purity,
            "homogeneity": metrics.homogeneity,
            "completeness": metrics.completeness,
            "v_measure": metrics.v_measure,
            "adjusted_rand_index": metrics.adjusted_rand_index,
            "normalized_mutual_info": metrics.normalized_mutual_info
        }

