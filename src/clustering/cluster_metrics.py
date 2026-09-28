"""
Clustering Benchmark Evaluation and Purity Metrics Suite.
Evaluates cluster purity (>= 90.0% benchmark target), homogeneity, completeness,
V-measure, Adjusted Rand Index (ARI), Normalized Mutual Information (NMI), and Silhouette scores.
Authoritative Specifications: ORIGINAL_REQUEST.md § Acceptance Data, PROJECT.md, m2_blueprint.md
"""

from __future__ import annotations

from typing import Any, Optional, Sequence, Union
import numpy as np
from sklearn.metrics import (
    homogeneity_completeness_v_measure,
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
    davies_bouldin_score
)

from src.clustering.schemas import ThermalCluster, ClusterEvaluationMetrics


class ClusterMetricsEvaluator:
    """
    Evaluates cluster purity, information-theoretic metrics, and spatial separation against ground truth benchmarks.
    """

    @staticmethod
    def calculate_cluster_purity(
        predicted_labels: Sequence[int],
        ground_truth_labels: Sequence[Union[int, str]]
    ) -> float:
        """
        Compute exact cluster purity: (1/N) * sum_k max_j |omega_k cap c_j|.
        Ignores unclustered noise points labeled -1.
        Returns float in [0.0, 1.0].
        """
        if len(predicted_labels) == 0 or len(ground_truth_labels) == 0:
            return 0.0

        pred = np.array(predicted_labels)
        truth = np.array(ground_truth_labels)

        # Filter out unclustered noise (-1)
        valid_mask = pred >= 0
        if not np.any(valid_mask):
            return 0.0

        valid_pred = pred[valid_mask]
        valid_truth = truth[valid_mask]
        n_total = len(valid_pred)

        unique_clusters = np.unique(valid_pred)
        purity_sum = 0

        for c in unique_clusters:
            c_mask = valid_pred == c
            c_truth = valid_truth[c_mask]
            
            # Find majority class count in cluster c
            _, counts = np.unique(c_truth, return_counts=True)
            max_class_count = int(np.max(counts))
            purity_sum += max_class_count

        return float(purity_sum / n_total)

    @classmethod
    def evaluate_benchmark(
        cls,
        predicted_labels: Sequence[int],
        ground_truth_labels: Sequence[Union[int, str]],
        coords_rad: Optional[np.ndarray] = None
    ) -> ClusterEvaluationMetrics:
        """
        Compute full benchmark metric suite: Purity, Homogeneity, Completeness, V-Measure, ARI, NMI.
        """
        pred = np.array(predicted_labels)
        truth = np.array(ground_truth_labels)

        purity = cls.calculate_cluster_purity(pred, truth)

        # Filter valid points for scikit-learn metrics
        valid_mask = pred >= 0
        if np.any(valid_mask):
            try:
                h, c, v = homogeneity_completeness_v_measure(truth[valid_mask], pred[valid_mask])
                ari = float(adjusted_rand_score(truth[valid_mask], pred[valid_mask]))
                nmi = float(normalized_mutual_info_score(truth[valid_mask], pred[valid_mask]))
            except Exception:
                h, c, v, ari, nmi = 0.0, 0.0, 0.0, 0.0, 0.0
        else:
            h, c, v, ari, nmi = 0.0, 0.0, 0.0, 0.0, 0.0

        # Unsupervised Silhouette score and Davies-Bouldin Index if coordinates provided
        sil: Optional[float] = None
        db_idx: Optional[float] = None
        if coords_rad is not None and np.any(valid_mask):
            unique_valid = np.unique(pred[valid_mask])
            if len(unique_valid) > 1 and len(valid_mask) > len(unique_valid):
                try:
                    sil = float(silhouette_score(coords_rad[valid_mask], pred[valid_mask], metric="haversine"))
                    db_idx = float(davies_bouldin_score(coords_rad[valid_mask], pred[valid_mask]))
                except Exception:
                    sil, db_idx = None, None

        return ClusterEvaluationMetrics(
            total_samples=len(pred),
            total_clusters=int(len(np.unique(pred[pred >= 0]))),
            noise_count=int(np.sum(pred == -1)),
            cluster_purity=purity,
            homogeneity=float(h),
            completeness=float(c),
            v_measure=float(v),
            adjusted_rand_index=ari,
            normalized_mutual_info=nmi,
            silhouette_score=sil,
            davies_bouldin_index=db_idx
        )
