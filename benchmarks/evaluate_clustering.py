"""
Clustering Benchmark Evaluation Script.
Evaluates spatio-temporal clustering engine against realistic multi-corridor benchmark datasets.
Authoritative Specifications: ORIGINAL_REQUEST.md § Acceptance Criteria, PROJECT.md
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from src.data_pipeline.synthetic_generator import SyntheticDataGenerator
from src.clustering.dbscan_engine import SpatioTemporalClusteringEngine
from src.clustering.cluster_metrics import ClusterMetricsEvaluator


def run_clustering_benchmark() -> dict[str, float]:
    """Execute master clustering benchmark suite and evaluate cluster purity."""
    print("=" * 70)
    print("NTRO THERMAL CLUSTERING BENCHMARK SUITE (SIH26162)")
    print("=" * 70)

    gen = SyntheticDataGenerator()
    suite = gen.generate_master_benchmark_suite()

    all_detections = []
    ground_truth = []

    for name, ds in suite.items():
        print(f"Loaded corridor '{name}': {len(ds.detections)} detections, {len(ds.facilities)} facilities.")
        for d in ds.detections:
            all_detections.append(d)
            ground_truth.append(name)

    print(f"\nTotal benchmark detections: {len(all_detections)}")

    engine = SpatioTemporalClusteringEngine()
    clusters = engine.run_dbscan_clustering(all_detections, eps_m=1000.0, min_samples=3)

    print(f"Discovered {len(clusters)} spatio-temporal clusters.")

    metrics = engine.evaluate_cluster_purity(clusters, ground_truth, detections=all_detections)

    print("\n--- CLUSTERING PERFORMANCE METRICS ---")
    print(f"  * Cluster Purity:         {metrics['cluster_purity'] * 100.0:.2f}% (Target: >= 90.0%)")
    print(f"  * Homogeneity Score:      {metrics['homogeneity']:.4f}")
    print(f"  * Completeness Score:     {metrics['completeness']:.4f}")
    print(f"  * V-Measure Score:        {metrics['v_measure']:.4f}")
    print(f"  * Adjusted Rand Index:    {metrics['adjusted_rand_index']:.4f}")
    print(f"  * Normalized Mutual Info: {metrics['normalized_mutual_info']:.4f}")
    print("=" * 70)

    purity_pass = metrics["cluster_purity"] >= 0.90
    print(f"VERDICT: {'PASSED [AC-DATA-2 SATISFIED]' if purity_pass else 'FAILED'}")
    print("=" * 70)

    return metrics


if __name__ == "__main__":
    metrics = run_clustering_benchmark()
    if metrics["cluster_purity"] < 0.90:
        sys.exit(1)
