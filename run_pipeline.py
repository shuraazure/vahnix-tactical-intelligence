"""
End-to-End Tactical Pipeline Runner for NTRO Thermal Detection System (SIH26162).
Executes complete automated processing pipeline:
Ingestion -> OSM Indexing -> Spatio-Temporal Clustering -> 28D Feature Extraction
-> Multi-Class AI Classification -> Risk Scoring -> SHAP XAI -> Intelligence Dossier Export.
Authoritative Specifications: ORIGINAL_REQUEST.md, PROJECT.md, TEST_INFRA.md
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from src.data_pipeline.schemas import FIRMSDetection, IndustrialFacility
from src.data_pipeline.firms_ingestion import FIRMSIngestionEngine
from src.data_pipeline.osm_indexer import SpatialIndexEngine
from src.data_pipeline.synthetic_generator import SyntheticDataGenerator
from src.clustering.dbscan_engine import SpatioTemporalClusteringEngine
from src.features.feature_extractor import FeatureExtractor
from src.classification.ensemble_classifier import ThermalEnsembleClassifier
from src.classification.model_trainer import ModelTrainer
from src.classification.heuristic_baseline import CLASS_NAMES
from src.explainability.shap_engine import SHAPExplainerEngine
from src.reporting.pdf_exporter import PDFReportExporter
from src.reporting.html_exporter import HTMLReportExporter
from src.reporting.gis_exporter import GISDataExporter


def run_e2e_pipeline(
    firms_path: Optional[str] = None,
    osm_path: Optional[str] = None,
    output_dir: str = "data/outputs",
    model_path: Optional[str] = "data/models/thermal_classifier.pkl",
    generate_reports: bool = True,
    verbose: bool = True
) -> dict[str, Any]:
    """
    Execute complete end-to-end NTRO Thermal Processing Pipeline.
    """
    t_start = time.perf_counter()
    os.makedirs(output_dir, exist_ok=True)
    reports_dir = os.path.join(output_dir, "reports")
    os.makedirs(reports_dir, exist_ok=True)

    if verbose:
        print("=" * 80)
        print("NTRO THERMAL DETECTION & AI CLASSIFICATION PIPELINE (SIH26162)")
        print(f"Timestamp: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}")
        print("=" * 80)

    # -------------------------------------------------------------------------
    # Step 1: Spatial Indexing & OSM Ingestion
    # -------------------------------------------------------------------------
    t0 = time.perf_counter()
    spatial_engine = SpatialIndexEngine()
    if osm_path and os.path.exists(osm_path):
        spatial_engine.load_osm_boundaries(osm_path)
    else:
        # Default bundled OSM polygons
        default_osm = "data/osm/industrial_polygons_master.geojson"
        if os.path.exists(default_osm):
            spatial_engine.load_osm_boundaries(default_osm)
    t_osm = time.perf_counter() - t0
    if verbose:
        print(f"[1/7] Ingested {len(spatial_engine.facilities)} OSM industrial facilities in {t_osm*1000:.1f}ms.")

    # -------------------------------------------------------------------------
    # Step 2: FIRMS Ingestion
    # -------------------------------------------------------------------------
    t0 = time.perf_counter()
    detections: list[FIRMSDetection] = []
    if firms_path and os.path.exists(firms_path):
        if firms_path.endswith(".csv"):
            detections = FIRMSIngestionEngine.ingest_csv(firms_path)
        elif firms_path.endswith(".geojson") or firms_path.endswith(".json"):
            detections = FIRMSIngestionEngine.ingest_geojson(firms_path)
    else:
        # Load sample multi-corridor benchmark suite
        gen = SyntheticDataGenerator(seed=42)
        suite = gen.generate_master_benchmark_suite()
        for ds in suite.values():
            detections.extend(ds.detections)
            for fac in ds.facilities:
                if fac.facility_id not in spatial_engine.facilities:
                    spatial_engine.add_facility(fac)

    t_ingest = time.perf_counter() - t0
    if verbose:
        print(f"[2/7] Ingested {len(detections)} active thermal detections in {t_ingest*1000:.1f}ms.")

    # -------------------------------------------------------------------------
    # Step 3: Spatio-Temporal Clustering & Persistence Analysis
    # -------------------------------------------------------------------------
    t0 = time.perf_counter()
    clustering_engine = SpatioTemporalClusteringEngine(spatial_index_engine=spatial_engine)
    clusters = clustering_engine.run_dbscan_clustering(detections, eps_m=1000.0, min_samples=3)
    t_cluster = time.perf_counter() - t0
    if verbose:
        print(f"[3/7] Discovered {len(clusters)} spatio-temporal clusters in {t_cluster*1000:.1f}ms.")

    # -------------------------------------------------------------------------
    # Step 4: 28D Feature Extraction
    # -------------------------------------------------------------------------
    t0 = time.perf_counter()
    feature_extractor = FeatureExtractor(spatial_index=spatial_engine)
    X = feature_extractor.extract_batch(detections, clusters)
    t_feat = time.perf_counter() - t0
    if verbose:
        print(f"[4/7] Extracted 28D feature matrix {X.shape} in {t_feat*1000:.1f}ms.")

    # -------------------------------------------------------------------------
    # Step 5: Multi-Class Classification & Risk Scoring
    # -------------------------------------------------------------------------
    t0 = time.perf_counter()
    if model_path and os.path.exists(model_path):
        classifier = ThermalEnsembleClassifier.load(model_path)
    else:
        trainer = ModelTrainer(random_state=42)
        classifier, _ = trainer.train_and_save_model(output_path=model_path or "data/models/thermal_classifier.pkl")

    predictions = classifier.predict_batch(detections, clusters)
    t_infer = time.perf_counter() - t0
    if verbose:
        crit_count = sum(1 for p in predictions if p.is_critical_alert)
        emerg_count = sum(1 for p in predictions if p.predicted_class.value == 1)
        print(f"[5/7] Executed ML classification in {t_infer*1000:.1f}ms ({emerg_count} Industrial Emergencies, {crit_count} Critical Alerts).")

    # -------------------------------------------------------------------------
    # Step 6: SHAP Explainability & Tactical Narratives
    # -------------------------------------------------------------------------
    t0 = time.perf_counter()
    explainer = SHAPExplainerEngine(classifier=classifier, background_data=X[:25])
    explanations = {}
    for i, p in enumerate(predictions[:10]):  # Explain top priority incidents
        exp = explainer.explain_instance(X[i], p, generate_plot=generate_reports)
        explanations[p.detection_id] = exp
    t_xai = time.perf_counter() - t0
    if verbose:
        print(f"[6/7] Generated SHAP XAI TreeExplainer attributions in {t_xai*1000:.1f}ms.")

    # -------------------------------------------------------------------------
    # Step 7: Export Multi-Format Tactical Artifacts
    # -------------------------------------------------------------------------
    t0 = time.perf_counter()
    geojson_incidents = os.path.join(output_dir, "classified_incidents.geojson")
    geojson_clusters = os.path.join(output_dir, "thermal_clusters.geojson")
    csv_incidents = os.path.join(output_dir, "classified_incidents.csv")

    GISDataExporter.export_detections_geojson(detections, predictions, output_path=geojson_incidents)
    GISDataExporter.export_clusters_geojson(clusters, output_path=geojson_clusters)
    GISDataExporter.export_detections_csv(detections, predictions, output_path=csv_incidents)

    if generate_reports and predictions:
        # Generate PDF & HTML for highest risk incident
        top_risk_pred = max(predictions, key=lambda p: p.risk_severity_index)
        top_det = next(d for d in detections if d.detection_id == top_risk_pred.detection_id)
        top_exp = explanations.get(top_risk_pred.detection_id)

        pdf_out = os.path.join(reports_dir, f"incident_{top_risk_pred.detection_id}.pdf")
        html_out = os.path.join(reports_dir, f"incident_{top_risk_pred.detection_id}.html")
        PDFReportExporter.generate_incident_pdf_report(top_det, top_risk_pred, top_exp, output_path=pdf_out)
        HTMLReportExporter.generate_incident_html_dossier(top_det, top_risk_pred, top_exp, output_path=html_out)

    t_export = time.perf_counter() - t0
    t_total = time.perf_counter() - t_start

    if verbose:
        print(f"[7/7] Exported GIS vector bundles, CSV tables, and dossier reports in {t_export*1000:.1f}ms.")
        print("-" * 80)
        print(f"PIPELINE EXECUTION COMPLETE: Total Latency = {t_total:.3f}s")
        print(f"Artifacts saved to: {output_dir}/")
        print("=" * 80)

    return {
        "total_detections": len(detections),
        "total_clusters": len(clusters),
        "total_facilities": len(spatial_engine.facilities),
        "total_predictions": len(predictions),
        "pipeline_latency_sec": t_total,
        "output_dir": output_dir
    }


def main():
    parser = argparse.ArgumentParser(description="NTRO Thermal Detection & AI Classification Pipeline CLI")
    parser.add_argument("--firms", type=str, default=None, help="Path to FIRMS CSV or GeoJSON dataset")
    parser.add_argument("--osm", type=str, default=None, help="Path to OSM industrial boundaries GeoJSON")
    parser.add_argument("--output-dir", type=str, default="data/outputs", help="Output directory for generated artifacts")
    parser.add_argument("--model-path", type=str, default="data/models/thermal_classifier.pkl", help="Path to serialized ML model")
    parser.add_argument("--no-reports", action="store_true", help="Skip PDF/HTML report generation")
    args = parser.parse_args()

    run_e2e_pipeline(
        firms_path=args.firms,
        osm_path=args.osm,
        output_dir=args.output_dir,
        model_path=args.model_path,
        generate_reports=not args.no_reports,
        verbose=True
    )


if __name__ == "__main__":
    main()
