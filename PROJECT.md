# Project: NTRO SIH26162 Master System Architecture

## Architecture Overview
AI-Based Detection and Classification of Industrial Fires and Persistent Thermal Sources Using NASA FIRMS, OSM & Satellite Data.
A unified multi-source, multi-modal geospatial intelligence platform integrating:
- NASA FIRMS (MODIS, VIIRS NRT/Ultra-Real-Time) active fire products
- OpenStreetMap (OSM) Overpass API industrial boundary polygons & infrastructure tags
- Multi-spectral remote sensing imagery (Sentinel-2 MSI SWIR B11/B12, Landsat 8/9 OLI-2/TIRS-2 B10/B11)
- Uber H3 discrete global grid spatial indexing & PostGIS spatial correlation
- Spatio-temporal AI/ML anomaly classification engine with Focal/Dice loss & SHAP explainability
- Real-time Kafka + Spark Streaming architecture with offline local cache fallback

## Feature Inventory
| # | Feature | Description | Milestone | Status | Source |
|---|---------|-------------|-----------|--------|--------|
| 1 | Executive Summary & Problem Analysis | Flare/kiln false alarm physics, Planck/Wien/Dozier derivations vs emergency anomalies | M1 | DONE | ORIGINAL_REQUEST §R1 |
| 2 | Data Source Deep-Dive & Availability Matrix | FIRMS, OSM, Sentinel-2, Landsat 8/9 APIs, schemas, queries, 100% free availability matrix | M1 | DONE | ORIGINAL_REQUEST §R1, R3 |
| 3 | Big Data Pipeline & Geospatial Architecture | Kafka 7-topic design, Spark Streaming, PostGIS/TimeScaleDB/Redis/MinIO, H3 Res 7/8/9, spatial math | M2 | DONE | ORIGINAL_REQUEST §R1 |
| 4 | AI/ML Classification Engine & Mathematics | 6-class taxonomy, 38-dim feature vector, ST-GNN/Transformer, GBDT spatial ensemble, loss math, SHAP, SOTA table | M3 | DONE | ORIGINAL_REQUEST §R2 |
| 5 | System Diagrams (5 Valid Mermaid) | Architecture, ingestion, inference, alert flow, component interaction | M4 | DONE | ORIGINAL_REQUEST §R1 |
| 6 | Scalability, Offline Mode & 36-Hour Blueprint | National/global scale, DuckDB/GeoPackage offline caching, 0-36h hackathon timeline | M5 | DONE | ORIGINAL_REQUEST §R1 |
| 7 | SIH Criteria Mapping & Jury Defense Prep | 12 deep technical defense Q&A with rigorous quantitative model answers | M6 | DONE | ORIGINAL_REQUEST §R2 |

## Milestones
| # | Name | Scope | Dependencies | Status |
|---|------|-------|-------------|--------|
| M1 | Survey & Technical Specifications | Deep-dive research across all 4 data sources, mathematical models, and system pipelines | None | DONE |
| M2 | Master Architecture Document Authoring | Full synthesis of sections 1-7 into `c:/SIH/architecture.md` | M1 | DONE |
| M3 | Review & Challenger Verification | 2 Reviewers + 2 Challengers checking technical rigor, Mermaid validity, and completeness | M2 | DONE |
| M4 | Forensic Audit & Gate Clearance | Forensic audit verifying authenticity, zero hardcoding, genuine derivations | M3 | DONE |

## Interface Contracts & Layout
- Target deliverable: `c:/SIH/architecture.md` (1,533 lines, 110 KB, verified)
- Verification test suite: `c:/SIH/ntro_thermal_detection/tests/verify_math_models.py`
- Working agent metadata directories: `c:/SIH/.agents/`
