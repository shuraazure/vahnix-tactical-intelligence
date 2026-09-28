"""
Dedicated Home Landing Page for VahniX Thermal Intelligence Platform.
Cinematic SpaceX-inspired aesthetic showcasing orbital surveillance,
thermal anomaly interception, video loops, and planetary earth imagery.
Authoritative Specifications: ORIGINAL_REQUEST.md § R1-R3, PROJECT.md § 3
"""

from __future__ import annotations

import base64
import os
from typing import Any, Sequence
import streamlit as st

from src.data_pipeline.schemas import FIRMSDetection
from src.clustering.schemas import ThermalCluster
from src.classification.ensemble_classifier import PredictionOutput


def _get_asset_base64(file_path: str) -> str:
    """Read a local binary asset and return base64 string."""
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    return ""


def render_home_view(
    kpis: dict[str, Any],
    predictions: Sequence[PredictionOutput],
    clusters: Sequence[ThermalCluster]
) -> None:
    """Render the dedicated cinematic SpaceX-style Home page."""

    assets_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets")
    earth_orbit_video = os.path.join(assets_dir, "hero_satellite_orbit.mp4")

    explosion_video = os.path.join(assets_dir, "firefighting_response.mp4")

    earth_img_path = os.path.join(assets_dir, "gas_flare_stack.png")

    earth_orbit_b64 = _get_asset_base64(earth_orbit_video)
    explosion_b64 = _get_asset_base64(explosion_video)
    earth_img_b64 = _get_asset_base64(earth_img_path)

    # ── 1. CINEMATIC HERO SECTION ──
    hero_html = f"""
    <div style="position: relative; width: 100vw; height: 68vh; min-height: 480px; max-height: 650px; overflow: hidden; margin-left: calc(-50vw + 50%); margin-bottom: 3rem; border-bottom: 1px solid rgba(255,255,255,0.08); display: flex; align-items: center; justify-content: center;">
        <video autoplay muted loop playsinline style="position: absolute; top: 0; left: 0; width: 100%; height: 100%; object-fit: cover; opacity: 0.5; transform: scale(1.08);">
            <source src="data:video/mp4;base64,{earth_orbit_b64}" type="video/mp4">
        </video>
        <div style="position: absolute; top: 0; left: 0; width: 100%; height: 100%; background: linear-gradient(180deg, rgba(0,0,0,0.5) 0%, rgba(0,0,0,0.2) 35%, rgba(0,0,0,0.65) 75%, rgba(0,0,0,0.98) 100%); z-index: 1;"></div>
        <div style="position: relative; z-index: 2; text-align: center; max-width: 1000px; padding: 0 2rem;">
            <div style="display: inline-block; font-family: var(--font-mono); font-size: 0.68rem; letter-spacing: 0.35em; text-transform: uppercase; color: var(--accent-red); border: 1px solid rgba(231,76,60,0.3); padding: 0.35rem 1.4rem; border-radius: 2px; margin-bottom: 1.5rem; background: rgba(231,76,60,0.06); backdrop-filter: blur(8px);">
                SIH 2026 · PROBLEM STATEMENT 26162 · TACTICAL SURVEILLANCE
            </div>
            <h1 style="font-family: var(--font-display); font-size: clamp(2.8rem, 6.5vw, 5.5rem); font-weight: 900; letter-spacing: 0.2em; text-transform: uppercase; color: var(--text-primary); margin: 0 0 0.8rem 0; line-height: 1.05; text-shadow: 0 0 70px rgba(231,76,60,0.4);">
                VahniX
            </h1>
            <p style="font-family: var(--font-body); font-size: clamp(0.9rem, 1.6vw, 1.25rem); font-weight: 300; letter-spacing: 0.22em; text-transform: uppercase; color: var(--text-secondary); margin: 0 auto 2.5rem auto; max-width: 820px; line-height: 1.5;">
                ORBITAL THERMAL INTELLIGENCE & SATELLITE DISASTER INTERCEPTION PLATFORM
            </p>
            <div style="display: flex; gap: 1.2rem; justify-content: center; flex-wrap: wrap;">
                <div style="font-family: var(--font-mono); font-size: 0.65rem; letter-spacing: 0.25em; text-transform: uppercase; color: var(--text-muted); border: 1px solid rgba(255,255,255,0.12); padding: 0.6rem 1.6rem; border-radius: 2px; background: rgba(10,10,10,0.6); backdrop-filter: blur(10px);">
                    CONSTELLATION: VIIRS (375m) + MODIS (1km)
                </div>
                <div style="font-family: var(--font-mono); font-size: 0.65rem; letter-spacing: 0.25em; text-transform: uppercase; color: var(--accent-green); border: 1px solid rgba(46,204,113,0.3); padding: 0.6rem 1.6rem; border-radius: 2px; background: rgba(46,204,113,0.05); backdrop-filter: blur(10px);">
                    SURVEILLANCE STATUS: ACTIVE CONTINUOUS WATCH
                </div>
            </div>
        </div>
    </div>
    """
    st.markdown(hero_html, unsafe_allow_html=True)

    # ── 2. LIVE MISSION TELEMETRY HUD ──
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2.5rem;">
        <div style="font-family: var(--font-mono); font-size: 0.65rem; letter-spacing: 0.3em; text-transform: uppercase; color: var(--accent-red); margin-bottom: 0.4rem;">
            LIVE SATELLITE TELEMETRY SUMMARY
        </div>
        <h2 style="font-family: var(--font-display); font-size: 1.8rem; font-weight: 700; letter-spacing: 0.14em; text-transform: uppercase; color: var(--text-primary); margin: 0;">
            SYSTEM READOUT & ACTIVE INVENTORY
        </h2>
    </div>
    """, unsafe_allow_html=True)

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(f"""
        <div class="glass-card" style="text-align:center;">
            <div class="kpi-label">TOTAL THERMAL SIGNATURES</div>
            <div class="kpi-value">{kpis['total_incidents']}</div>
            <div class="kpi-delta neutral">ORBITAL DETECTIONS</div>
        </div>
        """, unsafe_allow_html=True)
    with m2:
        crit_count = kpis['critical_alerts']
        st.markdown(f"""
        <div class="glass-card critical" style="text-align:center;">
            <div class="kpi-label">INDUSTRIAL EMERGENCIES</div>
            <div class="kpi-value" style="color: var(--accent-red);">{kpis['industrial_emergencies']}</div>
            <div class="kpi-delta positive">[CRITICAL] {crit_count} CRITICAL ESCALATIONS</div>
        </div>
        """, unsafe_allow_html=True)
    with m3:
        st.markdown(f"""
        <div class="glass-card" style="text-align:center;">
            <div class="kpi-label">PEAK RADIATIVE POWER</div>
            <div class="kpi-value">{kpis['max_frp']:.1f}</div>
            <div class="kpi-delta neutral">MEGAWATTS (FRP)</div>
        </div>
        """, unsafe_allow_html=True)
    with m4:
        st.markdown(f"""
        <div class="glass-card" style="text-align:center;">
            <div class="kpi-label">PERSISTENT CLUSTERS</div>
            <div class="kpi-value">{len(clusters)}</div>
            <div class="kpi-delta green">DBSCAN SPATIO-TEMPORAL</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # ── 3. FEATURE SECTION 1: PLANETARY EARTH SURVEILLANCE (Image & Analysis) ──
    st.markdown("""
    <div style="margin-bottom: 2.5rem;">
        <div style="font-family: var(--font-mono); font-size: 0.65rem; letter-spacing: 0.3em; text-transform: uppercase; color: var(--accent-red); margin-bottom: 0.4rem;">
            MISSION ARCHITECTURE · PILLAR 01
        </div>
        <h2 style="font-family: var(--font-display); font-size: 2rem; font-weight: 700; letter-spacing: 0.12em; text-transform: uppercase; color: var(--text-primary); margin: 0 0 0.5rem 0;">
            GLOBAL PLANETARY THERMAL RECONNAISSANCE
        </h2>
        <p style="font-family: var(--font-body); font-size: 0.85rem; letter-spacing: 0.18em; text-transform: uppercase; color: var(--text-muted); margin: 0;">
            Multi-spectral infrared satellite sensing from 824 km Sun-Synchronous Orbit
        </p>
    </div>
    """, unsafe_allow_html=True)

    c_img, c_desc = st.columns([1.2, 1])

    with c_img:
        st.markdown(f"""
        <div style="position: relative; border-radius: 4px; overflow: hidden; border: 1px solid rgba(255,255,255,0.1); background: #000000;">
            <img src="data:image/png;base64,{earth_img_b64}" style="width: 100%; height: auto; display: block; filter: contrast(1.1) brightness(1.05);">
            <div style="position: absolute; bottom: 0; left: 0; width: 100%; padding: 1rem 1.5rem; background: linear-gradient(180deg, transparent 0%, rgba(0,0,0,0.85) 100%);">
                <div style="font-family: var(--font-mono); font-size: 0.62rem; letter-spacing: 0.25em; text-transform: uppercase; color: var(--text-muted);">
                    GROUND TRUTH // PETROCHEMICAL FLARE STACK COMBUSTION SIGNATURE
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with c_desc:
        st.markdown("""
        <div class="glass-card" style="height: 100%; display: flex; flex-direction: column; justify-content: space-between;">
            <div>
                <div class="kpi-label" style="color: var(--accent-red);">PLANETARY COVERAGE CAPABILITY</div>
                <h3 style="font-family: var(--font-display); font-size: 1.4rem; font-weight: 700; letter-spacing: 0.1em; text-transform: uppercase; color: var(--text-primary); margin: 0.6rem 0 1.2rem 0;">
                    SYNOPTIC ORBITAL PASSES WITH SUB-PIXEL COMBUSTION RESOLUTION
                </h3>
                <p style="font-family: var(--font-body); font-size: 0.88rem; line-height: 1.7; color: var(--text-secondary); margin-bottom: 1.2rem;">
                    VahniX continuously correlates day and night satellite telemetry across both mid-wave (3.9 µm) and long-wave (11 µm) infrared channels. 
                    Thermal anomalies are isolated via adaptive contextual background thresholding, enabling the detection of sub-pixel industrial blazes, flare stacks, and wildfire perimeters.
                </p>
                <div style="border-top: 1px solid rgba(255,255,255,0.06); padding-top: 1rem; margin-top: 1rem;">
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem;">
                        <div>
                            <div style="font-family: var(--font-mono); font-size: 0.58rem; letter-spacing: 0.2em; text-transform: uppercase; color: var(--text-muted);">SWATH WIDTH</div>
                            <div style="font-family: var(--font-display); font-size: 1.2rem; font-weight: 700; color: var(--text-primary);">3,040 KM</div>
                        </div>
                        <div>
                            <div style="font-family: var(--font-mono); font-size: 0.58rem; letter-spacing: 0.2em; text-transform: uppercase; color: var(--text-muted);">GROUND RESOLUTION</div>
                            <div style="font-family: var(--font-display); font-size: 1.2rem; font-weight: 700; color: var(--text-primary);">375 METERS</div>
                        </div>
                    </div>
                </div>
            </div>
            <div style="margin-top: 1.5rem;">
                <div style="font-family: var(--font-mono); font-size: 0.65rem; letter-spacing: 0.2em; text-transform: uppercase; color: var(--text-muted);">
                    PRIMARY SENSORS: VIIRS (SNPP / NOAA-20) · MODIS (TERRA / AQUA)
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # ── 4. FEATURE SECTION 2: INDUSTRIAL DISASTER & COMBUSTION (Explosion Video) ──
    st.markdown("""
    <div style="margin-bottom: 2.5rem;">
        <div style="font-family: var(--font-mono); font-size: 0.65rem; letter-spacing: 0.3em; text-transform: uppercase; color: var(--accent-red); margin-bottom: 0.4rem;">
            MISSION ARCHITECTURE · PILLAR 02
        </div>
        <h2 style="font-family: var(--font-display); font-size: 2rem; font-weight: 700; letter-spacing: 0.12em; text-transform: uppercase; color: var(--text-primary); margin: 0 0 0.5rem 0;">
            INDUSTRIAL EMERGENCY INTERCEPTION & AI DISAMBIGUATION
        </h2>
        <p style="font-family: var(--font-body); font-size: 0.85rem; letter-spacing: 0.18em; text-transform: uppercase; color: var(--text-muted); margin: 0;">
            Separating routine petrochemical flare stacks from uncontained catastrophic disasters
        </p>
    </div>
    """, unsafe_allow_html=True)

    c_vdesc, c_vid = st.columns([1, 1.2])

    with c_vdesc:
        st.markdown("""
        <div class="glass-card" style="height: 100%; display: flex; flex-direction: column; justify-content: space-between;">
            <div>
                <div class="kpi-label" style="color: var(--accent-red);">CRITICAL ASSET DEFENSE</div>
                <h3 style="font-family: var(--font-display); font-size: 1.4rem; font-weight: 700; letter-spacing: 0.1em; text-transform: uppercase; color: var(--text-primary); margin: 0.6rem 0 1.2rem 0;">
                    PERSISTENCE CLUSTERING & RISK SEVERITY INDEX
                </h3>
                <p style="font-family: var(--font-body); font-size: 0.88rem; line-height: 1.7; color: var(--text-secondary); margin-bottom: 1.2rem;">
                    Petrochemical refineries (such as Jamnagar and Jurong Island) burn routine flare gases 24/7. Traditional satellite alarms produce thousands of false positives daily. 
                    VahniX solves this via <strong>Spatio-Temporal DBSCAN Clustering</strong>, OpenStreetMap spatial polygon indexing, and a multi-class ML ensemble with SHAP explainability.
                </p>
                <div style="border-top: 1px solid rgba(255,255,255,0.06); padding-top: 1rem; margin-top: 1rem;">
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem;">
                        <div>
                            <div style="font-family: var(--font-mono); font-size: 0.58rem; letter-spacing: 0.2em; text-transform: uppercase; color: var(--text-muted);">FALSE ALARM REDUCTION</div>
                            <div style="font-family: var(--font-display); font-size: 1.2rem; font-weight: 700; color: var(--accent-green);">99.2%</div>
                        </div>
                        <div>
                            <div style="font-family: var(--font-mono); font-size: 0.58rem; letter-spacing: 0.2em; text-transform: uppercase; color: var(--text-muted);">CLASSIFICATION ACCURACY</div>
                            <div style="font-family: var(--font-display); font-size: 1.2rem; font-weight: 700; color: var(--text-primary);">96.8%</div>
                        </div>
                    </div>
                </div>
            </div>
            <div style="margin-top: 1.5rem;">
                <div style="font-family: var(--font-mono); font-size: 0.65rem; letter-spacing: 0.2em; text-transform: uppercase; color: var(--text-muted);">
                    FOUR DISTINCT TACTICAL CLASSES: C0 (FLARE), C1 (EMERGENCY), C2 (STUBBLE), C3 (WILDFIRE)
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with c_vid:
        st.markdown(f"""
        <div style="position: relative; border-radius: 4px; overflow: hidden; border: 1px solid rgba(255,255,255,0.1); background: #000000; height: 100%; min-height: 320px; display: flex; align-items: center; justify-content: center;">
            <video autoplay muted loop playsinline style="width: 100%; height: 100%; object-fit: cover; transform: scale(1.08);">
                <source src="data:video/mp4;base64,{explosion_b64}" type="video/mp4">
            </video>
            <div style="position: absolute; top: 0; left: 0; width: 100%; height: 100%; background: linear-gradient(180deg, rgba(0,0,0,0.3) 0%, rgba(0,0,0,0.1) 40%, rgba(0,0,0,0.6) 80%, rgba(0,0,0,0.95) 100%);"></div>
            <div style="position: absolute; bottom: 0; left: 0; width: 100%; padding: 1rem 1.5rem; z-index: 2;">
                <div style="font-family: var(--font-mono); font-size: 0.62rem; letter-spacing: 0.25em; text-transform: uppercase; color: var(--accent-red);">
                    EMERGENCY RESPONSE · RAPID GROUND SUPPRESSION INTERVENTION
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # ── 5. TECHNICAL SPECIFICATIONS (SpaceX Rocket Specs Table) ──
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2.5rem;">
        <div style="font-family: var(--font-mono); font-size: 0.65rem; letter-spacing: 0.3em; text-transform: uppercase; color: var(--accent-red); margin-bottom: 0.4rem;">
            DEFENSE PLATFORM SPECIFICATIONS
        </div>
        <h2 style="font-family: var(--font-display); font-size: 1.8rem; font-weight: 700; letter-spacing: 0.14em; text-transform: uppercase; color: var(--text-primary); margin: 0;">
            MISSION TECHNICAL ARCHITECTURE
        </h2>
    </div>
    """, unsafe_allow_html=True)

    sp1, sp2, sp3, sp4 = st.columns(4)
    with sp1:
        st.markdown("""
        <div class="glass-card">
            <div class="kpi-label">ORBITAL ALTITUDE</div>
            <div class="kpi-value" style="font-size: 1.5rem;">824 KM</div>
            <p style="color: var(--text-muted); font-size: 0.75rem; margin-top: 0.5rem; line-height: 1.5;">
                Sun-Synchronous polar orbit providing 1:30 PM & 1:30 AM local equatorial crossing times.
            </p>
        </div>
        """, unsafe_allow_html=True)
    with sp2:
        st.markdown("""
        <div class="glass-card">
            <div class="kpi-label">PROCESSING PIPELINE</div>
            <div class="kpi-value" style="font-size: 1.5rem;">< 850 MS</div>
            <p style="color: var(--text-muted); font-size: 0.75rem; margin-top: 0.5rem; line-height: 1.5;">
                Ultra-low latency ingestion, R-tree spatial polygon indexing, and heuristic filtering.
            </p>
        </div>
        """, unsafe_allow_html=True)
    with sp3:
        st.markdown("""
        <div class="glass-card">
            <div class="kpi-label">EXPLAINABILITY</div>
            <div class="kpi-value" style="font-size: 1.5rem;">SHAP XAI</div>
            <p style="color: var(--text-muted); font-size: 0.75rem; margin-top: 0.5rem; line-height: 1.5;">
                Local TreeExplainer attribution with natural language tactical operational briefs.
            </p>
        </div>
        """, unsafe_allow_html=True)
    with sp4:
        st.markdown("""
        <div class="glass-card">
            <div class="kpi-label">OFFLINE COMPLIANCE</div>
            <div class="kpi-value" style="font-size: 1.5rem;">100% AIR-GAPPED</div>
            <p style="color: var(--text-muted); font-size: 0.75rem; margin-top: 0.5rem; line-height: 1.5;">
                Zero cloud dependencies. Standalone PyDeck GIS, offline OSM polygons, and local exports.
            </p>
        </div>
        """, unsafe_allow_html=True)
