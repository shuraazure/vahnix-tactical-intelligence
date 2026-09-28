"""
VahniX Tactical Thermal Anomaly Detection & AI Classification Dashboard.
SpaceX-inspired cinematic dark UI/UX redesign.
Standalone offline Streamlit web application.
Authoritative Specifications: ORIGINAL_REQUEST.md § R3, PROJECT.md § 3, TEST_INFRA.md
"""

from __future__ import annotations

import base64
import os
import sys
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import streamlit as st

from src.dashboard.state_manager import DashboardStateManager, DashboardFilterState
from src.dashboard.views.overview import render_overview_view
from src.dashboard.views.clusters_view import render_clusters_view
from src.dashboard.views.classifier_view import render_classifier_view
from src.dashboard.views.explainability_view import render_explainability_view
from src.dashboard.views.reporting_view import render_reporting_view
from src.dashboard.views.nix_chatbot_view import render_nix_chatbot_view
from src.dashboard.views.home_view import render_home_view
from src.dashboard.views.detailed_analysis_view import render_detailed_analysis_view
from src.dashboard.views.live_stream_view import render_live_stream_view
from src.dashboard.views.globe_view import render_globe_view
from src.dashboard.views.technology_view import render_technology_view


# Browser tab icon — the VahniX X-mark. Falls back to a clean tactical dot if the
# asset is missing, ensuring the app always starts without emoji dependency.
_FAVICON_PATH = os.path.join(os.path.dirname(__file__), "assets", "favicon_vahnix.png")
try:
    from PIL import Image as _PILImage
    if os.path.exists(_FAVICON_PATH):
        _PAGE_ICON = _PILImage.open(_FAVICON_PATH)
    else:
        _PAGE_ICON = _PILImage.new("RGBA", (32, 32), (56, 189, 248, 255))
except Exception:
    _PAGE_ICON = None

# Configure Streamlit page
st.set_page_config(
    page_title="VahniX · Thermal Intelligence Platform",
    page_icon=_PAGE_ICON,
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ---------------------------------------------------------------------------
#  SPACEX-INSPIRED GLOBAL THEME CSS
# ---------------------------------------------------------------------------

SPACEX_CSS = """
<style>
/* ===== IMPORTS ===== */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&display=swap');
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600&display=swap');

/* ===== CSS VARIABLES ===== */
:root {
    --bg-primary: #000000;
    --bg-secondary: #0a0a0a;
    --bg-card: rgba(255, 255, 255, 0.03);
    --bg-card-hover: rgba(255, 255, 255, 0.06);
    --glass-bg: rgba(10, 10, 10, 0.6);
    --glass-border: rgba(255, 255, 255, 0.08);
    --glass-glow: rgba(255, 255, 255, 0.02);
    --text-primary: rgba(240, 240, 250, 1);
    --text-secondary: rgba(240, 240, 250, 0.6);
    --text-muted: rgba(240, 240, 250, 0.35);
    --accent-red: #e74c3c;
    --accent-orange: #e67e22;
    --accent-blue: #3498db;
    --accent-green: #2ecc71;
    --border-subtle: rgba(255, 255, 255, 0.06);
    --border-ghost: rgba(255, 255, 255, 0.2);
    --font-display: 'Space Grotesk', 'Inter', 'D-DIN', Arial, sans-serif;
    --font-body: 'Inter', 'D-DIN', Arial, sans-serif;
    --font-mono: 'JetBrains Mono', 'Courier New', monospace;
}

/* ===== RESET & GLOBAL ===== */
html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"],
.main, .block-container, [data-testid="stMainBlockContainer"] {
    background-color: var(--bg-primary) !important;
    color: var(--text-primary) !important;
    font-family: var(--font-body) !important;
}

/* Hide Streamlit default chrome */
#MainMenu, footer, [data-testid="stToolbar"],
header[data-testid="stHeader"],
[data-testid="stSidebar"],
[data-testid="collapsedControl"] {
    display: none !important;
    visibility: hidden !important;
    height: 0 !important;
}

/* ===== SITE-WIDE PADDING & LAYOUT ===== */
.block-container, [data-testid="stMainBlockContainer"] {
    padding-top: 0 !important;
    padding-bottom: 3.5rem !important;
    padding-left: 2.4rem !important;
    padding-right: 2.4rem !important;
    max-width: 100% !important;
    box-sizing: border-box !important;
}

/* Prevent unwanted horizontal scrollbars from full-bleed elements */
html, body, [data-testid="stAppViewContainer"], .main {
    overflow-x: hidden !important;
}

/* ===== TOP BRAND BAR ===== */
/* Fixed header shell — hides on scroll down, slides back on scroll up */
.nav-shell {
    position: fixed;
    top: 0;
    left: 0;
    width: 100vw;
    z-index: 10000;
    background: #000000;
    transform: translateY(0);
    transition: transform 0.35s cubic-bezier(0.4, 0, 0.2, 1),
                box-shadow 0.35s ease;
}

.nav-shell.nav-hidden {
    transform: translateY(-105%);
}

.nav-shell.nav-floating {
    box-shadow: 0 6px 24px rgba(0, 0, 0, 0.65);
    border-bottom: 1px solid rgba(255, 255, 255, 0.07);
}

.nav-shell .top-status-row,
.nav-shell .top-brand-bar {
    width: 100%;
    margin-left: 0;
}

/* Reserves the space the fixed shell no longer occupies in the flow */
.nav-shell-spacer {
    height: 86px;
}

.top-status-row {
    display: flex;
    align-items: center;
    justify-content: flex-end;
    background: #000000;
    padding: 0.4rem 2.4rem;
    width: 100vw;
    margin-left: calc(-50vw + 50%);
    box-sizing: border-box;
}

.top-brand-bar {
    display: flex;
    align-items: center;
    justify-content: flex-start;
    background: #000000;
    padding: 0.9rem 2.4rem;
    width: 100vw;
    margin-left: calc(-50vw + 50%);
    box-sizing: border-box;
}

.top-brand-logo {
    font-family: var(--font-display) !important;
    font-size: 1.15rem;
    font-weight: 800;
    letter-spacing: 0.22em;
    color: var(--text-primary);
    display: flex;
    align-items: center;
    gap: 0.5rem;
    text-transform: uppercase;
    flex-shrink: 0;
    text-decoration: none !important;
    margin-left: 2.1rem;
    margin-right: 1.3rem;
}

.logo-flame {
    color: var(--accent-red);
    font-size: 1.1rem;
}

.brand-logo-img {
    height: 60px;
    width: auto;
    display: block;
    cursor: pointer;
    transition: opacity 0.15s ease;
}

.brand-logo-img:hover {
    opacity: 0.8;
}

.top-brand-bar .nav-item {
    font-family: var(--font-display) !important;
    font-size: 0.78rem;
    font-weight: 400;
    letter-spacing: 0.03em;
    text-transform: uppercase;
    color: var(--text-primary);
    cursor: pointer;
    white-space: nowrap;
    padding: 0.35rem 0;
    border-bottom: 2px solid transparent;
    transition: all 0.2s ease;
    text-decoration: none !important;
    margin-right: 2.1rem;
}

.top-brand-bar .nav-item:hover {
    color: var(--text-primary);
}

.top-brand-bar .nav-item.active {
    color: var(--text-primary);
    border-bottom: 2px solid var(--accent-red);
}

.top-brand-bar .nav-item.nav-item-tech {
    margin-left: auto !important;
    margin-right: 0 !important;
    color: #38bdf8 !important;
    font-weight: 600 !important;
    letter-spacing: 0.06em !important;
    padding: 0.35rem 0.85rem !important;
    border: 1px solid rgba(56, 189, 248, 0.4) !important;
    border-bottom: 2px solid #38bdf8 !important;
    border-radius: 4px !important;
    background: rgba(56, 189, 248, 0.08) !important;
    text-shadow: 0 0 10px rgba(56, 189, 248, 0.35) !important;
    transition: all 0.2s ease !important;
}

.top-brand-bar .nav-item.nav-item-tech:hover {
    color: #ffffff !important;
    background: rgba(56, 189, 248, 0.22) !important;
    border-color: rgba(56, 189, 248, 0.7) !important;
    border-bottom: 2px solid #38bdf8 !important;
    box-shadow: 0 0 14px rgba(56, 189, 248, 0.45) !important;
}

.top-brand-bar .nav-item.nav-item-tech.active {
    color: #38bdf8 !important;
    background: rgba(56, 189, 248, 0.25) !important;
    border: 1px solid #38bdf8 !important;
    border-bottom: 2px solid #38bdf8 !important;
    box-shadow: 0 0 18px rgba(56, 189, 248, 0.6) !important;
}

.top-brand-logo .sub-text {
    font-size: 0.75rem;
    font-weight: 400;
    letter-spacing: 0.18em;
    color: var(--text-secondary);
}

.top-brand-status {
    font-family: var(--font-mono) !important;
    font-size: 0.58rem;
    letter-spacing: 0.12em;
    color: var(--text-muted);
    display: flex;
    align-items: center;
    gap: 0.5rem;
    text-transform: uppercase;
    white-space: nowrap;
    flex-shrink: 0;
}

.status-pulse-dot {
    width: 7px;
    height: 7px;
    background: var(--accent-green);
    border-radius: 50%;
    box-shadow: 0 0 8px var(--accent-green);
    animation: status-pulse 2s infinite;
}

@keyframes status-pulse {
    0%, 100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.35; transform: scale(0.85); }
}

/* ===== INTRO TEXT SECTION ===== */
.text-intro-section {
    padding: 4.5rem 3rem;
    background: #000000;
}

.text-intro-tag {
    font-family: var(--font-mono) !important;
    font-size: 0.75rem;
    letter-spacing: 0.25em;
    text-transform: uppercase;
    color: var(--text-muted);
    margin-bottom: 1.5rem;
}

.text-intro-title {
    font-family: var(--font-display) !important;
    font-weight: 700;
    font-size: clamp(2.1rem, 5vw, 4.1rem);
    line-height: 1.15;
    margin: 0;
}

.text-intro-title .line-white {
    color: var(--text-primary);
    display: block;
}

.text-intro-title .line-grey {
    color: var(--text-muted);
    display: block;
    font-family: var(--font-body) !important;
    font-size: clamp(1rem, 1.5vw, 1.35rem);
    font-weight: 400;
    line-height: 1.6;
    max-width: 46rem;
    margin-top: 2rem;
}

.text-intro-typed {
    margin-top: 1.8rem;
}

.text-intro-typed .typed-line {
    display: block;
    font-family: var(--font-body) !important;
    font-size: clamp(1.45rem, 2.25vw, 2.15rem);
    font-weight: 400;
    line-height: 1.5;
    color: var(--text-muted);
    white-space: nowrap;
    overflow: hidden;
    width: 0;
    border-right: 2px solid var(--text-muted);
    animation-fill-mode: forwards;
    animation-timing-function: steps(60, end);
}

.text-intro-typed .typed-line-2 {
    border-right-color: transparent;
}

@keyframes intro-caret {
    0% { border-right-color: var(--text-muted); }
    50% { border-right-color: transparent; }
    99% { border-right-color: var(--text-muted); }
    100% { border-right-color: transparent; }
}

/* ===== HERO BANNER SECTION ===== */
.hero-container {
    position: relative;
    width: 100vw;
    height: 80vh;
    overflow: hidden;
    display: flex;
    align-items: center;
    justify-content: center;
    margin-left: calc(-50vw + 50%);
    margin-bottom: 2rem;
    border-bottom: 1px solid var(--border-subtle);
}

.hero-video {
    position: absolute;
    top: 0; left: 0;
    width: 100%; height: 100%;
    object-fit: cover;
    z-index: 0;
    opacity: 0.95;
    filter: brightness(1.15);
}

.hero-overlay {
    position: absolute;
    top: 0; left: 0;
    width: 100%; height: 100%;
    background: linear-gradient(
        180deg,
        rgba(0,0,0,0.2) 0%,
        rgba(0,0,0,0.08) 40%,
        rgba(0,0,0,0.35) 80%,
        rgba(0,0,0,0.75) 100%
    );
    z-index: 1;
}

/* Top-right hero link out to the standalone tech page */
.hero-tech-link {
    position: absolute;
    top: 1.8rem;
    right: 2.2rem;
    z-index: 5;
    display: inline-block;
    font-family: var(--font-mono) !important;
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 0.22em;
    text-transform: uppercase;
    text-decoration: none !important;
    color: var(--text-primary) !important;
    padding: 0.6rem 1.5rem;
    border: 1px solid rgba(245, 247, 250, 0.35);
    border-radius: 999px;
    background: rgba(10, 12, 18, 0.55);
    backdrop-filter: blur(6px);
    transition: background 0.25s ease, border-color 0.25s ease,
                box-shadow 0.25s ease;
}

.hero-tech-link:hover {
    background: rgba(231, 76, 60, 0.9);
    border-color: rgba(231, 76, 60, 1);
    box-shadow: 0 0 22px rgba(231, 76, 60, 0.4);
}

@media (max-width: 640px) {
    .hero-tech-link {
        top: 1rem;
        right: 1rem;
        font-size: 0.6rem;
        padding: 0.5rem 1.1rem;
    }
}

.hero-content {
    position: relative;
    z-index: 2;
    text-align: center;
    padding: 0 2rem;
}

.hero-title {
    font-family: var(--font-display) !important;
    font-size: clamp(4rem, 11vw, 9rem);
    font-weight: 800;
    letter-spacing: 0.03em;
    text-transform: uppercase;
    color: var(--text-primary);
    margin-bottom: 0.5rem;
    line-height: 1.1;
    text-shadow: 0 0 60px rgba(231, 76, 60, 0.35);
}

/* Hero wordmark — transparent PNG logo sitting on the cosmic video */
.hero-logo {
    display: block;
    width: min(54vw, 700px);
    max-width: 100%;
    height: auto;
    margin: 0 auto 0.7rem;
    filter: drop-shadow(0 0 70px rgba(231, 76, 60, 0.38))
            drop-shadow(0 0 26px rgba(0, 0, 0, 0.65));
    animation: hero-logo-in 1.5s cubic-bezier(0.16, 1, 0.3, 1) both;
}

@keyframes hero-logo-in {
    from { opacity: 0; transform: translateY(22px) scale(0.965); }
    to   { opacity: 1; transform: translateY(0) scale(1); }
}

@media (max-width: 640px) {
    .hero-logo { width: 78vw; }
}

.hero-subtitle {
    font-family: var(--font-body) !important;
    font-size: clamp(0.7rem, 1.2vw, 0.95rem);
    font-weight: 500;
    letter-spacing: 0.22em;
    text-transform: uppercase;
    color: var(--text-secondary);
    margin-bottom: 1.5rem;
}

/* Hero call-to-action — real link, uses the same ?page= navigation as the nav bar */
.hero-cta {
    display: inline-block;
    font-family: var(--font-mono) !important;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.26em;
    text-transform: uppercase;
    text-decoration: none !important;
    color: var(--text-primary) !important;
    padding: 0.85rem 2.6rem;
    border: 1px solid rgba(231, 76, 60, 0.85);
    border-radius: 2px;
    background: rgba(231, 76, 60, 0.38);
    backdrop-filter: blur(4px);
    transition: background 0.25s ease, border-color 0.25s ease,
                box-shadow 0.25s ease, transform 0.25s ease;
}

.hero-cta:hover {
    background: rgba(231, 76, 60, 1);
    border-color: rgba(231, 76, 60, 1);
    box-shadow: 0 0 28px rgba(231, 76, 60, 0.45);
    transform: translateY(-2px);
}

.hero-badge {
    display: inline-block;
    font-family: var(--font-mono) !important;
    font-size: 0.65rem;
    letter-spacing: 0.22em;
    text-transform: uppercase;
    color: var(--accent-red);
    border: 1px solid rgba(231, 76, 60, 0.3);
    padding: 0.3rem 1.2rem;
    border-radius: 2px;
    margin-bottom: 1.2rem;
    background: rgba(231, 76, 60, 0.05);
    /* nudged down; relative so the logo below stays where it is */
    position: relative;
    top: 4.2rem;
}

/* ===== GLASSMORPHISM CARDS ===== */
.glass-card {
    background: var(--bg-card);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border: 1px solid var(--glass-border);
    border-radius: 4px;
    padding: 1.5rem;
    transition: all 0.3s ease;
}

.glass-card:hover {
    background: var(--bg-card-hover);
    border-color: rgba(255, 255, 255, 0.12);
}

.glass-card.critical {
    border-color: rgba(231, 76, 60, 0.3);
    box-shadow: 0 0 20px rgba(231, 76, 60, 0.08);
    animation: critical-pulse 3s infinite;
}

@keyframes critical-pulse {
    0%, 100% { box-shadow: 0 0 20px rgba(231, 76, 60, 0.08); }
    50% { box-shadow: 0 0 35px rgba(231, 76, 60, 0.15); }
}

.kpi-label {
    font-family: var(--font-mono) !important;
    font-size: 0.6rem;
    font-weight: 500;
    letter-spacing: 0.25em;
    text-transform: uppercase;
    color: var(--text-muted);
    margin-bottom: 0.5rem;
}

.kpi-value {
    font-family: var(--font-display) !important;
    font-size: 2.2rem;
    font-weight: 700;
    color: var(--text-primary);
    line-height: 1;
    margin-bottom: 0.3rem;
}

.kpi-delta {
    font-family: var(--font-mono) !important;
    font-size: 0.65rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
}

.kpi-delta.positive { color: var(--accent-red); }
.kpi-delta.neutral { color: var(--text-muted); }
.kpi-delta.green { color: var(--accent-green); }

/* ===== SECTION HEADERS ===== */
.section-header {
    font-family: var(--font-display) !important;
    font-size: 1.5rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--text-primary);
    margin: 2.5rem 0 0.4rem 0;
}

.section-subtitle {
    font-family: var(--font-body) !important;
    font-size: 0.8rem;
    font-weight: 300;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: var(--text-muted);
    margin-bottom: 1.8rem;
}

.section-divider {
    border: none;
    height: 1px;
    background: linear-gradient(90deg, transparent, var(--border-subtle), transparent);
    margin: 2.2rem 0;
}

/* ===== STREAMLIT ELEMENTS OVERRIDES ===== */

/* Metric cards */
[data-testid="stMetric"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--glass-border) !important;
    border-radius: 4px !important;
    padding: 1.2rem !important;
    backdrop-filter: blur(10px) !important;
}

[data-testid="stMetricLabel"] {
    font-family: var(--font-mono) !important;
    font-size: 0.6rem !important;
    letter-spacing: 0.2em !important;
    text-transform: uppercase !important;
    color: var(--text-muted) !important;
}

[data-testid="stMetricValue"] {
    font-family: var(--font-display) !important;
    font-size: 1.8rem !important;
    font-weight: 700 !important;
    color: var(--text-primary) !important;
}

/* DataFrames */
[data-testid="stDataFrame"],
.stDataFrame {
    border: 1px solid var(--glass-border) !important;
    border-radius: 4px !important;
}

/* Form Controls */
[data-baseweb="select"] {
    background: var(--bg-card) !important;
    border-color: var(--glass-border) !important;
}

[data-baseweb="select"] > div {
    background: var(--bg-card) !important;
    color: var(--text-primary) !important;
}

/* Buttons */
[data-testid="stButton"] button {
    font-family: var(--font-display) !important;
    font-size: 0.7rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.18em !important;
    text-transform: uppercase !important;
    color: var(--text-primary) !important;
    background: transparent !important;
    border: 1px solid var(--border-ghost) !important;
    border-radius: 2px !important;
    padding: 0.6rem 1.5rem !important;
    transition: all 0.3s ease !important;
}

[data-testid="stButton"] button:hover {
    background: rgba(255, 255, 255, 0.08) !important;
    border-color: rgba(255, 255, 255, 0.5) !important;
    color: var(--text-primary) !important;
    box-shadow: 0 0 20px rgba(255, 255, 255, 0.05) !important;
}

/* Download buttons */
[data-testid="stDownloadButton"] button {
    font-family: var(--font-display) !important;
    font-size: 0.7rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.18em !important;
    text-transform: uppercase !important;
    color: var(--text-primary) !important;
    background: transparent !important;
    border: 1px solid var(--border-ghost) !important;
    border-radius: 2px !important;
    transition: all 0.3s ease !important;
}

[data-testid="stDownloadButton"] button:hover {
    background: rgba(255, 255, 255, 0.08) !important;
    border-color: rgba(255, 255, 255, 0.5) !important;
}

/* Expander */
[data-testid="stExpander"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--glass-border) !important;
    border-radius: 4px !important;
}

[data-testid="stExpander"] summary {
    font-family: var(--font-display) !important;
    font-size: 0.75rem !important;
    letter-spacing: 0.15em !important;
    text-transform: uppercase !important;
    color: var(--text-secondary) !important;
}

/* JSON viewer */
[data-testid="stJson"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--glass-border) !important;
    border-radius: 4px !important;
}

/* Scrollbar styling */
::-webkit-scrollbar {
    width: 6px;
}

::-webkit-scrollbar-track {
    background: var(--bg-primary);
}

::-webkit-scrollbar-thumb {
    background: rgba(255, 255, 255, 0.1);
    border-radius: 3px;
}

::-webkit-scrollbar-thumb:hover {
    background: rgba(255, 255, 255, 0.2);
}

/* ===== VIDEO SECTION DIVIDERS ===== */
.video-section {
    position: relative;
    width: 100vw;
    height: 80vh;
    overflow: hidden;
    margin-left: calc(-50vw + 50%);
    display: flex;
    align-items: center;
    justify-content: center;
    margin-bottom: 2rem;
    border-bottom: 1px solid var(--border-subtle);
}

.video-section.flush {
    margin-bottom: 0;
    border-bottom: none;
}

.video-section video {
    position: absolute;
    top: 0; left: 0;
    width: 100%; height: 100%;
    object-fit: cover;
    opacity: 0.3;
}

.video-section img {
    position: absolute;
    top: 0; left: 0;
    width: 100%; height: 100%;
    object-fit: cover;
    opacity: 0.3;
}

.video-section .overlay {
    position: absolute;
    top: 0; left: 0;
    width: 100%; height: 100%;
    background: linear-gradient(
        180deg,
        rgba(0,0,0,0.8) 0%,
        rgba(0,0,0,0.3) 50%,
        rgba(0,0,0,0.8) 100%
    );
    z-index: 1;
}

.video-section .section-text {
    position: relative;
    z-index: 2;
    text-align: center;
}

.video-section .section-text h2 {
    font-family: var(--font-display) !important;
    font-size: 2.2rem;
    font-weight: 700;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--text-primary);
    margin-bottom: 0.5rem;
}

.video-section .section-text p {
    font-family: var(--font-body) !important;
    font-size: 0.85rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: var(--text-muted);
}

/* ===== CORNER STYLE (opt-in, Home page only) ===== */
.video-section.corner {
    align-items: flex-start;
    justify-content: flex-end;
    padding: 5rem 4rem;
    box-sizing: border-box;
}

.video-section.corner.align-left {
    justify-content: flex-start;
}

.video-section.corner video,
.video-section.corner img {
    opacity: 0.92;
    filter: brightness(1.1);
}

.video-section.corner .overlay {
    background: linear-gradient(
        180deg,
        rgba(0,0,0,0.45) 0%,
        rgba(0,0,0,0.1) 40%,
        rgba(0,0,0,0.35) 100%
    );
}

.video-section.corner .section-text {
    text-align: right;
    max-width: 48rem;
}

.video-section.corner.align-left .section-text {
    text-align: left;
}

.video-section.corner .section-text h2 {
    font-size: clamp(1.8rem, 3.4vw, 2.8rem);
    font-weight: 800;
    letter-spacing: 0.01em;
    line-height: 1.15;
    margin-bottom: 1rem;
}

.video-section.corner .section-text p {
    font-size: clamp(0.95rem, 1.3vw, 1.25rem);
    font-weight: 600;
    letter-spacing: 0.01em;
    text-transform: none;
    color: rgba(240, 240, 250, 0.88);
    line-height: 1.5;
    text-wrap: balance;
    text-shadow: 0 1px 12px rgba(0, 0, 0, 0.55);
}

.video-section.corner.dark-text .section-text p {
    text-shadow: 0 1px 12px rgba(255, 255, 255, 0.45);
}

.video-section.dark-text .section-text h2,
.video-section.dark-text .section-text p {
    color: #111111;
}

/* ===== FOOTER ===== */
.spacex-footer {
    text-align: center;
    padding: 3rem 0;
    margin-top: 3rem;
    border-top: 1px solid var(--border-subtle);
}

.spacex-footer p {
    font-family: var(--font-mono) !important;
    font-size: 0.62rem;
    letter-spacing: 0.28em;
    text-transform: uppercase;
    color: var(--text-muted);
}
</style>
"""


# ---------------------------------------------------------------------------
#  HELPER: Encode video to base64 for inline embedding
# ---------------------------------------------------------------------------

def get_video_base64(video_path: str) -> str:
    """Read a video file and return base64-encoded string."""
    if os.path.exists(video_path):
        with open(video_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    return ""


def render_intro_text_section(white_text: str, grey_text: str, grey_text_2: str = "", tag: str = "") -> None:
    """Render a black-background section with a white headline and typed grey line(s) below it."""
    tag_html = f'<div class="text-intro-tag">{tag}</div>' if tag else ""

    lines = [t for t in (grey_text, grey_text_2) if t]
    line_css = ""
    lines_html = ""
    delay = 0.4
    for i, text in enumerate(lines, start=1):
        # ch-based width so the steps() reveal lands on the end of the text.
        width_ch = len(text)
        duration = round(width_ch * 0.075, 2)
        caret_cycles = max(1, int(duration / 0.8))
        line_css += f"""
        @keyframes intro-type-{i} {{
            from {{ width: 0; }}
            to {{ width: {width_ch}ch; }}
        }}
        .typed-line-{i} {{
            animation: intro-type-{i} {duration}s steps({width_ch}, end) {delay}s forwards,
                       intro-caret 0.8s step-end {delay}s {caret_cycles} forwards;
        }}
        """
        lines_html += f'<span class="typed-line typed-line-{i}">{text}</span>'
        delay += duration + 0.3

    html = f"""
    <style>{line_css}</style>
    <div class="text-intro-section">
        {tag_html}
        <h2 class="text-intro-title">
            <span class="line-white">{white_text}</span>
        </h2>
        <div class="text-intro-typed">{lines_html}</div>
    </div>
    """
    # Strip blank lines so an empty tag_html doesn't create a gap that
    # markdown misreads as the start of a code block.
    html = "\n".join(line for line in html.splitlines() if line.strip())
    st.markdown(html, unsafe_allow_html=True)


def render_hero_section(video_b64: str, logo_b64: str = "") -> None:
    """Render the cinematic hero banner section with video background.

    The VahniX wordmark is shown as a transparent PNG on top of the cosmic
    video. If the logo asset is missing, it falls back to the text wordmark.
    """
    if logo_b64:
        brand_html = f'<img src="data:image/png;base64,{logo_b64}" class="hero-logo" alt="VahniX">'
    else:
        brand_html = '<h1 class="hero-title">VAHNI X</h1>'

    hero_html = f"""
    <div class="hero-container">
        <video class="hero-video" autoplay muted loop playsinline>
            <source src="data:video/mp4;base64,{video_b64}" type="video/mp4">
        </video>
        <div class="hero-overlay"></div>
        <div class="hero-content">
            <div class="hero-badge">SIH 2026 · PROBLEM STATEMENT 26162</div>
            {brand_html}
            <p class="hero-subtitle">Clarity For Every Anomaly</p>
            <a href="?page=overview" target="_self" class="hero-cta">Explore</a>
        </div>
    </div>
    """
    hero_html = "\n".join(line for line in hero_html.splitlines() if line.strip())
    st.markdown(hero_html, unsafe_allow_html=True)


def render_video_section(video_b64: str, title: str, subtitle: str, align: str = "right", flush: bool = False, corner: bool = False) -> None:
    """Render a cinematic video interstitial. Classic centered style by default; pass corner=True for the SpaceX-style corner-positioned layout."""
    classes = "video-section"
    if corner:
        classes += " corner"
        if align == "left":
            classes += " align-left"
    if flush:
        classes += " flush"
    html = f"""
    <div class="{classes}">
        <video autoplay muted loop playsinline>
            <source src="data:video/mp4;base64,{video_b64}" type="video/mp4">
        </video>
        <div class="overlay"></div>
        <div class="section-text">
            <h2>{title}</h2>
            <p>{subtitle}</p>
        </div>
    </div>
    """
    html = "\n".join(line for line in html.splitlines() if line.strip())
    st.markdown(html, unsafe_allow_html=True)


def render_image_section(image_b64: str, title: str, subtitle: str, mime: str = "image/png", align: str = "right", flush: bool = False, dark_text: bool = False, corner: bool = False) -> None:
    """Render a cinematic image interstitial. Classic centered style by default; pass corner=True for the SpaceX-style corner-positioned layout."""
    classes = "video-section"
    if corner:
        classes += " corner"
    if align == "left":
        classes += " align-left"
    if flush:
        classes += " flush"
    if dark_text:
        classes += " dark-text"
    html = f"""
    <div class="{classes}">
        <img src="data:{mime};base64,{image_b64}">
        <div class="overlay"></div>
        <div class="section-text">
            <h2>{title}</h2>
            <p>{subtitle}</p>
        </div>
    </div>
    """
    html = "\n".join(line for line in html.splitlines() if line.strip())
    st.markdown(html, unsafe_allow_html=True)


def render_footer() -> None:
    """Render the SpaceX-style minimal footer."""
    st.markdown("""
    <div class="spacex-footer">
        <p>VahniX THERMAL INTELLIGENCE PLATFORM · SIH 2026 · CLASSIFIED TACTICAL SYSTEM</p>
        <p style="margin-top: 0.5rem;">INDUSTRIAL FIRE & PERSISTENT THERMAL ANOMALY DETECTION</p>
    </div>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
#  MAIN APPLICATION
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Initializing VahniX Tactical Engine...")
def get_state_manager() -> DashboardStateManager:
    """Initialize cached state manager with offline reference datasets."""
    return DashboardStateManager()


def _render_nav_scroll_script(current_page: str = "") -> None:
    """Hide the header on downward scroll, slide it back on upward scroll.

    Runs inside a zero-height component iframe and drives the parent document,
    because Streamlit strips <script> tags out of st.markdown.
    """
    import streamlit.components.v1 as components

    components.html(
        """
        <script>
        (function () {
          const doc = window.parent.document;
          const PAGE = "__CURRENT_PAGE__";

          function scrollerNow() {
            return doc.querySelector('section.main')
                || doc.querySelector('[data-testid="stMain"]')
                || doc.querySelector('[data-testid="stAppViewContainer"]')
                || doc.scrollingElement;
          }

          // A new page always opens at its top, never at the old scroll offset.
          if (doc.__vahnixPage !== PAGE) {
            doc.__vahnixPage = PAGE;
            setTimeout(function () {
              const el = scrollerNow();
              if (el) el.scrollTo({ top: 0, behavior: 'auto' });
              window.parent.scrollTo({ top: 0, behavior: 'auto' });
              const sh = doc.getElementById('vahnix-nav-shell');
              if (sh) sh.classList.remove('nav-hidden');
            }, 40);
          }

          if (doc.__vahnixNavScroll) return;   // only wire the listeners up once
          doc.__vahnixNavScroll = true;

          const TOP_GUARD = 90;   // always show near the very top
          const DELTA = 6;        // ignore sub-pixel jitter
          let lastY = 0, ticking = false;

          function scroller() {
            return doc.querySelector('section.main')
                || doc.querySelector('[data-testid="stMain"]')
                || doc.querySelector('[data-testid="stAppViewContainer"]')
                || doc.scrollingElement;
          }

          function apply() {
            const shell = doc.getElementById('vahnix-nav-shell');
            const el = scroller();
            if (!shell || !el) { ticking = false; return; }

            const y = el.scrollTop || window.parent.scrollY || 0;
            shell.classList.toggle('nav-floating', y > TOP_GUARD);

            if (Math.abs(y - lastY) > DELTA) {
              const down = y > lastY;
              shell.classList.toggle('nav-hidden', down && y > TOP_GUARD);
              lastY = y;
            }
            ticking = false;
          }

          function onScroll() {
            if (!ticking) { ticking = true; window.requestAnimationFrame(apply); }
          }

          function attach() {
            const el = scroller();
            if (el) el.addEventListener('scroll', onScroll, { passive: true });
            window.parent.addEventListener('scroll', onScroll, { passive: true });
          }

          attach();
          // Streamlit re-renders swap the DOM out, so re-attach on mutation.
          new MutationObserver(attach).observe(doc.body, { childList: true, subtree: false });
        })();
        </script>
        """.replace("__CURRENT_PAGE__", current_page),
        height=0,
    )


def main():
    # Inject global SpaceX theme CSS
    st.markdown(SPACEX_CSS, unsafe_allow_html=True)

    # Global parent-iframe communication bridge for incident synchronization
    st.html("""
    <script>
    (function() {
      if (window.__vahnix_sync_listener_active) return;
      window.__vahnix_sync_listener_active = true;

      function activateIncidentInStreamlit(detId, targetHash) {
        if (!detId) return;
        try {
          var u = new URL(window.location.href);
          var currentId = u.searchParams.get('id');
          var currentPage = u.searchParams.get('page');
          if (currentId === detId && (currentPage === 'overview' || !currentPage)) {
            if (targetHash) {
              var tgtId = targetHash.replace('#', '');
              var tgt = document.getElementById(tgtId) || document.querySelector('[data-testid="stSelectbox"]');
              if (tgt) tgt.scrollIntoView({ behavior: 'smooth' });
            }
            return;
          }
          u.searchParams.set('page', 'overview');
          u.searchParams.set('id', detId);
          if (targetHash) {
            u.hash = targetHash;
          }
          window.location.href = u.toString();
        } catch(err) {
          window.location.href = '?page=overview&id=' + encodeURIComponent(detId) + (targetHash || '');
        }
      }

      window.vahnixSelectIncident = activateIncidentInStreamlit;

      window.addEventListener('message', function(e) {
        if (!e.data) return;
        if (e.data.type === 'VAHNIX_SELECT_INCIDENT_SILENT' && e.data.id) {
          try {
            var u = new URL(window.location.href);
            u.searchParams.set('page', 'overview');
            u.searchParams.set('id', e.data.id);
            window.history.replaceState(null, '', u.toString());
          } catch(err) {}
        } else if (e.data.type === 'VAHNIX_FOCUS_COPILOT' && e.data.id) {
          var tgt = document.getElementById('co-pilot-section') || document.querySelector('[data-testid="stSelectbox"]');
          if (tgt) {
            tgt.scrollIntoView({ behavior: 'smooth' });
          } else {
            activateIncidentInStreamlit(e.data.id, e.data.hash);
          }
        } else if (e.data.type === 'VAHNIX_NAVIGATE' && e.data.url) {
          window.location.href = e.data.url;
        }
      });

      function checkAnchorScroll() {
        if (window.location.hash) {
          var elId = window.location.hash.replace('#', '');
          var el = document.getElementById(elId) || document.querySelector('[data-testid="stSelectbox"]');
          if (el) el.scrollIntoView({ behavior: 'smooth' });
        }
      }
      setTimeout(checkAnchorScroll, 350);
      setTimeout(checkAnchorScroll, 850);
    })();
    </script>
    """, unsafe_allow_javascript=True)

    # Load video assets
    assets_dir = os.path.join(os.path.dirname(__file__), "assets")
    hero_video_path = os.path.join(assets_dir, "hero_cosmic_portal.mp4")
    satellite_video_path = os.path.join(assets_dir, "hero_satellite_orbit.mp4")
    flare_img_path = os.path.join(assets_dir, "gas_flare_stack.jpg")
    firefight_video_path = os.path.join(assets_dir, "firefighting_response.mp4")

    # Cache video base64 in session state
    if "hero_video_b64" not in st.session_state:
        st.session_state.hero_video_b64 = get_video_base64(hero_video_path)
    if "satellite_video_b64" not in st.session_state:
        st.session_state.satellite_video_b64 = get_video_base64(satellite_video_path)
    if "cluster_img_b64" not in st.session_state:
        st.session_state.cluster_img_b64 = get_video_base64(
            os.path.join(assets_dir, "cluster_network.jpg")
        )
    if "flare_img_b64" not in st.session_state:
        st.session_state.flare_img_b64 = get_video_base64(flare_img_path)
    if "firefight_video_b64" not in st.session_state:
        st.session_state.firefight_video_b64 = get_video_base64(firefight_video_path)

    if "logo_b64" not in st.session_state:
        st.session_state.logo_b64 = get_video_base64(os.path.join(assets_dir, "logo_vahnix.png"))

    # Transparent-background wordmark used on the cosmic hero.
    # Falls back to the nav-bar logo if the hero asset isn't present.
    if "hero_logo_b64" not in st.session_state:
        st.session_state.hero_logo_b64 = (
            get_video_base64(os.path.join(assets_dir, "logo_vahnix_hero.png"))
            or st.session_state.logo_b64
        )

    # ── 1. NAVIGATION STATE (URL query param — real browser navigation, no JS hacks) ──
    valid_pages = [
        "home", "overview", "live_stream", "clusters", "classifier",
        "explainability", "analysis", "chatbot", "reports",
        "technology", "tech", "our-technology",
        "goal", "goals", "boost", "boosting", "teamwork", "teamwork-preview", "team", "c2", "math", "mathworks"
    ]
    current_page = st.query_params.get("page", "").lower().strip()

    # Support direct query param flags like ?goal, ?boost, ?teamwork-preview, ?technology
    if not current_page:
        for p in (
            "goal", "goals", "boost", "boosting", "teamwork-preview", "teamwork",
            "team", "c2", "math", "mathworks", "technology", "tech", "our-technology"
        ):
            if p in st.query_params:
                current_page = p
                break

    # Also check URL path from st.context if available
    if not current_page and hasattr(st, "context") and hasattr(st.context, "url") and st.context.url:
        url_str = str(st.context.url).lower()
        for p in (
            "teamwork-preview", "teamwork", "team", "c2", "goal", "goals",
            "boost", "boosting", "math", "mathworks", "technology", "tech", "our-technology"
        ):
            if f"/{p}" in url_str:
                current_page = p
                break

    if not current_page:
        if "id" in st.query_params:
            current_page = "overview"
        else:
            current_page = "home"
    elif current_page not in valid_pages:
        current_page = "overview" if current_page == "globe" else "home"

    # ── 2. TOP BRAND HEADER (logo + nav as real links) ──
    nav_labels = [
        ("overview", "TACTICAL OVERVIEW"),
        ("live_stream", "LIVE STREAM"),
        ("clusters", "SPATIO-TEMPORAL CLUSTERS"),
        ("classifier", "AI CLASSIFICATION"),
        ("explainability", "SHAP EXPLAINABILITY"),
        ("analysis", "38-D ANALYSIS"),
        ("chatbot", "NIX CHATBOT"),
        ("reports", "TACTICAL REPORTING"),
        ("technology", "OUR TECHNOLOGY"),
    ]
    nav_items_html = ""
    for page_id, label in nav_labels:
        is_active = (
            current_page == page_id
            or (page_id == "technology" and current_page in (
                "tech", "our-technology", "goal", "goals", "boost", "boosting",
                "teamwork", "teamwork-preview", "team", "c2", "math", "mathworks"
            ))
        )
        active_class = " active" if is_active else ""
        tech_class = " nav-item-tech" if page_id == "technology" else ""
        nav_items_html += f'<a href="?page={page_id}" class="nav-item{active_class}{tech_class}" target="_self">{label}</a>'

    st.markdown(f"""
    <div class="nav-shell" id="vahnix-nav-shell">
        <div class="top-status-row">
            <div class="top-brand-status">
                <span class="status-pulse-dot"></span> ORBITAL SURVEILLANCE ACTIVE · VIIRS/MODIS LIVE
            </div>
        </div>
        <div class="top-brand-bar">
            <a href="?page=home" target="_self" class="top-brand-logo">
                <img src="data:image/png;base64,{st.session_state.logo_b64}" class="brand-logo-img">
            </a>
            {nav_items_html}
        </div>
    </div>
    <div class="nav-shell-spacer"></div>
    """, unsafe_allow_html=True)

    _render_nav_scroll_script(current_page)

    # ── 3. INITIALIZE STATE MANAGER ──
    state_mgr = get_state_manager()

    # ── 4. FILTER STATE (defaults pre-seeded; visible controls render at the bottom) ──
    if "filter_corridor" not in st.session_state:
        st.session_state.filter_corridor = "all"
    if "filter_sensor" not in st.session_state:
        st.session_state.filter_sensor = "all"
    if "filter_conf" not in st.session_state:
        st.session_state.filter_conf = 0.50
    if "filter_frp" not in st.session_state:
        st.session_state.filter_frp = 0.0
    if "filter_risks" not in st.session_state:
        st.session_state.filter_risks = ["GREEN", "YELLOW", "ORANGE", "RED"]
    if "filter_crit" not in st.session_state:
        st.session_state.filter_crit = False

    corridor = st.session_state.filter_corridor
    sensor = st.session_state.filter_sensor
    min_conf = st.session_state.filter_conf
    min_frp = st.session_state.filter_frp
    selected_risks = st.session_state.filter_risks
    only_crit = st.session_state.filter_crit

    # Build Filter State
    filters = DashboardFilterState(
        selected_corridor=corridor,
        selected_sensor=sensor,
        min_confidence=min_conf,
        min_frp=min_frp,
        selected_risk_levels=selected_risks,
        only_critical=only_crit
    )

    # Apply filters
    filtered_dets, filtered_preds, filtered_clusters = state_mgr.get_filtered_data(filters)
    kpis = state_mgr.get_kpi_summary(filtered_preds)

    # ── 5. RENDER CURRENT PAGE ──
    if current_page == "home":
        if st.session_state.hero_video_b64:
            render_hero_section(st.session_state.hero_video_b64, st.session_state.hero_logo_b64)
        render_intro_text_section(
            "How We Tell a Flare From a Fire",
            "The AI doesn't guess from one data point.",
            "it fuses space, ground, and time into a single classification."
        )
        if st.session_state.satellite_video_b64:
            render_video_section(
                st.session_state.satellite_video_b64,
                "ORBITAL SURVEILLANCE",
                "Space doesn't sleep. Neither does our detection pipeline.",
                align="right",
                flush=True,
                corner=True
            )
        if st.session_state.flare_img_b64:
            render_image_section(
                st.session_state.flare_img_b64,
                "GROUND TRUTH",
                "We map every refinery, kiln, and flare stack in advance, so their daily burn never triggers a false alarm.",
                mime="image/jpeg",
                align="left",
                flush=True,
                dark_text=True,
                corner=True
            )
        if st.session_state.firefight_video_b64:
            render_video_section(
                st.session_state.firefight_video_b64,
                "EMERGENCY RESPONSE",
                "Real-time detection means real-time response.",
                align="right",
                corner=True
            )

    elif current_page in ("overview", "globe"):
        render_overview_view(filtered_dets, filtered_preds, filtered_clusters, kpis, state_mgr=state_mgr)

    elif current_page == "analysis":
        render_detailed_analysis_view(state_mgr, filtered_dets, filtered_preds, filtered_clusters)

    elif current_page == "live_stream":
        render_live_stream_view(filtered_dets, filtered_preds, filtered_clusters, kpis, state_mgr=state_mgr)

    elif current_page == "clusters":
        if st.session_state.cluster_img_b64:
            render_image_section(
                st.session_state.cluster_img_b64,
                "CLUSTER ANALYSIS",
                "Spatio-temporal pattern recognition & recurrence mapping",
                mime="image/jpeg"
            )
        render_clusters_view(filtered_clusters)

    elif current_page == "classifier":
        render_classifier_view(filtered_preds)

    elif current_page == "explainability":
        render_explainability_view(state_mgr, filtered_preds)

    elif current_page == "chatbot":
        render_nix_chatbot_view(state_mgr, filtered_dets, filtered_preds, filtered_clusters, kpis)

    elif current_page == "reports":
        render_reporting_view(state_mgr, filtered_dets, filtered_preds, filtered_clusters)

    elif current_page in (
        "technology", "tech", "our-technology", "goal", "goals",
        "boost", "boosting", "teamwork", "teamwork-preview", "team", "c2", "math", "mathworks"
    ):
        initial_tab = "pipeline"
        if current_page in ("boost", "boosting", "math", "mathworks"):
            initial_tab = "math"
        elif current_page in ("teamwork", "teamwork-preview", "team", "c2"):
            initial_tab = "teamwork"
        elif current_page in ("goal", "goals"):
            initial_tab = "pipeline"
        tab_param = st.query_params.get("tab", initial_tab)
        render_technology_view(initial_tab=tab_param, kpis=kpis)

    # ── 6. MISSION CONTROL FILTERS (COLLAPSIBLE, AT THE BOTTOM) ──
    with st.expander("[SETTINGS] MISSION CONTROL — TACTICAL FILTERS", expanded=False):
        fc1, fc2, fc3, fc4 = st.columns(4)
        with fc1:
            st.selectbox(
                "Corridor Focus",
                ["all", "jamnagar", "jurong", "punjab", "simlipal"],
                key="filter_corridor"
            )
        with fc2:
            st.selectbox(
                "Sensor Platform",
                ["all", "VIIRS", "MODIS"],
                key="filter_sensor"
            )
        with fc3:
            st.slider("Min Confidence", 0.0, 1.0, step=0.05, key="filter_conf")
        with fc4:
            st.slider("Min FRP (MW)", 0.0, 300.0, step=10.0, key="filter_frp")

        fc5, fc6 = st.columns(2)
        with fc5:
            st.multiselect(
                "Alert Level",
                ["GREEN", "YELLOW", "ORANGE", "RED"],
                key="filter_risks"
            )
        with fc6:
            st.checkbox("Show Only Critical Emergencies", key="filter_crit")

    # ── 7. FOOTER ──
    render_footer()


if __name__ == "__main__":
    main()
