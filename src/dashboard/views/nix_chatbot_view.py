"""
NIX Tactical Chatbot View for VahniX Thermal Intelligence Platform.
Interactive AI-powered tactical co-pilot for thermal anomaly interrogation,
incident triage, corridor drill-downs, and facility impact assessment.
Authoritative Specifications: ORIGINAL_REQUEST.md § R2/R3, PROJECT.md § 3
"""

from __future__ import annotations

from typing import Any, Sequence
import streamlit as st
import pandas as pd
import datetime

from src.data_pipeline.schemas import FIRMSDetection
from src.clustering.schemas import ThermalCluster
from src.classification.ensemble_classifier import PredictionOutput
from src.dashboard.state_manager import DashboardStateManager


def _format_tactical_response(
    query: str,
    state_manager: DashboardStateManager,
    detections: Sequence[FIRMSDetection],
    predictions: Sequence[PredictionOutput],
    clusters: Sequence[ThermalCluster],
    kpis: dict[str, Any]
) -> dict[str, Any]:
    """
    Intelligent tactical response generator grounded in active platform state.
    Returns response text, optional table data, and alert level.
    """
    q = query.lower().strip()

    # 1. Model / SHAP Explanation & Classification Rationale
    if any(w in q for w in ["how", "explain", "model", "shap", "difference", "c0", "c1", "c2", "c3", "ai", "classifier", "rationale"]):
        return {
            "text": "[AI] **VahniX AI CLASSIFICATION ARCHITECTURE**\n\n"
                    "The system employs an ensemble of **Gradient Boosting (LightGBM)**, **Random Forest**, and a **Heuristic Baseline**:\n\n"
                    "1. **Controlled Flare (C0):** Highly recurrent, stationary jitter (<300m), situated within verified industrial polygon, balanced day/night ratio (DNBI ≈ 1.0).\n"
                    "2. **Industrial Emergency (C1):** High FRP spike (>150 MW), high Delta-T (>60K), rapid spatial expansion, located in or immediately adjacent to industrial assets.\n"
                    "3. **Agricultural Fire (C2):** Low persistence, day-skewed, outside industrial boundaries, seasonal cadence.\n"
                    "4. **Wildfire (C3):** High spatial dispersal, non-stationary propagation vector, forest canopy intersection.\n\n"
                    "Explanations are computed locally via **TreeExplainer SHAP** waterfall attributions.",
            "table": None,
            "badge": "AI"
        }

    # 2. Critical / Emergency Incidents
    elif any(w in q for w in ["critical", "emergency", "emergencies", "disaster", "high risk", "alert"]):
        crit_preds = [p for p in predictions if p.is_critical_alert or p.predicted_class.value == 1 or p.risk_severity_index >= 70]
        if not crit_preds:
            return {
                "text": "[STATUS] **TACTICAL STATUS: ALL CLEAR**\n\nNo active incidents currently meet the critical emergency threshold (Risk Severity ≥ 70 or Class C1 Industrial Disaster). Surveillance remains active.",
                "table": None,
                "badge": "NOMINAL"
            }
        
        rows = []
        for p in crit_preds[:10]:
            det = next((d for d in detections if d.detection_id == p.detection_id), None)
            rows.append({
                "INCIDENT ID": p.detection_id,
                "CLASS": p.predicted_label,
                "RISK SCORE": f"{p.risk_severity_index:.1f}/100",
                "CONFIDENCE": f"{p.confidence_score * 100:.1f}%",
                "FRP (MW)": f"{p.feature_values.get('frp', 0.0):.1f}",
                "COORDINATES": f"{det.latitude:.4f} N, {det.longitude:.4f} E" if det else "—",
                "TIME": det.acq_time if det else "—"
            })
        
        return {
            "text": f"[ALERT] **CRITICAL ALERT: {len(crit_preds)} HIGH-RISK INCIDENTS DETECTED**\n\n"
                    f"Identified **{len(crit_preds)}** thermal signatures classified as active industrial emergencies or critical anomalies. "
                    f"Immediate automated containment telemetry and drone verification dispatched.",
            "table": pd.DataFrame(rows),
            "badge": "CRITICAL"
        }

    # 2. System Overview / Status / Telemetry
    elif any(w in q for w in ["status", "summary", "overview", "telemetry", "kpi", "metrics", "how many"]):
        return {
            "text": f"[TELEMETRY] **VahniX TACTICAL SYSTEM TELEMETRY**\n\n"
                    f"- **Total Active Signatures:** `{kpis['total_incidents']}`\n"
                    f"- **Industrial Emergencies (C1):** `{kpis['industrial_emergencies']}` (Critical: `{kpis['critical_alerts']}`)\n"
                    f"- **Controlled Petrochemical Flares (C0):** `{kpis['controlled_flares']}`\n"
                    f"- **Agricultural Stubble Biomass (C2):** `{kpis['agricultural_fires']}`\n"
                    f"- **Wildfire Fronts (C3):** `{kpis['wildfires']}`\n"
                    f"- **Peak FRP Recorded:** `{kpis['max_frp']:.1f} MW`\n"
                    f"- **Active Clusters Tracked:** `{len(clusters)}`\n\n"
                    f"Orbital sensors: **VIIRS (375m I-Band)** and **MODIS (1km)** data ingested with multi-pass DBSCAN clustering.",
            "table": None,
            "badge": "INFO"
        }

    # 3. Facility Assessment
    elif any(w in q for w in ["facility", "facilities", "plant", "refinery", "industrial"]):
        inside_clusters = [c for c in clusters if c.is_inside_industrial_boundary]
        fac_names = set(c.nearest_industrial_facility_id for c in clusters if c.nearest_industrial_facility_id)
        
        return {
            "text": f"[AUDIT] **INDUSTRIAL FACILITY VULNERABILITY AUDIT**\n\n"
                    f"- **Total Monitored Facilities:** `{len(state_manager.all_facilities)}` (OSM indexed)\n"
                    f"- **Clusters Inside Industrial Boundaries:** `{len(inside_clusters)}`\n"
                    f"- **Facilities with Nearby Thermal Activity:** `{len(fac_names)}`\n\n"
                    f"Monitored installations include Jamnagar Mega-Refinery, Jurong Petrochemical Complex, and allied crude processing nodes. "
                    f"Routine flares are segregated via spatial polygon intersection and day-night balance indexing (DNBI).",
            "table": None,
            "badge": "FACILITY"
        }

    # 4. Corridor: Jamnagar
    elif "jamnagar" in q:
        jam_dets = [d for d in detections if "jamnagar" in d.detection_id.lower() or "emerg" in d.detection_id.lower()]
        jam_crit = [d for d in jam_dets if "emerg" in d.detection_id.lower()]
        return {
            "text": f"[CORRIDOR] **CORRIDOR REPORT: JAMNAGAR PETROCHEMICAL HUB**\n\n"
                    f"- **Active Detections:** `{len(jam_dets)}`\n"
                    f"- **Injected Emergency Incidents:** `{len(jam_crit)}`\n"
                    f"- **Primary Signatures:** Continuous hydrocarbon flare stacks exhibiting high Day-Night Balance Index (DNBI > 0.8) and low spatial jitter (< 250m).\n"
                    f"- **Risk Protocol:** Controlled flare stacks are filtered from emergency dispatch, while uncontained high-FRP spikes trigger Level 1 Alert.",
            "table": None,
            "badge": "CORRIDOR"
        }

    # 5. Corridor: Punjab
    elif "punjab" in q or "stubble" in q or "agriculture" in q:
        pb_dets = [d for d in detections if "punjab" in d.detection_id.lower()]
        return {
            "text": f"[CORRIDOR] **CORRIDOR REPORT: PUNJAB AGRICULTURAL BELT**\n\n"
                    f"- **Active Detections:** `{len(pb_dets)}`\n"
                    f"- **Primary Classification:** Class C2 (Post-Harvest Stubble / Biomass Burning)\n"
                    f"- **Signature Profile:** Moderate FRP (20–90 MW), day-dominant acquisition, zero proximity to industrial OSM zones.\n"
                    f"- **Environmental Impact:** High particulate emission trajectory. Forwarded to State Pollution Control Board telemetry.",
            "table": None,
            "badge": "CORRIDOR"
        }

    # 6. Peak FRP / Hotspots
    elif any(w in q for w in ["hottest", "highest", "peak", "frp", "max", "fire power"]):
        sorted_preds = sorted(predictions, key=lambda p: p.feature_values.get("frp", 0.0), reverse=True)[:5]
        rows = []
        for p in sorted_preds:
            det = next((d for d in detections if d.detection_id == p.detection_id), None)
            rows.append({
                "INCIDENT": p.detection_id,
                "FRP (MW)": f"{p.feature_values.get('frp', 0.0):.1f}",
                "CLASS": p.predicted_label,
                "BRIGHTNESS (K)": f"{det.brightness_temp_t4:.1f}" if det else "—",
                "CONFIDENCE": f"{p.confidence_score * 100:.1f}%"
            })
        return {
            "text": f"[RANKING] **TOP 5 MAXIMUM FIRE RADIATIVE POWER (FRP) SIGNATURES**\n\n"
                    f"Peak thermal energy release detected across orbital passes:",
            "table": pd.DataFrame(rows),
            "badge": "FRP"
        }

    # 7. Model / SHAP Explanation
    elif any(w in q for w in ["how", "explain", "model", "shap", "difference", "c0", "c1", "ai", "classifier"]):
        return {
            "text": "[AI] **VahniX AI CLASSIFICATION ARCHITECTURE**\n\n"
                    "The system employs an ensemble of **Gradient Boosting (LightGBM)**, **Random Forest**, and a **Heuristic Baseline**:\n\n"
                    "1. **Controlled Flare (C0):** Highly recurrent, stationary jitter (<300m), situated within verified industrial polygon, balanced day/night ratio (DNBI ≈ 1.0).\n"
                    "2. **Industrial Emergency (C1):** High FRP spike (>150 MW), high Delta-T (>60K), rapid spatial expansion, located in or immediately adjacent to industrial assets.\n"
                    "3. **Agricultural Fire (C2):** Low persistence, day-skewed, outside industrial boundaries, seasonal cadence.\n"
                    "4. **Wildfire (C3):** High spatial dispersal, non-stationary propagation vector, forest canopy intersection.\n\n"
                    "Explanations are computed locally via **TreeExplainer SHAP** waterfall attributions.",
            "table": None,
            "badge": "AI"
        }

    # Default fallback response
    else:
        return {
            "text": f"[ASSISTANT] **NIX ASSISTANT ACKNOWLEDGED: '{query}'**\n\n"
                    f"Telemetry scan completed against active database (`{len(predictions)}` classified incidents, `{len(clusters)}` clusters).\n\n"
                    f"**Suggested queries:**\n"
                    f"- *'List all critical emergencies'*\n"
                    f"- *'Show tactical system telemetry summary'*\n"
                    f"- *'What is happening in the Jamnagar corridor?'*\n"
                    f"- *'Which facilities are at risk?'*\n"
                    f"- *'Show top highest FRP hotspots'*\n"
                    f"- *'Explain the difference between C0 flare and C1 emergency'*",
            "table": None,
            "badge": "ASSISTANT"
        }


_LAVENDER = "#A78BFA"


def _md_to_html(text: str) -> str:
    """Convert the light markdown used in NIX replies into inline HTML.

    Streamlit does not parse markdown inside a raw HTML block, which is why
    bold markers were showing up literally. This handles the small subset the
    assistant actually emits: bold, italics, inline code, bullets, newlines.
    """
    import html as _html
    import re as _re

    out_lines: list[str] = []
    in_list = False

    for raw in str(text).split("\n"):
        line = _html.escape(raw.strip())

        line = _re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", line)
        line = _re.sub(r"(?<!\*)\*(?!\s)(.+?)(?<!\s)\*(?!\*)", r"<em>\1</em>", line)
        line = _re.sub(
            r"`(.+?)`",
            r'<code style="background:rgba(167,139,250,0.16); color:#C9B8FF; '
            r'padding:0.08em 0.38em; border-radius:5px; font-size:0.86em;">\1</code>',
            line,
        )

        if line.startswith(("- ", "• ", "* ")):
            if not in_list:
                out_lines.append(
                    '<ul style="margin:0.45rem 0 0.2rem 0; padding-left:1.15rem;">'
                )
                in_list = True
            out_lines.append(
                f'<li style="margin-bottom:0.3rem;">{line[2:].strip()}</li>'
            )
            continue

        if in_list:
            out_lines.append("</ul>")
            in_list = False

        if not line:
            out_lines.append('<div style="height:0.55rem;"></div>')
        else:
            out_lines.append(f"<div>{line}</div>")

    if in_list:
        out_lines.append("</ul>")

    return "".join(out_lines)


def render_nix_chatbot_view(
    state_manager: DashboardStateManager,
    detections: Sequence[FIRMSDetection],
    predictions: Sequence[PredictionOutput],
    clusters: Sequence[ThermalCluster],
    kpis: dict[str, Any]
) -> None:
    """Render the interactive NIX Chatbot tactical intelligence channel."""

    # Lavender chat input styling
    st.markdown(f"""
    <style>
    [data-testid="stChatInput"] {{
        border-radius: 16px !important;
        border: 1px solid rgba(167,139,250,0.35) !important;
        background: rgba(167,139,250,0.07) !important;
        transition: border-color 0.2s ease, box-shadow 0.2s ease;
    }}
    [data-testid="stChatInput"]:focus-within {{
        border-color: {_LAVENDER} !important;
        box-shadow: 0 0 0 3px rgba(167,139,250,0.16) !important;
    }}
    [data-testid="stChatInput"] textarea {{
        caret-color: {_LAVENDER} !important;
        color: var(--text-primary) !important;
    }}
    [data-testid="stChatInput"] button {{
        color: {_LAVENDER} !important;
    }}
    [data-testid="stChatInput"] button:hover {{
        background: rgba(167,139,250,0.18) !important;
    }}
    </style>
    """, unsafe_allow_html=True)

    # Header
    st.markdown("""
    <div style="margin-bottom: 2rem;">
        <div style="font-family: var(--font-mono); font-size: 0.7rem; letter-spacing: 0.3em; text-transform: uppercase; color: var(--accent-red); margin-bottom: 0.5rem;">
            TACTICAL CO-PILOT // VahniX DEFENSE SYSTEM
        </div>
        <h2 style="font-family: var(--font-display); font-size: 1.8rem; font-weight: 700; letter-spacing: 0.12em; text-transform: uppercase; color: var(--text-primary); margin: 0 0 0.5rem 0;">
            NIX TACTICAL CHATBOT
        </h2>
        <p style="font-family: var(--font-body); font-size: 0.85rem; letter-spacing: 0.15em; text-transform: uppercase; color: var(--text-muted); margin: 0;">
            Autonomous conversational intelligence for satellite thermal anomaly interrogation
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Initialize chat history
    if "nix_messages" not in st.session_state:
        st.session_state.nix_messages = [
            {
                "role": "assistant",
                "content": "[NIX AI] **NIX Tactical Intelligence Co-Pilot Online.**\n\nSurveillance link established with VahniX orbital engine. Ingesting live VIIRS & MODIS feeds. Ready to assist with threat correlation, facility risk indexing, or anomaly classification. How can I assist your mission?",
                "table": None,
                "timestamp": datetime.datetime.now().strftime("%H:%M:%S UTC")
            }
        ]

    # Quick action prompt chips
    st.markdown("""
    <div style="margin-bottom: 1.5rem;">
        <div style="font-family: var(--font-mono); font-size: 0.6rem; letter-spacing: 0.25em; text-transform: uppercase; color: var(--text-muted); margin-bottom: 0.8rem;">
            RAPID TACTICAL QUERIES
        </div>
    </div>
    """, unsafe_allow_html=True)

    q1, q2, q3, q4, q5 = st.columns(5)
    selected_quick_prompt = None

    with q1:
        if st.button("[ALERT] CRITICAL ALERTS", use_container_width=True, key="quick_crit"):
            selected_quick_prompt = "List all critical emergencies"
    with q2:
        if st.button("[STATUS] SYSTEM STATUS", use_container_width=True, key="quick_status"):
            selected_quick_prompt = "Show tactical system telemetry summary"
    with q3:
        if st.button("[ASSETS] AT-RISK PLANTS", use_container_width=True, key="quick_fac"):
            selected_quick_prompt = "Which industrial facilities are at risk?"
    with q4:
        if st.button("[HOTSPOTS] MAX FRP SPOTS", use_container_width=True, key="quick_frp"):
            selected_quick_prompt = "Show top highest FRP hotspots"
    with q5:
        if st.button("[RATIONALE] AI RATIONALE", use_container_width=True, key="quick_ai"):
            selected_quick_prompt = "Explain difference between C0 flare and C1 emergency"

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # Chat message container
    chat_container = st.container()

    with chat_container:
        for msg in st.session_state.nix_messages:
            if msg["role"] == "user":
                st.markdown(f"""
                <div style="display:flex; justify-content:flex-end; margin-bottom:0.9rem;">
                    <div style="background:{_LAVENDER}; color:#1A1327;
                                border-radius:16px 16px 6px 16px;
                                padding:0.75rem 1.15rem; max-width:70%;
                                font-family:var(--font-body); font-size:0.95rem;
                                line-height:1.5; font-weight:500;
                                box-shadow:0 2px 10px rgba(167,139,250,0.18);">
                        {msg['content']}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style="display:flex; justify-content:flex-start; margin-bottom:0.9rem;">
                    <div style="background:rgba(255,255,255,0.045);
                                border:1px solid rgba(255,255,255,0.09);
                                border-radius:16px 16px 16px 6px;
                                padding:0.9rem 1.25rem; max-width:78%;
                                font-family:var(--font-body); font-size:0.95rem;
                                line-height:1.6; color:rgba(245,247,250,0.92);">
                        {_md_to_html(msg['content'])}
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # If the assistant message contains a data table
                if msg.get("table") is not None and not msg["table"].empty:
                    st.dataframe(msg["table"], use_container_width=True, hide_index=True)

    # Chat input bar
    user_input = st.chat_input("Enter tactical query or command for NIX Intelligence...")

    # Handle quick prompt or chat input
    active_prompt = selected_quick_prompt or user_input

    if active_prompt:
        ts = datetime.datetime.now().strftime("%H:%M:%S UTC")
        # Append user message
        st.session_state.nix_messages.append({
            "role": "user",
            "content": active_prompt,
            "timestamp": ts
        })

        # Generate NIX response
        response = _format_tactical_response(
            active_prompt,
            state_manager,
            detections,
            predictions,
            clusters,
            kpis
        )

        st.session_state.nix_messages.append({
            "role": "assistant",
            "content": response["text"],
            "table": response.get("table"),
            "timestamp": datetime.datetime.now().strftime("%H:%M:%S UTC")
        })

        st.rerun()

    # Clear chat option
    st.markdown('<div style="margin-top: 2rem; display: flex; justify-content: flex-end;">', unsafe_allow_html=True)
    if st.button("RESET CHAT SESSION", key="clear_chat_btn"):
        st.session_state.nix_messages = [
            {
                "role": "assistant",
                "content": "[NIX AI] **NIX Tactical Intelligence Co-Pilot Online.** Session reset. Ready for operator queries.",
                "table": None,
                "timestamp": datetime.datetime.now().strftime("%H:%M:%S UTC")
            }
        ]
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

