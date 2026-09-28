"""
AI Classifier & Multi-Class ML Performance View for VahniX Tactical Dashboard.
SpaceX-inspired dark glassmorphism treatment.
Renders predicted category distributions across 6 tactical tiers, risk severity matrices, and baseline comparisons.
Authoritative Specifications: ORIGINAL_REQUEST.md § R2, PROJECT.md § 3
"""

from __future__ import annotations

from typing import Sequence, Dict, Any
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from src.classification.ensemble_classifier import PredictionOutput
from src.classification.heuristic_baseline import CLASS_NAMES


# ── 6-CLASS TAXONOMY CONFIGURATION ──
AI_CLASS_DEFINITIONS = [
    {
        "id": 0,
        "code": "CLASS 01",
        "name": "Controlled Industrial Flare",
        "short_name": "Flare",
        "color": "#00D2FF",
        "border_color": "#38BDF8",
        "glow": "rgba(0, 210, 255, 0.40)",
        "delta": "ROUTINE PROCESS HEAT",
        "delta_type": "neutral",
        "is_critical": False,
        "icon": "[FLARE]",
        "protocol": "Continuous telemetry logging · Baseline steady process heat",
        "description": "Regulated industrial flaring, FCCU exhaust, rotary kilns, metal smelters.",
    },
    {
        "id": 1,
        "code": "CLASS 02",
        "name": "Industrial Fire Emergency",
        "short_name": "Emergency",
        "color": "#FF3344",
        "border_color": "#EF4444",
        "glow": "rgba(255, 51, 68, 0.50)",
        "delta": "HIGH PRIORITY HAZARD",
        "delta_type": "positive",
        "is_critical": True,
        "icon": "[EMERGENCY]",
        "protocol": "IMMEDIATE EMERGENCY DISPATCH · Notify NDRF / SDMA command",
        "description": "Storage tank boilover, BLEVE, petrochemical unit deflagration, runaway fires.",
    },
    {
        "id": 2,
        "code": "CLASS 03",
        "name": "Agricultural Burning",
        "short_name": "Agricultural",
        "color": "#F59E0B",
        "border_color": "#FBBF24",
        "glow": "rgba(245, 158, 11, 0.40)",
        "delta": "SEASONAL CROP RESIDUE",
        "delta_type": "neutral",
        "is_critical": False,
        "icon": "[AGRI]",
        "protocol": "Environmental compliance tracking · Biomass smoke plume modeling",
        "description": "Post-harvest paddy/wheat crop residue and seasonal agricultural stubble burning.",
    },
    {
        "id": 3,
        "code": "CLASS 04",
        "name": "Wildfire",
        "short_name": "Wildfire",
        "color": "#FF6B35",
        "border_color": "#FF8A65",
        "glow": "rgba(255, 107, 53, 0.40)",
        "delta": "VEGETATION CANOPY",
        "delta_type": "neutral",
        "is_critical": False,
        "icon": "[WILDFIRE]",
        "protocol": "Forestry containment task force · Rapid perimeter advancement check",
        "description": "Forest canopy fires, bushfire fronts, rapid perimeter advancement.",
    },
    {
        "id": 4,
        "code": "CLASS 05",
        "name": "Volcanic Activity",
        "short_name": "Volcanic",
        "color": "#B066FF",
        "border_color": "#C084FC",
        "glow": "rgba(176, 102, 255, 0.40)",
        "delta": "GEOTHERMAL & LAVA FLOW",
        "delta_type": "neutral",
        "is_critical": False,
        "icon": "[VOLCANO]",
        "protocol": "Geological Survey notification · Caldera thermal venting monitor",
        "description": "Active caldera lava lakes, basaltic flows, geothermal fumaroles.",
    },
    {
        "id": 5,
        "code": "CLASS 06",
        "name": "Noise",
        "short_name": "Noise / Glint",
        "color": "#94A3B8",
        "border_color": "#CBD5E1",
        "glow": "rgba(148, 163, 184, 0.35)",
        "delta": "SOLAR GLINT / ARTIFACT",
        "delta_type": "neutral",
        "is_critical": False,
        "icon": "[NOISE]",
        "protocol": "Automated rejection filter · Solar PV / specular reflection discard",
        "description": "Specular solar glint off metal roofs, solar PV panels, sensor telemetry noise.",
    },
]


def _resolve_prediction_class_index(p: PredictionOutput) -> int:
    """Resolve any prediction instance to one of the 6 canonical class indices [0..5]."""
    did = str(getattr(p, "detection_id", "") or "").upper()
    label = str(getattr(p, "predicted_label", "") or "").upper()

    if any(k in did or k in label for k in ["VOLCAN", "BARREN", "LAVA", "GEOTHERMAL"]):
        return 4
    if any(k in did or k in label for k in ["NOISE", "GLINT", "FALSE", "ARTIFACT"]):
        return 5
    if any(k in did or k in label for k in ["EMERG", "DISASTER", "EXPLOSION", "BLEVE", "HAZARD"]):
        return 1
    if any(k in did or k in label for k in ["FLARE", "JAMNAGAR", "JURONG", "REFINERY", "CONTROLLED"]):
        return 0
    if any(k in did or k in label for k in ["AGRICULTUR", "STUBBLE", "PUNJAB", "CROP", "FARM"]):
        return 2
    if any(k in did or k in label for k in ["WILD", "FOREST", "SIMLIPAL", "VEGETATION", "CANOPY"]):
        return 3

    cls_obj = getattr(p, "predicted_class", None)
    val = getattr(cls_obj, "value", cls_obj)
    if isinstance(val, int) and 0 <= val <= 5:
        return val

    return 3


def _render_section_header(title: str, subtitle: str = "") -> None:
    """Render a SpaceX-style section header."""
    html = f'<h2 class="section-header">{title}</h2>'
    if subtitle:
        html += f'<p class="section-subtitle">{subtitle}</p>'
    st.markdown(html, unsafe_allow_html=True)


def _render_class_kpi_card(
    class_code: str,
    full_name: str,
    count: int,
    pct: float,
    color: str,
    border_color: str,
    glow_color: str,
    delta: str = "",
    delta_type: str = "neutral",
    is_critical: bool = False,
    icon: str = "",
) -> None:
    """
    Render a high-impact, SpaceX-inspired class metric card.
    The class name is strictly horizontal, rendered in a prominent bigger font size.
    """
    crit_pulse = " critical" if (is_critical and count > 0) else ""
    border_style = (
        f"border: 1px solid {color} !important; box-shadow: 0 0 22px {glow_color} !important;"
        if (is_critical and count > 0)
        else f"border-top: 2px solid {color}; border-left: 1px solid rgba(255,255,255,0.08); border-right: 1px solid rgba(255,255,255,0.08); border-bottom: 1px solid rgba(255,255,255,0.08);"
    )

    delta_html = (
        f'<div class="kpi-delta {delta_type}" style="margin-top:0.4rem; font-size:0.65rem; '
        f'font-family:var(--font-mono); letter-spacing:0.08em; text-transform:uppercase;">'
        f'{delta}</div>'
        if delta else ""
    )

    html = f"""
    <div class="glass-card{crit_pulse}" style="height: 100%; min-height: 175px; padding: 1.15rem 1rem; display: flex; flex-direction: column; justify-content: space-between; {border_style} background: rgba(12, 17, 29, 0.65); border-radius: 8px;">
        <div>
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.55rem;">
                <span style="font-family: var(--font-mono); font-size: 0.68rem; font-weight: 700; letter-spacing: 0.16em; color: {color}; background: rgba(255,255,255,0.04); padding: 2px 7px; border-radius: 4px; border: 1px solid {border_color}33;">
                    {class_code}
                </span>
                <span style="font-family: var(--font-mono, monospace); font-size: 0.68rem; font-weight: 800; color: {color}; border: 1px solid {border_color}44; padding: 2px 6px; border-radius: 4px;">{icon}</span>
            </div>
            <div style="font-family: var(--font-display); font-size: 1.02rem; font-weight: 700; color: #FFFFFF; line-height: 1.25; min-height: 2.6rem; display: flex; align-items: center; white-space: normal; letter-spacing: 0.01em;">
                {full_name}
            </div>
        </div>
        <div style="margin-top: 0.7rem;">
            <div style="display: flex; align-items: baseline; gap: 0.45rem;">
                <div style="font-family: var(--font-mono); font-size: 2.2rem; font-weight: 800; color: #FFFFFF; line-height: 1;">
                    {count}
                </div>
                <div style="font-family: var(--font-mono); font-size: 0.8rem; font-weight: 600; color: {color};">
                    ({pct:.1f}%)
                </div>
            </div>
            {delta_html}
        </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def _render_distribution_graph(counts: Dict[int, int], total_count: int) -> None:
    """
    Render a visually appealing SpaceX dark-glassmorphism Plotly distribution chart.
    Displays all 6 classes horizontally with full names in large, bold font size.
    """
    labels = []
    values = []
    percentages = []
    colors = []
    border_colors = []
    protocols = []

    for item in AI_CLASS_DEFINITIONS:
        cid = item["id"]
        c_count = counts.get(cid, 0)
        c_pct = (c_count / total_count * 100.0) if total_count > 0 else 0.0

        # Numbered full name for crystal clarity (1 to 6)
        labels.append(f"{cid + 1}. {item['name']}")
        values.append(c_count)
        percentages.append(c_pct)
        colors.append(item["color"])
        border_colors.append(item["border_color"])
        protocols.append(item["protocol"])

    max_val = max(values) if values and max(values) > 0 else 10
    x_range = [0, max_val * 1.35]

    customdata = list(zip(percentages, protocols))

    # Build Horizontal Bar Figure
    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=values,
        y=labels,
        orientation="h",
        marker=dict(
            color=colors,
            line=dict(color=border_colors, width=2.0),
            cornerradius=5
        ),
        text=[
            f"  <b>{v}</b>  <span style='font-size:12px; color:rgba(255,255,255,0.72);'>({p:.1f}%)</span>"
            for v, p in zip(values, percentages)
        ],
        textposition="outside",
        textfont=dict(
            family="JetBrains Mono, Courier New, monospace",
            size=13,
            color="#FFFFFF"
        ),
        customdata=customdata,
        hovertemplate=(
            "<span style='font-family:Space Grotesk, sans-serif; font-size:15px; font-weight:700; color:#FFFFFF;'>%{y}</span><br>"
            "<hr style='border:none; border-top:1px solid rgba(255,255,255,0.18); margin:6px 0;'>"
            "<span style='font-family:JetBrains Mono, monospace; font-size:12px;'>"
            "Hotspot Detections: <b>%{x}</b><br>"
            "Distribution Share: <b>%{customdata[0]:.1f}%</b><br>"
            "Operational Directive: <span style='color:#00D2FF;'>%{customdata[1]}</span>"
            "</span><extra></extra>"
        ),
        showlegend=False,
    ))

    fig.update_layout(
        height=450,
        margin=dict(l=260, r=50, t=25, b=45),
        bargap=0.32,
        plot_bgcolor="rgba(6, 10, 20, 0.55)",
        paper_bgcolor="rgba(0, 0, 0, 0)",
        xaxis=dict(
            title=dict(
                text="ACTIVE SATELLITE DETECTIONS (MWIR / LWIR THERMAL HOTSPOTS)",
                font=dict(family="JetBrains Mono, monospace", size=10, color="rgba(240, 240, 250, 0.45)"),
            ),
            range=x_range,
            showgrid=True,
            gridcolor="rgba(255, 255, 255, 0.05)",
            gridwidth=1,
            zeroline=True,
            zerolinecolor="rgba(255, 255, 255, 0.18)",
            zerolinewidth=1,
            tickfont=dict(family="JetBrains Mono, monospace", size=11, color="rgba(240, 240, 250, 0.65)"),
        ),
        yaxis=dict(
            autorange="reversed",  # Class 1 at top down to Class 6 at bottom
            showgrid=False,
            zeroline=False,
            showline=False,
            ticksuffix="   ",
            tickfont=dict(
                family="Space Grotesk, Inter, sans-serif",
                size=14,
                color="#FFFFFF",
                weight="bold"
            ),
        ),
        hoverlabel=dict(
            bgcolor="#0B111E",
            bordercolor="rgba(255, 255, 255, 0.18)",
            font=dict(family="JetBrains Mono, monospace", size=12, color="#FFFFFF"),
        ),
    )

    # ── Interactive Sub-View Selector ──
    chart_view = st.radio(
        "Chart Mode",
        ["Horizontal Distribution Bar", "Proportions Donut Breakdown"],
        index=0,
        horizontal=True,
        label_visibility="collapsed",
        key="classifier_chart_mode"
    )

    if chart_view == "Horizontal Distribution Bar":
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    else:
        # Secondary Donut View
        donut_fig = go.Figure(data=[go.Pie(
            labels=labels,
            values=values,
            hole=0.62,
            marker=dict(colors=colors, line=dict(color=border_colors, width=2.0)),
            textinfo="percent+label",
            textposition="outside",
            textfont=dict(family="Space Grotesk, sans-serif", size=13, color="#FFFFFF"),
            hovertemplate=(
                "<b>%{label}</b><br>"
                "Hotspots: <b>%{value}</b><br>"
                "Share: <b>%{percent}</b><extra></extra>"
            ),
        )])
        donut_fig.update_layout(
            height=450,
            margin=dict(l=40, r=40, t=30, b=40),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            showlegend=True,
            legend=dict(
                font=dict(family="Space Grotesk, sans-serif", size=13, color="#FFFFFF"),
                orientation="h",
                yanchor="bottom",
                y=-0.25,
                xanchor="center",
                x=0.5
            ),
            annotations=[dict(
                text=f"<span style='font-size:26px; font-weight:800; color:#FFFFFF;'>{total_count}</span><br><span style='font-size:11px; color:rgba(255,255,255,0.5); letter-spacing:0.12em;'>TOTAL HOTSPOTS</span>",
                x=0.5, y=0.5,
                font=dict(family="JetBrains Mono, monospace"),
                showarrow=False
            )]
        )
        st.plotly_chart(donut_fig, use_container_width=True, config={"displayModeBar": False})

    # Proportional Segmented Progress Strip
    if total_count > 0:
        segment_htmls = []
        for item in AI_CLASS_DEFINITIONS:
            c_val = counts.get(item["id"], 0)
            c_pct = (c_val / total_count * 100.0)
            if c_pct > 0:
                segment_htmls.append(
                    f'<div title="{item["name"]}: {c_val} ({c_pct:.1f}%)" '
                    f'style="flex: {c_pct}; background: {item["color"]}; height: 8px; '
                    f'box-shadow: 0 0 8px {item["glow"]};"></div>'
                )
        st.markdown(
            f'<div style="display: flex; width: 100%; border-radius: 6px; overflow: hidden; '
            f'margin-top: -0.5rem; margin-bottom: 1.2rem; background: rgba(255,255,255,0.05);">'
            f'{"".join(segment_htmls)}</div>',
            unsafe_allow_html=True
        )


def render_classifier_view(predictions: Sequence[PredictionOutput]) -> None:
    """Render AI Classifier View with 6-class architecture and SpaceX styling."""

    _render_section_header(
        "AI CLASSIFICATION ENGINE",
        "Multi-class thermal anomaly classification & tactical threat intelligence (6-Class Core Architecture)"
    )

    if not predictions:
        st.markdown("""
        <div class="glass-card" style="text-align:center; padding:3rem;">
            <div class="kpi-label">NO PREDICTIONS AVAILABLE</div>
            <p style="color: var(--text-muted); font-size: 0.85rem; margin-top: 0.5rem;">
                No prediction outputs available for the current filter configuration.
            </p>
        </div>
        """, unsafe_allow_html=True)
        return

    # ── 1. COMPUTE 6-CLASS COUNTS ──
    counts: Dict[int, int] = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
    for p in predictions:
        idx = _resolve_prediction_class_index(p)
        counts[idx] = counts.get(idx, 0) + 1

    total_predictions = sum(counts.values()) or 1

    # ── 2. 6-CLASS KPI METRIC CARDS (HORIZONTAL, FULL NAME IN BIGGER FONT SIZE) ──
    cols = st.columns(6)
    for col, defn in zip(cols, AI_CLASS_DEFINITIONS):
        cid = defn["id"]
        c_val = counts.get(cid, 0)
        c_pct = (c_val / total_predictions * 100.0)

        with col:
            _render_class_kpi_card(
                class_code=defn["code"],
                full_name=defn["name"],
                count=c_val,
                pct=c_pct,
                color=defn["color"],
                border_color=defn["border_color"],
                glow_color=defn["glow"],
                delta=defn["delta"],
                delta_type=defn["delta_type"],
                is_critical=defn["is_critical"],
                icon=defn["icon"],
            )

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # ── 3. VISUALLY APPEALING DISTRIBUTION GRAPH (HORIZONTAL BARS WITH BIGGER FONT) ──
    _render_section_header(
        "PREDICTED CLASS DISTRIBUTION",
        "Ensemble model classification breakdown"
    )

    _render_distribution_graph(counts, total_predictions)

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # ── 5. HIGH RISK DETECTIONS TABLE ──
    _render_section_header(
        "HIGH-RISK TACTICAL DETECTIONS",
        "Risk severity index ≥ 60 — requires immediate assessment"
    )

    high_risk = [p for p in predictions if p.risk_severity_index >= 60.0]

    if high_risk:
        hr_rows = []
        for p in high_risk:
            # Map to canonical full name if available
            c_idx = _resolve_prediction_class_index(p)
            c_name = AI_CLASS_DEFINITIONS[c_idx]["name"]

            hr_rows.append({
                "INCIDENT ID": p.detection_id,
                "CLASS": c_name,
                "CONFIDENCE": f"{p.confidence_score * 100:.1f}%",
                "RISK SEVERITY": f"{p.risk_severity_index:.1f}",
                "FRP (MW)": f"{p.feature_values.get('frp', 0.0):.1f}",
                "DELTA-T (K)": f"{p.feature_values.get('brightness_delta', 0.0):.1f}",
                "DIST TO INDUSTRIAL (M)": f"{p.feature_values.get('dist_to_industrial_m', 0.0):.0f}"
            })
        st.dataframe(pd.DataFrame(hr_rows), use_container_width=True, hide_index=True)
    else:
        st.markdown("""
        <div class="glass-card" style="text-align:center; padding:2rem;">
            <div class="kpi-value" style="font-size:1.5rem; color: var(--accent-green);">ALL CLEAR</div>
            <div class="kpi-label" style="margin-top:0.5rem;">ZERO HIGH-RISK THERMAL ANOMALIES ACTIVE</div>
        </div>
        """, unsafe_allow_html=True)
