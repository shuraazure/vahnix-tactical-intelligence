"""
Explainability (SHAP XAI) & Tactical Intelligence View for VahniX Dashboard.
SpaceX-inspired dark glassmorphism treatment.
Renders local SHAP waterfall feature attribution, intelligence narratives, and parameter breakdowns.
Authoritative Specifications: ORIGINAL_REQUEST.md § R2, PROJECT.md § 3
"""

from __future__ import annotations

from typing import Any, Sequence
import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from src.classification.ensemble_classifier import PredictionOutput
from src.dashboard.state_manager import DashboardStateManager


def _render_kpi_card(label: str, value, delta: str = "", delta_type: str = "neutral",
                     is_critical: bool = False) -> None:
    """Render a glassmorphism KPI metric card."""
    crit_class = " critical" if is_critical else ""
    delta_html = f'<div class="kpi-delta {delta_type}">{delta}</div>' if delta else ""
    html = f"""
    <div class="glass-card{crit_class}" style="height:100%;">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        {delta_html}
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def _render_section_header(title: str, subtitle: str = "") -> None:
    """Render a SpaceX-style section header."""
    html = f'<h2 class="section-header">{title}</h2>'
    if subtitle:
        html += f'<p class="section-subtitle">{subtitle}</p>'
    st.markdown(html, unsafe_allow_html=True)


def _shap_bar_colour(value: float, magnitude: float) -> str:
    """Warm palette for positive contributors, cool palette for negative ones."""
    if value >= 0:
        stops = [(245, 158, 11), (231, 76, 60)]     # amber → ember red
    else:
        stops = [(125, 211, 252), (59, 130, 246)]   # ice → deep blue
    (r1, g1, b1), (r2, g2, b2) = stops
    t = max(0.0, min(1.0, magnitude))
    r = round(r1 + (r2 - r1) * t)
    g = round(g1 + (g2 - g1) * t)
    b = round(b1 + (b2 - b1) * t)
    return f"rgb({r},{g},{b})"


def _render_shap_chart(exp: Any, detection_id: str, target_label: str) -> bool:
    """Render SHAP attribution as a transparent, dark-theme horizontal bar chart.

    Uses the same feature/SHAP/observed values as the static plot. Returns False
    if the explanation carries no feature lists, so the caller can fall back.
    """
    rows = list(getattr(exp, "top_positive_features", []) or []) + \
           list(getattr(exp, "top_negative_features", []) or [])
    if not rows:
        return False

    # Strongest absolute contribution at the top of the chart.
    rows.sort(key=lambda item: abs(item[1]))
    peak = max(abs(item[1]) for item in rows) or 1.0

    labels = [f"{item[0]}  =  {item[2]:.2f}" for item in rows]
    values = [item[1] for item in rows]
    colours = [_shap_bar_colour(v, abs(v) / peak) for v in values]
    texts = [f"{v:+.4f}" for v in values]

    fig = go.Figure(
        go.Bar(
            x=values,
            y=labels,
            orientation="h",
            marker=dict(color=colours, line=dict(width=0)),
            text=texts,
            textposition="outside",
            textfont=dict(family="JetBrains Mono, monospace", size=12,
                          color="rgba(245,247,250,0.85)"),
            hovertemplate="<b>%{y}</b><br>SHAP  %{x:+.4f}<extra></extra>",
            cliponaxis=False,
        )
    )

    span = max(abs(v) for v in values)
    lo = min(0.0, min(values)) - span * 0.22
    hi = max(0.0, max(values)) + span * 0.22

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=max(300, 74 * len(rows) + 130),
        margin=dict(l=10, r=30, t=74, b=54),
        bargap=0.42,
        showlegend=False,
        title=dict(
            text=(f"<span style='color:#F5F7FA'>{detection_id}</span>"
                  f"<br><span style='font-size:12px;color:rgba(245,247,250,0.45);"
                  f"letter-spacing:0.14em'>TARGET · {target_label.upper()}</span>"),
            font=dict(family="Space Grotesk, sans-serif", size=17,
                      color="#F5F7FA"),
            x=0, xanchor="left", y=0.97,
        ),
        xaxis=dict(
            title=dict(
                text="SHAP VALUE · IMPACT ON MODEL LOG-ODDS",
                font=dict(family="JetBrains Mono, monospace", size=10,
                          color="rgba(245,247,250,0.40)"),
            ),
            range=[lo, hi],
            showgrid=True,
            gridcolor="rgba(255,255,255,0.05)",
            gridwidth=1,
            zeroline=True,
            zerolinecolor="rgba(255,255,255,0.22)",
            zerolinewidth=1,
            showline=False,
            tickfont=dict(family="JetBrains Mono, monospace", size=11,
                          color="rgba(245,247,250,0.45)"),
        ),
        yaxis=dict(
            showgrid=False,
            zeroline=False,
            showline=False,
            ticksuffix="   ",
            tickfont=dict(family="JetBrains Mono, monospace", size=12,
                          color="rgba(245,247,250,0.78)"),
        ),
        hoverlabel=dict(
            bgcolor="#0B111E",
            bordercolor="rgba(255,255,255,0.12)",
            font=dict(family="JetBrains Mono, monospace", size=12,
                      color="#F5F7FA"),
        ),
    )

    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    st.markdown(
        '<p style="font-family:var(--font-mono); font-size:0.68rem; '
        'letter-spacing:0.14em; color:var(--text-muted); text-align:center; '
        'margin-top:-0.4rem;">'
        f'LOCAL SHAP WATERFALL ATTRIBUTION — {detection_id}</p>',
        unsafe_allow_html=True,
    )
    return True


def render_explainability_view(
    state_manager: DashboardStateManager,
    predictions: Sequence[PredictionOutput]
) -> None:
    """Render SHAP XAI and Intelligence Narrative view with SpaceX styling."""

    _render_section_header(
        "EXPLAINABLE AI",
        "SHAP-powered feature attribution & tactical intelligence narratives"
    )

    if not predictions:
        st.markdown("""
        <div class="glass-card" style="text-align:center; padding:3rem;">
            <div class="kpi-label">NO ACTIVE INCIDENTS</div>
            <p style="color: var(--text-muted); font-size: 0.85rem; margin-top: 0.5rem;">
                No active incidents available for explainability analysis.
            </p>
        </div>
        """, unsafe_allow_html=True)
        return

    # ── INCIDENT SELECTOR ──
    incident_ids = [p.detection_id for p in predictions]
    target_id_param = st.query_params.get("id") or st.session_state.get("selected_detection_id")
    default_idx = incident_ids.index(target_id_param) if target_id_param in incident_ids else 0

    col_sel, col_back = st.columns([3.6, 1.2], gap="small")
    with col_sel:
        selected_id = st.selectbox(
            "Select Thermal Incident to Inspect",
            incident_ids,
            index=default_idx,
            label_visibility="collapsed"
        )
    with col_back:
        if st.button("← 38-D Analysis", key="shap_back_to_38d", use_container_width=True, help=f"Return to 38-D Feature Vector Analysis for {selected_id}"):
            st.query_params["page"] = "analysis"
            st.query_params["id"] = selected_id
            st.session_state.selected_detection_id = selected_id
            st.rerun()

    # Sync selection to session state
    st.session_state.selected_detection_id = selected_id

    target_pred = next((p for p in predictions if p.detection_id == selected_id), None)
    if not target_pred:
        return

    # Fetch or compute explanation
    exp = state_manager.get_or_create_explanation(selected_id)

    # ── INCIDENT METRICS ──
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        _render_kpi_card("PREDICTED CATEGORY", target_pred.predicted_label)
    with c2:
        _render_kpi_card(
            "CONFIDENCE SCORE",
            f"{target_pred.confidence_score * 100:.1f}%"
        )
    with c3:
        _render_kpi_card(
            "RISK SEVERITY",
            f"{target_pred.risk_severity_index:.1f}",
            delta="/ 100",
            delta_type="neutral"
        )
    with c4:
        is_crit = target_pred.is_critical_alert
        _render_kpi_card(
            "ALERT STATUS",
            "CRITICAL" if is_crit else "ROUTINE",
            delta="IMMEDIATE ACTION REQUIRED" if is_crit else "MONITORING",
            delta_type="positive" if is_crit else "green",
            is_critical=is_crit
        )

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # ── OPERATIONAL INTELLIGENCE NARRATIVE ──
    _render_section_header(
        "OPERATIONAL INTELLIGENCE SUMMARY",
        "AI-generated tactical assessment narrative"
    )

    if exp and exp.natural_language_summary:
        st.markdown(f"""
        <div class="glass-card" style="border-left: 3px solid var(--accent-blue);">
            <div class="kpi-label" style="margin-bottom: 1rem;">INTELLIGENCE BRIEF</div>
            <p style="color: var(--text-secondary); font-size: 0.9rem; line-height: 1.6;">
                {exp.natural_language_summary}
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # ── SHAP WATERFALL PLOT ──
    if exp:
        _render_section_header(
            "SHAP FEATURE ATTRIBUTION",
            "TreeExplainer local waterfall decomposition"
        )
        rendered = _render_shap_chart(exp, selected_id, target_pred.predicted_label)
        if not rendered and exp.waterfall_plot_base64:
            # Fallback to the pre-rendered static plot.
            st.image(
                f"data:image/png;base64,{exp.waterfall_plot_base64}",
                caption=f"Local SHAP Waterfall Attribution — {selected_id}",
                width="stretch"
            )

        st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # ── TOP FEATURE DRIVERS ──
    if exp:
        _render_section_header(
            "TOP FEATURE DRIVERS",
            "Positive & negative SHAP value contributors"
        )

        pos_df = pd.DataFrame([
            {
                "FEATURE": item[0],
                "SHAP VALUE": f"+{item[1]:.4f}",
                "OBSERVED": f"{item[2]:.2f}",
                "DIRECTION": "↑ INCREASES CLASS PROBABILITY"
            }
            for item in exp.top_positive_features
        ])
        neg_df = pd.DataFrame([
            {
                "FEATURE": item[0],
                "SHAP VALUE": f"{item[1]:.4f}",
                "OBSERVED": f"{item[2]:.2f}",
                "DIRECTION": "↓ DECREASES CLASS PROBABILITY"
            }
            for item in exp.top_negative_features
        ])
        combined_df = pd.concat([pos_df, neg_df], ignore_index=True)
        st.dataframe(combined_df, use_container_width=True, hide_index=True)
