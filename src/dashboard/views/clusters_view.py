"""
Spatio-Temporal Clusters View for VahniX Tactical Dashboard.
SpaceX-inspired dark glassmorphism treatment.
Renders cluster persistence analytics, recurrence histograms, DNBI distributions, and cluster drill-down.
Authoritative Specifications: ORIGINAL_REQUEST.md § R1, PROJECT.md § 3
"""

from __future__ import annotations

from typing import Sequence
import streamlit as st
import pandas as pd
import numpy as np

from src.clustering.schemas import ThermalCluster


def _render_kpi_card(label: str, value, delta: str = "", delta_type: str = "neutral") -> None:
    """Render a glassmorphism KPI metric card."""
    delta_html = f'<div class="kpi-delta {delta_type}">{delta}</div>' if delta else ""
    html = f"""
    <div class="glass-card" style="height:100%;">
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


def render_clusters_view(clusters: Sequence[ThermalCluster]) -> None:
    """Render Spatio-Temporal Clusters analytics view with SpaceX styling."""

    _render_section_header(
        "SPATIO-TEMPORAL CLUSTERING",
        "Recurrence analysis & persistence profiling"
    )

    if not clusters:
        st.markdown("""
        <div class="glass-card" style="text-align:center; padding:3rem;">
            <div class="kpi-label">NO CLUSTERS DETECTED</div>
            <p style="color: var(--text-muted); font-size: 0.85rem; margin-top: 0.5rem;">
                No spatio-temporal clusters available for current filter criteria.
            </p>
        </div>
        """, unsafe_allow_html=True)
        return

    # ── KPI ROW ──
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        _render_kpi_card("TOTAL CLUSTERS", len(clusters))
    with c2:
        stationary_count = sum(1 for c in clusters if c.cluster_type == "stationary_persistent")
        _render_kpi_card("STATIONARY PERSISTENT", stationary_count)
    with c3:
        inside_fac = sum(1 for c in clusters if c.is_inside_industrial_boundary)
        _render_kpi_card("INSIDE FACILITIES", inside_fac)
    with c4:
        avg_span = np.mean([c.temporal_span_days for c in clusters])
        _render_kpi_card("MEAN SPAN", f"{avg_span:.1f}", delta="DAYS", delta_type="neutral")

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # ── CLUSTER INVENTORY TABLE ──
    _render_section_header(
        "CLUSTER INVENTORY",
        "Persistence profile & facility proximity analysis"
    )

    rows = []
    for c in clusters:
        rows.append({
            "CLUSTER ID": c.cluster_id,
            "TYPE": c.cluster_type.upper().replace("_", " "),
            "DETECTIONS": c.total_detections,
            "RECURRENCE": f"{c.recurrence_rate * 100:.1f}%",
            "DNBI": f"{c.day_night_balance_index:.2f}",
            "SPATIAL JITTER (M)": f"{c.spatial_jitter_m:.1f}",
            "MEAN FRP (MW)": f"{c.mean_frp:.1f}",
            "MAX FRP (MW)": f"{c.max_frp:.1f}",
            "FACILITY PROXIMITY": "INSIDE" if c.is_inside_industrial_boundary else "OUTSIDE",
            "NEAREST FACILITY": c.nearest_industrial_facility_id or "—"
        })
    df_clus = pd.DataFrame(rows)
    st.dataframe(df_clus, use_container_width=True, hide_index=True)

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # ── CLUSTER DRILL-DOWN ──
    _render_section_header(
        "CLUSTER DETAIL INSPECTOR",
        "Select a cluster for full tactical intelligence"
    )

    selected_id = st.selectbox(
        "Select Cluster",
        [c.cluster_id for c in clusters],
        label_visibility="collapsed"
    )
    target_cluster = next((c for c in clusters if c.cluster_id == selected_id), None)

    if target_cluster:
        tc = target_cluster

        # ── Human-readable timestamps ──
        def _fmt_dt(dt):
            d = str(dt.day)
            return f"{dt.strftime('%a')}, {d} {dt.strftime('%b %Y')} at {dt.strftime('%I:%M %p').lstrip('0') or '12:00 AM'} UTC"

        first_str = _fmt_dt(tc.first_seen)
        last_str = _fmt_dt(tc.last_seen)

        # ── Cluster type badge styling ──
        type_label = tc.cluster_type.upper().replace("_", " ")
        if tc.cluster_type == "stationary_persistent":
            type_bg = "rgba(239, 68, 68, 0.15)"
            type_border = "rgba(239, 68, 68, 0.4)"
            type_color = "#ef4444"
            type_icon = "[FACILITY]"
        elif tc.cluster_type == "dynamic_front":
            type_bg = "rgba(249, 115, 22, 0.15)"
            type_border = "rgba(249, 115, 22, 0.4)"
            type_color = "#f97316"
            type_icon = "[FIRE]"
        else:
            type_bg = "rgba(100, 116, 139, 0.15)"
            type_border = "rgba(100, 116, 139, 0.4)"
            type_color = "#94a3b8"
            type_icon = "[SURGE]"

        # ── Facility proximity ──
        if tc.is_inside_industrial_boundary:
            fac_badge = '<span style="background:rgba(239,68,68,0.15); border:1px solid rgba(239,68,68,0.4); color:#ef4444; padding:2px 8px; border-radius:4px; font-size:10px; font-weight:800;">[ALERT] INSIDE BOUNDARY</span>'
            fac_dist_display = "0 m (Inside facility)"
        else:
            d_m = tc.distance_to_nearest_industrial_m
            if d_m < 1000:
                fac_dist_display = f"{d_m:.0f} m away"
            else:
                fac_dist_display = f"{d_m / 1000:.1f} km away"
            fac_badge = '<span style="background:rgba(16,185,129,0.15); border:1px solid rgba(16,185,129,0.4); color:#10b981; padding:2px 8px; border-radius:4px; font-size:10px; font-weight:800;">OUTSIDE</span>'

        fac_name = tc.nearest_industrial_facility_id or "No facility linked"

        # ── Day/Night balance visual ──
        total_dn = max(tc.day_count + tc.night_count, 1)
        day_pct = (tc.day_count / total_dn) * 100
        night_pct = 100 - day_pct

        # ── Recurrence rate color ──
        rec_pct = tc.recurrence_rate * 100
        rec_color = "#ef4444" if rec_pct >= 70 else ("#f59e0b" if rec_pct >= 40 else "#10b981")

        # ── Bounding box as coverage area ──
        bb = tc.bounding_box
        lat_span = abs(bb[3] - bb[1]) * 111.0  # rough km
        lon_span = abs(bb[2] - bb[0]) * 111.0 * 0.85  # rough km at mid-lat
        area_desc = f"~{lat_span:.1f} km × {lon_span:.1f} km"

        # ── Render structured UI ──
        st.html(f"""<div style="background:#0b111e; border:1px solid #1e293b; border-radius:12px; padding:22px 24px; font-family:-apple-system,BlinkMacSystemFont,'Inter','Segoe UI',sans-serif; color:#f8fafc; box-shadow:0 12px 32px -4px rgba(0,0,0,0.6);">

    <!-- Header: Cluster ID + Type Badge -->
    <div style="display:flex; justify-content:space-between; align-items:flex-start; margin-bottom:16px;">
        <div>
            <div style="font-size:10px; color:#64748b; font-weight:700; letter-spacing:0.16em; text-transform:uppercase;">CLUSTER INTELLIGENCE</div>
            <div style="font-family:'Space Grotesk',sans-serif; font-size:1.3rem; font-weight:800; color:#ffffff; letter-spacing:0.02em; margin-top:2px;">{tc.cluster_id}</div>
            <div style="font-size:11px; color:#64748b; margin-top:4px;">First seen: {first_str}</div>
            <div style="font-size:11px; color:#64748b; margin-top:1px;">Last seen: {last_str}</div>
        </div>
        <div>
            <span style="background:{type_bg}; border:1px solid {type_border}; color:{type_color}; font-size:10px; font-weight:800; letter-spacing:0.1em; padding:5px 12px; border-radius:4px;">{type_icon} {type_label}</span>
        </div>
    </div>

    <!-- Quick Stats Row: 4 KPI cards -->
    <div style="display:grid; grid-template-columns:1fr 1fr 1fr 1fr; gap:10px; margin-bottom:14px;">
        <div style="background:#080d17; border:1px solid #172338; border-radius:8px; padding:12px; text-align:center;">
            <div style="font-size:9px; color:#64748b; font-weight:700; letter-spacing:0.12em;">TOTAL DETECTIONS</div>
            <div style="font-size:22px; font-weight:800; color:#38bdf8; margin-top:4px;">{tc.total_detections}</div>
            <div style="font-size:9px; color:#475569; margin-top:2px;">satellite passes</div>
        </div>
        <div style="background:#080d17; border:1px solid #172338; border-radius:8px; padding:12px; text-align:center;">
            <div style="font-size:9px; color:#64748b; font-weight:700; letter-spacing:0.12em;">ACTIVE DAYS</div>
            <div style="font-size:22px; font-weight:800; color:#f59e0b; margin-top:4px;">{tc.active_days_count}</div>
            <div style="font-size:9px; color:#475569; margin-top:2px;">of {tc.temporal_span_days:.0f} day span</div>
        </div>
        <div style="background:#080d17; border:1px solid #172338; border-radius:8px; padding:12px; text-align:center;">
            <div style="font-size:9px; color:#64748b; font-weight:700; letter-spacing:0.12em;">RECURRENCE</div>
            <div style="font-size:22px; font-weight:800; color:{rec_color}; margin-top:4px;">{rec_pct:.0f}%</div>
            <div style="font-size:9px; color:#475569; margin-top:2px;">repeat rate</div>
        </div>
        <div style="background:#080d17; border:1px solid #172338; border-radius:8px; padding:12px; text-align:center;">
            <div style="font-size:9px; color:#64748b; font-weight:700; letter-spacing:0.12em;">SPATIAL JITTER</div>
            <div style="font-size:22px; font-weight:800; color:#a78bfa; margin-top:4px;">{tc.spatial_jitter_m:.0f}</div>
            <div style="font-size:9px; color:#475569; margin-top:2px;">meters (1σ)</div>
        </div>
    </div>

    <!-- Thermal Profile -->
    <div style="background:#080d17; border:1px solid #172338; border-radius:8px; padding:14px 16px; margin-bottom:14px;">
        <div style="font-size:10px; color:#64748b; font-weight:700; letter-spacing:0.14em; margin-bottom:10px;">[THERMAL] THERMAL PROFILE (FIRE RADIATIVE POWER)</div>
        <div style="display:grid; grid-template-columns:1fr 1fr 1fr; gap:12px;">
            <div>
                <div style="font-size:9px; color:#475569; font-weight:600;">AVERAGE FRP</div>
                <div style="font-size:18px; font-weight:800; color:#f97316; margin-top:2px;">{tc.mean_frp:.1f} <span style="font-size:11px; color:#64748b;">MW</span></div>
            </div>
            <div>
                <div style="font-size:9px; color:#475569; font-weight:600;">PEAK FRP</div>
                <div style="font-size:18px; font-weight:800; color:#ef4444; margin-top:2px;">{tc.max_frp:.1f} <span style="font-size:11px; color:#64748b;">MW</span></div>
            </div>
            <div>
                <div style="font-size:9px; color:#475569; font-weight:600;">VARIABILITY</div>
                <div style="font-size:18px; font-weight:800; color:#94a3b8; margin-top:2px;">±{tc.frp_std:.1f} <span style="font-size:11px; color:#64748b;">MW</span></div>
            </div>
        </div>
    </div>

    <!-- Facility Proximity -->
    <div style="background:#080d17; border:1px solid #172338; border-radius:8px; padding:14px 16px; margin-bottom:14px;">
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
            <div style="font-size:10px; color:#64748b; font-weight:700; letter-spacing:0.14em;">[ASSET] NEAREST INDUSTRIAL FACILITY</div>
            {fac_badge}
        </div>
        <div style="font-size:13px; font-weight:700; color:#38bdf8;">{fac_name}</div>
        <div style="font-size:11px; color:#64748b; margin-top:2px;">{fac_dist_display}</div>
    </div>

    <!-- Day/Night Balance -->
    <div style="background:#080d17; border:1px solid #172338; border-radius:8px; padding:14px 16px; margin-bottom:14px;">
        <div style="font-size:10px; color:#64748b; font-weight:700; letter-spacing:0.14em; margin-bottom:8px;">[BALANCE] DAY / NIGHT DETECTION BALANCE</div>
        <div style="display:flex; gap:6px; align-items:center; margin-bottom:6px;">
            <span style="font-size:10px; color:#f59e0b; font-weight:700;">DAY: {tc.day_count}</span>
            <div style="flex:1; height:8px; border-radius:4px; overflow:hidden; background:#172338; display:flex;">
                <div style="width:{day_pct:.1f}%; background:linear-gradient(90deg, #f59e0b, #fbbf24); border-radius:4px 0 0 4px;"></div>
                <div style="width:{night_pct:.1f}%; background:linear-gradient(90deg, #3b82f6, #6366f1); border-radius:0 4px 4px 0;"></div>
            </div>
            <span style="font-size:10px; color:#6366f1; font-weight:700;">NIGHT: {tc.night_count}</span>
        </div>
        <div style="font-size:10px; color:#64748b;">Balance Index (DNBI): <span style="color:#fff; font-weight:700;">{tc.day_night_balance_index:.2f}</span> <span style="font-size:9px; color:#475569;">({('Balanced — active day & night' if tc.day_night_balance_index > 0.6 else ('Mostly daytime' if tc.day_count > tc.night_count else 'Mostly nighttime'))})</span></div>
    </div>

    <!-- Location Summary -->
    <div style="background:#080d17; border:1px solid #172338; border-radius:8px; padding:14px 16px;">
        <div style="font-size:10px; color:#64748b; font-weight:700; letter-spacing:0.14em; margin-bottom:8px;">[LOCATION] LOCATION SUMMARY</div>
        <div style="display:grid; grid-template-columns:1fr 1fr; gap:10px;">
            <div>
                <div style="font-size:9px; color:#475569; font-weight:600;">CENTER COORDINATES</div>
                <div style="font-size:12px; font-weight:700; color:#fff; margin-top:2px;">{tc.centroid_lat:.4f}° N, {tc.centroid_lon:.4f}° E</div>
            </div>
            <div>
                <div style="font-size:9px; color:#475569; font-weight:600;">COVERAGE AREA</div>
                <div style="font-size:12px; font-weight:700; color:#fff; margin-top:2px;">{area_desc}</div>
            </div>
        </div>
    </div>

</div>""")
