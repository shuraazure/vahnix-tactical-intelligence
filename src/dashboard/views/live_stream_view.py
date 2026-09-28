"""
Thermal Live Incident Stream View for VahniX Tactical Dashboard.
SpaceX-inspired glassmorphism treatment. Dedicated, continuously-refreshable feed of
thermal anomaly detections pulled out of the Tactical Overview registry into its own page,
with per-incident quick-inspect routing back into the Event Intelligence card.
Authoritative Specifications: architecture.md § 4.2, § 4.7, ORIGINAL_REQUEST.md § R3, PROJECT.md § 3
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional, Sequence

import pandas as pd
import streamlit as st

from src.data_pipeline.schemas import FIRMSDetection
from src.clustering.schemas import ThermalCluster
from src.classification.ensemble_classifier import PredictionOutput
from src.dashboard.views.overview import resolve_detection_region


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


def _stream_style(c_name: str, is_crit: bool) -> dict[str, str]:
    """Resolve badge colors / icon / status tag for a class label, matching the
    Tactical Overview Event Intelligence card's color language."""
    if is_crit:
        return {
            "tag": "CRITICAL INDUSTRIAL EMERGENCY",
            "bg": "rgba(239, 68, 68, 0.15)",
            "border": "rgba(239, 68, 68, 0.4)",
            "color": "#ef4444",
            "icon": "[CRITICAL]",
        }
    if "Controlled" in c_name:
        return {
            "tag": "ROUTINE INDUSTRIAL FLARE",
            "bg": "rgba(16, 185, 129, 0.15)",
            "border": "rgba(16, 185, 129, 0.4)",
            "color": "#10b981",
            "icon": "[FLARE]",
        }
    if "Agricultural" in c_name:
        return {
            "tag": "AGRICULTURAL STUBBLE BURNING",
            "bg": "rgba(245, 158, 11, 0.15)",
            "border": "rgba(245, 158, 11, 0.4)",
            "color": "#f59e0b",
            "icon": "[AGRI]",
        }
    return {
        "tag": "NATURAL WILDFIRE BURNING",
        "bg": "rgba(249, 115, 22, 0.15)",
        "border": "rgba(249, 115, 22, 0.4)",
        "color": "#f97316",
        "icon": "[FIRE]",
    }


def _time_ago(ts: datetime, reference: datetime) -> str:
    """Human-readable relative time label vs. the latest signal in the stream."""
    delta = (reference - ts).total_seconds()
    if delta < 0:
        delta = 0
    if delta < 60:
        return "JUST IN"
    if delta < 3600:
        return f"{int(delta // 60)}M AGO"
    if delta < 86400:
        return f"{int(delta // 3600)}H AGO"
    return f"{int(delta // 86400)}D AGO"


def render_live_stream_view(
    detections: Sequence[FIRMSDetection],
    predictions: Sequence[PredictionOutput],
    clusters: Sequence[ThermalCluster],
    kpis: dict[str, Any],
    state_mgr: Optional[Any] = None
) -> None:
    """Render the dedicated Thermal Live Incident Stream page."""

    _render_section_header(
        "THERMAL LIVE INCIDENT STREAM",
        "Continuous orbital feed of thermal anomaly detections · VIIRS & MODIS · sorted by acquisition time"
    )

    if not detections:
        st.markdown("""
        <div class="glass-card" style="text-align:center; padding:3rem;">
            <div class="kpi-label">NO ACTIVE SIGNALS</div>
            <p style="color: var(--text-muted); font-size: 0.85rem; margin-top: 0.5rem;">
                No thermal detections match the current filter criteria.
            </p>
        </div>
        """, unsafe_allow_html=True)
        return

    pred_map = {p.detection_id: p for p in predictions}
    reference_time = max(d.timestamp for d in detections)

    # ── LIVE STATUS STRIP ──
    if "live_stream_last_sync" not in st.session_state:
        st.session_state.live_stream_last_sync = datetime.now()

    sync_col, btn_col = st.columns([5, 1])
    with sync_col:
        st.markdown(f"""
        <div class="glass-card" style="padding:0.9rem 1.4rem; display:flex; align-items:center; gap:0.9rem;">
            <span class="status-pulse-dot"></span>
            <span style="font-family:'JetBrains Mono',monospace; font-size:0.68rem; letter-spacing:0.14em;
                         text-transform:uppercase; color:var(--text-primary); font-weight:600;">STREAM ACTIVE</span>
            <span style="color:var(--text-muted); font-size:0.68rem;">·</span>
            <span style="font-family:'JetBrains Mono',monospace; font-size:0.65rem; letter-spacing:0.08em;
                         color:var(--text-secondary); text-transform:uppercase;">
                {len(detections)} SIGNALS IN FEED · LATEST SYNC {reference_time.strftime('%Y-%m-%d %H:%M')} UTC
            </span>
            <span style="color:var(--text-muted); font-size:0.68rem;">·</span>
            <span style="font-family:'JetBrains Mono',monospace; font-size:0.65rem; letter-spacing:0.08em;
                         color:var(--text-muted); text-transform:uppercase;">
                LAST OPERATOR SYNC {st.session_state.live_stream_last_sync.strftime('%H:%M:%S')} LOCAL
            </span>
        </div>
        """, unsafe_allow_html=True)
    with btn_col:
        if st.button("SYNC NOW", key="live_stream_sync_btn", use_container_width=True):
            st.session_state.live_stream_last_sync = datetime.now()
            st.rerun()

    # ── KPI ROW ──
    crit_count = sum(1 for p in predictions if p.is_critical_alert)
    avg_conf = (sum(p.confidence_score for p in predictions) / len(predictions) * 100.0) if predictions else 0.0
    sensors_online = sorted(set(d.sensor for d in detections))

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        _render_kpi_card("SIGNALS IN STREAM", len(detections), delta_type="neutral")
    with c2:
        _render_kpi_card(
            "CRITICAL IN STREAM",
            crit_count,
            delta="[ALERT] REQUIRES TRIAGE" if crit_count else "NOMINAL",
            delta_type="positive",
            is_critical=crit_count > 0
        )
    with c3:
        _render_kpi_card("AVG CONFIDENCE", f"{avg_conf:.1f}%", delta_type="neutral")
    with c4:
        _render_kpi_card("SENSORS ONLINE", len(sensors_online), delta=", ".join(sensors_online), delta_type="neutral")

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # ── REGION RESOLUTION & LOCATION FILTERING ──
    det_region_cache = {}
    for det in detections:
        det_region_cache[det.detection_id] = resolve_detection_region(det, state_mgr)

    unique_regions = sorted(list(set(r[0] for r in det_region_cache.values() if r[0])))
    region_filter_options = ["All Regions (Nationwide Stream)"] + unique_regions

    corridor_options = [
        "Jamnagar Petrochemical Complex, Gujarat",
        "Trombay Petrochemical Terminal, Mumbai",
        "Simlipal Forest Reserve, Odisha",
        "Punjab Agricultural Fire Belt",
        "Haldia Industrial Belt, West Bengal",
        "Visakhapatnam Steel Zone, AP",
        "Mathura Refinery Corridor, UP",
    ]
    for corr in corridor_options:
        if corr not in region_filter_options:
            region_filter_options.append(corr)

    # 1. Quick Region Navigation Pills
    st.markdown('<div style="font-size:10px; font-weight:800; color:#64748b; letter-spacing:0.1em; text-transform:uppercase; margin-bottom:6px;">QUICK REGION NAVIGATION:</div>', unsafe_allow_html=True)
    pill_cols = st.columns(7)
    quick_pills = [
        ("All", "All"),
        ("Jamnagar", "Jamnagar"),
        ("Trombay", "Maharashtra"),
        ("Simlipal", "Odisha"),
        ("Punjab", "Punjab"),
        ("Haldia", "West Bengal"),
        ("Vizag", "Andhra Pradesh")
    ]
    current_stream_reg = st.session_state.get("live_page_filter_region", "All")
    for i, (pill_lbl, pill_val) in enumerate(quick_pills):
        with pill_cols[i]:
            is_active = (current_stream_reg == pill_val) or (pill_val == "All" and current_stream_reg == "All")
            btn_type = "primary" if is_active else "secondary"
            if st.button(pill_lbl, key=f"live_page_pill_{i}", use_container_width=True, type=btn_type):
                st.session_state["live_page_filter_region"] = pill_val
                st.rerun()

    # 2. Detailed Filter & Feed Controls Row
    ctrl1, ctrl2, ctrl3, ctrl4, ctrl5 = st.columns([2.2, 1.6, 1.2, 1.1, 0.7], gap="small")
    with ctrl1:
        dd_idx = 0
        if current_stream_reg != "All":
            for idx, opt in enumerate(region_filter_options):
                if current_stream_reg.lower() in opt.lower():
                    dd_idx = idx
                    break
        chosen_reg_opt = st.selectbox(
            "Filter by Region / Corridor",
            region_filter_options,
            index=dd_idx,
            key="live_page_region_dropdown",
            label_visibility="collapsed",
            help="Filter the incident feed by State, Union Territory, or Industrial Corridor"
        )
        if chosen_reg_opt.startswith("All Regions"):
            sel_clean = "All"
        else:
            sel_clean = chosen_reg_opt
        
        if sel_clean != current_stream_reg and not any(p[1] == current_stream_reg for p in quick_pills if current_stream_reg != "All" and current_stream_reg in chosen_reg_opt):
            st.session_state["live_page_filter_region"] = sel_clean
            st.rerun()

    with ctrl2:
        search_txt = st.text_input(
            "Search Feed",
            placeholder="Search place, facility, ID...",
            key="live_page_search_input",
            label_visibility="collapsed"
        )

    with ctrl3:
        sort_mode = st.selectbox(
            "Sort Feed By",
            ["Newest First", "Highest Risk First", "Highest FRP First"],
            key="live_stream_sort_mode",
            label_visibility="collapsed"
        )

    with ctrl4:
        feed_limit_label = st.selectbox(
            "Show",
            ["Latest 10", "Latest 25", "Latest 50", "All Signals"],
            index=1,
            key="live_stream_feed_limit",
            label_visibility="collapsed"
        )

    with ctrl5:
        if st.button("Reset", key="live_page_reset_filters_btn", use_container_width=True, help="Reset all filters to All Regions"):
            st.session_state["live_page_filter_region"] = "All"
            st.session_state.pop("live_page_search_input", None)
            st.rerun()

    # 3. Apply Filtering
    active_filter = st.session_state.get("live_page_filter_region", "All")
    filtered_detections = []
    for det in detections:
        reg_st, reg_loc = det_region_cache.get(det.detection_id, ("All India", "—"))
        p = pred_map.get(det.detection_id)
        
        # Region filter
        if active_filter != "All":
            af_low = active_filter.lower()
            matches_reg = (
                af_low in reg_st.lower() or 
                af_low in reg_loc.lower() or
                ("jamnagar" in af_low and "jamnagar" in reg_loc.lower()) or
                ("trombay" in af_low and ("trombay" in reg_loc.lower() or "mumbai" in reg_loc.lower())) or
                ("simlipal" in af_low and "simlipal" in reg_loc.lower()) or
                ("punjab" in af_low and "punjab" in reg_loc.lower()) or
                ("haldia" in af_low and "haldia" in reg_loc.lower()) or
                ("vizag" in af_low and ("vizag" in reg_loc.lower() or "visakhapatnam" in reg_loc.lower())) or
                ("mathura" in af_low and "mathura" in reg_loc.lower())
            )
            if not matches_reg:
                continue

        # Text search filter
        if search_txt and search_txt.strip():
            st_low = search_txt.strip().lower()
            matches_txt = (
                st_low in det.detection_id.lower() or
                st_low in reg_st.lower() or
                st_low in reg_loc.lower() or
                (p and st_low in p.predicted_label.lower())
            )
            if not matches_txt:
                continue

        filtered_detections.append(det)

    # 4. Regional Radar Status Ribbon
    display_reg = "Nationwide (All India)" if active_filter == "All" else active_filter
    crit_in_filtered = sum(1 for d in filtered_detections if pred_map.get(d.detection_id) and pred_map[d.detection_id].is_critical_alert)
    st.markdown(
        f'<div style="font-size:11px; color:#38bdf8; margin:8px 0 14px 0; padding:8px 14px; '
        f'background:rgba(56,189,248,0.06); border:1px solid rgba(56,189,248,0.2); border-radius:6px; display:flex; justify-content:space-between; align-items:center;">'
        f'<span>REGION RADAR: <b>{display_reg}</b> &nbsp;·&nbsp; <b style="color:#ffffff;">{len(filtered_detections)}</b> signals matching filter</span>'
        f'<span style="color:{"#ef4444" if crit_in_filtered else "#10b981"}; font-weight:700;">'
        f'{"[CRITICAL] " + str(crit_in_filtered) + " CRITICAL ALERTS" if crit_in_filtered else "[NOMINAL] ALL SIGNALS NOMINAL"}'
        f'</span></div>',
        unsafe_allow_html=True
    )

    # 5. Sorting & Slicing
    def _risk_key(d: FIRMSDetection) -> float:
        p = pred_map.get(d.detection_id)
        return p.risk_severity_index if p else 0.0

    if sort_mode == "Highest Risk First":
        ordered = sorted(filtered_detections, key=_risk_key, reverse=True)
    elif sort_mode == "Highest FRP First":
        ordered = sorted(filtered_detections, key=lambda d: d.frp, reverse=True)
    else:
        ordered = sorted(filtered_detections, key=lambda d: d.timestamp, reverse=True)

    limit_map = {"Latest 10": 10, "Latest 25": 25, "Latest 50": 50, "All Signals": len(ordered)}
    feed_slice = ordered[: limit_map[feed_limit_label]]

    if not feed_slice:
        st.markdown(f"""
        <div class="glass-card" style="text-align:center; padding:2.5rem; margin:1rem 0;">
            <div class="kpi-label">NO SIGNALS IN {display_reg.upper()}</div>
            <p style="color: var(--text-muted); font-size: 0.85rem; margin-top: 0.5rem;">
                No active thermal detections match the selected region or search query.
            </p>
        </div>
        """, unsafe_allow_html=True)

    # ── LIVE FEED CARDS ──
    for det in feed_slice:
        p = pred_map.get(det.detection_id)
        c_name = p.predicted_label if p else "Unclassified"
        is_crit = p.is_critical_alert if p else False
        conf_pct = (p.confidence_score * 100.0) if p else (det.confidence * 100.0)
        rsi = p.risk_severity_index if p else 0.0
        style = _stream_style(c_name, is_crit)
        ago = _time_ago(det.timestamp, reference_time)
        dn_label = "DAY" if det.daynight == "D" else "NIGHT"
        is_focused = det.detection_id == st.session_state.get("selected_detection_id")
        focus_ring = "border:1px solid rgba(255,255,255,0.35); box-shadow:0 0 16px rgba(255,255,255,0.06);" if is_focused else ""
        reg_st, reg_loc = det_region_cache.get(det.detection_id, ("All India", "—"))

        row_card, row_btn = st.columns([7.0, 1.5], gap="small")
        with row_card:
            st.markdown(f"""
            <div class="glass-card" style="padding:0.85rem 1.2rem; margin-bottom:0.55rem; {focus_ring}">
                <div style="display:flex; align-items:center; justify-content:space-between; gap:1rem; flex-wrap:wrap;">
                    <div style="display:flex; align-items:center; gap:0.7rem; min-width:0; flex-wrap:wrap;">
                        <span style="width:8px; height:8px; border-radius:50%; background:{style['color']};
                                     box-shadow:0 0 8px {style['color']}; flex-shrink:0;
                                     animation: status-pulse 2s infinite;"></span>
                        <span style="font-family:'JetBrains Mono',monospace; font-size:0.8rem; font-weight:700;
                                     color:var(--text-primary); white-space:nowrap;">{det.detection_id}</span>
                        <span style="background:{style['bg']}; border:1px solid {style['border']}; color:{style['color']};
                                     font-size:0.6rem; font-weight:800; letter-spacing:0.08em; padding:2px 8px;
                                     border-radius:4px; text-transform:uppercase; white-space:nowrap;">
                            {style['icon']} {c_name}
                        </span>
                        <span style="background:rgba(56,189,248,0.1); border:1px solid rgba(56,189,248,0.25); color:#38bdf8;
                                     font-size:0.6rem; font-weight:700; letter-spacing:0.06em; padding:2px 8px;
                                     border-radius:4px; text-transform:uppercase; white-space:nowrap;">
                            {reg_st} · {reg_loc}
                        </span>
                    </div>
                    <span style="font-family:'JetBrains Mono',monospace; font-size:0.62rem; letter-spacing:0.1em;
                                 color:var(--text-muted); text-transform:uppercase; white-space:nowrap;">{ago}</span>
                </div>
                <div style="display:flex; gap:1.6rem; margin-top:0.55rem; flex-wrap:wrap;
                            font-family:'JetBrains Mono',monospace; font-size:0.66rem; color:var(--text-secondary);">
                    <span>[{dn_label}] {det.sensor} ({det.satellite})</span>
                    <span>FRP <b style="color:var(--text-primary);">{det.frp:.1f} MW</b></span>
                    <span>CONF <b style="color:var(--text-primary);">{conf_pct:.1f}%</b></span>
                    <span>RISK <b style="color:{style['color']};">{rsi:.0f}/100</b></span>
                    <span>{det.latitude:.4f}, {det.longitude:.4f}</span>
                    <span>{det.acq_date} {det.acq_time[:2]}:{det.acq_time[2:]} UTC</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
        with row_btn:
            st.markdown('<div style="height:0.3rem;"></div>', unsafe_allow_html=True)
            if st.button("Inspect Map →", key=f"stream_inspect_map_{det.detection_id}", use_container_width=True, help="Center 3D Globe map on this incident in Tactical Overview"):
                st.session_state.selected_detection_id = det.detection_id
                st.session_state["map_location_override"] = {
                    "lat": float(det.latitude),
                    "lon": float(det.longitude),
                    "zoom": 12,
                    "label": f"{det.detection_id} ({reg_loc})"
                }
                st.query_params["page"] = "overview"
                st.rerun()
            if st.button("38-D Vector →", key=f"stream_audit_38d_{det.detection_id}", use_container_width=True, help="View 38-D feature vector breakdown for this incident"):
                st.session_state.selected_detection_id = det.detection_id
                st.query_params["page"] = "analysis"
                st.query_params["id"] = det.detection_id
                st.rerun()

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # ── FULL STREAM REGISTRY (sortable / searchable table) ──
    _render_section_header(
        "STREAM REGISTRY",
        "Full tabular export of every signal currently in the live stream"
    )

    reg_rows = []
    for det in ordered:
        p = pred_map.get(det.detection_id)
        reg_st, reg_loc = det_region_cache.get(det.detection_id, ("All India", "—"))
        reg_rows.append({
            "DETECTION ID": det.detection_id,
            "REGION / STATE": reg_st,
            "LOCATION / FACILITY": reg_loc,
            "TIME (UTC)": f"{det.acq_date} {det.acq_time[:2]}:{det.acq_time[2:]}",
            "SENSOR": det.sensor,
            "SATELLITE": det.satellite,
            "D/N": det.daynight,
            "CLASS": p.predicted_label if p else "Unclassified",
            "CONFIDENCE": f"{(p.confidence_score * 100.0) if p else det.confidence * 100.0:.1f}%",
            "RISK SCORE": f"{p.risk_severity_index:.1f}" if p else "—",
            "ALERT LEVEL": "[ALERT] CRITICAL" if (p and p.is_critical_alert) else "NORMAL",
            "FRP (MW)": f"{det.frp:.1f}",
            "ΔT (K)": f"{det.bright_delta:.1f}",
            "COORDINATES": f"{det.latitude:.4f}, {det.longitude:.4f}",
        })
    st.dataframe(pd.DataFrame(reg_rows), use_container_width=True, hide_index=True, height=420)
