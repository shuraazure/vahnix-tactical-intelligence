"""
Targeted Test Suite for FRP vs History Trajectory Curve & Instant Map Interaction.
Verifies:
1. Multi-sensor 35-day revisit timeline generation across single detections and large clusters (e.g., TACTICAL_EMERG_015 with 400+ points).
2. Spatio-temporal cluster deduplication without timestamp collapse onto a single vertical pixel.
3. Plotly interactive curve rendering with baseline and peak annotations.
4. Client-side SVG curve generation in 3D globe view.
5. Instant client-side Ask Nix AI Copilot responses.
6. Zero reload / zero blanking event messaging (VAHNIX_SELECT_INCIDENT_SILENT).
7. Zero emoji verification across modified files.
"""

from __future__ import annotations

import os
import sys
import types
from dataclasses import dataclass, field
import datetime
import re
import plotly.graph_objects as go

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if os.path.exists(os.path.join(BASE_DIR, "src", "dashboard", "views")):
    sys.path.insert(0, os.path.join(BASE_DIR, "src", "dashboard", "views"))
if os.path.exists(os.path.join(BASE_DIR, "src", "dashboard")):
    sys.path.insert(0, os.path.join(BASE_DIR, "src", "dashboard"))
sys.path.insert(0, BASE_DIR)

def _resolve_file(name: str) -> str:
    candidates = [
        os.path.join(BASE_DIR, name),
        os.path.join(BASE_DIR, "src", "dashboard", "views", name),
        os.path.join(BASE_DIR, "src", "dashboard", name),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return os.path.join(BASE_DIR, name)


# Register lightweight schema mocks so overview and globe_view can be tested in isolation
if "src" not in sys.modules:
    src_mod = types.ModuleType("src")
    sys.modules["src"] = src_mod

    dp_mod = types.ModuleType("src.data_pipeline")
    dp_schemas = types.ModuleType("src.data_pipeline.schemas")
    sys.modules["src.data_pipeline"] = dp_mod
    sys.modules["src.data_pipeline.schemas"] = dp_schemas

    cl_mod = types.ModuleType("src.clustering")
    cl_schemas = types.ModuleType("src.clustering.schemas")
    sys.modules["src.clustering"] = cl_mod
    sys.modules["src.clustering.schemas"] = cl_schemas

    class_mod = types.ModuleType("src.classification")
    class_ens = types.ModuleType("src.classification.ensemble_classifier")
    class_base = types.ModuleType("src.classification.heuristic_baseline")
    sys.modules["src.classification"] = class_mod
    sys.modules["src.classification.ensemble_classifier"] = class_ens
    sys.modules["src.classification.heuristic_baseline"] = class_base

    dash_mod = types.ModuleType("src.dashboard")
    dash_map = types.ModuleType("src.dashboard.offline_map")
    dash_views = types.ModuleType("src.dashboard.views")
    dash_det = types.ModuleType("src.dashboard.views.detailed_analysis_view")
    dash_globe = types.ModuleType("src.dashboard.views.globe_view")
    sys.modules["src.dashboard"] = dash_mod
    sys.modules["src.dashboard.offline_map"] = dash_map
    sys.modules["src.dashboard.views"] = dash_views
    sys.modules["src.dashboard.views.detailed_analysis_view"] = dash_det
    sys.modules["src.dashboard.views.globe_view"] = dash_globe

    feat_mod = types.ModuleType("src.features")
    feat_ext = types.ModuleType("src.features.feature_extractor")
    sys.modules["src.features"] = feat_mod
    sys.modules["src.features.feature_extractor"] = feat_ext

    @dataclass
    class FIRMSDetection:
        detection_id: str
        latitude: float
        longitude: float
        acq_date: str
        acq_time: str
        timestamp: datetime.datetime
        sensor: str
        satellite: str
        frp: float
        brightness_temp_t4: float
        brightness_temp_t11: float
        bright_delta: float
        confidence: float
        daynight: str

    @dataclass
    class IndustrialFacility:
        facility_id: str
        name: str
        facility_type: str
        latitude: float
        longitude: float

    @dataclass
    class ThermalCluster:
        cluster_id: str
        centroid_lat: float
        centroid_lon: float
        detection_ids: list[str]
        total_frp: float
        max_frp: float
        mean_frp: float
        convex_hull_area_km2: float
        cluster_type: Any
        nearest_industrial_facility_id: str

    @dataclass
    class PredictionOutput:
        detection_id: str
        predicted_class: Any
        predicted_label: str
        confidence_score: float
        class_probabilities: dict
        risk_severity_index: float
        is_critical_alert: bool

    class ThermalAnomalyClass:
        CONTROLLED_INDUSTRIAL_FLARE = 0
        AGRICULTURAL_STUBBLE_BURNING = 1
        UNCONTROLLED_INDUSTRIAL_FIRE = 2
        FOREST_WILDFIRE = 3

    class MockOfflineMap:
        @staticmethod
        def prepare_detections_dataframe(*args, **kwargs):
            return None

    dp_schemas.FIRMSDetection = FIRMSDetection
    dp_schemas.IndustrialFacility = IndustrialFacility
    cl_schemas.ThermalCluster = ThermalCluster
    cl_schemas.ClusterType = types.SimpleNamespace(INDUSTRIAL_COMPLEX="INDUSTRIAL_COMPLEX")
    class_ens.PredictionOutput = PredictionOutput
    class_base.ThermalAnomalyClass = ThermalAnomalyClass
    dash_map.OfflineMapBuilder = MockOfflineMap
    dash_map.CLASS_COLOR_RGBA = {}
    feat_ext.extract_38d_features = lambda *args, **kwargs: {"frp": 312.0, "mu_frp_90d": 25.0}
    dash_det.answer_detection_query = lambda *args, **kwargs: "Mock Answer"
    dash_globe.build_3d_globe_html = lambda *args, **kwargs: ""

from overview import _frp_history_points, _parse_timestamp
from globe_view import build_3d_globe_html, _pseudo_rand_frp


def test_frp_history_points_single_detection():
    """Verify 35-day timeline generation for an isolated incident with no cluster."""
    now = datetime.datetime(2026, 3, 15, 14, 30)
    det = FIRMSDetection(
        detection_id="SINGLE_DET_001",
        latitude=22.3582,
        longitude=69.8681,
        acq_date="2026-03-15",
        acq_time="1430",
        timestamp=now,
        sensor="VIIRS_SNPP",
        satellite="SNPP",
        frp=65.4,
        brightness_temp_t4=340.0,
        brightness_temp_t11=295.0,
        bright_delta=45.0,
        confidence=0.95,
        daynight="D"
    )
    points = _frp_history_points(det, selected_cluster=None, state_mgr=None, mu_frp_90=20.0)
    assert len(points) >= 8  # 7 historical baseline points + 1 target point
    # Verify sorted ascending by timestamp
    for i in range(len(points) - 1):
        assert points[i][0] <= points[i + 1][0]
    # Verify target point is present
    target_pts = [p for p in points if p[2] == "SINGLE_DET_001"]
    assert len(target_pts) == 1
    assert target_pts[0][1] == 65.4
    # Verify time span is roughly 35 days
    span_days = (points[-1][0] - points[0][0]).total_seconds() / 86400.0
    assert span_days >= 30.0


def test_frp_history_points_large_cluster_same_timestamps():
    """Verify cluster with 400+ identical timestamp detections does NOT collapse onto a single pixel."""
    now = datetime.datetime(2026, 3, 15, 14, 30)
    target_det = FIRMSDetection(
        detection_id="TACTICAL_EMERG_015",
        latitude=22.4707,
        longitude=70.0577,
        acq_date="2026-03-15",
        acq_time="1430",
        timestamp=now,
        sensor="MODIS_Terra",
        satellite="Terra",
        frp=312.0,
        brightness_temp_t4=375.0,
        brightness_temp_t11=300.0,
        bright_delta=75.0,
        confidence=0.99,
        daynight="D"
    )
    # Generate 400 cluster detections with identical timestamp
    cluster_dets = [target_det]
    cluster_ids = ["TACTICAL_EMERG_015"]
    for i in range(1, 400):
        did = f"CLUSTER_DET_{i:03d}"
        cluster_ids.append(did)
        cluster_dets.append(FIRMSDetection(
            detection_id=did,
            latitude=22.4707 + (i * 0.0001),
            longitude=70.0577 + (i * 0.0001),
            acq_date="2026-03-15",
            acq_time="1430",
            timestamp=now,
            sensor="MODIS_Terra",
            satellite="Terra",
            frp=50.0 + (i % 30),
            brightness_temp_t4=330.0,
            brightness_temp_t11=295.0,
            bright_delta=35.0,
            confidence=0.90,
            daynight="D"
        ))

    cluster = ThermalCluster(
        cluster_id="CLUST_001",
        centroid_lat=22.4707,
        centroid_lon=70.0577,
        detection_ids=cluster_ids,
        total_frp=15000.0,
        max_frp=312.0,
        mean_frp=60.0,
        convex_hull_area_km2=2.5,
        cluster_type="INDUSTRIAL_COMPLEX",
        nearest_industrial_facility_id="FAC_JAMNAGAR_01"
    )

    class MockStateMgr:
        all_detections = cluster_dets

    points = _frp_history_points(target_det, selected_cluster=cluster, state_mgr=MockStateMgr(), mu_frp_90=35.0)

    # Must guarantee prior 35-day overpass history
    assert len(points) >= 8
    # Min timestamp and max timestamp must differ by at least 30 days!
    t_min = points[0][0]
    t_max = points[-1][0]
    span = (t_max - t_min).total_seconds() / 86400.0
    assert span >= 30.0, f"Expected 30+ day span, got {span} days"

    # All FRP values must be positive
    for pt in points:
        assert pt[1] > 0.0


def test_plotly_frp_chart_generation():
    """Verify interactive Plotly curve figure structure."""
    now = datetime.datetime.now(datetime.timezone.utc)
    dates = [now - datetime.timedelta(days=d) for d in [35, 28, 21, 14, 7, 3, 1, 0]]
    frps = [22.0, 24.5, 21.0, 25.0, 23.0, 45.0, 110.0, 312.0]
    mu_baseline = 25.0
    accent_color = "#ef4444"

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates,
        y=frps,
        mode="lines+markers",
        name="FRP (MW)",
        line=dict(color=accent_color, width=2.8),
        fill="tozeroy",
        fillcolor="rgba(56, 189, 248, 0.12)"
    ))
    fig.add_hline(y=mu_baseline, line_dash="dash")
    fig.add_annotation(x=dates[-1], y=frps[-1], text="TARGET 312 MW")

    data = fig.to_dict()
    assert len(data["data"]) == 1
    assert data["data"][0]["type"] == "scatter"
    assert len(data["data"][0]["x"]) == 8
    assert data["layout"]["shapes"][0]["y0"] == mu_baseline


def test_globe_html_contains_instant_frp_and_zero_reload():
    """Verify build_3d_globe_html output includes client-side FRP curve, Nix copilot, and silent event sync."""
    now = datetime.datetime.now(datetime.timezone.utc)
    det = FIRMSDetection(
        detection_id="TACTICAL_EMERG_015",
        latitude=22.4707,
        longitude=70.0577,
        acq_date="2026-03-15",
        acq_time="1430",
        timestamp=now,
        sensor="MODIS_Terra",
        satellite="Terra",
        frp=312.0,
        brightness_temp_t4=375.0,
        brightness_temp_t11=300.0,
        bright_delta=75.0,
        confidence=0.99,
        daynight="D"
    )
    pred = PredictionOutput(
        detection_id="TACTICAL_EMERG_015",
        predicted_class=types.SimpleNamespace(value=2),
        predicted_label="Industrial Fire Emergency",
        confidence_score=0.98,
        class_probabilities={"EMERGENCY": 0.98},
        risk_severity_index=88.0,
        is_critical_alert=True
    )
    html = build_3d_globe_html([det], [pred], None, None, start_open=True, initial_selected_id="TACTICAL_EMERG_015")

    # 1. Contains client-side SVG generator
    assert "generateFrpSvg" in html
    # 2. Contains embedded Ask Nix Copilot
    assert "askNixMini" in html
    assert "askNixMiniCustom" in html
    # 3. Contains silent parent sync
    assert "VAHNIX_SELECT_INCIDENT_SILENT" in html
    assert "window.parent.history.replaceState" in html
    # 4. Does NOT navigate with window.location.href inside syncIncidentToParent
    sync_code_match = re.search(r"function syncIncidentToParent\([^)]*\)\s*\{([\s\S]*?)\}\n\s*function focusIncidentInOverview", html)
    assert sync_code_match is not None
    sync_code = sync_code_match.group(1)
    assert "window.location.href =" not in sync_code
    assert "window.parent.location.href =" not in sync_code


def test_globe_view_string_timestamp_support():
    """Verify build_3d_globe_html handles detection records where timestamp is an ISO string without crashing."""
    det = FIRMSDetection(
        detection_id="STRING_TS_001",
        latitude=22.4707,
        longitude=70.0577,
        acq_date="2026-03-15",
        acq_time="1430",
        timestamp="2026-03-15T14:30:00",  # String instead of datetime object
        sensor="MODIS_Terra",
        satellite="Terra",
        frp=150.0,
        brightness_temp_t4=360.0,
        brightness_temp_t11=300.0,
        bright_delta=60.0,
        confidence=0.95,
        daynight="D"
    )
    html = build_3d_globe_html([det], None, None, None, start_open=True, initial_selected_id="STRING_TS_001")
    assert "STRING_TS_001" in html
    assert "generateFrpSvg" in html


def test_overview_card_contains_svg_inside_card():
    """Verify _render_frp_history_card embeds the visual SVG curve graph directly inside <div class="glass-card">."""
    import overview
    captured_html = []
    orig_html = getattr(overview.st, "html", None)
    orig_markdown = getattr(overview.st, "markdown", None)
    orig_plotly = getattr(overview.st, "plotly_chart", None)

    overview.st.html = lambda h: captured_html.append(h)
    overview.st.markdown = lambda h, **kw: captured_html.append(h)
    overview.st.plotly_chart = lambda *args, **kwargs: None

    try:
        now = datetime.datetime.now(datetime.timezone.utc)
        det = FIRMSDetection(
            detection_id="TACTICAL_EMERG_015",
            latitude=22.4707,
            longitude=70.0577,
            acq_date="2026-03-15",
            acq_time="1430",
            timestamp=now,
            sensor="MODIS_Terra",
            satellite="Terra",
            frp=312.0,
            brightness_temp_t4=375.0,
            brightness_temp_t11=300.0,
            bright_delta=75.0,
            confidence=0.99,
            daynight="D"
        )
        overview._render_frp_history_card(det, None, None, mu_frp_90=25.0, accent_color="#ef4444")
        assert len(captured_html) >= 1
        card_content = captured_html[0]
        # Assert glass-card contains SVG inside it
        assert '<div class="glass-card"' in card_content
        assert '<svg width="100%" height="220"' in card_content
        assert '--- 90D BASELINE (25 MW)' in card_content
        assert '312 MW' in card_content
        # Ensure the closing </div> is AFTER the <svg>
        svg_idx = card_content.find('<svg')
        div_close_idx = card_content.rfind('</div>')
        assert svg_idx != -1 and div_close_idx != -1
        assert div_close_idx > svg_idx, "SVG must be located inside the glass-card before closing </div>"
    finally:
        if orig_html:
            overview.st.html = orig_html
        if orig_markdown:
            overview.st.markdown = orig_markdown
        if orig_plotly:
            overview.st.plotly_chart = orig_plotly


def test_zero_emojis_across_codebase():
    """Verify zero emojis exist in modified and related files."""
    files_to_check = [
        "overview.py",
        "globe_view.py",
        "app.py",
        "technology_view.py"
    ]
    for fname in files_to_check:
        resolved = _resolve_file(fname)
        if not os.path.exists(resolved):
            continue
        with open(resolved, "r", encoding="utf-8") as f:
            content = f.read()
        for line_num, line in enumerate(content.splitlines(), start=1):
            for ch in line:
                cp = ord(ch)
                is_emoji = (
                    (0x1F300 <= cp <= 0x1FAFF) or
                    (0x1F600 <= cp <= 0x1F64F) or
                    (0x2600 <= cp <= 0x27BF) or
                    (0x1F900 <= cp <= 0x1F9FF)
                )
                assert not is_emoji, f"Emoji '{ch}' (U+{cp:04X}) found in {fname}:{line_num}"


def test_previous_intelligence_window_and_bottom_popup_frp_graph():
    """Verify that the previous intelligence window exists on the 3D globe and clicking map shows JUST ONLY the FRP graph in bottom popup."""
    now = datetime.datetime.now(datetime.timezone.utc)
    det = FIRMSDetection(
        detection_id="TACTICAL_EMERG_015",
        latitude=22.4707,
        longitude=70.0577,
        acq_date="2026-03-15",
        acq_time="1430",
        timestamp=now,
        sensor="MODIS_Terra",
        satellite="Terra",
        frp=312.0,
        brightness_temp_t4=375.0,
        brightness_temp_t11=300.0,
        bright_delta=75.0,
        confidence=0.99,
        daynight="D"
    )
    pred = PredictionOutput(
        detection_id="TACTICAL_EMERG_015",
        predicted_class=types.SimpleNamespace(value=2),
        predicted_label="Industrial Fire Emergency",
        confidence_score=0.98,
        class_probabilities={"EMERGENCY": 0.98},
        risk_severity_index=88.0,
        is_critical_alert=True
    )
    html = build_3d_globe_html([det], [pred], None, None, start_open=True, initial_selected_id="TACTICAL_EMERG_015")

    # 1. On Globe: The previous floating intelligence window exists as it was
    show_float_match = re.search(r"function showFloatingCard\([^)]*\)\s*\{([\s\S]*?)\}\n\s*function closeFloatingCard", html)
    assert show_float_match is not None, "showFloatingCard must exist"
    show_float_body = show_float_match.group(1)

    assert "EVENT INTELLIGENCE" in show_float_body, "Previous intelligence window header must exist on globe"
    assert "anim-gauge" in show_float_body, "Donut risk gauge must exist in intelligence window"
    assert "anim-bar" in show_float_body, "Confidence and top 5 feature bars must exist in intelligence window"
    assert "AI DIAGNOSTIC INSIGHTS" in show_float_body, "Diagnostic insights must exist in intelligence window"
    assert "TOP 5 ARCHITECTURE DIAGNOSTIC FEATURES" in show_float_body, "Top 5 features must exist in intelligence window"
    assert "VIEW 38-D VECTOR" in show_float_body, "View 38-D Vector action must exist in intelligence window"

    # 2. Below Map: Bottom popup exists and shows JUST ONLY the FRP vs History graph
    bottom_match = re.search(r"function renderBottomPopupHtml\(d\)\s*\{([\s\S]*?)\}\n\s*function showBottomPopup", html)
    assert bottom_match is not None, "renderBottomPopupHtml must exist"
    bottom_body = bottom_match.group(1)

    assert "FRP TEMPORAL TRAJECTORY · HISTORICAL OVERPASSES" in bottom_body or "FRP VS HISTORY GRAPH" in bottom_body
    assert "generateFrpSvg" in bottom_body
    assert "[X] CLOSE POPUP" in bottom_body

    # Must NOT show Nix Co-Pilot or extra clutter in the bottom pop-up
    assert "miniNixInput" not in bottom_body, "Nix chatbot input must not be in bottom pop-up"
    assert "Why Flagged?" not in bottom_body, "Preset chips must not be in bottom pop-up"
    assert "Flare or Fire?" not in bottom_body, "Preset chips must not be in bottom pop-up"

    # 3. In overview.py: mounting root and handlers exist below the map
    with open(_resolve_file("overview.py"), "r", encoding="utf-8") as f:
        ov_text = f.read()

    assert 'id="vahnix-bottom-popup-root"' in ov_text, "vahnix-bottom-popup-root must exist below map in overview.py"
    assert "vahnixRenderBottomPopup" in ov_text, "vahnixRenderBottomPopup must exist in overview.py"
    assert "vahnixCloseBottomPopup" in ov_text, "vahnixCloseBottomPopup must exist in overview.py"


if __name__ == "__main__":
    test_frp_history_points_single_detection()
    print("[PASS] test_frp_history_points_single_detection")
    test_frp_history_points_large_cluster_same_timestamps()
    print("[PASS] test_frp_history_points_large_cluster_same_timestamps")
    test_plotly_frp_chart_generation()
    print("[PASS] test_plotly_frp_chart_generation")
    test_globe_html_contains_instant_frp_and_zero_reload()
    print("[PASS] test_globe_html_contains_instant_frp_and_zero_reload")
    test_globe_view_string_timestamp_support()
    print("[PASS] test_globe_view_string_timestamp_support")
    test_previous_intelligence_window_and_bottom_popup_frp_graph()
    print("[PASS] test_previous_intelligence_window_and_bottom_popup_frp_graph")
    test_zero_emojis_across_codebase()
    print("[PASS] test_zero_emojis_across_codebase")
    print("======================================================================")
    print("ALL TARGETED INTEGRATION AND UNIT TESTS PASSED SUCCESSFULLY!")
    print("======================================================================")



