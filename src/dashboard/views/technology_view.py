"""
Technology & Architecture View for VahniX Tactical Dashboard.
Integrated multi-source physical filtering, thermal fingerprinting,
SHAP explainability, gradient-boosted tree ensemble mathworks, and
Tactical Teamwork & Command Center (C2) Interoperability Preview.
Authoritative Specifications: architecture.md § 1-4, PROJECT.md § 3, ORIGINAL_REQUEST.md § R1-R3
"""

from __future__ import annotations

import os
import re
from typing import Any, Optional
import streamlit as st
import streamlit.components.v1 as components


TAB_ALIASES: dict[str, str] = {
    "pipeline": "pipeline",
    "goal": "pipeline",
    "goals": "pipeline",
    "data-pipeline": "pipeline",
    "fingerprint": "fingerprint",
    "thermal": "fingerprint",
    "thermal-fingerprint": "fingerprint",
    "explain": "explain",
    "explainability": "explain",
    "shap": "explain",
    "math": "math",
    "mathworks": "math",
    "boost": "math",
    "boosting": "math",
    "ensemble": "math",
    "teamwork": "teamwork",
    "teamwork-preview": "teamwork",
    "team": "teamwork",
    "c2": "teamwork",
    "command": "teamwork",
}


@st.cache_data
def load_tech_html() -> str:
    """Read the standalone technology HTML asset into memory.
    
    Cached with st.cache_data to eliminate file I/O latency across page transitions.
    """
    base_dir = os.path.dirname(__file__)
    candidate_paths = [
        os.path.join(base_dir, "static", "vahnix-tech-page.html"),
        os.path.join(base_dir, "..", "static", "vahnix-tech-page.html"),
        os.path.join(base_dir, "vahnix-tech-page.html"),
        os.path.join(base_dir, "..", "vahnix-tech-page.html"),
        os.path.join(os.path.dirname(os.path.dirname(__file__)), "static", "vahnix-tech-page.html"),
        "/Users/kirtikarawat/Documents/ED1/sih/static/vahnix-tech-page.html",
        "/Users/kirtikarawat/Documents/ED1/sih/vahnix-tech-page.html",
        "/Users/kirtikarawat/Downloads/VahniX-master 11/src/dashboard/static/vahnix-tech-page.html",
        "/Users/kirtikarawat/Downloads/VahniX-master 11/static/vahnix-tech-page.html",
    ]
    for p in candidate_paths:
        abs_p = os.path.abspath(p)
        if os.path.exists(abs_p):
            try:
                with open(abs_p, "r", encoding="utf-8") as f:
                    content = f.read()
                    if content and len(content) > 1000:
                        return content
            except Exception:
                continue
    return ""


TEAMWORK_SECTION_HTML = """
  <!-- TAB 5 — TEAMWORK & C2 INTEROPERABILITY PREVIEW -->
  <section class="tab" id="tab-teamwork">
    <p class="kicker">COMMAND &amp; CONTROL · MULTI-AGENCY INCIDENT RESPONSE · AIR-GAPPED FIELD CACHE</p>
    <h1 class="tab-title">Tactical Teamwork &amp; C2 Interoperability</h1>
    <p class="lead">How VahniX synchronizes multi-agency disaster response—bridging orbital surveillance, district fire command, and air-gapped field responders with sub-3 minute tactical dispatch and zero false-alarm latency.</p>

    <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:24px;margin-bottom:36px;">
      <div class="card" style="border-left:3px solid var(--explain);">
        <p class="mono-label" style="color:var(--explain);font-size:11px;letter-spacing:0.18em;margin-bottom:8px;">TIER 1 · STRATEGIC COMMAND</p>
        <h3 style="font-size:18px;margin:0 0 10px;color:var(--text);">National &amp; State Crisis Centers</h3>
        <p style="font-size:13.5px;color:#c9c6bf;line-height:1.5;">Direct telemetry feed to PMO, National Crisis Management Committee (NCMC), and State Disaster Management Authorities (SDMA). Provides verified tactical classification and prevents mobilization panic over routine refinery flares.</p>
      </div>

      <div class="card" style="border-left:3px solid var(--cool);">
        <p class="mono-label" style="color:var(--cool);font-size:11px;letter-spacing:0.18em;margin-bottom:8px;">TIER 2 · FIRST RESPONDERS</p>
        <h3 style="font-size:18px;margin:0 0 10px;color:var(--text);">NDRF &amp; District Fire Command</h3>
        <p style="font-size:13.5px;color:#c9c6bf;line-height:1.5;">Automated Situation Report (SitRep) generation with exact GPS vectors, perimeter estimates, and hazardous materials risk scoring. Dispatches units in &lt;3 minutes with TreeSHAP decision explanations.</p>
      </div>

      <div class="card" style="border-left:3px solid var(--hot);">
        <p class="mono-label" style="color:var(--hot);font-size:11px;letter-spacing:0.18em;margin-bottom:8px;">TIER 3 · INDUSTRIAL ERT</p>
        <h3 style="font-size:18px;margin:0 0 10px;color:var(--text);">Refinery &amp; Plant Safety Crews</h3>
        <p style="font-size:13.5px;color:#c9c6bf;line-height:1.5;">Real-time boundary monitoring around flare stacks and industrial furnaces. Automatic verification against scheduled maintenance burns eliminates 98.4% of false alarms without manual inquiry.</p>
      </div>
    </div>

    <!-- C2 Timeline & Workflow -->
    <div class="card" style="margin-bottom:36px;">
      <p class="mono-label" style="color:var(--math);font-size:11px;letter-spacing:0.18em;margin-bottom:12px;">SUB-3 MINUTE END-TO-END TELEMETRY PIPELINE</p>
      <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:16px;">
        <div style="background:rgba(255,255,255,0.02);padding:14px;border-radius:8px;border:1px solid var(--border);">
          <span style="font-family:'IBM Plex Mono',monospace;font-size:12px;color:var(--cool);">00:00 · SATELLITE PASS</span>
          <p style="font-size:12.5px;color:var(--muted);margin:6px 0 0;">VIIRS/MODIS active fire detection ingestion via NASA FIRMS near-real-time API.</p>
        </div>
        <div style="background:rgba(255,255,255,0.02);padding:14px;border-radius:8px;border:1px solid var(--border);">
          <span style="font-family:'IBM Plex Mono',monospace;font-size:12px;color:var(--cool);">00:45 · 4-STAGE FILTER</span>
          <p style="font-size:12.5px;color:var(--muted);margin:6px 0 0;">Dozier sub-pixel solver, cloud/glint masking, and spatial landcover validation.</p>
        </div>
        <div style="background:rgba(255,255,255,0.02);padding:14px;border-radius:8px;border:1px solid var(--border);">
          <span style="font-family:'IBM Plex Mono',monospace;font-size:12px;color:var(--cool);">01:15 · AI CLASSIFIER</span>
          <p style="font-size:12.5px;color:var(--muted);margin:6px 0 0;">Sub-2ms GBDT ensemble inference with TreeSHAP feature attribution breakdown.</p>
        </div>
        <div style="background:rgba(255,255,255,0.02);padding:14px;border-radius:8px;border:1px solid var(--border);">
          <span style="font-family:'IBM Plex Mono',monospace;font-size:12px;color:var(--cool);">02:00 · C2 SITREP READY</span>
          <p style="font-size:12.5px;color:var(--muted);margin:6px 0 0;">Automated SitRep &amp; GIS GeoJSON packages generated for tactical command briefing.</p>
        </div>
        <div style="background:rgba(255,255,255,0.02);padding:14px;border-radius:8px;border:1px solid var(--border);">
          <span style="font-family:'IBM Plex Mono',monospace;font-size:12px;color:var(--cool);">02:45 · FIELD DISPATCH</span>
          <p style="font-size:12.5px;color:var(--muted);margin:6px 0 0;">Encrypted telemetry delivered to air-gapped field caches and mobile units.</p>
        </div>
      </div>
    </div>

    <!-- Air-Gapped Field Resiliency -->
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:24px;">
      <div class="card">
        <h3 style="font-size:17px;margin:0 0 10px;color:var(--text);">Air-Gapped Field Cache Architecture</h3>
        <p style="font-size:13px;color:#c9c6bf;line-height:1.55;">Designed for disaster zones with severed communications infrastructure. Local SQLite/IndexedDB vector store caches high-resolution satellite basemaps and pre-computed plant polygons. When mobile units enter comms blackouts, tactical maps, classification rules, and chatbot assistant remain 100% operational offline.</p>
      </div>
      <div class="card">
        <h3 style="font-size:17px;margin:0 0 10px;color:var(--text);">Multi-Sensor Cross-Verification Matrix</h3>
        <p style="font-size:13px;color:#c9c6bf;line-height:1.55;">Combines polar-orbiting Suomi NPP, NOAA-20, and NOAA-21 VIIRS (375m) with MODIS Terra/Aqua (1km) and Sentinel-2 MSI (20m SWIR). Every anomaly is corroborated against OpenStreetMap industrial geometries to prevent false-positive emergency alerts.</p>
      </div>
    </div>
  </section>
"""


def prepare_tech_html(html_raw: str, initial_tab: str = "pipeline") -> str:
    """Prepare and instrument the Technology HTML page for embedded Streamlit rendering.
    
    Enhancements:
    1. Sets initial active tab (Physical Filter, Thermal Fingerprint, Explainability, Mathworks, Teamwork & C2).
    2. Adds dedicated 'Teamwork & C2' tab to the HTML navigation and content sections.
    3. Injects auto-resizing postMessage hooks (streamlit:setFrameHeight).
    4. Handles parent window navigation so links/buttons navigate the main Streamlit application.
    5. Adds a Return to Dashboard header action button and clickable brand logo.
    6. Injects offline KaTeX fallbacks so mathematical formulas remain readable without CDN.
    """
    if not html_raw:
        return ""

    canonical_tab = TAB_ALIASES.get(initial_tab.lower().strip(), "pipeline")

    html = html_raw

    # 1. Inject Teamwork & C2 tab button and section if not already present
    if 'data-tab="teamwork"' not in html:
        # Add Teamwork tab button right after math button
        math_btn = '<button data-tab="math">MATHWORKS</button>'
        teamwork_btn = '<button data-tab="math">MATHWORKS</button>\n      <button data-tab="teamwork">TEAMWORK &amp; C2</button>'
        if math_btn in html:
            html = html.replace(math_btn, teamwork_btn, 1)

        # Add Teamwork section before </main>
        if '</main>' in html:
            html = html.replace('</main>', f'{TEAMWORK_SECTION_HTML}\n</main>', 1)

    # 2. Update initial active classes if not the default "pipeline"
    if canonical_tab != "pipeline":
        # Remove active from pipeline button and section
        html = html.replace(
            '<button data-tab="pipeline" class="active">',
            '<button data-tab="pipeline">'
        )
        html = html.replace(
            '<section class="tab active" id="tab-pipeline">',
            '<section class="tab" id="tab-pipeline">'
        )
        # Add active to the target tab
        html = html.replace(
            f'<button data-tab="{canonical_tab}">',
            f'<button data-tab="{canonical_tab}" class="active">'
        )
        html = html.replace(
            f'<section class="tab" id="tab-{canonical_tab}">',
            f'<section class="tab active" id="tab-{canonical_tab}">'
        )

    # 3. Add clickable brand logo linking to home in parent window (hidden from view per user requirement)
    brand_anchor = """
    <a href="?page=home" target="_parent" class="brand" style="display:none !important;text-decoration:none;cursor:pointer;" title="Return to VahniX Tactical Dashboard">
      <img class="brand-logo" src="data:image/png;base64,"""
    html = re.sub(
        r'<div class="brand"[^>]*>\s*<img class="brand-logo" src="data:image/png;base64,',
        brand_anchor,
        html,
        count=1
    )
    # Close the brand <a> tag for all header text variations
    html = html.replace(
        '<div class="brand-sub">TACTICAL DEFENSE<br>THERMAL INTELLIGENCE</div>\n    </div>',
        '<div class="brand-sub">TACTICAL DEFENSE<br>THERMAL INTELLIGENCE</div>\n    </a>',
        1
    )
    html = html.replace(
        '<span class="brand-sub">Technology overview</span>\n    </div>',
        '<span class="brand-sub">Technology overview</span>\n    </a>',
        1
    )

    # 4. Add 'Return to Dashboard' button in the side viewbar right before </nav> if not already present
    if 'class="tech-return-btn"' not in html:
        return_btn_html = """
      <a href="?page=home" target="_parent" class="tech-return-btn" style="display:flex;align-items:center;justify-content:flex-start;gap:8px;width:100%;box-sizing:border-box;margin:18px 0 0 0;padding:11px 14px;font-family:'Space Grotesk',sans-serif;font-size:11px;font-weight:600;letter-spacing:0.12em;text-transform:uppercase;color:#38bdf8;text-decoration:none;border:1px solid rgba(56,189,248,0.35);border-radius:6px;background:rgba(56,189,248,0.08);transition:all .2s ease;" onmouseover="this.style.color='#ffffff';this.style.borderColor='rgba(56,189,248,0.7)';this.style.background='rgba(56,189,248,0.2)';" onmouseout="this.style.color='#38bdf8';this.style.borderColor='rgba(56,189,248,0.35)';this.style.background='rgba(56,189,248,0.08)';">
        <span>←</span> RETURN TO DASHBOARD
      </a>
    </nav>
        """
        html = html.replace("</nav>", return_btn_html, 1)

    # 5. Inject side viewbar layout and full-width styling overrides into <head>
    layout_override_css = """
<style id="vahnix-tech-sidebar-layout">
  /* 1. Hide logo and technological text header */
  .brand, .brand-logo, .brand-sub {
    display: none !important;
    visibility: hidden !important;
    height: 0 !important;
    width: 0 !important;
    margin: 0 !important;
    padding: 0 !important;
    overflow: hidden !important;
    position: absolute !important;
  }

  /* 2. Side viewbar layout */
  body {
    display: flex !important;
    flex-direction: row !important;
    align-items: flex-start !important;
    min-height: 100vh !important;
    margin: 0 !important;
    padding: 0 !important;
    background: #0a0b0d !important;
    color: #ece9e2 !important;
    width: 100% !important;
    overflow-x: hidden !important;
  }

  header.top {
    position: sticky !important;
    top: 0 !important;
    left: 0 !important;
    width: 250px !important;
    min-width: 250px !important;
    max-width: 250px !important;
    height: 100vh !important;
    max-height: 100vh !important;
    background: #0d0f13 !important;
    backdrop-filter: blur(16px) !important;
    border-right: 1px solid #262a31 !important;
    border-bottom: none !important;
    z-index: 100 !important;
    display: flex !important;
    flex-direction: column !important;
    padding: 24px 16px !important;
    box-sizing: border-box !important;
    overflow-y: auto !important;
    flex-shrink: 0 !important;
  }

  header.top .shell, header.top .top-inner {
    padding: 0 !important;
    margin: 0 !important;
    width: 100% !important;
    max-width: 100% !important;
  }

  .top-inner {
    display: flex !important;
    flex-direction: column !important;
    align-items: stretch !important;
    justify-content: flex-start !important;
    width: 100% !important;
    padding: 0 !important;
    margin: 0 !important;
    gap: 12px !important;
  }

  nav.tabs {
    display: flex !important;
    flex-direction: column !important;
    gap: 8px !important;
    width: 100% !important;
    background: transparent !important;
    border: none !important;
    padding: 0 !important;
  }

  nav.tabs button {
    appearance: none !important;
    border: 1px solid transparent !important;
    background: rgba(255, 255, 255, 0.02) !important;
    color: #8b9099 !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 11.5px !important;
    font-weight: 600 !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    padding: 12px 14px !important;
    border-radius: 6px !important;
    cursor: pointer !important;
    transition: all 0.2s ease !important;
    white-space: normal !important;
    text-align: left !important;
    line-height: 1.35 !important;
    width: 100% !important;
    box-sizing: border-box !important;
  }

  nav.tabs button:hover {
    color: #ece9e2 !important;
    background: rgba(255, 255, 255, 0.05) !important;
    border-color: rgba(255, 255, 255, 0.1) !important;
  }

  nav.tabs button.active {
    background: rgba(56, 189, 248, 0.12) !important;
    color: #38bdf8 !important;
    border: 1px solid rgba(56, 189, 248, 0.35) !important;
    border-left: 3px solid #38bdf8 !important;
    box-shadow: 0 2px 10px rgba(56, 189, 248, 0.15) !important;
  }

  @media (max-width: 768px) {
    body {
      flex-direction: column !important;
    }
    header.top {
      position: static !important;
      width: 100% !important;
      min-width: 100% !important;
      max-width: 100% !important;
      height: auto !important;
      max-height: none !important;
      border-right: none !important;
      border-bottom: 1px solid #262a31 !important;
      padding: 16px 20px !important;
    }
    nav.tabs {
      flex-direction: row !important;
      flex-wrap: wrap !important;
    }
    nav.tabs button {
      width: auto !important;
      flex: 1 1 auto !important;
    }
    .tech-return-btn {
      width: auto !important;
      margin: 8px 0 0 0 !important;
    }
    .tech-content-container {
      width: 100% !important;
    }
    main, main.shell {
      padding: 24px 20px 60px !important;
    }
  }

  .tech-content-container {
    flex: 1 1 0% !important;
    min-width: 0 !important;
    width: calc(100% - 250px) !important;
    display: flex !important;
    flex-direction: column !important;
  }

  /* 3. Full-width display taking entire space */
  main, main.shell {
    flex: 1 0 auto !important;
    min-width: 0 !important;
    width: 100% !important;
    max-width: 100% !important;
    padding: 36px 48px 80px !important;
    box-sizing: border-box !important;
  }

  .shell {
    width: 100% !important;
    max-width: 100% !important;
    margin: 0 !important;
    padding: 0 36px !important;
    box-sizing: border-box !important;
  }

  h1.tab-title {
    max-width: none !important;
    width: 100% !important;
  }

  p.lead {
    max-width: none !important;
    width: 100% !important;
  }

  .pf-intro, .pf-table-wrap, .fv, .mw-flow, .mw-pair, .mw-out, .mw-steps, .loss-wrap, .tier-wrap, .ph-wrap {
    width: 100% !important;
    max-width: 100% !important;
  }

  .pf-copy p:not(.kicker) {
    max-width: none !important;
    width: 100% !important;
  }

  .pf-table-lead, .fv-lead, .ph-foot, .mw-model p, .mw-stage p, .mw-step p, .loss-copy p, .loss-side p {
    max-width: none !important;
    width: 100% !important;
  }

  section.tab {
    width: 100% !important;
    max-width: 100% !important;
  }

  .card {
    width: 100% !important;
    box-sizing: border-box !important;
  }

  footer {
    width: 100% !important;
    padding: 24px 36px 36px !important;
    box-sizing: border-box !important;
  }
</style>
"""
    if 'id="vahnix-tech-sidebar-layout"' not in html and "</head>" in html:
        html = html.replace("</head>", f"{layout_override_css}\n</head>", 1)

    if 'class="tech-content-container"' not in html:
        if '<main' in html:
            html = re.sub(r'(<main[^>]*>)', r'<div class="tech-content-container">\n\1', html, count=1)
            if '</footer>' in html:
                html = html.replace('</footer>', '</footer>\n</div>', 1)
            elif '</main>' in html:
                html = html.replace('</main>', '</main>\n</div>', 1)

    # 5. Inject height synchronization, parent scroll, and link handling scripts before </body>
    injected_js = f"""
<script>
(function() {{
  const TARGET_TAB = "{canonical_tab}";

  let lastReportedHeight = 0;
  function syncHeight() {{
    try {{
      const activeSection = document.querySelector('section.tab.active');
      const footer = document.querySelector('footer');
      const contentContainer = document.querySelector('.tech-content-container');
      const sidebarNav = document.querySelector('nav.tabs');
      
      let contentH = 1200;
      if (contentContainer) {{
        contentH = contentContainer.scrollHeight || contentContainer.offsetHeight || 1200;
      }} else if (activeSection) {{
        const secH = activeSection.scrollHeight || activeSection.offsetHeight || 1000;
        const footH = footer ? (footer.offsetHeight || 80) : 80;
        contentH = secH + footH + 100;
      }}
      
      let sidebarH = 600;
      if (sidebarNav) {{
        sidebarH = (sidebarNav.scrollHeight || 400) + 120;
      }}
      
      const targetHeight = Math.max(contentH, sidebarH, 1000) + 60;

      if (Math.abs(targetHeight - lastReportedHeight) > 20) {{
        lastReportedHeight = targetHeight;
        window.parent.postMessage({{
          type: "streamlit:setFrameHeight",
          height: targetHeight
        }}, "*");
      }}
    }} catch (e) {{}}
  }}

  // Scroll parent Streamlit page to top smoothly
  function scrollParentTop() {{
    try {{
      window.parent.scrollTo({{ top: 0, behavior: 'smooth' }});
      const scroller = window.parent.document.querySelector('section.main')
                    || window.parent.document.querySelector('[data-testid="stMain"]')
                    || window.parent.document.querySelector('[data-testid="stAppViewContainer"]');
      if (scroller) scroller.scrollTo({{ top: 0, behavior: 'smooth' }});
    }} catch (e) {{}}
  }}

  // Ensure all links to ?page=... navigate the parent Streamlit window
  function fixLinks() {{
    document.querySelectorAll('a[href^="?page="]').forEach(a => {{
      a.setAttribute('target', '_parent');
    }});
  }}

  // Select initial tab if specified
  function applyInitialTab() {{
    if (TARGET_TAB && TARGET_TAB !== "pipeline") {{
      const btn = document.querySelector(`#tabs button[data-tab="${{TARGET_TAB}}"]`);
      if (btn) {{
        document.querySelectorAll('#tabs button').forEach(b => b.classList.remove('active'));
        document.querySelectorAll('section.tab').forEach(s => s.classList.remove('active'));
        btn.classList.add('active');
        const sec = document.getElementById('tab-' + TARGET_TAB);
        if (sec) sec.classList.add('active');
      }}
    }}
  }}

  // Wire tab buttons to trigger height sync & parent scroll
  document.querySelectorAll('#tabs button').forEach(btn => {{
    btn.addEventListener('click', () => {{
      scrollParentTop();
      setTimeout(syncHeight, 50);
      setTimeout(syncHeight, 250);
      setTimeout(syncHeight, 600);
    }});
  }});

  // Offline KaTeX formula fallback if cdnjs is unreachable
  function applyKatexFallbacks() {{
    if (typeof katex === 'undefined') {{
      const fallbacks = {{
        'eq-planck': 'B(λ,T) = (2hc²) / (λ⁵ · [exp(hc / (λ k_B T)) - 1])',
        'eq-wien': 'λ_max = 2897.77 μm·K / T',
        'eq-sb': 'E = ε · σ · T⁴',
        'eq-dozier': 'L_λ = p · ε_f · B(λ, T_f) + (1 - p) · ε_b · B(λ, T_b)',
        'eq-focal': 'L_Focal = - Σ α_k (1 - p_k)^γ log(p_k)   [γ = 2.0]',
        'eq-focusterm': '(1 - p_k)^γ',
        'eq-shapley': 'φ_i = Σ [|S|!(|F|-|S|-1)! / |F|!] · [v(S ∪ {{i}}) - v(S)]',
        'eq-eff': 'Σ φ_i = f(x) - E[f(X)]',
        'eq-treeshap': 'O(T · L · D²)   [T=1200, L=63, D=7]'
      }};
      for (const [id, mathText] of Object.entries(fallbacks)) {{
        const el = document.getElementById(id);
        if (el && !el.textContent.trim()) {{
          el.innerHTML = '<span style="font-family:\\'IBM Plex Mono\\',monospace;font-size:13px;color:#4cd98a;">' + mathText + '</span>';
        }}
      }}
    }}
  }}

  window.addEventListener('load', () => {{
    applyInitialTab();
    fixLinks();
    applyKatexFallbacks();
    syncHeight();
    setTimeout(syncHeight, 200);
    setTimeout(syncHeight, 600);
    setTimeout(syncHeight, 1500);
  }});

  window.addEventListener('resize', syncHeight);
  new MutationObserver(syncHeight).observe(document.body, {{ childList: true, subtree: true }});
}})();
</script>
"""
    html = html.replace("</body>", f"{injected_js}\n</body>")
    return html


def render_technology_view(
    initial_tab: str = "pipeline",
    kpis: Optional[dict[str, Any]] = None
) -> None:
    """Render the integrated VahniX Technology & Architecture View."""

    # 1. Check for query parameter override
    query_tab = st.query_params.get("tab", "").strip()
    if query_tab:
        initial_tab = query_tab

    canonical_tab = TAB_ALIASES.get(initial_tab.lower().strip(), "pipeline")

    # 2. Contextual Telemetry Banner for specialized entrypoints
    page_param = st.query_params.get("page", "").lower().strip()

    if page_param in ("goal", "goals", "pipeline"):
        st.markdown("""
        <div style="background: rgba(63, 214, 208, 0.08); border: 1px solid rgba(63, 214, 208, 0.35); border-radius: 8px; padding: 0.85rem 1.4rem; margin-bottom: 1.2rem; display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 10px;">
            <div style="display: flex; align-items: center; gap: 10px;">
                <span style="font-family: var(--font-mono); font-size: 0.72rem; font-weight: 800; color: #3fd6d0; border: 1px solid rgba(63, 214, 208, 0.4); padding: 3px 7px; border-radius: 4px; letter-spacing: 0.08em;">[GOAL]</span>
                <div>
                    <div style="font-family: var(--font-mono); font-size: 0.72rem; letter-spacing: 0.2em; text-transform: uppercase; color: #3fd6d0; font-weight: 600;">
                        MISSION GOAL · SIH 2026 PROBLEM STATEMENT 26162
                    </div>
                    <div style="font-family: var(--font-body); font-size: 0.85rem; color: #ece9e2; margin-top: 2px;">
                        Zero False-Alarm Tactical Surveillance: Distinguishing Routine Industrial Flaring from Explosive Thermal Disasters
                    </div>
                </div>
            </div>
            <div style="font-family: var(--font-mono); font-size: 0.7rem; color: #8b9099; background: rgba(0,0,0,0.4); padding: 4px 10px; border-radius: 4px; border: 1px solid #262a31;">
                100% FREE SATELLITE STACK: NASA FIRMS + SENTINEL-2 + LANDSAT + OSM
            </div>
        </div>
        """, unsafe_allow_html=True)

    elif page_param in ("boost", "boosting", "math", "mathworks"):
        st.markdown("""
        <div style="background: rgba(255, 122, 69, 0.08); border: 1px solid rgba(255, 122, 69, 0.35); border-radius: 8px; padding: 0.85rem 1.4rem; margin-bottom: 1.2rem; display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 10px;">
            <div style="display: flex; align-items: center; gap: 10px;">
                <span style="font-family: var(--font-mono); font-size: 0.72rem; font-weight: 800; color: #ff7a45; border: 1px solid rgba(255, 122, 69, 0.4); padding: 3px 7px; border-radius: 4px; letter-spacing: 0.08em;">[BOOST]</span>
                <div>
                    <div style="font-family: var(--font-mono); font-size: 0.72rem; letter-spacing: 0.2em; text-transform: uppercase; color: #ff7a45; font-weight: 600;">
                        GRADIENT-BOOSTED SPATIAL ENSEMBLE · SUB-2MS CPU INFERENCE
                    </div>
                    <div style="font-family: var(--font-body); font-size: 0.85rem; color: #ece9e2; margin-top: 2px;">
                        LightGBM / XGBoost 1,200-Tree Architecture with Multi-Class Focal Loss (γ = 2.0) and Stratified H3 Res-6 GroupKFold
                    </div>
                </div>
            </div>
            <div style="font-family: var(--font-mono); font-size: 0.7rem; color: #8b9099; background: rgba(0,0,0,0.4); padding: 4px 10px; border-radius: 4px; border: 1px solid #262a31;">
                TIER 1 ACCURACY: 96.4% · EMERGENCY RECALL: ≥ 0.96
            </div>
        </div>
        """, unsafe_allow_html=True)

    elif page_param in ("teamwork", "teamwork-preview", "team", "c2", "command"):
        st.markdown("""
        <div style="background: rgba(155, 123, 255, 0.08); border: 1px solid rgba(155, 123, 255, 0.35); border-radius: 8px; padding: 0.85rem 1.4rem; margin-bottom: 1.2rem; display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 10px;">
            <div style="display: flex; align-items: center; gap: 10px;">
                <span style="font-family: var(--font-mono); font-size: 0.72rem; font-weight: 800; color: #9b7bff; border: 1px solid rgba(155, 123, 255, 0.4); padding: 3px 7px; border-radius: 4px; letter-spacing: 0.08em;">[TEAM]</span>
                <div>
                    <div style="font-family: var(--font-mono); font-size: 0.72rem; letter-spacing: 0.2em; text-transform: uppercase; color: #9b7bff; font-weight: 600;">
                        TACTICAL TEAMWORK &amp; C2 INTEROPERABILITY PREVIEW
                    </div>
                    <div style="font-family: var(--font-body); font-size: 0.85rem; color: #ece9e2; margin-top: 2px;">
                        Coordinated Command Center Dispatch, Multi-Agency Incident Response (NDRF / SDMA), and Air-Gapped Field Cache
                    </div>
                </div>
            </div>
            <div style="font-family: var(--font-mono); font-size: 0.7rem; color: #8b9099; background: rgba(0,0,0,0.4); padding: 4px 10px; border-radius: 4px; border: 1px solid #262a31;">
                STANDALONE OFFLINE C2 READY
            </div>
        </div>
        """, unsafe_allow_html=True)

    # 3. Load & Process HTML
    raw_html = load_tech_html()
    if not raw_html:
        st.error("Technology view asset (`vahnix-tech-page.html`) could not be loaded from static assets.")
        return

    processed_html = prepare_tech_html(raw_html, initial_tab=canonical_tab)

    # 4. Render interactive component with dynamic frame height synchronization
    components.html(processed_html, height=1800, scrolling=True)
