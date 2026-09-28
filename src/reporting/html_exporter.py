"""
Self-Contained Tactical HTML Report Generator for NTRO Thermal Detection System.
Generates standalone interactive HTML dossiers with embedded CSS styling and base64 SHAP charts.
Authoritative Specifications: ORIGINAL_REQUEST.md § R3, PROJECT.md § 3, TEST_INFRA.md
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional
from datetime import datetime

from src.data_pipeline.schemas import FIRMSDetection
from src.classification.ensemble_classifier import PredictionOutput
from src.explainability.shap_engine import SHAPExplanation


class HTMLReportExporter:
    """
    Exports self-contained standalone HTML incident dossiers.
    """

    @classmethod
    def generate_incident_html_dossier(
        cls,
        detection: FIRMSDetection,
        prediction: PredictionOutput,
        explanation: Optional[SHAPExplanation] = None,
        output_path: str = "data/reports/incident_dossier.html"
    ) -> str:
        """
        Generate standalone HTML dossier.
        """
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        chart_html = ""
        if explanation and explanation.waterfall_plot_base64:
            chart_html = f"""
            <div class="card">
                <h3>SHAP Feature Attribution Chart</h3>
                <div style="text-align: center;">
                    <img src="data:image/png;base64,{explanation.waterfall_plot_base64}" style="max-width: 100%; border-radius: 6px; box-shadow: 0 2px 8px rgba(0,0,0,0.15);" alt="SHAP Waterfall Chart" />
                </div>
            </div>
            """

        pos_rows = ""
        if explanation:
            for item in explanation.top_positive_features[:5]:
                pos_rows += f"<tr><td>{item[0]}</td><td style='color: #27ae60; font-weight: bold;'>+{item[1]:.3f}</td><td>{item[2]:.2f}</td></tr>"
            for item in explanation.top_negative_features[:3]:
                pos_rows += f"<tr><td>{item[0]}</td><td style='color: #c0392b; font-weight: bold;'>{item[1]:.3f}</td><td>{item[2]:.2f}</td></tr>"

        alert_badge = f"<span class='badge {'badge-critical' if prediction.is_critical_alert else 'badge-routine'}'>{'CRITICAL ALERT' if prediction.is_critical_alert else 'ROUTINE'}</span>"

        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>NTRO Incident Dossier - {detection.detection_id}</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            background-color: #0f172a;
            color: #f1f5f9;
            margin: 0;
            padding: 24px;
        }}
        .container {{
            max-width: 960px;
            margin: 0 auto;
        }}
        .header {{
            background: linear-gradient(135deg, #1e293b, #0f172a);
            border: 1px solid #334155;
            padding: 24px;
            border-radius: 12px;
            margin-bottom: 20px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.3);
        }}
        .card {{
            background-color: #1e293b;
            border: 1px solid #334155;
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.2);
        }}
        h1, h2, h3 {{
            margin-top: 0;
            color: #38bdf8;
        }}
        .badge {{
            padding: 6px 14px;
            border-radius: 20px;
            font-weight: bold;
            font-size: 13px;
            display: inline-block;
        }}
        .badge-critical {{ background-color: #ef4444; color: white; }}
        .badge-routine {{ background-color: #10b981; color: white; }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 12px;
        }}
        th, td {{
            padding: 10px 14px;
            text-align: left;
            border-bottom: 1px solid #334155;
            font-size: 14px;
        }}
        th {{
            background-color: #0f172a;
            color: #94a3b8;
        }}
        .metric-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 16px;
            margin-bottom: 16px;
        }}
        .metric-box {{
            background-color: #0f172a;
            padding: 14px;
            border-radius: 8px;
            border: 1px solid #334155;
        }}
        .metric-value {{
            font-size: 22px;
            font-weight: bold;
            color: #f8fafc;
        }}
        .metric-label {{
            font-size: 12px;
            color: #94a3b8;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <h1>NATIONAL TECHNICAL RESEARCH ORGANISATION</h1>
                    <p style="margin: 0; color: #94a3b8;">Thermal Anomaly Intelligence Dossier // NTRO-SIH26162</p>
                </div>
                <div>{alert_badge}</div>
            </div>
        </div>

        <div class="card">
            <h2>Incident Classification: {prediction.predicted_label}</h2>
            <div class="metric-grid">
                <div class="metric-box">
                    <div class="metric-label">Incident ID</div>
                    <div class="metric-value">{detection.detection_id}</div>
                </div>
                <div class="metric-box">
                    <div class="metric-label">AI Confidence</div>
                    <div class="metric-value">{prediction.confidence_score * 100:.1f}%</div>
                </div>
                <div class="metric-box">
                    <div class="metric-label">Risk Severity Index</div>
                    <div class="metric-value" style="color: {'#ef4444' if prediction.risk_severity_index >= 70 else '#38bdf8'};">{prediction.risk_severity_index:.1f} / 100</div>
                </div>
                <div class="metric-box">
                    <div class="metric-label">Fire Radiative Power</div>
                    <div class="metric-value">{detection.frp:.1f} MW</div>
                </div>
            </div>
            <table>
                <tr><th>Parameter</th><th>Value</th><th>Parameter</th><th>Value</th></tr>
                <tr><td>Coordinates</td><td>{detection.latitude:.5f}° N, {detection.longitude:.5f}° E</td><td>Sensor / Platform</td><td>{detection.satellite} ({detection.sensor})</td></tr>
                <tr><td>Acquisition Time</td><td>{detection.acq_date} {detection.acq_time} UTC</td><td>Day/Night Flag</td><td>{detection.daynight}</td></tr>
                <tr><td>Brightness Temp (T4)</td><td>{detection.brightness_temp_t4:.1f} K</td><td>Brightness Temp (T11)</td><td>{detection.brightness_temp_t11:.1f} K</td></tr>
            </table>
        </div>

        <div class="card">
            <h3>Tactical AI Intelligence Narrative</h3>
            <p style="font-size: 15px; line-height: 1.6; color: #cbd5e1;">
                {explanation.natural_language_summary if explanation else "Incident analyzed via multi-class ensemble."}
            </p>
        </div>

        {chart_html}

        <div class="card">
            <h3>Model Attribution Breakdown</h3>
            <table>
                <thead>
                    <tr><th>Feature Name</th><th>Attribution (SHAP)</th><th>Observed Value</th></tr>
                </thead>
                <tbody>
                    {pos_rows}
                </tbody>
            </table>
        </div>
    </div>
</body>
</html>
"""
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html_content)

        return output_path
