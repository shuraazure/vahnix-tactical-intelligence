"""
Natural Language Tactical Explainability Generator for NTRO Thermal Detection System.
Synthesizes machine learning predictions, SHAP attribution vectors, and geospatial intelligence
into human-readable tactical dossiers.
Authoritative Specifications: ORIGINAL_REQUEST.md § R2, PROJECT.md § 2.3, TEST_INFRA.md
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
from src.classification.heuristic_baseline import ThermalAnomalyClass, CLASS_NAMES


class NaturalLanguageXAI:
    """
    Translates mathematical model features and SHAP attribution vectors
    into operational intelligence summaries.
    """

    @classmethod
    def generate_tactical_narrative(
        cls,
        detection_id: str,
        predicted_class: ThermalAnomalyClass,
        confidence_score: float,
        top_positive: List[Tuple[str, float, float]],
        top_negative: List[Tuple[str, float, float]],
        feature_dict: Dict[str, float],
        risk_severity: float
    ) -> str:
        """
        Generate operational intelligence narrative for an active detection or incident.
        """
        class_name = CLASS_NAMES.get(predicted_class.value, predicted_class.name)
        conf_pct = confidence_score * 100.0
        
        frp = feature_dict.get("frp", 0.0)
        delta_t = feature_dict.get("brightness_delta", 0.0)
        dist_ind = feature_dict.get("dist_to_industrial_m", 5000.0)
        is_inside = feature_dict.get("is_inside_industrial", 0.0) > 0.5
        recur_freq = feature_dict.get("recurrence_freq_per_month", 0.0)
        dn_ratio = feature_dict.get("day_night_ratio", 1.0)
        t_span = feature_dict.get("temporal_span_days", 0.0)
        c_radius = feature_dict.get("cluster_radius_m", 0.0)

        # Build paragraph header
        if predicted_class == ThermalAnomalyClass.INDUSTRIAL_FIRE_DISASTER:
            header = f"[CRITICAL TACTICAL ALERT] Detection {detection_id} is classified as {class_name} with {conf_pct:.1f}% confidence (Risk Severity: {risk_severity:.1f}/100)."
            body = (
                f"High-confidence emergency thermal signature detected. FRP observed at {frp:.1f} MW with extreme brightness temperature delta ({delta_t:.1f} K). "
                f"Incident is located {'directly inside an industrial boundary' if is_inside else f'{dist_ind:.0f}m from industrial perimeter'}. "
                "Immediate tactical verification and emergency response protocols recommended."
            )
        elif predicted_class == ThermalAnomalyClass.CONTROLLED_INDUSTRIAL_FLARE:
            header = f"[ROUTINE INDUSTRIAL MONITORING] Detection {detection_id} is classified as {class_name} with {conf_pct:.1f}% confidence (Risk Severity: {risk_severity:.1f}/100)."
            body = (
                f"Stationary persistent thermal emission collocated {'within an authorized industrial sector' if is_inside else f'within {dist_ind:.0f}m of facility boundary'}. "
                f"Exhibits recurrent signature ({recur_freq:.1f} detections/month over {t_span:.1f} days) and stable diurnal emission profile (Day/Night ratio {dn_ratio:.2f})."
            )
        elif predicted_class == ThermalAnomalyClass.AGRICULTURAL_STUBBLE_BURNING:
            header = f"[AGRICULTURAL MONITORING] Detection {detection_id} is classified as {class_name} with {conf_pct:.1f}% confidence (Risk Severity: {risk_severity:.1f}/100)."
            body = (
                f"Thermal anomaly situated in rural cropland ({dist_ind / 1000.0:.1f} km from industrial facilities). "
                f"Characterized by strong daytime diurnal dominance (Day/Night ratio {dn_ratio:.2f}) and transient lifespan ({t_span:.1f} days)."
            )
        else:  # WILDFIRE_VEGETATION_BURNING
            header = f"[ENVIRONMENTAL DISASTER ALERT] Detection {detection_id} is classified as {class_name} with {conf_pct:.1f}% confidence (Risk Severity: {risk_severity:.1f}/100)."
            body = (
                f"Expansive non-industrial thermal anomaly with cluster dispersion radius of {c_radius:.0f}m located {dist_ind / 1000.0:.1f} km from infrastructure. "
                f"Sustained burning front observed with total FRP of {frp:.1f} MW."
            )

        # Append top SHAP attribution drivers
        pos_drivers = [f"{item[0]} (+{item[1]:.2f} SHAP, raw: {item[2]:.1f})" for item in top_positive[:3]]
        drivers_text = f" Primary model attribution drivers: {', '.join(pos_drivers)}." if pos_drivers else ""

        return f"{header} {body}{drivers_text}"
