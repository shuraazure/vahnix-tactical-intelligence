"""
Risk Severity Index (RSI) and Tactical Threat Scorer for NTRO Thermal Detection System.
Calculates multi-factor risk scores [0.0 - 100.0], tactical alert levels, and critical notifications.
Authoritative Specifications: ORIGINAL_REQUEST.md § R2, PROJECT.md § 2.2, TEST_INFRA.md
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, Optional, Union
import numpy as np

from src.data_pipeline.schemas import FIRMSDetection
from src.clustering.schemas import ThermalCluster
from src.classification.heuristic_baseline import ThermalAnomalyClass


class AlertLevel(Enum):
    GREEN = "GREEN"      # Routine / Low Risk (< 30.0)
    YELLOW = "YELLOW"    # Elevated / Advisory (30.0 - 59.9)
    ORANGE = "ORANGE"    # High / Warning (60.0 - 79.9)
    RED = "RED"          # Critical Emergency (>= 80.0)


def calculate_risk_severity_index(
    frp: float,
    bright_delta: float,
    dist_to_industrial_m: float,
    is_inside_industrial: bool,
    night_fraction: float = 0.5,
    frp_zscore: float = 0.0
) -> float:
    """
    Computes composite Risk Severity Index (RSI) [0.0 - 100.0].
    Weighted multi-factor index combining FRP, Delta-T, Industrial Proximity, and Surge Z-score.
    """
    w_frp = min(max(frp, 0.0) / 500.0, 1.0) * 35.0
    w_delta = min(max(bright_delta, 0.0) / 100.0, 1.0) * 25.0
    
    if is_inside_industrial or dist_to_industrial_m <= 350.0:
        w_prox = 20.0
    elif dist_to_industrial_m <= 750.0:
        w_prox = 10.0
    else:
        w_prox = max(0.0, 20.0 - (dist_to_industrial_m / 100.0))
        
    w_surge = min(max(frp_zscore, 0.0) / 4.0, 1.0) * 20.0
    total = w_frp + w_delta + w_prox + w_surge
    return float(np.clip(total, 0.0, 100.0))


class RiskScorer:
    """
    Tactical Threat & Risk Evaluation Engine.
    """

    @staticmethod
    def compute_risk(
        detection: FIRMSDetection,
        cluster: Optional[ThermalCluster] = None,
        dist_to_industrial_m: float = 5000.0,
        is_inside_industrial: bool = False,
        predicted_class: Optional[ThermalAnomalyClass] = None,
        confidence_score: float = 0.80
    ) -> tuple[float, AlertLevel, bool]:
        """
        Evaluate full risk profile for a detection.
        Returns (risk_severity_index, alert_level, is_critical_alert).
        """
        frp = float(getattr(detection, "frp", 20.0))
        delta_t = float(getattr(detection, "bright_delta", 30.0))
        
        mean_frp = cluster.mean_frp if cluster else frp
        std_frp = cluster.frp_std if cluster and cluster.frp_std > 0 else 1.0
        frp_zscore = float((frp - mean_frp) / (std_frp + 1e-4))
        
        night_cnt = cluster.night_count if cluster else (1 if getattr(detection, "daynight", "D") == "N" else 0)
        total_cnt = cluster.total_detections if cluster else 1
        night_frac = float(night_cnt / max(total_cnt, 1))

        if cluster:
            dist_to_industrial_m = cluster.distance_to_nearest_industrial_m
            is_inside_industrial = cluster.is_inside_industrial_boundary

        rsi = calculate_risk_severity_index(
            frp=frp,
            bright_delta=delta_t,
            dist_to_industrial_m=dist_to_industrial_m,
            is_inside_industrial=is_inside_industrial,
            night_fraction=night_frac,
            frp_zscore=frp_zscore
        )

        # Alert Level determination
        if rsi >= 80.0:
            level = AlertLevel.RED
        elif rsi >= 60.0:
            level = AlertLevel.ORANGE
        elif rsi >= 30.0:
            level = AlertLevel.YELLOW
        else:
            level = AlertLevel.GREEN

        # Critical Alert logic
        is_critical = False
        if rsi >= 75.0:
            is_critical = True
        elif predicted_class == ThermalAnomalyClass.INDUSTRIAL_FIRE_DISASTER and confidence_score >= 0.70:
            is_critical = True
        elif is_inside_industrial and frp >= 150.0:
            is_critical = True

        return rsi, level, is_critical
