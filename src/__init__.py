"""
VOLLEY-REF AI - Système d'arbitrage assisté par IA pour le volleyball

Ce package contient les modules principaux pour la détection IN/OUT:
- court_detector: Détection des keypoints du terrain et homographie
- ball_tracker: Détection et tracking de la balle
- landing_predictor: Prédiction du point d'atterrissage
- decision_engine: Moteur de décision IN/OUT
- visualizer: Visualisation et overlays
- pipeline: Pipeline principal orchestrant tous les modules
"""

from .config import CourtConfig, ModelConfig, DecisionConfig, TrainingConfig
from .court_detector import CourtDetector
from .ball_tracker import BallTracker
from .landing_predictor import LandingPredictor
from .decision_engine import DecisionEngine, Decision
from .visualizer import Visualizer
from .pipeline import VolleyRefAI

__version__ = "1.0.0"
__author__ = "VOLLEY-REF AI Team"

__all__ = [
    "CourtConfig",
    "ModelConfig",
    "DecisionConfig",
    "TrainingConfig",
    "CourtDetector",
    "BallTracker",
    "LandingPredictor",
    "DecisionEngine",
    "Decision",
    "Visualizer",
    "VolleyRefAI",
]
