"""
Configuration du système VOLLEY-REF AI

Ce module contient toutes les configurations pour:
- Les dimensions du terrain de volleyball (normes FIVB)
- Les paramètres des modèles YOLO
- Les seuils de décision IN/OUT
- Les hyperparamètres d'entraînement
"""

from dataclasses import dataclass, field
from typing import Tuple, List
from pathlib import Path
import numpy as np


@dataclass
class CourtConfig:
    """
    Dimensions officielles du terrain de volleyball (en cm) selon les normes FIVB.

    Le terrain fait 18m x 9m avec une ligne d'attaque à 3m du filet de chaque côté.
    Les lignes ont une largeur de 5cm et font partie intégrante du terrain.

    Attributes:
        LENGTH: Longueur totale du terrain (18m = 1800cm)
        WIDTH: Largeur totale du terrain (9m = 900cm)
        ATTACK_LINE: Distance de la ligne d'attaque au filet (3m = 300cm)
        LINE_WIDTH: Largeur des lignes de marquage (5cm)
        KEYPOINT_NAMES: Noms des 14 keypoints à détecter
    """
    LENGTH: int = 1800  # 18m
    WIDTH: int = 900    # 9m
    ATTACK_LINE: int = 300  # 3m depuis le filet
    LINE_WIDTH: int = 5

    KEYPOINT_NAMES: Tuple[str, ...] = (
        "corner_top_left", "corner_top_right",
        "corner_bottom_left", "corner_bottom_right",
        "attack_top_left", "attack_top_right",
        "attack_bottom_left", "attack_bottom_right",
        "net_left", "net_right",
        "midline_left", "midline_right",
        "center_top", "center_bottom"
    )

    @property
    def real_keypoints(self) -> np.ndarray:
        """
        Coordonnées réelles des keypoints du terrain (en cm).

        Ces coordonnées servent de référence pour calculer la matrice
        d'homographie permettant de projeter les points image vers
        les coordonnées terrain réelles.

        Returns:
            np.ndarray: Array (14, 2) contenant les coordonnées (x, y) de chaque keypoint
        """
        return np.array([
            # Coins extérieurs
            [0, 0],                          # corner_top_left
            [self.LENGTH, 0],                # corner_top_right
            [0, self.WIDTH],                 # corner_bottom_left
            [self.LENGTH, self.WIDTH],       # corner_bottom_right
            # Lignes d'attaque
            [self.ATTACK_LINE, 0],           # attack_top_left
            [self.ATTACK_LINE, self.WIDTH],  # attack_top_right
            [self.LENGTH - self.ATTACK_LINE, 0],         # attack_bottom_left
            [self.LENGTH - self.ATTACK_LINE, self.WIDTH], # attack_bottom_right
            # Points du filet (milieu)
            [self.LENGTH // 2, 0],           # net_left
            [self.LENGTH // 2, self.WIDTH],  # net_right
            # Points milieu lignes latérales
            [0, self.WIDTH // 2],            # midline_left
            [self.LENGTH, self.WIDTH // 2],  # midline_right
            # Points centre
            [self.LENGTH // 2, 0],           # center_top
            [self.LENGTH // 2, self.WIDTH],  # center_bottom
        ], dtype=np.float32)

    @property
    def court_corners(self) -> np.ndarray:
        """
        Retourne uniquement les 4 coins du terrain.

        Returns:
            np.ndarray: Array (4, 2) des coins du terrain
        """
        return self.real_keypoints[:4]


@dataclass
class ModelConfig:
    """
    Configuration des modèles YOLO.

    Attributes:
        COURT_MODEL: Chemin vers le modèle de détection des keypoints terrain
        BALL_MODEL: Chemin vers le modèle de détection de la balle
        CONFIDENCE: Seuil de confiance minimum pour les détections
        IOU: Seuil IoU pour le Non-Maximum Suppression
    """
    COURT_MODEL: str = "weights/yolo_court_keypoints.pt"
    BALL_MODEL: str = "weights/yolo_volleyball_ball.pt"
    CONFIDENCE: float = 0.55
    IOU: float = 0.45


@dataclass
class DecisionConfig:
    """
    Configuration des seuils de décision IN/OUT.

    Attributes:
        UNCERTAINTY_MARGIN: Marge d'incertitude en cm pour la décision
        TOO_CLOSE_THRESHOLD: Seuil en cm en dessous duquel le verdict est "TOO_CLOSE"
        MIN_TRAJECTORY_LENGTH: Nombre minimum de points pour prédire le landing
    """
    UNCERTAINTY_MARGIN: float = 5.0  # cm
    TOO_CLOSE_THRESHOLD: float = 3.0  # cm
    MIN_TRAJECTORY_LENGTH: int = 5


@dataclass
class TrainingConfig:
    """
    Configuration des hyperparamètres pour le fine-tuning.

    Attributes:
        EPOCHS: Nombre d'époques d'entraînement
        BATCH_SIZE: Taille du batch
        IMG_SIZE: Taille des images (redimensionnement)
        PATIENCE: Nombre d'époques sans amélioration avant early stopping
        OPTIMIZER: Optimiseur à utiliser
        LR0: Learning rate initial
        LRF: Learning rate final (facteur multiplicatif)
    """
    EPOCHS: int = 100
    BATCH_SIZE: int = 16
    IMG_SIZE: int = 640
    PATIENCE: int = 20
    OPTIMIZER: str = "AdamW"
    LR0: float = 0.001
    LRF: float = 0.01

    # Configuration spécifique pour la détection de balle
    BALL_EPOCHS: int = 150
    BALL_IMG_SIZE: int = 1280  # Plus grand pour détecter les petits objets


@dataclass
class PathConfig:
    """
    Configuration des chemins du projet.

    Attributes:
        ROOT: Répertoire racine du projet
        WEIGHTS: Répertoire des poids des modèles
        DATASETS: Répertoire des datasets
        OUTPUTS: Répertoire des sorties (vidéos générées)
        RUNS: Répertoire des runs d'entraînement
    """
    ROOT: Path = field(default_factory=lambda: Path(__file__).parent.parent)

    @property
    def WEIGHTS(self) -> Path:
        return self.ROOT / "weights"

    @property
    def DATASETS(self) -> Path:
        return self.ROOT / "datasets"

    @property
    def OUTPUTS(self) -> Path:
        return self.ROOT / "outputs"

    @property
    def RUNS(self) -> Path:
        return self.ROOT / "runs"
