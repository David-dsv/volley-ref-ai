"""
Module 4: Decision Engine (IN/OUT)

Ce module est responsable de:
- Déterminer si la balle est IN ou OUT
- Calculer la distance aux lignes du terrain
- Gérer les cas incertains (TOO_CLOSE)
"""

from dataclasses import dataclass
from typing import Tuple, Dict, Optional
from enum import Enum

from .config import CourtConfig, DecisionConfig


class Verdict(Enum):
    """Verdicts possibles pour une décision."""
    IN = "IN"
    OUT = "OUT"
    TOO_CLOSE = "TOO_CLOSE"
    UNKNOWN = "UNKNOWN"


@dataclass
class Decision:
    """
    Résultat d'une décision IN/OUT.

    Attributes:
        verdict: Verdict de la décision (IN, OUT, TOO_CLOSE)
        distance_to_line: Distance signée à la ligne la plus proche (cm)
                         Positif = à l'intérieur, Négatif = à l'extérieur
        confidence: Score de confiance de la décision (0.0 à 1.0)
        closest_line: Nom de la ligne la plus proche
        landing_point: Coordonnées du point d'atterrissage (cm)
    """
    verdict: str
    distance_to_line: float
    confidence: float
    closest_line: str
    landing_point: Tuple[float, float]

    def __str__(self) -> str:
        """Représentation textuelle de la décision."""
        sign = "inside" if self.distance_to_line >= 0 else "outside"
        return (
            f"{self.verdict} | {abs(self.distance_to_line):.1f}cm {sign} "
            f"{self.closest_line} | Confidence: {self.confidence:.0%}"
        )

    def to_dict(self) -> Dict:
        """Convertit la décision en dictionnaire."""
        return {
            "verdict": self.verdict,
            "distance_to_line": self.distance_to_line,
            "confidence": self.confidence,
            "closest_line": self.closest_line,
            "landing_point": self.landing_point,
        }


class DecisionEngine:
    """
    Moteur de décision IN/OUT pour le volleyball.

    Détermine si un point d'atterrissage est à l'intérieur ou à l'extérieur
    du terrain en calculant les distances aux lignes et en appliquant
    les marges d'incertitude.

    Attributes:
        court: Configuration du terrain (dimensions)
        config: Configuration des seuils de décision
    """

    def __init__(self, court: CourtConfig, config: DecisionConfig):
        """
        Initialise le moteur de décision.

        Args:
            court: Configuration du terrain de volleyball
            config: Configuration des seuils de décision
        """
        self.court = court
        self.config = config

    def is_in_court(
        self,
        point: Tuple[float, float]
    ) -> Tuple[bool, float, str]:
        """
        Vérifie si un point est à l'intérieur du terrain.

        Calcule la distance signée aux 4 lignes du terrain et détermine
        si le point est IN ou OUT.

        Note: Les lignes font partie du terrain! Un point touchant la ligne
        est considéré comme IN.

        Args:
            point: Coordonnées (x, y) en cm sur le terrain

        Returns:
            Tuple (is_in, distance, closest_line):
                - is_in: True si le point est dans le terrain
                - distance: Distance signée à la ligne la plus proche
                           (positif = intérieur, négatif = extérieur)
                - closest_line: Nom de la ligne la plus proche
        """
        x, y = point

        # Calculer les distances à chaque ligne
        # Distance positive = à l'intérieur du terrain
        distances = {
            "end_line_near": y,  # Distance à la ligne de fond proche (y=0)
            "end_line_far": self.court.WIDTH - y,  # Distance à la ligne de fond loin
            "side_line_left": x,  # Distance à la ligne latérale gauche (x=0)
            "side_line_right": self.court.LENGTH - x,  # Distance à la ligne latérale droite
        }

        # Trouver la ligne la plus proche
        closest_line = min(distances, key=lambda k: abs(distances[k]))
        min_distance = distances[closest_line]

        # Vérifier si le point est dans le terrain
        # Un point est IN si toutes les distances sont >= 0
        # (en tenant compte de la largeur des lignes)
        half_line_width = self.court.LINE_WIDTH / 2

        is_in = (
            -half_line_width <= x <= self.court.LENGTH + half_line_width and
            -half_line_width <= y <= self.court.WIDTH + half_line_width
        )

        # La distance est positive si IN, négative si OUT
        if not is_in:
            min_distance = -abs(min_distance)

        return (is_in, min_distance, closest_line)

    def decide(
        self,
        landing: Tuple[float, float],
        pred_confidence: float
    ) -> Decision:
        """
        Prend la décision finale IN/OUT.

        La décision tient compte de:
        - La position du point d'atterrissage
        - La confiance de la prédiction
        - Les marges d'incertitude configurées

        Args:
            landing: Coordonnées du point d'atterrissage (cm)
            pred_confidence: Confiance de la prédiction du landing (0-1)

        Returns:
            Decision: Objet contenant le verdict et les détails
        """
        is_in, distance, line = self.is_in_court(landing)

        # Ajuster l'incertitude en fonction de la confiance
        # Plus la confiance est basse, plus la marge est grande
        uncertainty = self.config.UNCERTAINTY_MARGIN / max(pred_confidence, 0.5)

        # Déterminer le verdict
        abs_distance = abs(distance)

        if abs_distance < self.config.TOO_CLOSE_THRESHOLD:
            # Trop proche de la ligne pour être certain
            verdict = Verdict.TOO_CLOSE.value
            final_confidence = 0.5
        elif abs_distance < uncertainty:
            # Dans la zone d'incertitude
            verdict = Verdict.TOO_CLOSE.value
            final_confidence = 0.5 + (abs_distance / uncertainty) * 0.3
        elif is_in:
            verdict = Verdict.IN.value
            # Confiance augmente avec la distance à la ligne
            final_confidence = min(0.99, pred_confidence + abs_distance / 100)
        else:
            verdict = Verdict.OUT.value
            final_confidence = min(0.99, pred_confidence + abs_distance / 100)

        return Decision(
            verdict=verdict,
            distance_to_line=distance,
            confidence=final_confidence,
            closest_line=line,
            landing_point=landing
        )

    def get_zone(self, point: Tuple[float, float]) -> str:
        """
        Détermine la zone du terrain où se trouve le point.

        Args:
            point: Coordonnées (x, y) en cm

        Returns:
            str: Nom de la zone (front_left, back_right, etc.)
        """
        x, y = point

        # Diviser le terrain en zones
        mid_x = self.court.LENGTH / 2
        mid_y = self.court.WIDTH / 2
        attack_line = self.court.ATTACK_LINE

        # Zone avant/arrière
        if x < attack_line:
            x_zone = "back_left"
        elif x < mid_x:
            x_zone = "front_left"
        elif x < self.court.LENGTH - attack_line:
            x_zone = "front_right"
        else:
            x_zone = "back_right"

        # Zone gauche/droite
        if y < mid_y:
            y_zone = "near"
        else:
            y_zone = "far"

        return f"{x_zone}_{y_zone}"

    def calculate_all_distances(
        self,
        point: Tuple[float, float]
    ) -> Dict[str, float]:
        """
        Calcule la distance à toutes les lignes du terrain.

        Args:
            point: Coordonnées (x, y) en cm

        Returns:
            Dict des distances à chaque ligne
        """
        x, y = point

        return {
            "end_line_near": y,
            "end_line_far": self.court.WIDTH - y,
            "side_line_left": x,
            "side_line_right": self.court.LENGTH - x,
            "attack_line_left": x - self.court.ATTACK_LINE,
            "attack_line_right": (self.court.LENGTH - self.court.ATTACK_LINE) - x,
            "center_line": abs(x - self.court.LENGTH / 2),
        }

    def is_service_valid(
        self,
        landing: Tuple[float, float],
        serving_team_side: str = "left"
    ) -> Tuple[bool, str]:
        """
        Vérifie si un service est valide (atterrit dans la zone adverse).

        Args:
            landing: Point d'atterrissage
            serving_team_side: Côté de l'équipe au service ("left" ou "right")

        Returns:
            Tuple (is_valid, reason)
        """
        x, y = landing

        # Zone de réception dépend du côté au service
        if serving_team_side == "left":
            # Service vers la droite, doit atterrir dans x > LENGTH/2
            valid_x = x > self.court.LENGTH / 2
        else:
            # Service vers la gauche, doit atterrir dans x < LENGTH/2
            valid_x = x < self.court.LENGTH / 2

        # Doit être dans les limites du terrain
        valid_y = 0 <= y <= self.court.WIDTH
        valid_bounds = 0 <= x <= self.court.LENGTH

        if not valid_bounds or not valid_y:
            return (False, "OUT - outside court boundaries")
        elif not valid_x:
            return (False, "FAULT - landed on serving team's side")
        else:
            return (True, "VALID SERVICE")
