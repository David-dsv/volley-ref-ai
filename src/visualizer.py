"""
Module 5: Visualization

Ce module est responsable de:
- Dessiner la trajectoire de la balle
- Afficher le point d'atterrissage
- Afficher le badge de décision
- Générer la vue 2D top-down du terrain
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict

from .config import CourtConfig
from .decision_engine import Decision


class Visualizer:
    """
    Gère la visualisation des résultats de l'arbitrage.

    Dessine les overlays sur les frames vidéo:
    - Trajectoire de la balle (ligne orange)
    - Point d'atterrissage (cercle + croix)
    - Badge de décision (IN/OUT avec distance)
    - Vue 2D miniature du terrain

    Attributes:
        config: Configuration du terrain
        COLORS: Dictionnaire des couleurs utilisées (BGR)
    """

    COLORS: Dict[str, Tuple[int, int, int]] = {
        "IN": (0, 255, 0),          # Vert
        "OUT": (0, 0, 255),         # Rouge
        "TOO_CLOSE": (0, 255, 255), # Jaune
        "trajectory": (0, 165, 255), # Orange
        "prediction": (255, 0, 255), # Magenta
        "court_lines": (255, 255, 255),  # Blanc
        "court_fill": (34, 139, 34),     # Vert terrain
        "text_bg": (0, 0, 0),            # Noir
    }

    def __init__(self, config: CourtConfig):
        """
        Initialise le visualiseur.

        Args:
            config: Configuration du terrain de volleyball
        """
        self.config = config

    def draw_trajectory(
        self,
        frame: np.ndarray,
        trajectory: List[Tuple[float, float]],
        color: Optional[Tuple[int, int, int]] = None,
        max_thickness: int = 4
    ) -> np.ndarray:
        """
        Dessine la trajectoire de la balle sur la frame.

        L'épaisseur de la ligne augmente progressivement pour
        montrer la direction du mouvement.

        Args:
            frame: Image BGR à modifier
            trajectory: Liste des positions (x, y)
            color: Couleur de la trajectoire (défaut: orange)
            max_thickness: Épaisseur maximale de la ligne

        Returns:
            Frame avec la trajectoire dessinée
        """
        if len(trajectory) < 2:
            return frame

        color = color or self.COLORS["trajectory"]

        for i in range(1, len(trajectory)):
            pt1 = tuple(map(int, trajectory[i - 1]))
            pt2 = tuple(map(int, trajectory[i]))

            # Épaisseur progressive
            progress = i / len(trajectory)
            thickness = max(1, int(progress * max_thickness))

            # Dessiner le segment
            cv2.line(frame, pt1, pt2, color, thickness, cv2.LINE_AA)

        # Dessiner un cercle au dernier point
        if trajectory:
            last_pt = tuple(map(int, trajectory[-1]))
            cv2.circle(frame, last_pt, 8, color, -1, cv2.LINE_AA)
            cv2.circle(frame, last_pt, 8, (255, 255, 255), 2, cv2.LINE_AA)

        return frame

    def draw_predicted_trajectory(
        self,
        frame: np.ndarray,
        predicted: List[Tuple[float, float]],
        color: Optional[Tuple[int, int, int]] = None
    ) -> np.ndarray:
        """
        Dessine la trajectoire prédite en pointillés.

        Args:
            frame: Image BGR
            predicted: Liste des positions prédites
            color: Couleur (défaut: magenta)

        Returns:
            Frame modifiée
        """
        if len(predicted) < 2:
            return frame

        color = color or self.COLORS["prediction"]

        # Dessiner en pointillés
        for i in range(1, len(predicted), 2):  # Sauter 1 segment sur 2
            if i >= len(predicted):
                break
            pt1 = tuple(map(int, predicted[i - 1]))
            pt2 = tuple(map(int, predicted[i]))
            cv2.line(frame, pt1, pt2, color, 2, cv2.LINE_AA)

        return frame

    def draw_landing(
        self,
        frame: np.ndarray,
        point: Tuple[float, float],
        decision: Decision,
        radius: int = 20
    ) -> np.ndarray:
        """
        Dessine le point d'atterrissage avec un marqueur.

        Affiche un cercle et une croix de la couleur du verdict.

        Args:
            frame: Image BGR
            point: Position du landing en pixels
            decision: Décision associée
            radius: Rayon du cercle

        Returns:
            Frame modifiée
        """
        color = self.COLORS.get(decision.verdict, self.COLORS["TOO_CLOSE"])
        center = tuple(map(int, point))

        # Cercle extérieur
        cv2.circle(frame, center, radius, color, 3, cv2.LINE_AA)

        # Cercle intérieur semi-transparent
        overlay = frame.copy()
        cv2.circle(overlay, center, radius - 5, color, -1, cv2.LINE_AA)
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)

        # Croix au centre
        cross_size = radius - 5
        cv2.drawMarker(
            frame, center, color,
            cv2.MARKER_CROSS, cross_size * 2, 3, cv2.LINE_AA
        )

        return frame

    def draw_badge(
        self,
        frame: np.ndarray,
        decision: Decision,
        pos: Tuple[int, int] = (50, 50),
        scale: float = 1.0
    ) -> np.ndarray:
        """
        Dessine le badge de décision avec le verdict et la distance.

        Args:
            frame: Image BGR
            decision: Décision à afficher
            pos: Position du badge (coin supérieur gauche)
            scale: Échelle du texte

        Returns:
            Frame modifiée
        """
        color = self.COLORS.get(decision.verdict, self.COLORS["TOO_CLOSE"])

        # Texte principal (verdict)
        verdict_text = decision.verdict
        font = cv2.FONT_HERSHEY_DUPLEX
        font_scale = 1.5 * scale
        thickness = 3

        # Obtenir la taille du texte
        (text_w, text_h), baseline = cv2.getTextSize(
            verdict_text, font, font_scale, thickness
        )

        # Fond semi-transparent
        padding = 15
        bg_rect = (
            pos[0] - padding,
            pos[1] - text_h - padding,
            pos[0] + text_w + padding,
            pos[1] + padding + 50
        )

        overlay = frame.copy()
        cv2.rectangle(
            overlay,
            (bg_rect[0], bg_rect[1]),
            (bg_rect[2], bg_rect[3]),
            self.COLORS["text_bg"],
            -1
        )
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        # Bordure colorée
        cv2.rectangle(
            frame,
            (bg_rect[0], bg_rect[1]),
            (bg_rect[2], bg_rect[3]),
            color,
            3
        )

        # Texte du verdict
        cv2.putText(
            frame, verdict_text, pos,
            font, font_scale, color, thickness, cv2.LINE_AA
        )

        # Distance
        dist_text = f"{abs(decision.distance_to_line):.1f} cm"
        dist_font_scale = 0.8 * scale
        cv2.putText(
            frame, dist_text,
            (pos[0], pos[1] + 40),
            font, dist_font_scale, (255, 255, 255), 2, cv2.LINE_AA
        )

        # Confiance
        conf_text = f"Conf: {decision.confidence:.0%}"
        cv2.putText(
            frame, conf_text,
            (pos[0], pos[1] + 70),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5 * scale, (200, 200, 200), 1, cv2.LINE_AA
        )

        return frame

    def draw_2d_court(
        self,
        width: int = 400,
        height: int = 200,
        landing: Optional[Tuple[float, float]] = None,
        decision: Optional[Decision] = None,
        trajectory_court: Optional[List[Tuple[float, float]]] = None
    ) -> np.ndarray:
        """
        Génère une vue 2D top-down du terrain.

        Args:
            width: Largeur de l'image générée
            height: Hauteur de l'image générée
            landing: Point d'atterrissage en coordonnées terrain (cm)
            decision: Décision associée au landing
            trajectory_court: Trajectoire en coordonnées terrain (optionnel)

        Returns:
            Image BGR du terrain 2D
        """
        # Créer l'image de fond
        court = np.zeros((height, width, 3), dtype=np.uint8)

        # Remplir avec la couleur du terrain
        court[:] = self.COLORS["court_fill"]

        # Calculer les facteurs d'échelle
        margin = 20
        usable_width = width - 2 * margin
        usable_height = height - 2 * margin

        scale_x = usable_width / self.config.LENGTH
        scale_y = usable_height / self.config.WIDTH

        def to_screen(x: float, y: float) -> Tuple[int, int]:
            """Convertit coordonnées terrain -> écran."""
            return (
                int(margin + x * scale_x),
                int(margin + y * scale_y)
            )

        # Dessiner les lignes du terrain
        line_color = self.COLORS["court_lines"]
        line_thickness = 2

        # Contour extérieur
        corners = [
            (0, 0), (self.config.LENGTH, 0),
            (self.config.LENGTH, self.config.WIDTH), (0, self.config.WIDTH)
        ]
        for i in range(4):
            pt1 = to_screen(*corners[i])
            pt2 = to_screen(*corners[(i + 1) % 4])
            cv2.line(court, pt1, pt2, line_color, line_thickness)

        # Ligne centrale (filet)
        mid_x = self.config.LENGTH / 2
        cv2.line(
            court,
            to_screen(mid_x, 0),
            to_screen(mid_x, self.config.WIDTH),
            line_color, line_thickness
        )

        # Lignes d'attaque
        attack_left = self.config.ATTACK_LINE
        attack_right = self.config.LENGTH - self.config.ATTACK_LINE

        cv2.line(
            court,
            to_screen(attack_left, 0),
            to_screen(attack_left, self.config.WIDTH),
            line_color, 1
        )
        cv2.line(
            court,
            to_screen(attack_right, 0),
            to_screen(attack_right, self.config.WIDTH),
            line_color, 1
        )

        # Dessiner la trajectoire si fournie
        if trajectory_court and len(trajectory_court) >= 2:
            for i in range(1, len(trajectory_court)):
                pt1 = to_screen(*trajectory_court[i - 1])
                pt2 = to_screen(*trajectory_court[i])
                cv2.line(court, pt1, pt2, self.COLORS["trajectory"], 2)

        # Dessiner le point d'atterrissage
        if landing and decision:
            pt = to_screen(*landing)
            color = self.COLORS.get(decision.verdict, self.COLORS["TOO_CLOSE"])

            # Cercle plein
            cv2.circle(court, pt, 8, color, -1, cv2.LINE_AA)
            cv2.circle(court, pt, 8, (255, 255, 255), 2, cv2.LINE_AA)

        # Ajouter une bordure
        cv2.rectangle(court, (0, 0), (width - 1, height - 1), (100, 100, 100), 1)

        return court

    def draw_court_overlay(
        self,
        frame: np.ndarray,
        keypoints: Optional[np.ndarray] = None,
        draw_lines: bool = True
    ) -> np.ndarray:
        """
        Dessine les lignes du terrain détecté sur la frame.

        Args:
            frame: Image BGR
            keypoints: Keypoints détectés (optionnel)
            draw_lines: Si True, dessine les lignes entre keypoints

        Returns:
            Frame modifiée
        """
        if keypoints is None:
            return frame

        # Dessiner les keypoints
        for i, kp in enumerate(keypoints):
            if kp[0] > 0 and kp[1] > 0:  # Point valide
                center = tuple(map(int, kp))
                cv2.circle(frame, center, 5, (0, 255, 0), -1)

                # Ajouter le nom du keypoint
                if i < len(self.config.KEYPOINT_NAMES):
                    name = self.config.KEYPOINT_NAMES[i][:3]
                    cv2.putText(
                        frame, name,
                        (center[0] + 10, center[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1
                    )

        # Dessiner les lignes du terrain si demandé
        if draw_lines and len(keypoints) >= 4:
            # Connecter les coins
            connections = [(0, 1), (1, 3), (3, 2), (2, 0)]
            for i, j in connections:
                if (keypoints[i][0] > 0 and keypoints[j][0] > 0):
                    pt1 = tuple(map(int, keypoints[i]))
                    pt2 = tuple(map(int, keypoints[j]))
                    cv2.line(frame, pt1, pt2, (0, 255, 0), 2)

        return frame

    def create_composite_view(
        self,
        main_frame: np.ndarray,
        court_2d: np.ndarray,
        position: str = "bottom_right",
        scale: float = 0.3
    ) -> np.ndarray:
        """
        Crée une vue composite avec la frame principale et le terrain 2D.

        Args:
            main_frame: Frame vidéo principale
            court_2d: Vue 2D du terrain
            position: Position de l'incrustation
            scale: Échelle de l'incrustation par rapport à la frame

        Returns:
            Frame composite
        """
        h, w = main_frame.shape[:2]
        court_h, court_w = court_2d.shape[:2]

        # Redimensionner si nécessaire
        target_w = int(w * scale)
        target_h = int(court_h * (target_w / court_w))
        court_resized = cv2.resize(court_2d, (target_w, target_h))

        # Calculer la position
        margin = 10
        if position == "bottom_right":
            x = w - target_w - margin
            y = h - target_h - margin
        elif position == "bottom_left":
            x = margin
            y = h - target_h - margin
        elif position == "top_right":
            x = w - target_w - margin
            y = margin
        else:  # top_left
            x = margin
            y = margin

        # Incruster
        result = main_frame.copy()
        result[y:y + target_h, x:x + target_w] = court_resized

        return result

    def add_frame_info(
        self,
        frame: np.ndarray,
        frame_num: int,
        fps: float = 30.0,
        additional_info: Optional[str] = None
    ) -> np.ndarray:
        """
        Ajoute des informations de frame (numéro, timestamp).

        Args:
            frame: Image BGR
            frame_num: Numéro de la frame
            fps: FPS de la vidéo
            additional_info: Information supplémentaire à afficher

        Returns:
            Frame modifiée
        """
        h, w = frame.shape[:2]

        # Timestamp
        timestamp = frame_num / fps
        time_str = f"Frame: {frame_num} | Time: {timestamp:.2f}s"

        cv2.putText(
            frame, time_str,
            (10, h - 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA
        )

        if additional_info:
            cv2.putText(
                frame, additional_info,
                (10, h - 45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA
            )

        # Logo/titre
        cv2.putText(
            frame, "VOLLEY-REF AI",
            (w - 150, 30),
            cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 165, 255), 2, cv2.LINE_AA
        )

        return frame
