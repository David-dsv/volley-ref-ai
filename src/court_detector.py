"""
Module 1: Court Detection + Homography

Ce module est responsable de:
- Détecter les 14 keypoints du terrain de volleyball
- Stabiliser les détections avec un filtre temporel
- Calculer la matrice d'homographie pour projeter les points image vers terrain
"""

import cv2
import numpy as np
from collections import deque
from typing import Optional, Tuple, List, Dict
from ultralytics import YOLO

from .config import CourtConfig


class CourtDetector:
    """
    Détecte les keypoints du terrain et calcule l'homographie.

    Cette classe utilise un modèle YOLO-pose fine-tuné pour détecter les 14 keypoints
    du terrain de volleyball, puis calcule la matrice d'homographie permettant
    de transformer les coordonnées image en coordonnées terrain réelles.

    Attributes:
        model: Modèle YOLO pour la détection des keypoints
        config: Configuration du terrain (dimensions, noms des keypoints)
        keypoint_history: Historique des détections pour stabilisation temporelle
        homography_matrix: Matrice d'homographie courante
    """

    def __init__(self, model_path: str, config: CourtConfig):
        """
        Initialise le détecteur de terrain.

        Args:
            model_path: Chemin vers le modèle YOLO-pose fine-tuné
            config: Configuration du terrain de volleyball
        """
        self.model = YOLO(model_path)
        self.config = config
        self.keypoint_history: Dict[str, deque] = {
            name: deque(maxlen=10)
            for name in config.KEYPOINT_NAMES
        }
        self.homography_matrix: Optional[np.ndarray] = None
        self._last_valid_keypoints: Optional[np.ndarray] = None

    def detect_keypoints(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Détecte les 14 keypoints du terrain dans une frame.

        Args:
            frame: Image BGR de la frame vidéo

        Returns:
            np.ndarray ou None: Array (14, 2) des coordonnées (x, y) des keypoints,
                               ou None si aucune détection
        """
        results = self.model(frame, verbose=False)

        if not results or results[0].keypoints is None:
            return self._last_valid_keypoints

        keypoints_data = results[0].keypoints

        # Vérifier si des keypoints ont été détectés
        if keypoints_data.xy is None or len(keypoints_data.xy) == 0:
            return self._last_valid_keypoints

        keypoints = keypoints_data.xy.cpu().numpy()

        if keypoints.shape[0] > 0:
            self._last_valid_keypoints = keypoints[0]
            return keypoints[0]

        return self._last_valid_keypoints

    def stabilize_keypoints(self, keypoints: np.ndarray) -> np.ndarray:
        """
        Applique un filtre temporel pour stabiliser les keypoints.

        Utilise une moyenne glissante sur les 10 dernières détections
        pour réduire le bruit et les oscillations.

        Args:
            keypoints: Keypoints bruts détectés

        Returns:
            np.ndarray: Keypoints stabilisés
        """
        stabilized = []

        for i, name in enumerate(self.config.KEYPOINT_NAMES):
            if i < len(keypoints):
                point = keypoints[i]
                # Ne pas ajouter les points invalides (0, 0)
                if point[0] > 0 and point[1] > 0:
                    self.keypoint_history[name].append(point)

            if self.keypoint_history[name]:
                # Moyenne des dernières détections valides
                avg = np.mean(list(self.keypoint_history[name]), axis=0)
                stabilized.append(avg)
            else:
                # Point non détecté
                stabilized.append([0, 0])

        return np.array(stabilized, dtype=np.float32)

    def compute_homography(
        self,
        image_keypoints: np.ndarray,
        method: int = cv2.RANSAC,
        reproj_threshold: float = 5.0
    ) -> Optional[np.ndarray]:
        """
        Calcule la matrice d'homographie image -> terrain réel.

        La matrice permet de transformer n'importe quel point des coordonnées
        image vers les coordonnées terrain en centimètres.

        Args:
            image_keypoints: Keypoints détectés dans l'image
            method: Méthode de calcul (RANSAC recommandé pour robustesse)
            reproj_threshold: Seuil de reprojection pour RANSAC

        Returns:
            np.ndarray ou None: Matrice d'homographie 3x3, ou None si échec
        """
        if image_keypoints is None or len(image_keypoints) < 4:
            return self.homography_matrix

        # Filtrer les points valides (non nuls)
        valid_mask = (image_keypoints[:, 0] > 0) & (image_keypoints[:, 1] > 0)
        valid_indices = np.where(valid_mask)[0]

        if len(valid_indices) < 4:
            return self.homography_matrix

        # Utiliser les 4 premiers points valides (coins du terrain)
        src_points = []
        dst_points = []

        for idx in valid_indices[:8]:  # Utiliser jusqu'à 8 points pour plus de précision
            src_points.append(image_keypoints[idx])
            dst_points.append(self.config.real_keypoints[idx])

        src = np.array(src_points, dtype=np.float32)
        dst = np.array(dst_points, dtype=np.float32)

        try:
            self.homography_matrix, mask = cv2.findHomography(
                src, dst, method, reproj_threshold
            )
        except cv2.error:
            pass

        return self.homography_matrix

    def transform_point(
        self,
        point: Tuple[float, float]
    ) -> Optional[Tuple[float, float]]:
        """
        Transforme un point image vers coordonnées terrain.

        Args:
            point: Coordonnées (x, y) dans l'image

        Returns:
            Tuple[float, float] ou None: Coordonnées terrain (x, y) en cm,
                                         ou None si pas d'homographie
        """
        if self.homography_matrix is None:
            return None

        pt = np.array([[point]], dtype=np.float32)
        transformed = cv2.perspectiveTransform(pt, self.homography_matrix)

        return tuple(transformed[0][0])

    def transform_points(
        self,
        points: List[Tuple[float, float]]
    ) -> Optional[List[Tuple[float, float]]]:
        """
        Transforme plusieurs points image vers coordonnées terrain.

        Args:
            points: Liste de coordonnées (x, y) dans l'image

        Returns:
            Liste de coordonnées terrain ou None si pas d'homographie
        """
        if self.homography_matrix is None or not points:
            return None

        pts = np.array([points], dtype=np.float32)
        transformed = cv2.perspectiveTransform(pts, self.homography_matrix)

        return [tuple(pt) for pt in transformed[0]]

    def inverse_transform_point(
        self,
        point: Tuple[float, float]
    ) -> Optional[Tuple[float, float]]:
        """
        Transforme un point terrain vers coordonnées image (transformation inverse).

        Args:
            point: Coordonnées terrain (x, y) en cm

        Returns:
            Tuple[float, float] ou None: Coordonnées image (x, y)
        """
        if self.homography_matrix is None:
            return None

        inv_matrix = np.linalg.inv(self.homography_matrix)
        pt = np.array([[point]], dtype=np.float32)
        transformed = cv2.perspectiveTransform(pt, inv_matrix)

        return tuple(transformed[0][0])

    def get_court_polygon_image(self) -> Optional[np.ndarray]:
        """
        Retourne les coins du terrain en coordonnées image.

        Returns:
            np.ndarray ou None: Array (4, 2) des coins du terrain dans l'image
        """
        if self.homography_matrix is None:
            return None

        corners_court = self.config.court_corners
        corners_image = []

        for corner in corners_court:
            img_pt = self.inverse_transform_point(tuple(corner))
            if img_pt:
                corners_image.append(img_pt)

        if len(corners_image) == 4:
            return np.array(corners_image, dtype=np.int32)

        return None

    def reset(self):
        """Réinitialise l'état du détecteur."""
        self.keypoint_history = {
            name: deque(maxlen=10)
            for name in self.config.KEYPOINT_NAMES
        }
        self.homography_matrix = None
        self._last_valid_keypoints = None
