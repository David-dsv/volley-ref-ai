"""
Module 2: Ball Detection + Tracking

Ce module est responsable de:
- Détecter la balle de volleyball avec YOLO
- Tracker la balle avec un filtre de Kalman
- Maintenir l'historique de la trajectoire
"""

import cv2
import numpy as np
from collections import deque
from typing import Optional, Tuple, List

from ultralytics import YOLO

from .config import ModelConfig


class BallTracker:
    """
    Détecte et track la balle de volleyball.

    Utilise un modèle YOLO fine-tuné pour la détection et un filtre de Kalman
    pour le tracking, permettant de maintenir le suivi même en cas de
    détections manquées.

    Attributes:
        model: Modèle YOLO pour la détection de la balle
        config: Configuration du modèle (seuils, etc.)
        trajectory: Historique des positions (30 derniers points)
        kalman: Filtre de Kalman pour le tracking
    """

    def __init__(self, model_path: str, config: ModelConfig):
        """
        Initialise le tracker de balle.

        Args:
            model_path: Chemin vers le modèle YOLO fine-tuné
            config: Configuration du modèle
        """
        self.model = YOLO(model_path)
        self.config = config
        self.trajectory: deque = deque(maxlen=30)
        self.kalman = self._init_kalman()
        self._frames_without_detection = 0
        self._max_frames_without_detection = 10
        self._last_valid_detection: Optional[Tuple[float, float]] = None

        # Paramètres de filtrage
        self._ignore_top_ratio = 0.0       # Pas de filtre en haut
        self._ignore_bottom_ratio = 0.17   # Ignorer les 17% du bas (HUD)
        self._min_ball_size = 8            # Taille min en pixels
        self._max_ball_size = 80           # Taille max en pixels
        self._max_jump_distance = 200      # Distance max entre deux détections consécutives

    def _init_kalman(self) -> cv2.KalmanFilter:
        """
        Initialise le filtre de Kalman à 4 états (x, y, vx, vy).

        Le filtre modélise la position et la vitesse de la balle,
        permettant de prédire sa position même sans détection.

        Returns:
            cv2.KalmanFilter: Filtre de Kalman initialisé
        """
        kf = cv2.KalmanFilter(4, 2)  # 4 états, 2 mesures

        # Matrice de transition (modèle de mouvement constant)
        # [x]     [1 0 1 0] [x]
        # [y]  =  [0 1 0 1] [y]
        # [vx]    [0 0 1 0] [vx]
        # [vy]    [0 0 0 1] [vy]
        kf.transitionMatrix = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ], dtype=np.float32)

        # Matrice de mesure (on mesure x et y directement)
        kf.measurementMatrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ], dtype=np.float32)

        # Bruit de processus (incertitude du modèle)
        kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03

        # Bruit de mesure (incertitude des détections)
        kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.5

        # Covariance initiale
        kf.errorCovPost = np.eye(4, dtype=np.float32)

        return kf

    def _detect_single(
        self,
        frame: np.ndarray,
        offset_x: int = 0,
        offset_y: int = 0,
        classes: Optional[List[int]] = None
    ) -> List[Tuple[float, np.ndarray]]:
        """
        Détecte la balle dans une frame ou tuile.

        Args:
            frame: Image BGR
            offset_x: Décalage X pour remettre en coordonnées originales
            offset_y: Décalage Y pour remettre en coordonnées originales
            classes: Classes à détecter

        Returns:
            Liste de (confidence_ajustée, box[x,y,w,h])
        """
        results = self.model(
            frame,
            conf=self.config.CONFIDENCE,
            iou=self.config.IOU,
            imgsz=640,  # Taille standard par tuile
            verbose=False,
            classes=classes
        )

        if not results or len(results[0].boxes) == 0:
            return []

        detections = []
        boxes = results[0].boxes

        for i in range(len(boxes)):
            box = boxes.xywh[i].cpu().numpy().copy()
            # Ajuster les coordonnées avec l'offset
            box[0] += offset_x
            box[1] += offset_y
            conf = float(boxes.conf[i])
            cls = int(boxes.cls[i])

            # Bonus si c'est la classe "ball" (classe 0)
            if cls == 0:
                conf *= 1.2

            detections.append((conf, box, cls))

        return detections

    def detect(
        self,
        frame: np.ndarray,
        classes: Optional[List[int]] = None
    ) -> Optional[Tuple[float, float, float, float]]:
        """
        Détecte la balle avec inférence par tuiles (2x2) pour meilleure précision.

        Utilise le tiled inference: découpe l'image en 4 tuiles avec overlap,
        détecte dans chaque tuile, puis fusionne les résultats.

        Filtres appliqués:
        1. Zone verticale: ignore haut (gradins) et bas (HUD)
        2. Taille: la balle doit avoir une taille cohérente
        3. Continuité: si on a une trajectoire, la détection doit être proche

        Args:
            frame: Image BGR de la frame vidéo
            classes: Liste des classes à détecter (None = toutes)

        Returns:
            Tuple (x_center, y_center, width, height) ou None si pas de détection
        """
        frame_height, frame_width = frame.shape[:2]
        min_y = frame_height * self._ignore_top_ratio
        max_y = frame_height * (1 - self._ignore_bottom_ratio)

        # Configuration des tuiles 2x2 avec overlap de 15%
        tiles = 2
        overlap = 0.15
        tile_h = frame_height // tiles
        tile_w = frame_width // tiles
        margin_h = int(tile_h * overlap)
        margin_w = int(tile_w * overlap)

        all_detections = []

        # Parcourir chaque tuile
        for row in range(tiles):
            for col in range(tiles):
                # Calculer les limites de la tuile avec overlap
                y1 = max(0, row * tile_h - margin_h)
                y2 = min(frame_height, (row + 1) * tile_h + margin_h)
                x1 = max(0, col * tile_w - margin_w)
                x2 = min(frame_width, (col + 1) * tile_w + margin_w)

                # Extraire la tuile
                tile = frame[y1:y2, x1:x2]

                # Détecter dans cette tuile
                tile_detections = self._detect_single(tile, x1, y1, classes)
                all_detections.extend(tile_detections)

        if not all_detections:
            return None

        # Appliquer NMS pour enlever les doublons des zones d'overlap
        all_detections = self._nms_detections(all_detections, iou_threshold=0.5)

        # Filtrer les détections selon tous les critères
        valid_detections = []

        for conf, box, cls in all_detections:
            x_center, y_center, width, height = box

            # 1. Filtre de zone (ignorer haut et bas)
            if y_center < min_y or y_center > max_y:
                continue

            # 2. Filtre de taille (la balle a une taille cohérente)
            ball_size = max(width, height)
            if ball_size < self._min_ball_size or ball_size > self._max_ball_size:
                continue

            # 3. Filtre de continuité (si on a une trajectoire)
            if self._last_valid_detection is not None:
                last_x, last_y = self._last_valid_detection
                distance = np.sqrt((x_center - last_x)**2 + (y_center - last_y)**2)
                if distance > self._max_jump_distance:
                    conf *= 0.3

            valid_detections.append((conf, box))

        if not valid_detections:
            return None

        # Prendre la détection avec le meilleur score ajusté
        best = max(valid_detections, key=lambda x: x[0])
        best_box = best[1]

        # Mettre à jour la dernière détection valide
        self._last_valid_detection = (best_box[0], best_box[1])

        return tuple(best_box)

    def _nms_detections(
        self,
        detections: List[Tuple[float, np.ndarray, int]],
        iou_threshold: float = 0.5
    ) -> List[Tuple[float, np.ndarray, int]]:
        """
        Applique Non-Maximum Suppression pour enlever les détections dupliquées.

        Args:
            detections: Liste de (confidence, box[x,y,w,h], class)
            iou_threshold: Seuil IoU pour considérer comme doublon

        Returns:
            Liste filtrée de détections
        """
        if len(detections) <= 1:
            return detections

        # Trier par confidence décroissante
        detections = sorted(detections, key=lambda x: x[0], reverse=True)

        kept = []
        while detections:
            best = detections.pop(0)
            kept.append(best)

            # Filtrer les détections qui overlap trop avec la meilleure
            remaining = []
            for det in detections:
                if self._compute_iou(best[1], det[1]) < iou_threshold:
                    remaining.append(det)
            detections = remaining

        return kept

    def _compute_iou(self, box1: np.ndarray, box2: np.ndarray) -> float:
        """
        Calcule l'Intersection over Union entre deux boxes (format xywh).

        Args:
            box1: [x_center, y_center, width, height]
            box2: [x_center, y_center, width, height]

        Returns:
            IoU entre 0 et 1
        """
        # Convertir xywh -> xyxy
        x1_min = box1[0] - box1[2] / 2
        y1_min = box1[1] - box1[3] / 2
        x1_max = box1[0] + box1[2] / 2
        y1_max = box1[1] + box1[3] / 2

        x2_min = box2[0] - box2[2] / 2
        y2_min = box2[1] - box2[3] / 2
        x2_max = box2[0] + box2[2] / 2
        y2_max = box2[1] + box2[3] / 2

        # Intersection
        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)

        if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
            return 0.0

        inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)

        # Union
        area1 = box1[2] * box1[3]
        area2 = box2[2] * box2[3]
        union_area = area1 + area2 - inter_area

        return inter_area / union_area if union_area > 0 else 0.0

    def update(
        self,
        detection: Optional[Tuple[float, float, float, float]]
    ) -> Tuple[float, float]:
        """
        Met à jour le tracker avec une nouvelle détection.

        Si une détection est fournie, corrige l'estimation du filtre de Kalman.
        Sinon, utilise uniquement la prédiction.

        Args:
            detection: Tuple (x, y, w, h) de la détection ou None

        Returns:
            Tuple (x, y) de la position estimée
        """
        if detection is not None:
            x, y = detection[0], detection[1]
            # Correction avec la mesure
            measurement = np.array([[x], [y]], dtype=np.float32)
            self.kalman.correct(measurement)
            self.trajectory.append((x, y))
            self._frames_without_detection = 0
        else:
            self._frames_without_detection += 1

        # Prédiction
        prediction = self.kalman.predict()
        pred_x, pred_y = prediction[0, 0], prediction[1, 0]

        # Ajouter la prédiction à la trajectoire si pas de détection
        # mais seulement si on n'a pas perdu la balle trop longtemps
        if detection is None and self._frames_without_detection <= self._max_frames_without_detection:
            if self.trajectory:
                self.trajectory.append((pred_x, pred_y))

        return (pred_x, pred_y)

    def get_trajectory(self) -> List[Tuple[float, float]]:
        """
        Retourne l'historique de la trajectoire.

        Returns:
            Liste des positions (x, y) des 30 derniers points
        """
        return list(self.trajectory)

    def get_velocity(self) -> Optional[Tuple[float, float]]:
        """
        Retourne la vitesse estimée de la balle.

        Returns:
            Tuple (vx, vy) ou None si pas assez de données
        """
        state = self.kalman.statePost
        if state is not None:
            return (state[2, 0], state[3, 0])
        return None

    def is_ball_descending(self) -> bool:
        """
        Détermine si la balle est en phase descendante.

        Utile pour déclencher la prédiction du landing uniquement
        quand la balle descend vers le sol.

        Returns:
            True si la balle descend (vy > 0 en coordonnées image)
        """
        velocity = self.get_velocity()
        if velocity is None:
            return False
        return velocity[1] > 0  # En coordonnées image, y augmente vers le bas

    def reset(self):
        """Réinitialise le tracker."""
        self.trajectory.clear()
        self.kalman = self._init_kalman()
        self._frames_without_detection = 0
        self._last_valid_detection = None

    def has_valid_trajectory(self, min_points: int = 5) -> bool:
        """
        Vérifie si la trajectoire est suffisante pour la prédiction.

        Args:
            min_points: Nombre minimum de points requis

        Returns:
            True si la trajectoire est valide
        """
        return len(self.trajectory) >= min_points

    def get_trajectory_direction(self) -> Optional[str]:
        """
        Détermine la direction générale de la trajectoire.

        Returns:
            'left', 'right', 'up', 'down', ou None
        """
        if len(self.trajectory) < 2:
            return None

        start = self.trajectory[0]
        end = self.trajectory[-1]

        dx = end[0] - start[0]
        dy = end[1] - start[1]

        if abs(dx) > abs(dy):
            return 'right' if dx > 0 else 'left'
        else:
            return 'down' if dy > 0 else 'up'
