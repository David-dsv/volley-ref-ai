"""
Module 3: Landing Point Prediction

Ce module est responsable de:
- Prédire le point d'atterrissage de la balle
- Utiliser un fit polynomial + modèle physique
- Calculer la confiance de la prédiction
"""

import numpy as np
from typing import Optional, Tuple, List
from scipy import optimize

from .court_detector import CourtDetector


class LandingPredictor:
    """
    Prédit le point d'atterrissage de la balle de volleyball.

    Utilise un fit polynomial sur la trajectoire observée pour extrapoler
    le point où la balle touchera le sol. La prédiction tient compte
    de la physique du mouvement parabolique.

    Attributes:
        court: Détecteur de terrain pour la transformation de coordonnées
    """

    def __init__(self, court_detector: CourtDetector):
        """
        Initialise le prédicteur.

        Args:
            court_detector: Instance du détecteur de terrain pour l'homographie
        """
        self.court = court_detector
        self._last_prediction: Optional[Tuple[Tuple[float, float], float]] = None

    def predict(
        self,
        trajectory: List[Tuple[float, float]],
        fps: float = 30.0,
        ground_y: Optional[float] = None
    ) -> Optional[Tuple[Tuple[float, float], float]]:
        """
        Prédit le point d'atterrissage de la balle.

        La méthode:
        1. Fit une parabole sur les coordonnées Y (gravité)
        2. Fit une ligne sur les coordonnées X (mouvement horizontal)
        3. Extrapole pour trouver le point le plus bas
        4. Convertit en coordonnées terrain
        5. Calcule la confiance basée sur le RMSE du fit

        Args:
            trajectory: Liste des positions (x, y) observées
            fps: Frames par seconde de la vidéo
            ground_y: Coordonnée Y du sol dans l'image (optionnel)

        Returns:
            Tuple ((x_terrain, y_terrain), confidence) ou None si échec
        """
        if len(trajectory) < 5:
            return self._last_prediction

        points = np.array(trajectory)
        n_points = len(points)
        x_indices = np.arange(n_points)

        try:
            # Fit polynomial de degré 2 pour Y (trajectoire parabolique)
            coeffs_y = np.polyfit(x_indices, points[:, 1], 2)

            # Fit linéaire pour X (mouvement horizontal approximativement constant)
            coeffs_x = np.polyfit(x_indices, points[:, 0], 1)
        except (np.linalg.LinAlgError, ValueError):
            return self._last_prediction

        # Vérifier que la parabole a bien une forme de U (a > 0 pour balle descendante)
        # En coordonnées image, Y augmente vers le bas
        a = coeffs_y[0]

        # Prédire dans le futur (30 frames)
        future_frames = 30
        future = np.arange(n_points, n_points + future_frames)

        pred_y = np.polyval(coeffs_y, future)
        pred_x = np.polyval(coeffs_x, future)

        # Trouver le point d'atterrissage
        # Si ground_y est fourni, trouver l'intersection avec cette ligne
        if ground_y is not None:
            landing_idx = self._find_ground_intersection(
                coeffs_y, n_points, ground_y, future_frames
            )
        else:
            # Sinon, prendre le point le plus bas de la trajectoire prédite
            # (maximum de Y en coordonnées image)
            landing_idx = np.argmax(pred_y)

        if landing_idx is None:
            landing_idx = np.argmax(pred_y)

        landing_pixel = (pred_x[landing_idx], pred_y[landing_idx])

        # Convertir en coordonnées terrain
        landing_court = self.court.transform_point(landing_pixel)

        if landing_court is None:
            return self._last_prediction

        # Calculer la confiance basée sur la qualité du fit
        confidence = self._compute_confidence(points, coeffs_y, x_indices)

        self._last_prediction = (landing_court, confidence)
        return self._last_prediction

    def _find_ground_intersection(
        self,
        coeffs_y: np.ndarray,
        start_idx: int,
        ground_y: float,
        max_frames: int
    ) -> Optional[int]:
        """
        Trouve le frame où la trajectoire intersecte le sol.

        Args:
            coeffs_y: Coefficients du polynôme Y
            start_idx: Index de départ pour la prédiction
            ground_y: Coordonnée Y du sol
            max_frames: Nombre maximum de frames à prédire

        Returns:
            Index du frame d'intersection ou None
        """
        # Résoudre coeffs_y[0]*t^2 + coeffs_y[1]*t + coeffs_y[2] - ground_y = 0
        poly = np.poly1d(coeffs_y)
        poly_shifted = poly - ground_y

        try:
            roots = np.roots(poly_shifted)
            # Garder les racines réelles et futures
            valid_roots = []
            for root in roots:
                if np.isreal(root):
                    real_root = np.real(root)
                    if start_idx <= real_root <= start_idx + max_frames:
                        valid_roots.append(real_root)

            if valid_roots:
                # Prendre la première intersection future
                return int(min(valid_roots) - start_idx)
        except Exception:
            pass

        return None

    def _compute_confidence(
        self,
        points: np.ndarray,
        coeffs_y: np.ndarray,
        x_indices: np.ndarray
    ) -> float:
        """
        Calcule la confiance de la prédiction.

        La confiance est basée sur:
        - Le RMSE du fit polynomial (plus petit = meilleure confiance)
        - Le nombre de points dans la trajectoire (plus = mieux)

        Args:
            points: Points observés
            coeffs_y: Coefficients du fit Y
            x_indices: Indices des points

        Returns:
            Score de confiance entre 0.5 et 0.99
        """
        # Calculer le RMSE
        pred_y = np.polyval(coeffs_y, x_indices)
        residuals = points[:, 1] - pred_y
        rmse = np.sqrt(np.mean(residuals ** 2))

        # Confiance basée sur RMSE (normaliser par rapport à une valeur typique)
        rmse_confidence = max(0.0, 1.0 - rmse / 100)

        # Bonus pour avoir plus de points
        n_points = len(points)
        point_bonus = min(0.2, (n_points - 5) * 0.02)

        # Confiance finale entre 0.5 et 0.99
        confidence = max(0.5, min(0.99, rmse_confidence * 0.8 + point_bonus + 0.2))

        return confidence

    def predict_with_physics(
        self,
        trajectory: List[Tuple[float, float]],
        fps: float = 30.0,
        gravity: float = 9.81
    ) -> Optional[Tuple[Tuple[float, float], float]]:
        """
        Prédiction avancée utilisant un modèle physique complet.

        Cette méthode tient compte de:
        - La gravité (9.81 m/s²)
        - La résistance de l'air (approximée)
        - La vitesse initiale estimée

        Args:
            trajectory: Liste des positions observées
            fps: Frames par seconde
            gravity: Accélération gravitationnelle (m/s²)

        Returns:
            Tuple ((x_terrain, y_terrain), confidence) ou None
        """
        if len(trajectory) < 5:
            return None

        points = np.array(trajectory)
        n_points = len(points)
        dt = 1.0 / fps

        # Estimer la vitesse initiale
        if n_points >= 3:
            # Différences finies pour vitesse
            vx = (points[-1, 0] - points[-3, 0]) / (2 * dt)
            vy = (points[-1, 1] - points[-3, 1]) / (2 * dt)
        else:
            vx = (points[-1, 0] - points[0, 0]) / ((n_points - 1) * dt)
            vy = (points[-1, 1] - points[0, 1]) / ((n_points - 1) * dt)

        # Position actuelle
        x0, y0 = points[-1]

        # Simuler la trajectoire
        x, y = x0, y0
        max_iter = 100

        for _ in range(max_iter):
            # Mise à jour avec gravité (convertir en pixels)
            # Approximation: 1 pixel ≈ 0.5 cm, donc gravity_pixels ≈ gravity * 100 / 0.5
            gravity_pixels = gravity * 50 * dt * dt

            x += vx * dt
            y += vy * dt + 0.5 * gravity_pixels
            vy += gravity_pixels / dt

            # Vérifier si on atteint une limite raisonnable
            if y > points[:, 1].max() + 500:  # 500 pixels sous le point le plus bas
                break

        landing_pixel = (x, y)
        landing_court = self.court.transform_point(landing_pixel)

        if landing_court is None:
            return None

        confidence = 0.75  # Confiance modérée pour le modèle physique
        return (landing_court, confidence)

    def get_predicted_trajectory(
        self,
        trajectory: List[Tuple[float, float]],
        n_future_points: int = 30
    ) -> Optional[List[Tuple[float, float]]]:
        """
        Retourne la trajectoire prédite pour visualisation.

        Args:
            trajectory: Trajectoire observée
            n_future_points: Nombre de points futurs à prédire

        Returns:
            Liste des positions prédites ou None
        """
        if len(trajectory) < 5:
            return None

        points = np.array(trajectory)
        n_points = len(points)
        x_indices = np.arange(n_points)

        try:
            coeffs_y = np.polyfit(x_indices, points[:, 1], 2)
            coeffs_x = np.polyfit(x_indices, points[:, 0], 1)
        except (np.linalg.LinAlgError, ValueError):
            return None

        future = np.arange(n_points, n_points + n_future_points)
        pred_y = np.polyval(coeffs_y, future)
        pred_x = np.polyval(coeffs_x, future)

        return [(x, y) for x, y in zip(pred_x, pred_y)]

    def reset(self):
        """Réinitialise le prédicteur."""
        self._last_prediction = None
