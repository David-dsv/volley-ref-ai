"""
Pipeline Principal - VOLLEY-REF AI

Ce module orchestre tous les composants du système:
- Court Detection
- Ball Tracking
- Landing Prediction
- Decision Engine
- Visualization
"""

import cv2
import numpy as np
from typing import Tuple, Optional, List, Callable
from pathlib import Path
from tqdm import tqdm

from .config import CourtConfig, ModelConfig, DecisionConfig
from .court_detector import CourtDetector
from .ball_tracker import BallTracker
from .landing_predictor import LandingPredictor
from .decision_engine import DecisionEngine, Decision
from .visualizer import Visualizer


class VolleyRefAI:
    """
    Pipeline principal du système d'arbitrage VOLLEY-REF AI.

    Orchestre tous les modules pour traiter des frames vidéo
    et produire des décisions IN/OUT avec visualisation.

    Attributes:
        court_config: Configuration du terrain
        model_config: Configuration des modèles
        decision_config: Configuration des décisions
        court_detector: Module de détection du terrain
        ball_tracker: Module de tracking de la balle
        landing_predictor: Module de prédiction d'atterrissage
        decision_engine: Module de décision IN/OUT
        visualizer: Module de visualisation
    """

    def __init__(
        self,
        court_model: str,
        ball_model: str,
        court_config: Optional[CourtConfig] = None,
        model_config: Optional[ModelConfig] = None,
        decision_config: Optional[DecisionConfig] = None,
    ):
        """
        Initialise le pipeline VOLLEY-REF AI.

        Args:
            court_model: Chemin vers le modèle YOLO de détection terrain
            ball_model: Chemin vers le modèle YOLO de détection balle
            court_config: Configuration du terrain (optionnel)
            model_config: Configuration des modèles (optionnel)
            decision_config: Configuration des décisions (optionnel)
        """
        # Configurations
        self.court_config = court_config or CourtConfig()
        self.model_config = model_config or ModelConfig()
        self.decision_config = decision_config or DecisionConfig()

        # Modules
        self.court_detector = CourtDetector(court_model, self.court_config)
        self.ball_tracker = BallTracker(ball_model, self.model_config)
        self.landing_predictor = LandingPredictor(self.court_detector)
        self.decision_engine = DecisionEngine(
            self.court_config, self.decision_config
        )
        self.visualizer = Visualizer(self.court_config)

        # État
        self._frame_count = 0
        self._last_decision: Optional[Decision] = None
        self._decision_history: List[Tuple[int, Decision]] = []

        # État pour détection d'atterrissage
        self._was_descending = False
        self._pending_landing: Optional[Tuple[Tuple[float, float], float]] = None
        self._last_decision_frame = 0
        self._decision_cooldown = 60  # Frames minimum entre deux décisions (2s à 30fps)

    def process_frame(
        self,
        frame: np.ndarray,
        visualize: bool = True,
        show_court_overlay: bool = False
    ) -> Tuple[np.ndarray, Optional[Decision]]:
        """
        Traite une frame complète du pipeline.

        Étapes:
        1. Détection du terrain et calcul de l'homographie
        2. Détection et tracking de la balle
        3. Prédiction du point d'atterrissage
        4. Décision IN/OUT
        5. Visualisation

        Args:
            frame: Image BGR de la frame vidéo
            visualize: Si True, ajoute les overlays visuels
            show_court_overlay: Si True, affiche les keypoints détectés

        Returns:
            Tuple (frame_output, decision):
                - frame_output: Frame avec visualisations
                - decision: Décision IN/OUT ou None si pas de prédiction
        """
        self._frame_count += 1
        output = frame.copy()
        decision = None

        # 1. Détection du terrain
        keypoints = self.court_detector.detect_keypoints(frame)
        if keypoints is not None:
            keypoints = self.court_detector.stabilize_keypoints(keypoints)
            self.court_detector.compute_homography(keypoints)

            if show_court_overlay:
                output = self.visualizer.draw_court_overlay(output, keypoints)

        # 2. Tracking de la balle
        detection = self.ball_tracker.detect(frame)
        ball_position = self.ball_tracker.update(detection)
        trajectory = self.ball_tracker.get_trajectory()

        # 3. Visualiser la trajectoire
        if visualize and len(trajectory) >= 2:
            output = self.visualizer.draw_trajectory(output, trajectory)

        # 4. Prédiction et décision (uniquement sur événement d'atterrissage)
        is_descending = self.ball_tracker.is_ball_descending()
        frames_since_last_decision = self._frame_count - self._last_decision_frame

        if len(trajectory) >= self.decision_config.MIN_TRAJECTORY_LENGTH:
            if is_descending:
                # Balle en descente : prédire le landing mais ne pas décider encore
                result = self.landing_predictor.predict(trajectory)
                if result:
                    self._pending_landing = result
                self._was_descending = True

            elif self._was_descending and self._pending_landing:
                # La balle ÉTAIT en descente et ne l'est plus = atterrissage !
                # Vérifier le cooldown pour éviter les décisions multiples
                if frames_since_last_decision >= self._decision_cooldown:
                    landing_court, confidence = self._pending_landing
                    decision = self.decision_engine.decide(landing_court, confidence)

                    # Stocker la décision
                    self._last_decision = decision
                    self._decision_history.append((self._frame_count, decision))
                    self._last_decision_frame = self._frame_count

                    # Reset
                    self._pending_landing = None
                    self._was_descending = False

        # Visualiser le badge de décision (sans le point d'atterrissage)
        if visualize:
            if self._pending_landing and is_descending:
                # Afficher le badge de prédiction en cours (sans enregistrer de décision)
                landing_court, confidence = self._pending_landing
                temp_decision = self.decision_engine.decide(landing_court, confidence)
                output = self.visualizer.draw_badge(output, temp_decision)
            elif self._last_decision:
                # Afficher la dernière décision finale
                output = self.visualizer.draw_badge(output, self._last_decision)

        # 6. Ajouter la vue 2D du terrain
        if visualize:
            court_2d = self.visualizer.draw_2d_court(
                width=400,
                height=200,
                landing=decision.landing_point if decision else None,
                decision=decision
            )
            output = self.visualizer.create_composite_view(
                output, court_2d, position="bottom_right", scale=0.25
            )

            # Ajouter les infos de frame
            output = self.visualizer.add_frame_info(output, self._frame_count)

        return output, decision

    def process_video(
        self,
        input_path: str,
        output_path: str,
        show_progress: bool = True,
        callback: Optional[Callable[[int, Optional[Decision]], None]] = None,
        max_frames: Optional[int] = None
    ) -> List[Tuple[int, Decision]]:
        """
        Traite une vidéo complète.

        Args:
            input_path: Chemin vers la vidéo d'entrée
            output_path: Chemin vers la vidéo de sortie
            show_progress: Afficher une barre de progression
            callback: Fonction appelée après chaque frame (frame_num, decision)
            max_frames: Nombre maximum de frames à traiter (None = toutes)

        Returns:
            Liste des décisions (frame_num, decision)
        """
        # Ouvrir la vidéo d'entrée
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {input_path}")

        # Propriétés de la vidéo
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Créer le writer de sortie
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        # Réinitialiser l'état
        self.reset()

        # Traitement
        decisions = []
        frames_to_process = min(total_frames, max_frames) if max_frames else total_frames
        iterator = range(frames_to_process)

        if show_progress:
            iterator = tqdm(iterator, desc="Processing video", unit="frame")

        for _ in iterator:
            ret, frame = cap.read()
            if not ret:
                break

            output, decision = self.process_frame(frame)

            if decision:
                decisions.append((self._frame_count, decision))

            out.write(output)

            if callback:
                callback(self._frame_count, decision)

        # Libérer les ressources
        cap.release()
        out.release()

        return decisions

    def process_webcam(
        self,
        camera_id: int = 0,
        window_name: str = "VOLLEY-REF AI",
        quit_key: str = 'q'
    ):
        """
        Traite le flux d'une webcam en temps réel.

        Args:
            camera_id: ID de la caméra
            window_name: Nom de la fenêtre d'affichage
            quit_key: Touche pour quitter
        """
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            raise ValueError(f"Cannot open camera: {camera_id}")

        self.reset()

        print(f"Press '{quit_key}' to quit")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            output, decision = self.process_frame(frame)

            cv2.imshow(window_name, output)

            if cv2.waitKey(1) & 0xFF == ord(quit_key):
                break

        cap.release()
        cv2.destroyAllWindows()

    def process_single_image(
        self,
        image: np.ndarray,
        ball_position: Optional[Tuple[float, float]] = None
    ) -> Tuple[np.ndarray, Optional[Decision]]:
        """
        Traite une seule image (utile pour le debugging).

        Args:
            image: Image BGR
            ball_position: Position manuelle de la balle (optionnel)

        Returns:
            Tuple (image_output, decision)
        """
        self.reset()

        if ball_position:
            # Simuler une trajectoire pour le test
            for i in range(10):
                self.ball_tracker.trajectory.append(
                    (ball_position[0] - i * 10, ball_position[1] - i * 5)
                )

        return self.process_frame(image)

    def reset(self):
        """Réinitialise l'état du pipeline."""
        self._frame_count = 0
        self._last_decision = None
        self._decision_history.clear()

        # Reset état d'atterrissage
        self._was_descending = False
        self._pending_landing = None
        self._last_decision_frame = 0

        self.court_detector.reset()
        self.ball_tracker.reset()
        self.landing_predictor.reset()

    def get_statistics(self) -> dict:
        """
        Retourne les statistiques de la session.

        Returns:
            Dict avec les statistiques
        """
        if not self._decision_history:
            return {
                "total_frames": self._frame_count,
                "total_decisions": 0,
                "in_count": 0,
                "out_count": 0,
                "too_close_count": 0,
            }

        verdicts = [d.verdict for _, d in self._decision_history]

        return {
            "total_frames": self._frame_count,
            "total_decisions": len(self._decision_history),
            "in_count": verdicts.count("IN"),
            "out_count": verdicts.count("OUT"),
            "too_close_count": verdicts.count("TOO_CLOSE"),
            "average_confidence": np.mean([
                d.confidence for _, d in self._decision_history
            ]),
            "decisions": self._decision_history
        }

    def export_decisions(self, output_path: str):
        """
        Exporte les décisions en CSV.

        Args:
            output_path: Chemin du fichier CSV
        """
        import csv

        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'frame', 'verdict', 'distance_cm', 'confidence',
                'closest_line', 'landing_x', 'landing_y'
            ])

            for frame_num, decision in self._decision_history:
                writer.writerow([
                    frame_num,
                    decision.verdict,
                    f"{decision.distance_to_line:.2f}",
                    f"{decision.confidence:.3f}",
                    decision.closest_line,
                    f"{decision.landing_point[0]:.2f}",
                    f"{decision.landing_point[1]:.2f}"
                ])

        print(f"Decisions exported to {output_path}")


def create_demo_pipeline(
    court_model: str = "weights/yolo_court_keypoints.pt",
    ball_model: str = "weights/yolo_volleyball_ball.pt"
) -> VolleyRefAI:
    """
    Fonction utilitaire pour créer un pipeline avec les paramètres par défaut.

    Args:
        court_model: Chemin vers le modèle terrain
        ball_model: Chemin vers le modèle balle

    Returns:
        Instance de VolleyRefAI configurée
    """
    return VolleyRefAI(
        court_model=court_model,
        ball_model=ball_model
    )
