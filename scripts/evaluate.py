#!/usr/bin/env python3
"""
Script d'évaluation des modèles VOLLEY-REF AI

Ce script évalue les performances des modèles de détection:
- Court keypoints detection (mAP, precision, recall)
- Ball detection (mAP, precision, recall)
- Pipeline end-to-end (accuracy des décisions IN/OUT)

Usage:
    python scripts/evaluate.py --model court
    python scripts/evaluate.py --model ball
    python scripts/evaluate.py --model all
"""

import argparse
import sys
import json
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

try:
    from ultralytics import YOLO
    import cv2
    import numpy as np
except ImportError as e:
    print(f"Error: Missing dependency - {e}")
    print("Install with: pip install ultralytics opencv-python numpy")
    sys.exit(1)


def evaluate_court_model(
    model_path: str,
    dataset_path: Path,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Évalue le modèle de détection des keypoints du terrain.

    Args:
        model_path: Chemin vers le modèle
        dataset_path: Chemin vers le dataset de validation
        verbose: Afficher les détails

    Returns:
        Dict avec les métriques
    """
    print("\n" + "="*60)
    print("EVALUATING COURT KEYPOINTS MODEL")
    print("="*60)

    if not Path(model_path).exists():
        print(f"Error: Model not found at {model_path}")
        return {"error": "Model not found"}

    # Charger le modèle
    model = YOLO(model_path)

    # Évaluer sur le dataset de validation
    data_yaml = dataset_path / "data.yaml"

    if not data_yaml.exists():
        print(f"Error: Dataset not found at {data_yaml}")
        return {"error": "Dataset not found"}

    # Lancer la validation
    results = model.val(
        data=str(data_yaml),
        verbose=verbose,
        plots=True
    )

    # Extraire les métriques
    metrics = {
        "model": model_path,
        "dataset": str(dataset_path),
        "timestamp": datetime.now().isoformat(),
    }

    # Métriques de pose (keypoints)
    if hasattr(results, 'pose'):
        metrics["pose_mAP50"] = float(results.pose.map50)
        metrics["pose_mAP50-95"] = float(results.pose.map)

    # Métriques de box (si disponible)
    if hasattr(results, 'box'):
        metrics["box_mAP50"] = float(results.box.map50)
        metrics["box_mAP50-95"] = float(results.box.map)
        metrics["precision"] = float(results.box.mp)
        metrics["recall"] = float(results.box.mr)

    return metrics


def evaluate_ball_model(
    model_path: str,
    dataset_path: Path,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Évalue le modèle de détection de la balle.

    Args:
        model_path: Chemin vers le modèle
        dataset_path: Chemin vers le dataset de validation
        verbose: Afficher les détails

    Returns:
        Dict avec les métriques
    """
    print("\n" + "="*60)
    print("EVALUATING BALL DETECTION MODEL")
    print("="*60)

    if not Path(model_path).exists():
        print(f"Error: Model not found at {model_path}")
        return {"error": "Model not found"}

    # Charger le modèle
    model = YOLO(model_path)

    # Évaluer sur le dataset de validation
    data_yaml = dataset_path / "data.yaml"

    if not data_yaml.exists():
        print(f"Error: Dataset not found at {data_yaml}")
        return {"error": "Dataset not found"}

    # Lancer la validation
    results = model.val(
        data=str(data_yaml),
        verbose=verbose,
        plots=True
    )

    # Extraire les métriques
    metrics = {
        "model": model_path,
        "dataset": str(dataset_path),
        "timestamp": datetime.now().isoformat(),
        "mAP50": float(results.box.map50),
        "mAP50-95": float(results.box.map),
        "precision": float(results.box.mp),
        "recall": float(results.box.mr),
    }

    # Métriques par classe
    if hasattr(results.box, 'ap_class_index'):
        class_names = model.names
        metrics["per_class"] = {}
        for i, cls_idx in enumerate(results.box.ap_class_index):
            cls_name = class_names[cls_idx]
            metrics["per_class"][cls_name] = {
                "AP50": float(results.box.ap50[i]),
                "AP": float(results.box.ap[i]),
            }

    return metrics


def evaluate_pipeline(
    court_model: str,
    ball_model: str,
    test_video: Optional[str] = None,
    ground_truth: Optional[str] = None
) -> Dict[str, Any]:
    """
    Évalue le pipeline complet end-to-end.

    Args:
        court_model: Chemin vers le modèle terrain
        ball_model: Chemin vers le modèle balle
        test_video: Vidéo de test (optionnel)
        ground_truth: Fichier JSON avec les annotations (optionnel)

    Returns:
        Dict avec les métriques
    """
    print("\n" + "="*60)
    print("EVALUATING FULL PIPELINE")
    print("="*60)

    # Importer le pipeline
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from src.pipeline import VolleyRefAI

    metrics = {
        "timestamp": datetime.now().isoformat(),
        "court_model": court_model,
        "ball_model": ball_model,
    }

    # Vérifier les modèles
    if not Path(court_model).exists():
        print(f"Error: Court model not found at {court_model}")
        metrics["error"] = "Court model not found"
        return metrics

    if not Path(ball_model).exists():
        print(f"Error: Ball model not found at {ball_model}")
        metrics["error"] = "Ball model not found"
        return metrics

    # Créer le pipeline
    try:
        pipeline = VolleyRefAI(
            court_model=court_model,
            ball_model=ball_model
        )
        metrics["pipeline_initialized"] = True
    except Exception as e:
        metrics["error"] = f"Pipeline initialization failed: {e}"
        return metrics

    # Si pas de vidéo de test, juste vérifier l'initialisation
    if not test_video:
        print("No test video provided. Pipeline initialization successful.")
        return metrics

    # Évaluer sur la vidéo de test
    if not Path(test_video).exists():
        print(f"Error: Test video not found at {test_video}")
        metrics["error"] = "Test video not found"
        return metrics

    print(f"Processing test video: {test_video}")

    # Traiter la vidéo
    cap = cv2.VideoCapture(test_video)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    processed_frames = 0
    decisions_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        output, decision = pipeline.process_frame(frame, visualize=False)
        processed_frames += 1

        if decision:
            decisions_count += 1

        if processed_frames % 100 == 0:
            print(f"Processed {processed_frames}/{total_frames} frames...")

    cap.release()

    # Statistiques
    stats = pipeline.get_statistics()
    metrics.update({
        "total_frames": processed_frames,
        "decisions_made": decisions_count,
        "in_count": stats.get("in_count", 0),
        "out_count": stats.get("out_count", 0),
        "too_close_count": stats.get("too_close_count", 0),
        "average_confidence": stats.get("average_confidence", 0),
    })

    # Comparer avec ground truth si fourni
    if ground_truth and Path(ground_truth).exists():
        with open(ground_truth) as f:
            gt_data = json.load(f)

        # Calculer l'accuracy
        correct = 0
        total = 0

        for frame_num, decision in stats.get("decisions", []):
            if str(frame_num) in gt_data:
                total += 1
                gt_verdict = gt_data[str(frame_num)]
                if decision.verdict == gt_verdict:
                    correct += 1

        if total > 0:
            metrics["accuracy"] = correct / total
            metrics["correct_decisions"] = correct
            metrics["total_annotated"] = total

    return metrics


def print_metrics(metrics: Dict[str, Any], title: str = "Results"):
    """
    Affiche les métriques de manière formatée.

    Args:
        metrics: Dict des métriques
        title: Titre à afficher
    """
    print("\n" + "="*60)
    print(title.upper())
    print("="*60)

    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        elif isinstance(value, dict):
            print(f"  {key}:")
            for k, v in value.items():
                if isinstance(v, dict):
                    print(f"    {k}:")
                    for kk, vv in v.items():
                        print(f"      {kk}: {vv:.4f}" if isinstance(vv, float) else f"      {kk}: {vv}")
                else:
                    print(f"    {k}: {v:.4f}" if isinstance(v, float) else f"    {k}: {v}")
        else:
            print(f"  {key}: {value}")


def main():
    """Point d'entrée principal."""
    parser = argparse.ArgumentParser(
        description="Evaluate VOLLEY-REF AI models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Evaluate court keypoints model
    python scripts/evaluate.py --model court

    # Evaluate ball detection model
    python scripts/evaluate.py --model ball

    # Evaluate both models
    python scripts/evaluate.py --model all

    # Evaluate pipeline with test video
    python scripts/evaluate.py --model pipeline --video test.mp4
        """
    )

    parser.add_argument(
        "--model", "-m",
        choices=["court", "ball", "pipeline", "all"],
        default="all",
        help="Model(s) to evaluate"
    )
    parser.add_argument(
        "--court-weights",
        default="weights/yolo_court_keypoints.pt",
        help="Path to court model weights"
    )
    parser.add_argument(
        "--ball-weights",
        default="weights/yolo_volleyball_ball.pt",
        help="Path to ball model weights"
    )
    parser.add_argument(
        "--video",
        help="Test video for pipeline evaluation"
    )
    parser.add_argument(
        "--ground-truth",
        help="JSON file with ground truth annotations"
    )
    parser.add_argument(
        "--output", "-o",
        help="Output JSON file for metrics"
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Reduce output verbosity"
    )

    args = parser.parse_args()

    # Chemins
    script_dir = Path(__file__).parent.parent

    all_metrics = {}
    verbose = not args.quiet

    # Évaluer selon le choix
    if args.model in ["court", "all"]:
        dataset_path = script_dir / "datasets" / "volleyball-court-keypoints-k6y7r"
        metrics = evaluate_court_model(
            args.court_weights,
            dataset_path,
            verbose
        )
        all_metrics["court"] = metrics
        print_metrics(metrics, "Court Model Metrics")

    if args.model in ["ball", "all"]:
        dataset_path = script_dir / "datasets" / "volleyball_detection"
        metrics = evaluate_ball_model(
            args.ball_weights,
            dataset_path,
            verbose
        )
        all_metrics["ball"] = metrics
        print_metrics(metrics, "Ball Model Metrics")

    if args.model in ["pipeline", "all"]:
        metrics = evaluate_pipeline(
            args.court_weights,
            args.ball_weights,
            args.video,
            args.ground_truth
        )
        all_metrics["pipeline"] = metrics
        print_metrics(metrics, "Pipeline Metrics")

    # Sauvegarder les résultats
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(all_metrics, f, indent=2)

        print(f"\n✓ Metrics saved to: {output_path}")


if __name__ == "__main__":
    main()
