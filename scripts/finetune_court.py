#!/usr/bin/env python3
"""
Fine-tuning du modèle YOLO pour la détection des keypoints du terrain

Ce script entraîne un modèle YOLO-pose pour détecter les 14 keypoints
du terrain de volleyball, permettant ensuite de calculer l'homographie.

Usage:
    python scripts/finetune_court.py
    python scripts/finetune_court.py --epochs 150 --batch 8
    python scripts/finetune_court.py --resume runs/court/yolo_court_keypoints/weights/last.pt
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime

try:
    from ultralytics import YOLO
except ImportError:
    print("Error: ultralytics package not installed.")
    print("Install it with: pip install ultralytics")
    sys.exit(1)


# Configuration par défaut
DEFAULT_CONFIG = {
    "model": "yolo11n-pose.pt",  # Modèle de base (pose estimation)
    "epochs": 100,
    "batch": 16,
    "imgsz": 640,
    "patience": 20,
    "optimizer": "AdamW",
    "lr0": 0.001,
    "lrf": 0.01,
    "device": "",  # Auto-detect (GPU si disponible)
    "workers": 8,
    "project": "runs/court",
    "name": "yolo_court_keypoints",
}


def check_dataset(dataset_path: Path) -> bool:
    """
    Vérifie que le dataset existe et est valide.

    Args:
        dataset_path: Chemin vers le dataset

    Returns:
        bool: True si le dataset est valide
    """
    data_yaml = dataset_path / "data.yaml"

    if not data_yaml.exists():
        print(f"Error: Dataset not found at {dataset_path}")
        print(f"Expected data.yaml at: {data_yaml}")
        print("\nRun first: python scripts/download_datasets.py")
        return False

    # Vérifier les dossiers d'images
    for split in ["train", "valid"]:
        split_dir = dataset_path / split / "images"
        if not split_dir.exists():
            alt_split_dir = dataset_path / split
            if not alt_split_dir.exists():
                print(f"Warning: {split} directory not found")

    return True


def train(args):
    """
    Lance l'entraînement du modèle de détection des keypoints.

    Args:
        args: Arguments de ligne de commande
    """
    # Chemins
    script_dir = Path(__file__).parent.parent
    dataset_path = script_dir / "datasets" / "volleyball-court-keypoints-k6y7r"
    data_yaml = dataset_path / "data.yaml"

    # Vérifier le dataset
    if not check_dataset(dataset_path):
        sys.exit(1)

    print("\n" + "="*60)
    print("VOLLEY-REF AI - Court Keypoints Fine-tuning")
    print("="*60)
    print(f"Dataset: {dataset_path}")
    print(f"Base model: {args.model}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch}")
    print(f"Image size: {args.imgsz}")
    print("="*60 + "\n")

    # Charger le modèle de base
    if args.resume:
        print(f"Resuming from: {args.resume}")
        model = YOLO(args.resume)
    else:
        print(f"Loading base model: {args.model}")
        model = YOLO(args.model)

    # Configuration de l'entraînement
    train_args = {
        "data": str(data_yaml),
        "epochs": args.epochs,
        "batch": args.batch,
        "imgsz": args.imgsz,
        "patience": args.patience,
        "optimizer": args.optimizer,
        "lr0": args.lr0,
        "lrf": args.lrf,
        "project": args.project,
        "name": args.name,
        "exist_ok": True,
        "pretrained": True,
        "verbose": True,
        "save": True,
        "save_period": 10,  # Sauvegarder tous les 10 epochs
        "plots": True,
    }

    # Ajouter le device si spécifié
    if args.device:
        train_args["device"] = args.device

    # Ajouter workers si spécifié
    if args.workers:
        train_args["workers"] = args.workers

    # Lancer l'entraînement
    print("Starting training...")
    results = model.train(**train_args)

    # Afficher les résultats
    print("\n" + "="*60)
    print("TRAINING COMPLETED")
    print("="*60)

    # Chemin des poids
    best_weights = Path(args.project) / args.name / "weights" / "best.pt"
    last_weights = Path(args.project) / args.name / "weights" / "last.pt"

    print(f"Best weights: {best_weights}")
    print(f"Last weights: {last_weights}")

    # Copier les meilleurs poids dans le dossier weights/
    weights_dir = script_dir / "weights"
    weights_dir.mkdir(exist_ok=True)

    target_path = weights_dir / "yolo_court_keypoints.pt"

    if best_weights.exists():
        import shutil
        shutil.copy(best_weights, target_path)
        print(f"\n✓ Best model copied to: {target_path}")

    print("\nNext step: python scripts/finetune_ball.py")

    return results


def main():
    """Point d'entrée principal."""
    parser = argparse.ArgumentParser(
        description="Fine-tune YOLO for volleyball court keypoint detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic training
    python scripts/finetune_court.py

    # Custom parameters
    python scripts/finetune_court.py --epochs 150 --batch 8

    # Resume from checkpoint
    python scripts/finetune_court.py --resume runs/court/yolo_court_keypoints/weights/last.pt

    # Use specific GPU
    python scripts/finetune_court.py --device 0
        """
    )

    parser.add_argument(
        "--model", "-m",
        default=DEFAULT_CONFIG["model"],
        help=f"Base model to fine-tune (default: {DEFAULT_CONFIG['model']})"
    )
    parser.add_argument(
        "--epochs", "-e",
        type=int,
        default=DEFAULT_CONFIG["epochs"],
        help=f"Number of training epochs (default: {DEFAULT_CONFIG['epochs']})"
    )
    parser.add_argument(
        "--batch", "-b",
        type=int,
        default=DEFAULT_CONFIG["batch"],
        help=f"Batch size (default: {DEFAULT_CONFIG['batch']})"
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=DEFAULT_CONFIG["imgsz"],
        help=f"Image size (default: {DEFAULT_CONFIG['imgsz']})"
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=DEFAULT_CONFIG["patience"],
        help=f"Early stopping patience (default: {DEFAULT_CONFIG['patience']})"
    )
    parser.add_argument(
        "--optimizer",
        default=DEFAULT_CONFIG["optimizer"],
        choices=["SGD", "Adam", "AdamW", "RMSProp"],
        help=f"Optimizer (default: {DEFAULT_CONFIG['optimizer']})"
    )
    parser.add_argument(
        "--lr0",
        type=float,
        default=DEFAULT_CONFIG["lr0"],
        help=f"Initial learning rate (default: {DEFAULT_CONFIG['lr0']})"
    )
    parser.add_argument(
        "--lrf",
        type=float,
        default=DEFAULT_CONFIG["lrf"],
        help=f"Final learning rate factor (default: {DEFAULT_CONFIG['lrf']})"
    )
    parser.add_argument(
        "--device",
        default=DEFAULT_CONFIG["device"],
        help="Device to use (e.g., 0, 0,1, cpu). Default: auto-detect"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_CONFIG["workers"],
        help=f"Number of data loading workers (default: {DEFAULT_CONFIG['workers']})"
    )
    parser.add_argument(
        "--project",
        default=DEFAULT_CONFIG["project"],
        help=f"Project directory (default: {DEFAULT_CONFIG['project']})"
    )
    parser.add_argument(
        "--name",
        default=DEFAULT_CONFIG["name"],
        help=f"Experiment name (default: {DEFAULT_CONFIG['name']})"
    )
    parser.add_argument(
        "--resume",
        help="Path to checkpoint to resume from"
    )

    args = parser.parse_args()

    # Lancer l'entraînement
    train(args)


if __name__ == "__main__":
    main()
