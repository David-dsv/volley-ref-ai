#!/usr/bin/env python3
"""
Fine-tuning du modèle YOLO pour la détection de la balle de volleyball

Ce script entraîne un modèle YOLO pour détecter la balle de volleyball
dans les frames vidéo. Utilise une résolution plus élevée (1280) pour
mieux détecter les petits objets.

Usage:
    python scripts/finetune_ball.py
    python scripts/finetune_ball.py --epochs 200 --batch 8
    python scripts/finetune_ball.py --resume runs/ball/yolo_volleyball_ball/weights/last.pt
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
    "model": "yolo11s.pt",  # Modèle small pour meilleure détection
    "epochs": 150,
    "batch": 16,
    "imgsz": 1280,  # Plus grand pour petits objets
    "patience": 20,
    "optimizer": "AdamW",
    "lr0": 0.001,
    "lrf": 0.01,
    "device": "",  # Auto-detect
    "workers": 8,
    "project": "runs/ball",
    "name": "yolo_volleyball_ball",
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

    return True


def get_class_mapping(data_yaml: Path) -> dict:
    """
    Lit le mapping des classes depuis data.yaml.

    Args:
        data_yaml: Chemin vers data.yaml

    Returns:
        dict: Mapping des classes
    """
    import yaml

    with open(data_yaml) as f:
        data = yaml.safe_load(f)

    return data.get("names", {})


def train(args):
    """
    Lance l'entraînement du modèle de détection de la balle.

    Args:
        args: Arguments de ligne de commande
    """
    # Chemins
    script_dir = Path(__file__).parent.parent
    dataset_path = script_dir / "datasets" / "volleyball_detection"
    data_yaml = dataset_path / "data.yaml"

    # Vérifier le dataset
    if not check_dataset(dataset_path):
        sys.exit(1)

    # Afficher le mapping des classes
    class_mapping = get_class_mapping(data_yaml)

    print("\n" + "="*60)
    print("VOLLEY-REF AI - Ball Detection Fine-tuning")
    print("="*60)
    print(f"Dataset: {dataset_path}")
    print(f"Base model: {args.model}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch}")
    print(f"Image size: {args.imgsz}")
    print(f"\nClasses: {class_mapping}")
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
        "save_period": 10,
        "plots": True,
        # Augmentations spécifiques pour la balle
        "mosaic": 1.0,
        "mixup": 0.1,
        "copy_paste": 0.1,
        "scale": 0.5,  # Augmentation d'échelle
        "flipud": 0.5,  # Flip vertical
        "fliplr": 0.5,  # Flip horizontal
    }

    # Ajouter le device si spécifié
    if args.device:
        train_args["device"] = args.device

    # Ajouter workers si spécifié
    if args.workers:
        train_args["workers"] = args.workers

    # Filtrer uniquement la classe ball si demandé
    if args.ball_only:
        print("Note: Training will focus on 'ball' class detection")
        # Trouver l'index de la classe ball
        ball_idx = None
        for idx, name in class_mapping.items():
            if name.lower() == "ball":
                ball_idx = idx
                break

        if ball_idx is not None:
            train_args["classes"] = [ball_idx]
            print(f"Filtering to class: ball (index {ball_idx})")

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

    target_path = weights_dir / "yolo_volleyball_ball.pt"

    if best_weights.exists():
        import shutil
        shutil.copy(best_weights, target_path)
        print(f"\n✓ Best model copied to: {target_path}")

    print("\n" + "="*60)
    print("SETUP COMPLETE!")
    print("="*60)
    print("Both models are now ready. You can run the demo:")
    print("  python scripts/demo.py --video path/to/your/video.mp4")

    return results


def main():
    """Point d'entrée principal."""
    parser = argparse.ArgumentParser(
        description="Fine-tune YOLO for volleyball ball detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic training
    python scripts/finetune_ball.py

    # Custom parameters (higher resolution for better small object detection)
    python scripts/finetune_ball.py --epochs 200 --batch 8 --imgsz 1280

    # Resume from checkpoint
    python scripts/finetune_ball.py --resume runs/ball/yolo_volleyball_ball/weights/last.pt

    # Train only on ball class
    python scripts/finetune_ball.py --ball-only
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
    parser.add_argument(
        "--ball-only",
        action="store_true",
        help="Train only on ball class (ignore players, net, etc.)"
    )

    args = parser.parse_args()

    # Lancer l'entraînement
    train(args)


if __name__ == "__main__":
    main()
