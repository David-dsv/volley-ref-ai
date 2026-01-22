#!/usr/bin/env python3
"""
Script de téléchargement des datasets depuis Roboflow

Ce script télécharge les datasets nécessaires pour l'entraînement:
1. volleyball-court-keypoints: Détection des keypoints du terrain
2. volleyball_detection: Détection de la balle et des joueurs

Usage:
    python scripts/download_datasets.py --api-key YOUR_ROBOFLOW_API_KEY

    # Ou avec la variable d'environnement
    export ROBOFLOW_API_KEY=your_key
    python scripts/download_datasets.py
"""

import os
import sys
import argparse
from pathlib import Path
from typing import List, Tuple

try:
    from roboflow import Roboflow
except ImportError:
    print("Error: roboflow package not installed.")
    print("Install it with: pip install roboflow")
    sys.exit(1)


# Configuration des datasets
DATASETS: List[Tuple[str, str, int, str]] = [
    # (workspace, project, version, format)
    ("volleyballcourt", "volleyball-court-keypoints-k6y7r", 1, "yolov8"),
    ("aivolleyballref", "volleyball_detection", 2, "yolo26"),
]

# Datasets optionnels
OPTIONAL_DATASETS: List[Tuple[str, str, int, str]] = [
    ("activity-graz-uni", "volleyball-activity-dataset", 1, "yolov8"),
]


def get_api_key(args_key: str = None) -> str:
    """
    Récupère la clé API Roboflow.

    Ordre de priorité:
    1. Argument de ligne de commande
    2. Variable d'environnement ROBOFLOW_API_KEY
    3. Demande interactive

    Args:
        args_key: Clé fournie en argument (optionnel)

    Returns:
        str: Clé API Roboflow
    """
    if args_key:
        return args_key

    env_key = os.environ.get("ROBOFLOW_API_KEY")
    if env_key:
        return env_key

    print("No API key found.")
    print("Get your free API key at: https://app.roboflow.com/settings/api")
    key = input("Enter your Roboflow API key: ").strip()

    if not key:
        print("Error: API key is required")
        sys.exit(1)

    return key


def download_dataset(
    rf: Roboflow,
    workspace: str,
    project_name: str,
    version: int,
    format: str,
    output_dir: Path
) -> bool:
    """
    Télécharge un dataset depuis Roboflow.

    Args:
        rf: Instance Roboflow authentifiée
        workspace: Nom du workspace
        project_name: Nom du projet
        version: Version du dataset
        format: Format de téléchargement (yolov8, coco, etc.)
        output_dir: Répertoire de sortie

    Returns:
        bool: True si succès, False sinon
    """
    try:
        print(f"\n{'='*60}")
        print(f"Downloading: {project_name} (v{version})")
        print(f"Workspace: {workspace}")
        print(f"Format: {format}")
        print(f"{'='*60}")

        # Accéder au workspace et projet
        project = rf.workspace(workspace).project(project_name)

        # Télécharger la version spécifiée
        dataset_path = output_dir / project_name
        dataset = project.version(version).download(
            format,
            location=str(dataset_path)
        )

        print(f"✓ Downloaded to: {dataset_path}")

        # Afficher les informations du dataset
        print(f"  - Images: {dataset.location}")

        return True

    except Exception as e:
        print(f"✗ Error downloading {project_name}: {e}")
        return False


def download_all(
    api_key: str,
    output_dir: Path,
    include_optional: bool = False
) -> Tuple[int, int]:
    """
    Télécharge tous les datasets configurés.

    Args:
        api_key: Clé API Roboflow
        output_dir: Répertoire de sortie
        include_optional: Inclure les datasets optionnels

    Returns:
        Tuple (succès, échecs)
    """
    # Initialiser Roboflow
    print("Initializing Roboflow...")
    rf = Roboflow(api_key=api_key)

    # Créer le répertoire de sortie
    output_dir.mkdir(parents=True, exist_ok=True)

    # Liste des datasets à télécharger
    datasets_to_download = list(DATASETS)
    if include_optional:
        datasets_to_download.extend(OPTIONAL_DATASETS)

    success_count = 0
    fail_count = 0

    for workspace, project, version, fmt in datasets_to_download:
        success = download_dataset(
            rf, workspace, project, version, fmt, output_dir
        )
        if success:
            success_count += 1
        else:
            fail_count += 1

    return success_count, fail_count


def verify_datasets(output_dir: Path) -> List[str]:
    """
    Vérifie quels datasets sont déjà téléchargés.

    Args:
        output_dir: Répertoire des datasets

    Returns:
        Liste des datasets manquants
    """
    missing = []

    for workspace, project, version, fmt in DATASETS:
        dataset_path = output_dir / project
        data_yaml = dataset_path / "data.yaml"

        if not data_yaml.exists():
            missing.append(project)

    return missing


def main():
    """Point d'entrée principal."""
    parser = argparse.ArgumentParser(
        description="Download volleyball datasets from Roboflow",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/download_datasets.py --api-key YOUR_KEY
    python scripts/download_datasets.py --include-optional
    python scripts/download_datasets.py --verify
        """
    )

    parser.add_argument(
        "--api-key", "-k",
        help="Roboflow API key (or set ROBOFLOW_API_KEY env var)"
    )
    parser.add_argument(
        "--output-dir", "-o",
        default="datasets",
        help="Output directory for datasets (default: datasets)"
    )
    parser.add_argument(
        "--include-optional",
        action="store_true",
        help="Include optional datasets (volleyball-activity)"
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Only verify which datasets are missing"
    )

    args = parser.parse_args()

    # Déterminer le répertoire de sortie
    script_dir = Path(__file__).parent.parent
    output_dir = script_dir / args.output_dir

    # Mode vérification
    if args.verify:
        print("Checking for existing datasets...")
        missing = verify_datasets(output_dir)

        if missing:
            print(f"\nMissing datasets ({len(missing)}):")
            for name in missing:
                print(f"  - {name}")
            print("\nRun without --verify to download them.")
        else:
            print("\n✓ All required datasets are present!")

        return

    # Téléchargement
    api_key = get_api_key(args.api_key)

    print("\n" + "="*60)
    print("VOLLEY-REF AI - Dataset Downloader")
    print("="*60)
    print(f"Output directory: {output_dir}")
    print(f"Include optional: {args.include_optional}")

    success, failures = download_all(
        api_key,
        output_dir,
        args.include_optional
    )

    # Résumé
    print("\n" + "="*60)
    print("DOWNLOAD SUMMARY")
    print("="*60)
    print(f"✓ Successful: {success}")
    print(f"✗ Failed: {failures}")

    if failures > 0:
        print("\nSome downloads failed. Check your API key and network connection.")
        sys.exit(1)
    else:
        print("\n✓ All datasets downloaded successfully!")
        print(f"\nNext steps:")
        print(f"  1. Fine-tune court model: python scripts/finetune_court.py")
        print(f"  2. Fine-tune ball model: python scripts/finetune_ball.py")


if __name__ == "__main__":
    main()
