#!/usr/bin/env python3
"""
Script de démonstration VOLLEY-REF AI

Ce script permet de:
- Traiter une vidéo de volleyball
- Afficher les décisions IN/OUT en temps réel
- Générer une vidéo avec les overlays visuels

Usage:
    python scripts/demo.py --video path/to/video.mp4
    python scripts/demo.py --video video.mp4 --output result.mp4
    python scripts/demo.py --webcam
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

# Ajouter le répertoire parent au path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import cv2
    import numpy as np
    from rich.console import Console
    from rich.table import Table
    from rich.progress import Progress, SpinnerColumn, TextColumn
except ImportError as e:
    print(f"Error: Missing dependency - {e}")
    print("Install with: pip install opencv-python numpy rich")
    sys.exit(1)

from src.pipeline import VolleyRefAI
from src.config import CourtConfig, ModelConfig, DecisionConfig


console = Console()


def check_models(court_model: str, ball_model: str) -> bool:
    """
    Vérifie que les modèles existent.

    Args:
        court_model: Chemin vers le modèle terrain
        ball_model: Chemin vers le modèle balle

    Returns:
        bool: True si les deux modèles existent
    """
    court_exists = Path(court_model).exists()
    ball_exists = Path(ball_model).exists()

    if not court_exists:
        console.print(f"[red]Error:[/red] Court model not found at {court_model}")
        console.print("Run: python scripts/finetune_court.py")

    if not ball_exists:
        console.print(f"[red]Error:[/red] Ball model not found at {ball_model}")
        console.print("Run: python scripts/finetune_ball.py")

    return court_exists and ball_exists


def process_video(
    video_path: str,
    output_path: str,
    court_model: str,
    ball_model: str,
    show_preview: bool = False,
    max_duration: Optional[int] = None
) -> None:
    """
    Traite une vidéo et génère la sortie avec overlays.

    Args:
        video_path: Chemin vers la vidéo d'entrée
        output_path: Chemin vers la vidéo de sortie
        court_model: Chemin vers le modèle terrain
        ball_model: Chemin vers le modèle balle
        show_preview: Afficher la prévisualisation en temps réel
    """
    console.print("\n[bold blue]═══════════════════════════════════════════════════════════[/bold blue]")
    console.print("[bold blue]           VOLLEY-REF AI - Video Processing[/bold blue]")
    console.print("[bold blue]═══════════════════════════════════════════════════════════[/bold blue]\n")

    # Vérifier les modèles
    if not check_models(court_model, ball_model):
        return

    # Vérifier la vidéo d'entrée
    if not Path(video_path).exists():
        console.print(f"[red]Error:[/red] Video not found at {video_path}")
        return

    # Créer le pipeline
    console.print("[cyan]Initializing pipeline...[/cyan]")
    pipeline = VolleyRefAI(
        court_model=court_model,
        ball_model=ball_model
    )

    console.print(f"[green]✓[/green] Pipeline initialized")
    console.print(f"  Input: {video_path}")
    console.print(f"  Output: {output_path}")
    if max_duration:
        console.print(f"  Duration limit: {max_duration} seconds")

    # Traiter la vidéo
    console.print("\n[cyan]Processing video...[/cyan]")

    def callback(frame_num: int, decision):
        if decision:
            verdict = decision.verdict
            color = "green" if verdict == "IN" else "red" if verdict == "OUT" else "yellow"
            console.print(
                f"  Frame {frame_num}: [{color}]{verdict}[/{color}] "
                f"({abs(decision.distance_to_line):.1f}cm, {decision.confidence:.0%})"
            )

    try:
        # Ouvrir la vidéo pour obtenir le FPS et calculer max_frames
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        max_frames = None
        if max_duration:
            max_frames = int(fps * max_duration)
            console.print(f"  Processing {max_frames} frames ({max_duration}s at {fps:.1f} fps)")

        decisions = pipeline.process_video(
            video_path,
            output_path,
            show_progress=True,
            callback=callback if not show_preview else None,
            max_frames=max_frames
        )
    except Exception as e:
        console.print(f"[red]Error processing video:[/red] {e}")
        return

    # Afficher le résumé
    stats = pipeline.get_statistics()

    console.print("\n[bold blue]═══════════════════════════════════════════════════════════[/bold blue]")
    console.print("[bold blue]                        SUMMARY[/bold blue]")
    console.print("[bold blue]═══════════════════════════════════════════════════════════[/bold blue]\n")

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right")

    table.add_row("Total Frames", str(stats["total_frames"]))
    table.add_row("Total Decisions", str(stats["total_decisions"]))
    table.add_row("[green]IN[/green]", str(stats["in_count"]))
    table.add_row("[red]OUT[/red]", str(stats["out_count"]))
    table.add_row("[yellow]TOO CLOSE[/yellow]", str(stats["too_close_count"]))

    if stats["total_decisions"] > 0:
        table.add_row("Avg Confidence", f"{stats['average_confidence']:.1%}")

    console.print(table)

    console.print(f"\n[green]✓[/green] Output saved to: {output_path}")

    # Exporter les décisions en CSV
    csv_path = Path(output_path).with_suffix('.csv')
    pipeline.export_decisions(str(csv_path))


def process_webcam(
    court_model: str,
    ball_model: str,
    camera_id: int = 0
) -> None:
    """
    Traite le flux webcam en temps réel.

    Args:
        court_model: Chemin vers le modèle terrain
        ball_model: Chemin vers le modèle balle
        camera_id: ID de la caméra
    """
    console.print("\n[bold blue]═══════════════════════════════════════════════════════════[/bold blue]")
    console.print("[bold blue]           VOLLEY-REF AI - Webcam Mode[/bold blue]")
    console.print("[bold blue]═══════════════════════════════════════════════════════════[/bold blue]\n")

    # Vérifier les modèles
    if not check_models(court_model, ball_model):
        return

    # Créer le pipeline
    console.print("[cyan]Initializing pipeline...[/cyan]")
    pipeline = VolleyRefAI(
        court_model=court_model,
        ball_model=ball_model
    )

    console.print("[green]✓[/green] Pipeline initialized")
    console.print("\nPress 'q' to quit, 's' to save screenshot")

    # Lancer le traitement webcam
    try:
        pipeline.process_webcam(
            camera_id=camera_id,
            window_name="VOLLEY-REF AI",
            quit_key='q'
        )
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")


def process_image(
    image_path: str,
    output_path: str,
    court_model: str,
    ball_model: str
) -> None:
    """
    Traite une seule image.

    Args:
        image_path: Chemin vers l'image
        output_path: Chemin de sortie
        court_model: Chemin vers le modèle terrain
        ball_model: Chemin vers le modèle balle
    """
    console.print("\n[bold blue]═══════════════════════════════════════════════════════════[/bold blue]")
    console.print("[bold blue]           VOLLEY-REF AI - Image Processing[/bold blue]")
    console.print("[bold blue]═══════════════════════════════════════════════════════════[/bold blue]\n")

    # Vérifier les modèles
    if not check_models(court_model, ball_model):
        return

    # Vérifier l'image
    if not Path(image_path).exists():
        console.print(f"[red]Error:[/red] Image not found at {image_path}")
        return

    # Charger l'image
    image = cv2.imread(image_path)
    if image is None:
        console.print(f"[red]Error:[/red] Could not read image at {image_path}")
        return

    # Créer le pipeline
    console.print("[cyan]Initializing pipeline...[/cyan]")
    pipeline = VolleyRefAI(
        court_model=court_model,
        ball_model=ball_model
    )

    # Traiter l'image
    output, decision = pipeline.process_frame(image)

    # Sauvegarder
    cv2.imwrite(output_path, output)

    if decision:
        verdict = decision.verdict
        color = "green" if verdict == "IN" else "red" if verdict == "OUT" else "yellow"
        console.print(
            f"[{color}]{verdict}[/{color}] - "
            f"{abs(decision.distance_to_line):.1f}cm from {decision.closest_line} "
            f"(Confidence: {decision.confidence:.0%})"
        )
    else:
        console.print("[yellow]No decision could be made (insufficient data)[/yellow]")

    console.print(f"\n[green]✓[/green] Output saved to: {output_path}")


def main():
    """Point d'entrée principal."""
    parser = argparse.ArgumentParser(
        description="VOLLEY-REF AI Demo - Volleyball IN/OUT Detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Process a video
    python scripts/demo.py --video match.mp4

    # Process video with custom output
    python scripts/demo.py --video match.mp4 --output result.mp4

    # Use webcam
    python scripts/demo.py --webcam

    # Process single image
    python scripts/demo.py --image frame.jpg --output result.jpg
        """
    )

    # Mode de fonctionnement
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        "--video", "-v",
        help="Path to input video file"
    )
    mode_group.add_argument(
        "--webcam", "-w",
        action="store_true",
        help="Use webcam input"
    )
    mode_group.add_argument(
        "--image", "-i",
        help="Path to input image file"
    )

    # Options
    parser.add_argument(
        "--output", "-o",
        help="Path to output file (default: outputs/output.mp4)"
    )
    parser.add_argument(
        "--court-model",
        default="weights/yolo_court_keypoints.pt",
        help="Path to court keypoints model"
    )
    parser.add_argument(
        "--ball-model",
        default="weights/yolo_volleyball_ball.pt",
        help="Path to ball detection model"
    )
    parser.add_argument(
        "--camera-id",
        type=int,
        default=0,
        help="Camera ID for webcam mode (default: 0)"
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Show preview while processing video"
    )
    parser.add_argument(
        "--duration", "-d",
        type=int,
        default=None,
        help="Max duration in seconds (e.g., --duration 60 for 1 minute)"
    )

    args = parser.parse_args()

    # Créer le dossier outputs si nécessaire
    script_dir = Path(__file__).parent.parent
    outputs_dir = script_dir / "outputs"
    outputs_dir.mkdir(exist_ok=True)

    # Traitement selon le mode
    if args.video:
        output_path = args.output or str(outputs_dir / "output.mp4")
        process_video(
            args.video,
            output_path,
            args.court_model,
            args.ball_model,
            args.preview,
            args.duration
        )

    elif args.webcam:
        process_webcam(
            args.court_model,
            args.ball_model,
            args.camera_id
        )

    elif args.image:
        output_path = args.output or str(outputs_dir / "output.jpg")
        process_image(
            args.image,
            output_path,
            args.court_model,
            args.ball_model
        )


if __name__ == "__main__":
    main()
