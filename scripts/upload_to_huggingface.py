#!/usr/bin/env python3
"""
Upload VOLLEY-REF AI models to Hugging Face Hub

Usage:
    # First login to Hugging Face
    huggingface-cli login

    # Then run this script
    python scripts/upload_to_huggingface.py --repo-id your-username/volley-ref-ai
"""

import argparse
from pathlib import Path

try:
    from huggingface_hub import HfApi, upload_file
except ImportError:
    print("Error: huggingface_hub not installed")
    print("Install with: pip install huggingface_hub")
    exit(1)


def main():
    parser = argparse.ArgumentParser(description="Upload models to Hugging Face Hub")
    parser.add_argument(
        "--repo-id",
        required=True,
        help="Hugging Face repo ID (e.g., username/volley-ref-ai)"
    )
    parser.add_argument(
        "--weights-dir",
        default="weights",
        help="Directory containing model weights"
    )
    args = parser.parse_args()

    api = HfApi()

    # Create repo if it doesn't exist
    print(f"Creating/checking repository: {args.repo_id}")
    api.create_repo(repo_id=args.repo_id, exist_ok=True)

    # Upload model card
    model_card_path = Path(__file__).parent.parent / "MODEL_CARD.md"
    if model_card_path.exists():
        print("Uploading model card (README.md)...")
        api.upload_file(
            path_or_fileobj=str(model_card_path),
            path_in_repo="README.md",
            repo_id=args.repo_id
        )

    # Upload weights
    weights_dir = Path(__file__).parent.parent / args.weights_dir

    models = [
        "yolo_court_keypoints.pt",
        "yolo_volleyball_ball.pt"
    ]

    for model_name in models:
        model_path = weights_dir / model_name
        if model_path.exists():
            print(f"Uploading {model_name}...")
            api.upload_file(
                path_or_fileobj=str(model_path),
                path_in_repo=model_name,
                repo_id=args.repo_id
            )
            print(f"  Uploaded: {model_name}")
        else:
            print(f"  Warning: {model_name} not found at {model_path}")

    print(f"\nDone! Models available at: https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
