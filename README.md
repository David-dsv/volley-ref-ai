# VOLLEY-REF AI

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![YOLO](https://img.shields.io/badge/YOLO-v11-green.svg)](https://github.com/ultralytics/ultralytics)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Hugging Face](https://img.shields.io/badge/HuggingFace-Models-orange.svg)](https://huggingface.co/vuong-vn/volley-ref-ai)

**AI-powered volleyball referee system** - Automatic IN/OUT detection using Computer Vision and Deep Learning.

---

## Features

- **Court Detection**: Automatic identification of 14 volleyball court keypoints
- **Ball Tracking**: Real-time tracking with Kalman filter and tiled inference
- **Landing Prediction**: Impact point anticipation using physics model
- **IN/OUT Decision**: Automatic verdict with confidence level
- **Advanced Visualization**: Real-time overlays + 2D court view

---

## Architecture

```
VOLLEY-REF AI PIPELINE
----------------------
MODULE 1: COURT DETECTION + HOMOGRAPHY
  - YOLO11 pose model for 14 keypoints detection
  - Homography matrix computation for perspective transform

MODULE 2: BALL DETECTION + TRACKING
  - YOLO11 fine-tuned for volleyball detection
  - Tiled inference (2x2 grid with overlap) for small object detection
  - Kalman filter for trajectory smoothing

MODULE 3: LANDING POINT PREDICTION
  - Polynomial trajectory fitting
  - Physics-based model with gravity

MODULE 4: DECISION ENGINE
  - Distance calculation to court lines
  - Verdict: IN / OUT / TOO_CLOSE

MODULE 5: VISUALIZATION
  - Trajectory overlay + decision badge
  - 2D minimap court view
```

---

## Quick Start

### Installation

```bash
git clone https://github.com/vuong-vn/volley-ref-ai.git
cd volley-ref-ai

python -m venv venv
source venv/bin/activate  # Linux/Mac
pip install -r requirements.txt
```

### Download Models

Download pre-trained models from Hugging Face:

```bash
# Install huggingface_hub
pip install huggingface_hub

# Download models
python -c "
from huggingface_hub import hf_hub_download
import shutil

# Court keypoints model
court = hf_hub_download('vuong-vn/volley-ref-ai', 'yolo_court_keypoints.pt')
shutil.copy(court, 'weights/yolo_court_keypoints.pt')

# Ball detection model
ball = hf_hub_download('vuong-vn/volley-ref-ai', 'yolo_volleyball_ball.pt')
shutil.copy(ball, 'weights/yolo_volleyball_ball.pt')

print('Models downloaded to weights/')
"
```

### Run Demo

```bash
# Process a video
python scripts/demo.py --video match.mp4 --output result.mp4

# Process first 60 seconds only
python scripts/demo.py --video match.mp4 --duration 60

# Webcam mode
python scripts/demo.py --webcam

# Single image
python scripts/demo.py --image frame.jpg --output result.jpg
```

---

## Training Your Own Models

### 1. Download Datasets

Get a free API key from [Roboflow](https://app.roboflow.com/settings/api):

```bash
export ROBOFLOW_API_KEY=your_api_key
python scripts/download_datasets.py
```

### 2. Train Court Model

```bash
python scripts/finetune_court.py --epochs 100 --batch 16
```

### 3. Train Ball Model

```bash
python scripts/finetune_ball.py --epochs 150 --batch 16
```

### Training on Mac (Apple Silicon)

```bash
# Uses MPS (Metal Performance Shaders) automatically
python scripts/finetune_ball.py --device mps
```

---

## Python API

```python
from src.pipeline import VolleyRefAI

# Initialize
system = VolleyRefAI(
    court_model="weights/yolo_court_keypoints.pt",
    ball_model="weights/yolo_volleyball_ball.pt"
)

# Process video
decisions = system.process_video("input.mp4", "output.mp4")

# Get statistics
stats = system.get_statistics()
print(f"Total decisions: {stats['total_decisions']}")
print(f"IN: {stats['in_count']}, OUT: {stats['out_count']}")

# Export to CSV
system.export_decisions("decisions.csv")
```

---

## Project Structure

```
volley-ref-ai/
├── src/
│   ├── config.py           # Configuration (court dimensions, thresholds)
│   ├── court_detector.py   # Court keypoints detection + homography
│   ├── ball_tracker.py     # Ball detection + Kalman tracking
│   ├── landing_predictor.py# Trajectory prediction
│   ├── decision_engine.py  # IN/OUT decision logic
│   ├── visualizer.py       # Drawing overlays
│   └── pipeline.py         # Main orchestration
├── scripts/
│   ├── demo.py             # Demo script
│   ├── download_datasets.py
│   ├── finetune_court.py
│   ├── finetune_ball.py
│   └── evaluate.py
├── weights/                # Model weights (download from HF)
├── datasets/               # Training data (download via script)
├── outputs/                # Generated videos
└── notebooks/              # Jupyter exploration
```

---

## Configuration

### Court Dimensions (FIVB Standard)

| Parameter | Value |
|-----------|-------|
| Length | 18m (1800 cm) |
| Width | 9m (900 cm) |
| Attack line | 3m from net |
| Line width | 5 cm |

### Decision Thresholds

| Parameter | Value | Description |
|-----------|-------|-------------|
| `UNCERTAINTY_MARGIN` | 5.0 cm | Measurement uncertainty |
| `TOO_CLOSE_THRESHOLD` | 3.0 cm | "Too close to call" zone |

---

## Model Performance

| Model | mAP@50 | Training |
|-------|--------|----------|
| Court Keypoints | 99% box, 29% pose | 100 epochs |
| Ball Detection | 98.8% | 57 epochs |

---

## License

MIT License - see [LICENSE](LICENSE)

---

## Acknowledgments

- [Ultralytics](https://github.com/ultralytics/ultralytics) for YOLOv11
- [Roboflow](https://roboflow.com/) for datasets
- OpenCV community

---

**Built for volleyball enthusiasts**
