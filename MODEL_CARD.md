---
license: mit
tags:
  - yolo
  - object-detection
  - pose-estimation
  - volleyball
  - sports
  - computer-vision
  - pytorch
datasets:
  - volleyball-court-keypoints
  - volleyball-detection
language:
  - en
pipeline_tag: object-detection
---

# VOLLEY-REF AI Models

AI-powered volleyball referee system for automatic IN/OUT line call detection.

## Models Included

### 1. Court Keypoints Model (`yolo_court_keypoints.pt`)
- **Architecture**: YOLOv11n-pose
- **Task**: Detect 14 keypoints of a volleyball court
- **Training**: 100 epochs on volleyball-court-keypoints dataset
- **Performance**: 99% box mAP@50, 29% pose mAP@50

### 2. Ball Detection Model (`yolo_volleyball_ball.pt`)
- **Architecture**: YOLOv11s
- **Task**: Detect volleyball in video frames
- **Training**: 57 epochs on volleyball_detection dataset
- **Performance**: 98.8% mAP@50

## Usage

### Download Models

```python
from huggingface_hub import hf_hub_download

# Download court model
court_model = hf_hub_download(
    repo_id="vuong-vn/volley-ref-ai",
    filename="yolo_court_keypoints.pt"
)

# Download ball model
ball_model = hf_hub_download(
    repo_id="vuong-vn/volley-ref-ai",
    filename="yolo_volleyball_ball.pt"
)
```

### Inference with Ultralytics

```python
from ultralytics import YOLO

# Court keypoints detection
court_model = YOLO("yolo_court_keypoints.pt")
results = court_model("volleyball_frame.jpg")

# Ball detection
ball_model = YOLO("yolo_volleyball_ball.pt")
results = ball_model("volleyball_frame.jpg", conf=0.7)
```

### Full Pipeline

See the [GitHub repository](https://github.com/vuong-vn/volley-ref-ai) for the complete VOLLEY-REF AI pipeline that combines both models for automatic IN/OUT detection.

## Training Details

### Court Model
- Base: `yolo11n-pose.pt`
- Dataset: volleyball-court-keypoints (495 images)
- Epochs: 100
- Image size: 640
- Augmentation: Default YOLO augmentations

### Ball Model
- Base: `yolo11s.pt`
- Dataset: volleyball_detection (1091 images)
- Epochs: 57 (early stopped from 150)
- Image size: 640
- Augmentation: Default YOLO augmentations

## Limitations

- Trained primarily on indoor volleyball footage
- Performance may vary with different camera angles
- Ball detection works best with clear visibility (no motion blur)
- Court detection requires visible court lines

## License

MIT License

## Citation

```bibtex
@software{volley_ref_ai_2025,
  author = {Vuong},
  title = {VOLLEY-REF AI: AI-Powered Volleyball Referee System},
  year = {2025},
  url = {https://github.com/vuong-vn/volley-ref-ai}
}
```

## Acknowledgments

- [Ultralytics](https://github.com/ultralytics/ultralytics) for YOLOv11
- [Roboflow](https://roboflow.com/) for the training datasets
