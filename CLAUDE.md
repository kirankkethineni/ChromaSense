# ChromaSense — Project Context for Claude

## What This Project Is
This is the code repository for the IEEE CAI 2025 paper:
**"Chroma-Sense: A Memory-Efficient Plant Leaf Disease Classification Model For Edge Devices"**

Authors: Kiran K. Kethineni, Samuel Y. Wu, Saraju P. Mohanty, Elias Kougianos
Institution: University of North Texas, Dept. of Computer Science and Engineering

## Key Idea
Chroma-Sense processes R, G, B image channels sequentially through a **single shared feature extractor** (not three separate ones), then concatenates the outputs and applies pointwise convolution for fusion. This achieves:
- 25% less peak RAM vs. MobileNetV3
- 60% less flash memory
- Only 15K parameters (vs. 97K in closest competitor)
- 8.4 FPS on OpenMV H7 (256 KB heap / 128 KB flash) — the only model that runs on it

## Repository Files
| File | Purpose |
|------|---------|
| `CAI.ipynb` | Full training notebook (TensorFlow/Keras, Tesla T4) |
| `Code.txt` | Text export of model architecture + training code |
| `ei_image_classification.py` | MicroPython inference script for OpenMV cameras |
| `trained.tflite` | Int8 quantized TFLite model (Apple: 4 classes, 54 KB) |
| `labels.txt` | Class labels: Black Rot, Black Scab, Cedar Rust, Healthy |
| `Metrics-Train.json` | Validation metrics (accuracy 92.18%, ROC-AUC 0.9909) |
| `Metrics-Test.json` | Test metrics (815 samples, accuracy 91.78%) |
| `Logs.txt` | Full training logs (250 epochs + 10 fine-tuning epochs) |
| `Memory.JPG` | RAM usage visualization on edge device |
| `docs/index.html` | GitHub Pages site |
| `*.bin` | Pre-built Edge Impulse firmware for OpenMV / Arduino Nicla Vision |

## Architecture Summary
```
Input (96×96×3)
  → Split into R, G, B channels (96×96×1 each)
  → Each through shared Feature Extractor (7,352 params):
      Conv2D(8) → SepConv2D(24) → SepConv2D(48) → SepConv2D(48) → SepConv2D(48)
      [MaxPooling2D after each, stride 1]
  → Concatenate → (3,3,144)
  → Conv2D 1×1 (48) → Conv2D 1×1 (16) → GlobalMaxPooling2D → Dense + Softmax
Total: ~15,164 parameters
```

## Training Details
- Dataset: PlantVillage (Apple, Tomato, Grape, Corn — ~18K images, 18 classes)
- This repo contains the Apple subset model (4 classes)
- Phase 1: 250 epochs, Adam lr=0.0005, batch=128
- Phase 2 fine-tuning: 10 epochs, lr=0.000045, final 65% layers unfrozen
- Quantization: TFLite Int8 via representative dataset calibration

## Edge Devices Tested
| Device | Heap | Flash | FPS |
|--------|------|-------|-----|
| OpenMV H7 | 256 KB | 128 KB | 8.4 |
| OpenMV H7 Plus | 4 MB | 32 MB | 8.4 |
| Arduino Nicla Vision | 256 KB | 16 MB | 7.2 |

## GitHub Pages
Site lives at `docs/index.html` — dark theme, sidebar TOC, inline F1 bar charts.
Enable via: repo Settings → Pages → Source: main branch, /docs folder.
URL: https://kirankkethineni.github.io/ChromaSense

## Paper PDF Location (local)
`C:\Users\kethi\OneDrive - UNT System\Kiran_Saraju_Shared\Papers\IEEE-CAI_2025_ChromaSense\IEEE-CAI_2025_ChromaSense_New.pdf`

## Related Work
- SprayCraft (same group): https://arxiv.org/abs/2412.12176
- PlantVillage dataset: https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset
