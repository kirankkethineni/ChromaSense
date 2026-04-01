# Chroma-Sense: A Memory-Efficient Plant Leaf Disease Classification Model For Edge Devices

> **Published:** IEEE CAI 2025
> **Authors:** Kiran K. Kethineni, Samuel Y. Wu, Saraju P. Mohanty, Elias Kougianos
> **Institution:** University of North Texas, Department of Computer Science and Engineering

---

## Table of Contents

1. [Overview](#overview)
2. [The Core Idea](#the-core-idea)
3. [Why This Matters](#why-this-matters)
4. [Architecture](#architecture)
5. [Technology Stack](#technology-stack)
6. [Repository Structure](#repository-structure)
7. [Code Walkthrough](#code-walkthrough)
8. [Dataset](#dataset)
9. [Training](#training)
10. [Results](#results)
11. [Edge Device Deployment](#edge-device-deployment)
12. [How to Use](#how-to-use)
13. [Limitations & Future Work](#limitations--future-work)
14. [Citation](#citation)

---

## Overview

Chroma-Sense is a novel TinyML model architecture for classifying plant leaf diseases directly on resource-constrained edge devices — cameras, drones, and microcontrollers — **without requiring a cloud connection**.

The key insight: plant disease patterns (spots, rings, patches, mosaics) are **semantically simple and consistent across RGB color channels**. Rather than learning complex multi-channel features simultaneously, Chroma-Sense processes the R, G, and B channels **sequentially through the same shared feature extractor**, then fuses the results for classification.

**Key results vs. the closest competitor (MobileNetV3):**

| Metric | Reduction |
|--------|-----------|
| Peak RAM usage | **25% less** |
| Flash memory (model size) | **60% less** |
| Parameter count | **15K vs. 97K** |
| MADDs (multiply-accumulate ops) | **8.3M vs. 20M** |
| Inference speed (OpenMV H7) | **8.4 FPS** (only model that runs on H7) |

---

## The Core Idea

### Problem: CNN Models Are Too Heavy for Edge Devices

Traditional plant disease detection pipelines push images from field sensors over the internet to cloud servers running large CNN models. This creates three problems:

1. **Latency** — network round-trips slow down real-time response
2. **Security** — crop/field imagery sent over the internet in rural areas
3. **Reliability** — remote farmlands often have poor or no connectivity

Edge computing (running inference locally on the sensor device) solves all three. But edge devices have extremely limited resources:

- **Flash memory** (to store the model): as low as 128 KB
- **SRAM / heap** (to run inference): as low as 256 KB
- **Compute**: ARM Cortex-M7/M4 microcontrollers, no GPU

Most CNN architectures — even "mobile" ones like MobileNet — are still too large or RAM-hungry to run on the smallest edge devices.

### Solution: Serial Multi-Channel Processing

The insight behind Chroma-Sense comes from examining **what plant diseases actually look like** per channel.

Consider Apple Cedar Rust disease:

| Channel | What you see |
|---------|-------------|
| RGB (combined) | Orange/yellow spot with red ring and black center |
| Red channel | Bright yellow region + red ring (both visible) |
| Green channel | Bright yellow region only |
| Blue channel | Only the dark black center |

The features per channel are **simpler than the combined image** — just spots, rings, dots, patches. And critically, **the same types of features appear in every channel**. A network that can detect a circular spot in grayscale will work equally well on the R, G, or B channel individually.

Chroma-Sense exploits this by:

1. **Sharing one feature extractor** across all three channels (reduces parameter count → smaller flash footprint)
2. **Processing channels sequentially**, one at a time (reduces peak RAM since only one channel's activations are in memory at a time)
3. **Concatenating the three channel outputs** and applying pointwise convolution to fuse cross-channel information for final classification

This is "Serial Multi-Channel Processing."

---

## Why This Matters

### Compared to Prior Approaches

| Approach | Problem |
|----------|---------|
| Standard CNNs (AlexNet, ResNet) | Way too large for edge devices |
| MobileNet / EfficientNet | Optimized for mobile, but not specifically for plant diseases — still too large for smallest devices |
| Existing edge-plant-disease models | Reduce layers but don't change the fundamental architecture |
| Multi-channel ensemble (Peker 2021) | Processes R, G, B in parallel — uses 3× the RAM of Chroma-Sense |

Chroma-Sense is the **only model in the comparison that runs on the OpenMV H7** (256 KB heap, 128 KB flash), the most constrained device tested.

### Four Novel Contributions

1. **Memory Efficiency** — Sequential channel processing reduces the CNN width by ~2/3, dramatically cutting peak RAM
2. **Parameter Reduction** — Reusing the same feature extractor for all three channels shrinks the model to just 15K parameters
3. **Simplified Feature Extraction** — Processing grayscale channel images forces the network to learn shapes and textures rather than color combinations, improving generalization
4. **Explainability** — Channel-specific feature maps are spatially interpretable; Grad-CAM visualizations clearly show which regions drive classification

---

## Architecture

### Feature Extractor (shared across all 3 channels)

```
Input: (96, 96, 1)  ← single grayscale channel

Conv2D 3×3         → (96, 96, 8)    [80 params]
MaxPooling 2×2     → (48, 48, 8)
SeparableConv2D    → (48, 48, 24)   [228 params]
MaxPooling 2×2     → (24, 24, 24)
SeparableConv2D    → (24, 24, 48)   [1,416 params]
MaxPooling 2×2     → (12, 12, 48)
SeparableConv2D    → (12, 12, 48)   [2,746 params]
MaxPooling 2×2     → (6, 6, 48)
SeparableConv2D    → (6, 6, 48)     [2,746 params]
MaxPooling 2×2     → (3, 3, 48)

Total: 7,352 parameters
```

**Why SeparableConv2D?** Depthwise separable convolutions (from MobileNet) factor a standard convolution into a depthwise spatial convolution + pointwise channel mixing, cutting compute and parameter count significantly.

**Why MaxPooling over AveragePooling?** Plant diseases appear randomly on leaves — multiple spots of varying sizes and densities. Average pooling dilutes activations when diseases are sparse (e.g., 2 spots → 0.2 average vs. 1.0 max). This dilution causes misclassifications, especially after Int8 quantization which amplifies rounding errors. Global Max Pooling focuses on the strongest disease signal regardless of how many instances are present.

### Full Classification Model

```
Input: (96, 96, 3)  ← full RGB image

Lambda → R channel (96, 96, 1) ─┐
Lambda → G channel (96, 96, 1) ─┼─ Feature Extractor (shared weights)
Lambda → B channel (96, 96, 1) ─┘

Concatenate → (3, 3, 144)   ← stack all three channel outputs
Conv2D 1×1  → (3, 3, 48)    ← fuse cross-channel info
Conv2D 1×1  → (3, 3, 16)    ← compress
GlobalMaxPooling2D → (16,)
Dense + Softmax → (N_classes,)

Total model: ~15,164 parameters
```

The pointwise (1×1) convolution after concatenation learns a **weighted linear combination** of channel features — effectively implementing a cross-channel attention mechanism that reconstructs the color correlation lost during independent processing.

---

## Technology Stack

| Component | Technology |
|-----------|-----------|
| Model development | Python, TensorFlow / Keras |
| Edge quantization | TensorFlow Lite (Int8) |
| Edge training/deployment pipeline | Edge Impulse |
| Inference runtime | OpenMV MicroPython |
| Hardware targets | OpenMV H7, OpenMV H7+, Arduino Nicla Vision |
| Training hardware | NVIDIA Tesla T4 GPU |
| Dataset | PlantVillage (Kaggle) |

---

## Repository Structure

```
ChromaSense/
├── CAI.ipynb                              # Full training notebook (Jupyter)
├── Code.txt                               # Model architecture + training code (text export)
├── ei_image_classification.py             # Edge Impulse OpenMV inference script
├── trained.tflite                         # Quantized Int8 TFLite model (ready to deploy)
├── labels.txt                             # Class labels for the Apple model
├── Metrics-Train.json                     # Validation metrics (accuracy, ROC-AUC, F1)
├── Metrics-Test.json                      # Test metrics across classes
├── Logs.txt                               # Full training logs (250 epochs + fine-tuning)
├── Memory.JPG                             # RAM usage visualization on edge device
├── edge_impulse_firmware_openmv_cam_h7.bin
├── edge_impulse_firmware_openmv_cam_h7plus.bin
├── edge_impulse_firmware_arduino_nicla_vision.bin
├── edge_impulse_firmware_arduino_portenta.bin
├── edge_impulse_firmware_openmv_cam_m7.bin
├── edge_impulse_firmware_openmv_rt1060.bin
└── edge_impulse_firmware_openmv_pure_thermal.bin
```

---

## Code Walkthrough

### Model Definition (`Code.txt` / `CAI.ipynb`)

**1. Build the shared feature extractor:**

```python
def build_feature_extractor(input_shape=(96, 96, 1)):
    inp = Input(shape=input_shape)
    x = Conv2D(8, (3,3), padding='same', activation='relu')(inp)
    x = MaxPooling2D((2,2))(x)
    x = SeparableConv2D(24, (3,3), padding='same', activation='relu')(x)
    x = MaxPooling2D((2,2))(x)
    x = SeparableConv2D(48, (3,3), padding='same', activation='relu')(x)
    x = MaxPooling2D((2,2))(x)
    x = SeparableConv2D(48, (3,3), padding='same', activation='relu')(x)
    x = MaxPooling2D((2,2))(x)
    x = SeparableConv2D(48, (3,3), padding='same', activation='relu')(x)
    x = MaxPooling2D((2,2))(x)
    return Model(inp, x)
```

**2. Build the full classification model:**

```python
def build_chroma_sense(num_classes):
    feature_extractor = build_feature_extractor()

    rgb_input = Input(shape=(96, 96, 3))
    r = Lambda(lambda x: x[:, :, :, 0:1])(rgb_input)
    g = Lambda(lambda x: x[:, :, :, 1:2])(rgb_input)
    b = Lambda(lambda x: x[:, :, :, 2:3])(rgb_input)

    # Same weights, sequential processing
    r_feat = feature_extractor(r)
    g_feat = feature_extractor(g)
    b_feat = feature_extractor(b)

    fused = Concatenate()([r_feat, g_feat, b_feat])  # (3, 3, 144)
    x = Conv2D(48, (1,1), activation='relu')(fused)
    x = Conv2D(16, (1,1), activation='relu')(x)
    x = GlobalMaxPooling2D()(x)
    output = Dense(num_classes, activation='softmax')(x)

    return Model(rgb_input, output)
```

**3. Data augmentation:**

```python
def augment_image(image, label):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    scale = tf.random.uniform([], 1.0, 1.2)
    new_size = tf.cast(96 * scale, tf.int32)
    image = tf.image.resize(image, [new_size, new_size])
    image = tf.image.random_crop(image, [96, 96, 3])
    image = tf.image.random_brightness(image, max_delta=0.2)
    return image, label
```

**4. Training configuration:**

```python
model.compile(
    optimizer=Adam(learning_rate=0.0005),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history = model.fit(
    train_dataset,
    epochs=250,
    validation_data=val_dataset,
    batch_size=128,
    callbacks=[ModelCheckpoint('best_model.h5', save_best_only=True)]
)
```

**5. Fine-tuning (after initial training):**

```python
# Unfreeze final 65% of layers
model = load_model('best_model.h5')
trainable_start = int(len(model.layers) * 0.35)
for layer in model.layers[trainable_start:]:
    layer.trainable = True

model.compile(optimizer=Adam(learning_rate=0.000045), loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_dataset, epochs=10, validation_data=val_dataset)
```

**6. Export to TFLite (Int8 quantization):**

```python
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

def representative_dataset():
    for images, _ in val_dataset.take(100):
        yield [images]

converter.representative_dataset = representative_dataset
tflite_model = converter.convert()

with open('trained.tflite', 'wb') as f:
    f.write(tflite_model)
```

### Edge Inference Script (`ei_image_classification.py`)

This script runs on the OpenMV camera using MicroPython via the Edge Impulse runtime:

```python
import sensor, image, time, os, tf, uos, gc

sensor.reset()
sensor.set_pixformat(sensor.RGB565)
sensor.set_framesize(sensor.QVGA)   # 320x240
sensor.set_windowing((240, 240))    # Center crop to 240x240
sensor.skip_frames(time=2000)       # Allow auto-exposure to settle

with open("/labels.txt", "r") as f:
    labels = [line.rstrip('\n') for line in f]

net = tf.load("/trained", load_to_fb=uos.stat('/trained')[6] > (gc.mem_free() - (64*1024)))

clock = time.clock()

while True:
    clock.tick()
    img = sensor.snapshot()

    for obj in net.classify(img, min_scale=1.0, scale_mul=0.8, x_overlap=0.5, y_overlap=0.5):
        predictions_list = list(zip(labels, obj.output()))
        predictions_list.sort(key=lambda x: x[1], reverse=True)
        for label, confidence in predictions_list:
            print(f"{label}: {confidence:.2f}")

    print(f"FPS: {clock.fps():.1f}")
```

---

## Dataset

- **Source:** [PlantVillage Dataset (Kaggle)](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset)
- **Plants used:** Apple, Tomato, Grape, Corn
- **Total images:** ~18,000 across 18 disease classes
- **Split:** 80% train / 20% validation (+ held-out test set)
- **Input resolution:** 96×96×3 (cropped from originals)
- **Preprocessing:** Images cropped to ensure only disease semantics are the consistent feature — prevents the model from learning background artifacts

**Apple subset classes (model included in this repo):**

| Class | Description |
|-------|-------------|
| Healthy | No disease present |
| Black Rot | Dark circular lesions with concentric rings |
| Black Scab | Olive-green to black scab lesions |
| Cedar Rust | Orange/yellow spots with red ring, black center |

---

## Training

**Hardware:** NVIDIA Tesla T4 GPU | **Framework:** TensorFlow / Keras

| Hyperparameter | Value |
|----------------|-------|
| Input size | 96 × 96 × 3 |
| Batch size | 128 |
| Initial epochs | 250 |
| Initial learning rate | 0.0005 |
| Optimizer | Adam |
| Loss | Categorical Crossentropy |
| Fine-tuning epochs | 10 |
| Fine-tuning LR | 0.000045 |
| Layers unfrozen (fine-tuning) | Final 65% |

**Training progression (Apple subset):**

- Final training accuracy: **98.08%** | Training loss: **0.0646**
- Final validation accuracy: **92.18%** (after fine-tuning) | Validation loss: **0.2587**

---

## Results

### F1 Scores by Plant Type (Int8 quantized model)

| Apple | F1 | Tomato | F1 | Grape | F1 | Corn | F1 |
|-------|----|--------|----|-------|----|------|----|
| Healthy | 0.93 | Healthy | 0.90 | Healthy | 0.97 | Healthy | 0.94 |
| Black Rot | 0.93 | Mold | 0.88 | Black Rot | 0.95 | Rust | 0.91 |
| Black Scab | 0.95 | Curls | 0.88 | Esca | 0.96 | Blight | 0.92 |
| Cedar Rust | 0.91 | Blight | 0.91 | Blight | 0.96 | Leaf Spots | 0.95 |
| | | Mosaic | 0.90 | | | | |
| | | Septoria Spot | 0.87 | | | | |

### Validation & Test Metrics (Apple subset)

| Split | Format | Accuracy | ROC-AUC | Weighted F1 |
|-------|--------|----------|---------|-------------|
| Validation | Int8 | 92.18% | 0.9909 | 0.922 |
| Validation | Float32 | 92.02% | 0.9920 | 0.920 |
| Test (815 samples) | Int8 | 91.78% | 0.9849 | 0.9177 |
| Test (815 samples) | Float32 | 91.53% | 0.9870 | 0.9153 |

### Full Edge Device Benchmark (Table VI from paper)

| Model | Acc Apple | Acc Tomato | RAM | Flash | Params | MADDs | H7 FPS | H7+ FPS | Nicla FPS |
|-------|-----------|------------|-----|-------|--------|-------|--------|---------|-----------|
| Conv2D | 95.2 | 88.8 | 336KB | 502KB | 491K | 130M | ❌ | 1.7 | ❌ |
| MobileNet | 86.7 | 86.6 | 341KB | 99KB | 66K | 21M | ❌ | 5.3 | ❌ |
| MobileNetV2 | 95.4 | 90.4 | 271KB | 203KB | 153K | 25M | ❌ | 7.0 | ❌ |
| MobileNetV3 | 92.8 | 92.6 | 216KB | 138KB | 97K | 20M | ❌ | 5.3 | 7.6 |
| EfficientNetV2 | 91.8 | 88.6 | 240KB | 160KB | 110K | 27M | ❌ | 5.1 | 6.9 |
| SqueezeNet | 97.2 | 83.5 | 338KB | 80KB | 56K | 23M | ❌ | 6.4 | ❌ |
| ShuffleNet | 90.2 | 83.0 | 347KB | 129KB | 88K | 13M | ❌ | 2.3 | ❌ |
| Multi-Ch. CNN | 88.8 | 86.1 | 160KB | 87KB | 62K | 43M | 4.7 | 4.7 | 4.0 |
| **Chroma-Sense** | **93.1** | **89.3** | **160KB** | **54KB** | **15K** | **8.3M** | **8.4** | **8.4** | **7.2** |

> ❌ = model did not fit in flash or exceeded heap; could not run on device.

Chroma-Sense is the **only model that runs on the OpenMV H7** (most constrained device: 256 KB heap, 128 KB flash) while maintaining competitive accuracy.

---

## Edge Device Deployment

### Supported Hardware

| Device | Heap | Flash | Processor | Chroma-Sense FPS |
|--------|------|-------|-----------|-----------------|
| OpenMV H7 | 256 KB | 128 KB | Cortex-M7 | **8.4 FPS** |
| OpenMV H7 Plus | 4 MB | 32 MB | Cortex-M7 | **8.4 FPS** |
| Arduino Nicla Vision | 256 KB | 16 MB | Cortex-M7 + M4 | **7.2 FPS** |

### Option A: Flash Pre-built Firmware (Easiest)

1. Download the `.bin` file for your device from this repo
2. Flash using the [Edge Impulse CLI](https://docs.edgeimpulse.com/docs/tools/edge-impulse-cli):
   ```bash
   edge-impulse-flash-tool --firmware edge_impulse_firmware_openmv_cam_h7.bin
   ```
3. Copy `trained.tflite` and `labels.txt` to the device's flash storage
4. Copy `ei_image_classification.py` to the device as `main.py`
5. Reset — the device will classify immediately at boot

### Option B: Edge Impulse Studio (GUI workflow)

1. Create a free account at [edgeimpulse.com](https://edgeimpulse.com)
2. Upload your image dataset and design the impulse: Image (96×96, RGB) → Chroma-Sense → Classification
3. Train and validate in the Studio
4. Deploy via CLI: `edge-impulse-run-impulse`

### Option C: Manual TFLite (OpenMV IDE)

1. Copy `trained.tflite` → `/trained` on device
2. Copy `labels.txt` → `/labels.txt` on device
3. Copy `ei_image_classification.py` → `main.py` on device
4. Connect in OpenMV IDE and run

---

## How to Use

### Requirements

```bash
pip install tensorflow numpy pillow matplotlib scikit-learn
```

### Train the Model

1. Open `CAI.ipynb` in Jupyter or Google Colab (T4 GPU recommended)
2. Point the dataset path to your PlantVillage download
3. Run all cells — ~2 hours on T4 for 250 epochs
4. Exports both `float32` and `int8` `.tflite` models

### Run Inference on a Local Image

```python
import tensorflow as tf
import numpy as np
from PIL import Image

interpreter = tf.lite.Interpreter(model_path="trained.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

img = Image.open("leaf.jpg").resize((96, 96))
img_array = np.expand_dims(np.array(img, dtype=np.float32) / 255.0, axis=0)

scale, zero_point = input_details[0]['quantization']
img_int8 = (img_array / scale + zero_point).astype(np.int8)

interpreter.set_tensor(input_details[0]['index'], img_int8)
interpreter.invoke()

output = interpreter.get_tensor(output_details[0]['index'])
labels = ["Black Rot", "Black Scab", "Cedar Rust", "Healthy"]
print(f"Prediction: {labels[np.argmax(output)]}")
```

---

## Limitations & Future Work

1. **Low resolution (96×96)** — Fine details in dense multi-leaf images may be lost
2. **Per-crop models** — Trained separately per plant type; a unified multi-crop model remains future work
3. **Localized receptive field** — Global leaf-level features are not captured, causing occasional misclassification with mixed plant types in frame

Future directions:
- Memory allocation algorithms to support **224×224** inputs on edge devices
- Unified multi-crop hierarchical classification models

---

## Citation

```bibtex
@inproceedings{kethineni2025chromasense,
  title     = {Chroma-Sense: A Memory-Efficient Plant Leaf Disease Classification Model For Edge Devices},
  author    = {Kethineni, Kiran K. and Wu, Samuel Y. and Mohanty, Saraju P. and Kougianos, Elias},
  booktitle = {Proceedings of the IEEE Conference on Artificial Intelligence (CAI)},
  year      = {2025},
  institution = {University of North Texas}
}
```

Related work: SprayCraft — Graph-Based Route Optimization for Variable Rate Precision Spraying — [arXiv:2412.12176](https://arxiv.org/abs/2412.12176)

---

## Contact

| Author | Email |
|--------|-------|
| Kiran K. Kethineni | kirankumar.kethineni@unt.edu |
| Samuel Y. Wu | samuelwu@my.unt.edu |
| Saraju P. Mohanty | saraju.mohanty@unt.edu |
| Elias Kougianos | elias.kougianos@unt.edu |

**Smart Systems and Platforms Group, University of North Texas**
