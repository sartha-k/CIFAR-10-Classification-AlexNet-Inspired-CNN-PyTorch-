# CIFAR-10 Classification — AlexNet-Inspired CNN (PyTorch)

An AlexNet-inspired CNN built in PyTorch to classify CIFAR-10 images into 10 categories, adapted for 32×32 RGB inputs instead of the original 227×227 ImageNet images.

---

## Dataset

**CIFAR-10** — 60,000 colored images (32×32 pixels, RGB) across 10 classes:

Airplane, Automobile, Bird, Cat, Deer, Dog, Frog, Horse, Ship, Truck

- **Training set:** 50,000 images
- **Test set:** 10,000 images

---

## Model Architecture

5 convolutional layers followed by a fully connected classifier — inspired by AlexNet (Krizhevsky et al., 2012), adapted for smaller input size.

```
Input (3, 32, 32)
    ↓
Conv(3→16) → ReLU → MaxPool(2×2)        # 32→16
    ↓
Conv(16→32) → ReLU → MaxPool(2×2)       # 16→8
    ↓
Conv(32→64) → ReLU
    ↓
Conv(64→128) → ReLU
    ↓
Conv(128→256) → ReLU
    ↓ (256, 4, 4)
Flatten → Linear(4096→1000) → ReLU → Dropout(0.5)
         → Linear(1000→516) → ReLU → Dropout(0.5)
         → Linear(516→128) → ReLU
         → Linear(128→10)
    ↓
Logits (10)
```

**Key design decision:** Only 2 MaxPool layers instead of 3 — CIFAR-10 images are 32×32, pooling after every layer shrinks feature maps to 1×1, destroying spatial information.

---

## Training Details

| Hyperparameter | Value |
|----------------|-------|
| Loss Function | CrossEntropyLoss |
| Optimizer | Adam |
| Learning Rate | 0.001 |
| Batch Size | 32 |
| Epochs | 30 |
| Hardware | NVIDIA T4 GPU |

---

## Results

| Configuration | Accuracy |
|---------------|----------|
| Without Dropout | 70% |
| With Dropout | **71%** |

Dropout provided a small but consistent improvement by reducing overfitting.

---

## Key Concepts Practiced

- 5-layer deep CNN architecture inspired by AlexNet
- **Dropout** (`p=0.5`) in FC layers — key AlexNet innovation
- Pooling placement strategy — why not every conv layer needs pooling
- Adapting architectures designed for large inputs to smaller datasets
- GPU training with `.to(device)`

---

## Differences from Original AlexNet

| | Original AlexNet | This Implementation |
|---|---|---|
| Input size | 227×227 | 32×32 |
| Dataset | ImageNet (1000 classes) | CIFAR-10 (10 classes) |
| First kernel | 11×11 | 3×3 |
| Normalization | Local Response Norm | None |
| MaxPool layers | 3 | 2 |

---
