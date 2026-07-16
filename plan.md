# Plan: 98% Forgery Detection — Dataset Migration + Full Optimization

## 1. Current Status & Root Cause Analysis

### CASIA v2 Bottlenecks

| Issue | Impact | Root Cause |
|---|---|---|
| **Format confound** | ~3-5% accuracy ceiling | Au: 99.3% JPEG, Tp: 40.3% TIFF. Model learns "TIFF=tampered" |
| **CutMix fractional labels** | Train accuracy stuck at ~47% | `accuracy` metric uses `round(pred) == fractional_label` — mathematically broken |
| **Threshold penalty** | Test accuracy capped at 87.53% | `-0.05*abs(thr-0.5)` penalizes deviation from 0.500 |
| **12K images** | Insufficient for deep learning | EfficientNetB3 has 12M params — 12K images is a 1:1 param:image ratio |
| **JPEG quality range 50–100** | Narrow enough to keep format signal | 50→100 is only ~2 stops — model can still distinguish compression origins |

**Verdict:** CASIA v2 cannot reach 98% regardless of model improvements. Dataset change is required.

---

## 2. Dataset Strategy

### Primary: DEFACTO (Splicing + Copy-Move)

| Property | Value |
|---|---|
| **Forged images** | 124K (105K splicing + 19K copy-move) |
| **Authentic images** | COCO 2017 train (118K) + val (5K) = 123K |
| **Total** | ~247K |
| **Format** | **All JPEG** — zero format confound |
| **Storage (original)** | DEFACTO: ~2GB | COCO: ~18GB |
| **Storage (cache 224×224)** | RAW (~5KB/image): ~1.2GB | ELA (~15KB/image): ~3.7GB | **Total: ~5GB** |
| **Kaggle datasets** | `defactodataset/defactosplicing`, `defactodataset/defactocopymove`, `awsaf49/coco-2017-dataset` |

### Directory Structure (expected on Kaggle)

```
/kaggle/input/defactosplicing/
└── splicing/
    ├── images/          # 105K JPEG forged images
    ├── probe_mask/      # binary forgery masks
    └── donor_mask/      # source region masks

/kaggle/input/defactocopymove/
└── copymove/
    ├── images/          # 19K JPEG forged images
    ├── probe_mask/      # binary forgery masks
    └── donor_mask/      # donor region masks

/kaggle/input/coco-2017-dataset/
└── coco2017/
    ├── train2017/       # 118K JPEG authentic images
    └── val2017/         # 5K JPEG authentic images
```

### Label Mapping

| Source | Label |
|---|---|
| COCO 2017 | 1 (authentic) |
| DEFACTO splicing + copy-move | 0 (tampered) |

### Why Not tampCOCO (for now)

| Factor | DEFACTO | tampCOCO |
|---|---|---|
| **Storage on Kaggle** | ~5GB cache (fits 20GB limit) | ~50GB cache (exceeds 20GB limit) |
| **Authentic source** | COCO 2017 on Kaggle (read-only) | COCO 2017, same issue but 118K images need full cache |
| **Forgery diversity** | 4 types (splicing, CM, inpainting, morphing) | 2 types (splicing, CM) |
| **Cache time** | ~1.5 hrs (8-thread parallel) | ~8-12 hrs |
| **Training time** | ~3 hrs | ~12+ hrs |
| **Free tier feasible?** | **Yes** | No (200GB storage, 24hr+ total time) |

tampCOCO is the **phase 2 fallback** if DEFACTO cannot reach 98%.

---

## 3. Platform Constraints & Strategy

### Kaggle Free Tier (Recommended)

| Resource | Limit | Our Usage | Headroom |
|---|---|---|---|
| GPU | P100 (16GB VRAM) 30hrs/week | ~4GB (fp16 batch=64) | 12GB spare |
| RAM | ~16GB | ~8GB (dataset + cache + model) | 8GB spare |
| Working disk | ~20GB | ~6GB (cache + checkpoints) | 14GB spare |
| Session | 9 hours | ~4-5 hours total | 4 hours spare |
| GPU hours/week | 30 hours | ~5 hours | 25 hours spare |

### Colab Free Tier (Fallback)

| Resource | Limit | Our Usage | Risk |
|---|---|---|---|
| GPU | T4/K80 (16GB/12GB VRAM) | ~4GB (fp16 batch=48) | K80 has 12GB → reduce batch to 32 |
| RAM | ~12GB | ~8GB | Tight — enable memory growth |
| Disk | ~68GB | ~25GB (download + cache) | Spare |
| Session | ~12 hours (disconnect if idle) | ~6 hours | Need keep-alive workaround |

### Key Differences

| Feature | Kaggle | Colab |
|---|---|---|
| **Dataset mounting** | Read-only (zero download cost) | Must download via kagglehub or URL |
| **COCO availability** | Mounted from `awsaf49/coco-2017-dataset` | Download 18GB (slow, may fail) |
| **DEFACTO availability** | Mounted from Kaggle datasets | Download 2GB (manageable) |
| **Session stability** | Reliable for 9 hours | Disconnects after 90 min idle |
| **GPU type** | P100 guaranteed | T4 or K80 (random) |

**Decision: Write code that works on BOTH platforms.**
- Use try/except to detect platform and switch data source
- On Kaggle: mount datasets as inputs
- On Colab: download via kagglehub with resume support + smaller COCO subset (5K val instead of 118K train)

---

## 4. Code Changes (Detailed)

### 4.1 Cell 2: Platform Detection + Dataset Setup

Add platform detection:
```python
IN_KAGGLE = 'KAGGLE_KERNEL_RUN_TYPE' in os.environ
IN_COLAB = 'COLAB_GPU' in os.environ
IS_KAGGLE = IN_KAGGLE
DATASET_MODE = 'casia'  # 'casia' | 'defacto'  — set to 'defacto' for final training
```

### 4.2 Cell 3: Data Root Detection

```
if DATASET_MODE == 'casia':
    # Keep existing CASIA code
    ...
elif DATASET_MODE == 'defacto':
    if IS_KAGGLE:
        DEFACTO_SPLICING = Path('/kaggle/input/defactosplicing/splicing')
        DEFACTO_COPYMOVE = Path('/kaggle/input/defactocopymove/copymove')
        COCO_ROOT = Path('/kaggle/input/coco-2017-dataset/coco2017')
    else:
        # Colab: download datasets
        ...
```

### 4.3 Cell 4: ELA Computation (keep as-is)

### 4.4 Cell 5: Path Collection — DEFACTO mode

Collect forged images from DEFACTO + authentic from COCO:
```python
if DATASET_MODE == 'defacto':
    forged_paths = collect_paths(DEFACTO_SPLICING / 'images') + \
                   collect_paths(DEFACTO_COPYMOVE / 'images')
    auth_paths = collect_paths(COCO_ROOT / 'train2017')
    if not IS_KAGGLE:
        # Colab: use smaller subset
        auth_paths = auth_paths[:5000]  # COCO val equivalent
    all_paths = forged_paths + auth_paths
    all_labels = [0]*len(forged_paths) + [1]*len(auth_paths)
```

### 4.5 Cell 6: Split (keep as-is)

### 4.6 Cell 7: Parallel Cache Precomputation

Replace sequential `prepare_cache` with 8-thread `ThreadPoolExecutor`:
```python
from concurrent.futures import ThreadPoolExecutor, as_completed

def prepare_cache_parallel(paths, desc, workers=8):
    ...
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(_process_one, p) for p in paths]
        for i, f in enumerate(as_completed(futures)):
            results.append(f.result())
            if (i+1) % 2000 == 0: print(f'  {desc}: {i+1}/{len(paths)}')
    ...
```

**Time savings:** Sequential: ~3 hrs | Parallel (8 threads): ~25 min ✔

### 4.7 Cell 8: Data Pipeline Changes

**4.7a: Widen JPEG quality range + replace py_function**

```python
# BEFORE
def random_jpeg(img):
    img_u8 = tf.cast(...)
    def _apply(x):
        q = np.random.randint(50, 100)
        return tf.io.decode_jpeg(tf.io.encode_jpeg(x, quality=int(q)), channels=3).numpy()
    res = tf.py_function(_apply, [img_u8], tf.uint8)

# AFTER
def random_jpeg(img):
    q = tf.random.uniform([], 10, 101, dtype=tf.int32)
    img_u8 = tf.cast(tf.clip_by_value(tf.cast(img, tf.int32), 0, 255), tf.uint8)
    return tf.image.random_jpeg_quality(img_u8, 10, 100)
```

**Benefit:** Pure TF op (no py_function overhead), wider range (10-100 vs 50-100)

**4.7b: Add SBI augmentation (replaces CutMix)**

```python
def sbi_batch(data, labels):
    raw, ela = data['raw_input'], data['ela_input']
    bs = tf.shape(raw)[0]
    h = w = 224

    # Broadcastable coordinate grid
    xs = tf.cast(tf.range(w), tf.float32)[tf.newaxis, tf.newaxis, :]  # [1,1,224]
    ys = tf.cast(tf.range(h), tf.float32)[tf.newaxis, :, tf.newaxis]  # [1,224,1]

    # Random ellipse parameters per image
    cx = tf.random.uniform([bs, 1, 1], 0.0, tf.cast(w, tf.float32))
    cy = tf.random.uniform([bs, 1, 1], 0.0, tf.cast(h, tf.float32))
    rx = tf.random.uniform([bs, 1, 1], 0.05, 0.35) * tf.cast(w, tf.float32)
    ry = tf.random.uniform([bs, 1, 1], 0.05, 0.35) * tf.cast(h, tf.float32)
    angle = tf.random.uniform([bs, 1, 1], 0.0, 2 * np.pi)

    cos_a, sin_a = tf.cos(angle), tf.sin(angle)
    X_rot = (xs - cx) * cos_a + (ys - cy) * sin_a
    Y_rot = -(xs - cx) * sin_a + (ys - cy) * cos_a
    ellipse = (X_rot / rx)**2 + (Y_rot / ry)**2
    mask = tf.cast(ellipse <= 1.0, tf.float32)[..., tf.newaxis]  # [bs, h, w, 1]

    # Apply to 50% of batch
    apply = tf.cast(tf.random.uniform([bs, 1, 1, 1]) > 0.5, tf.float32)
    mask = mask * apply

    # Unsharp mask (5x5 box blur)
    raw_blur = tf.nn.avg_pool2d(raw, 5, 1, 'SAME')
    ela_blur = tf.nn.avg_pool2d(ela, 5, 1, 'SAME')
    strength = tf.random.uniform([bs, 1, 1, 1], 0.5, 3.0)
    raw_source = tf.clip_by_value(raw + strength * (raw - raw_blur), -1.0, 1.0)
    ela_source = tf.clip_by_value(ela + strength * (ela - ela_blur), -1.0, 1.0)

    # Blend
    raw_out = raw * (1.0 - mask) + raw_source * mask
    ela_out = ela * (1.0 - mask) + ela_source * mask

    # SBI'd images → tampered (0)
    new_labels = labels * (1.0 - tf.squeeze(apply[:, 0, 0, :]))
    return {'raw_input': raw_out, 'ela_input': ela_out}, new_labels
```

**Why SBI > CutMix:**
- SBI produces binary labels (0 or 1) → train accuracy is meaningful
- SBI simulates realistic blending artifacts → better generalization
- SBI addresses the actual task (blending detection) instead of random mixing

**4.7c: Update pipeline**

```python
train_phase2_ds = train_ds.map(sbi_batch, num_parallel_calls=tf.data.AUTOTUNE)
```

### 4.8 Cell 9: Model Architecture — Expanded Head

| Layer | Before | After |
|---|---|---|
| Dropout | 0.4 | 0.5 |
| Dense 1 | 256 (L2=3e-4) | 512 (L2=4e-4) |
| Dropout | 0.4 | 0.5 |
| Dense 2 | 128 (L2=3e-4) | 256 (L2=4e-4) |
| Dropout | 0.3 | 0.3 |
| Output | 1 (sigmoid) | 1 (sigmoid) |
| **Total head params** | **~98K** | **~262K** |

**Rationale:** With 247K training images (vs 12K), we have 20x more data. The current head is bottlenecked at 98K params. Expanding to 262K provides enough capacity to learn the 4-forgery-type manifold without overfitting (regularized by L2=4e-4 + dropout 0.5).

### 4.9 Cell 10: Threshold Calibration Fix

```python
# BEFORE: score = 0.6*f1_t + 0.4*f1_a - 0.05*abs(thr-0.5)
# AFTER:
score = acc  # optimize accuracy directly, no penalty term
```

### 4.10 Cell 13: Phase 2 Training

| Stage | Blocks | LR | Epochs | BN trainable |
|---|---|---|---|---|
| Stage 1 | Block 7 | 1e-5 | 5 | No |
| Stage 2 | Blocks 5-7 | 5e-6 | 7 | No |
| Stage 3 | Blocks 3-7 | 2e-6 | 15 | **Yes** (last stage) |

**Why BN trainable in Stage 3:** SBI changes the input distribution (blended images have different statistics). Frozen BN with SBI causes train-test mismatch. Allowing BN to adapt in the last stage fixes this.

### 4.11 Mixed Precision (new addition)

```python
tf.keras.mixed_precision.set_global_policy('mixed_float16')
```

**Benefits:**
- 2x training speed (P100 fp16: ~60 TFLOPs vs fp32: ~10 TFLOPs)
- 50% less GPU memory (activations stored as fp16)
- Enables batch_size = 64 (vs 32 in fp32)

**Risk:** Loss underflow → add `loss_scale` via optimizer:
```python
opt = tf.keras.optimizers.AdamW(lr, weight_decay=1e-4, global_clipnorm=1.0)
opt = tf.keras.mixed_precision.LossScaleOptimizer(opt)
```
Wait — Keras 3 handles loss scaling internally with `mixed_float16`. The `LossScaleOptimizer` wrapper was for TF2.x Keras. Let me check...

Actually, in TF 2.12+, `mixed_float16` works with AdamW directly — Keras adds loss scaling automatically for the loss. We don't need the wrapper.

### 4.12 Error Handling (new additions)

**4.12a: Disk space check before cache**
```python
def check_disk_space(path, min_gb=5):
    import shutil
    _, _, free = shutil.disk_usage(path)
    free_gb = free / (1024**3)
    if free_gb < min_gb:
        raise RuntimeError(f'Only {free_gb:.1f}GB free, need {min_gb}GB')
```

**4.12b: GPU memory growth**
```python
gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
```

**4.12c: Fallback batch sizes**
```python
try:
    _ = model.predict(train_ds.take(1))
except tf.errors.ResourceExhaustedError:
    print('OOM at batch_size=64, falling back to 32')
    # Recreate datasets with smaller batch
    BATCH_SIZE = 32
```

**4.12d: Session keep-alive (Colab)**
```python
import time, threading
def keepalive():
    while True:
        time.sleep(60)
        print('.', end='', flush=True)
if IN_COLAB:
    t = threading.Thread(target=keepalive, daemon=True)
    t.start()
```

---

## 5. Full Training Configuration

### Hyperparameters

| Parameter | Phase 1 | Phase 2 S1 | Phase 2 S2 | Phase 2 S3 |
|---|---|---|---|---|
| Trainable | Backbones frozen | Block 7 | Blocks 5-7 | Blocks 3-7 + BN |
| Optimizer | Adam | AdamW | AdamW | AdamW |
| Learning rate | 1e-3 | 1e-5 | 5e-6 | 2e-6 |
| Schedule | — | CosineDecay | CosineDecay | CosineDecay |
| Weight decay | — | 1e-4 | 1e-4 | 1e-4 |
| Global clipnorm | — | 1.0 | 1.0 | 1.0 |
| Loss | BCE (label_smoothing=0.1) | Focal (γ=2, α=0.5) | Focal (γ=2, α=0.5) | Focal (γ=2, α=0.5) |
| Epochs | 10 | 5 | 7 | 15 |
| Mixed precision | fp16 | fp16 | fp16 | fp16 |
| Batch size | 64 | 64 | 64 | 64 |
| Augmentation | Basic | Basic + SBI | Basic + SBI | Basic + SBI |

### Callbacks

| Callback | Phase 1 | Phase 2 |
|---|---|---|
| EarlyStopping (val_auc, p=5) | ✓ | ✓ (last stage only) |
| ReduceLROnPlateau (val_auc, f=0.5, p=2) | ✓ | — (CosineDecay handles LR) |
| ModelCheckpoint (best val_auc) | ✓ | ✓ |
| OverfitGuard (gap=4%, p=3) | — | ✓ |
| SWACallback (start=1) | — | ✓ (last stage only) |

---

## 6. Estimated Timelines

### Kaggle (P100 GPU, 9hr session)

| Phase | Steps | Time |
|---|---|---|
| **Pre-cache** (8 threads, 247K images) | — | ~25 min |
| Phase 1 (10 epochs × 3860 steps × 50ms) | 38,600 | ~32 min |
| Phase 2 Stage 1 (5 × 3860 × 80ms) | 19,300 | ~26 min |
| Phase 2 Stage 2 (7 × 3860 × 100ms) | 27,020 | ~45 min |
| Phase 2 Stage 3 (15 × 3860 × 120ms) | 57,900 | ~116 min |
| SWA averaging + evaluation | — | ~5 min |
| **Total** | — | **~4.1 hours** |

✓ **Within 9hr session. Even with 2x safety margin: ~5 hours.**

### Colab (T4 GPU, ~50% slower than P100, 12hr session)

| Phase | Time |
|---|---|
| Dataset download (kagglehub) | ~1 hour |
| Pre-cache | ~40 min |
| Training | ~5 hours |
| **Total** | **~6.8 hours** |

✓ **Within 12hr session but tight. Risk of disconnect → use keepalive thread.**

---

## 7. Projected Accuracy Breakdown

| Change | Train Accuracy | Test Accuracy | Cumul. Test |
|---|---|---|---|
| **Current (CASIA v2)** | 47% (CutMix artifact) | 87.53% | 87.53% |
| Switch to DEFACTO + COCO | 85% | 82% | ~82% |
| + Fix threshold (optimize acc) | — | +3% | 85% |
| + SBI augmentation | 95% | +5% | 90% |
| + Expand head capacity | +3% | +2% | 92% |
| + Wider JPEG range 10-100 | — | +1% | 93% |
| + BN trainable in last stage | — | +1% | 94% |
| + Mixed precision (faster = more steps) | — | +1% | 95% |
| + Ensemble (SWA + best checkpoint avg) | — | +2% | **97%** |
| + 3-model ensemble (if needed) | — | +1% | **98%** |

### If 98% not reached on DEFACTO:

| Fallback | Expected lift | Cost |
|---|---|---|
| Add DEFACTO inpainting (+25K) | +0.5% | +20 min cache, +10 min training |
| Add DEFACTO morphing (+40K) | +0.5% | +30 min cache, +15 min training |
| Switch to tampCOCO (822K forged) | +1-2% | Cannot run on free tier |

---

## 8. Implementation Order

```
Step 1: Fix threshold calibration (Cell 10)          — no retraining needed
Step 2: Fix random_jpeg (Cell 8)                     — no retraining needed  
Step 3: Expand head capacity (Cell 9)                — needs retraining
Step 4: Add mixed precision (Cell 13)                — needs retraining
Step 5: Implement SBI (Cell 8)                       — needs retraining
Step 6: Add error handling (memory, disk, fallback)  — safety net
Step 7: Switch dataset to DEFACTO + COCO (Cells 3-6) — full retrain
Step 8: Train from scratch on DEFACTO
Step 9: Evaluate → if < 96%, add SWA ensemble
Step 10: If < 98%, add DEFACTO inpainting/morphing
```

---

## 9. Error Modes & Mitigations

| Error | When | Mitigation |
|---|---|---|
| **OOM** | Training (batch too large) | `tf.config.set_memory_growth` + fallback batch_size 64→48→32 |
| **Disk full** | Cache creation | `check_disk_space()` before cache → skip/truncate if < 5GB |
| **KeyError/GradCAM** | After training | Rebuild cell already added (line 552-556) |
| **SWA crash (Keras 3)** | Last training epoch | List comprehension fix already applied (line 431-435) |
| **Kaggle API timeout** | Colab dataset download | Add retry loop (3 attempts, 30s delay) |
| **Colab disconnect** | During long training | Keepalive thread + model checkpoint every epoch |
| **Mixed precision NaN** | Training (loss explosion) | Add `global_clipnorm=1.0` (already in code) |
| **FileNotFoundError** | Dataset path mismatch | Add fallback path search (multiple candidates) |
| **TF version incompatibility** | Keras 3 / TensorFlow mismatch | Pin tensorflow>=2.10,<2.16 in pip install |

---

## 10. Success Criteria

| Metric | Target | Minimum Acceptable |
|---|---|---|
| **Test accuracy** | 98% | 95% |
| **Train accuracy** | 98% | 95% (with SBI binary labels) |
| **Train-val gap** | <2% | <4% (OverfitGuard enforces this) |
| **Test AUC** | >0.99 | >0.98 |
| **Tampered F1** | >0.97 | >0.95 |
| **Authentic F1** | >0.97 | >0.95 |
| **No runtime errors** | Pass | Fatal exceptions = failure |

---

## 11. Decision Matrix

| Decision | Choice | Rationale |
|---|---|---|
| Primary dataset | **DEFACTO** (splicing + CM) | 247K all-JPEG, 2 forgery types, fits free tier |
| Fallback dataset | **DEFACTO** + inpainting + morphing | 312K all-JPEG, 4 types, +1% expected |
| Hard-fallback dataset | **tampCOCO** | 940K images, but needs paid tier (200GB) |
| Primary platform | **Kaggle** | Stable 9hr session, P100 GPU, zero dataset download |
| Batch size | **64** (fp16) | Fits in 16GB VRAM with EfficientNetB3 dual backbone |
| Augmentation | **SBI** replaces CutMix | Binary labels + realistic forgery simulation |
| Mixed precision | **Yes** (mixed_float16) | 2x speed, 50% less memory |
| Head capacity | **512→256→1** (L2=4e-4) | 262K params, matched to 247K training images |
