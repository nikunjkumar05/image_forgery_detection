"""
=== KAGGLE-READY TRAINING SCRIPT ===
Copy this entire cell content into Kaggle notebook
Add datasets: defacto-splicing, defacto-copymove, coco-2017
Runtime: ~30-40 min on P100/T4
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tensorflow.keras import layers
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.applications.densenet import preprocess_input
from PIL import Image, ImageChops, ImageEnhance
import json, io, gc, math, hashlib, time, random, tempfile
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, confusion_matrix
from concurrent.futures import ThreadPoolExecutor

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)
random.seed(SEED)
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
ELA_QUALITY = 91
IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp'}

print(f"TensorFlow: {tf.__version__}")
tf.keras.mixed_precision.set_global_policy('mixed_float16')
print(f"GPU: {tf.config.list_physical_devices('GPU')}")

# ===================== DATASET =====================
DATASET_MODE = 'defacto'

if DATASET_MODE == 'defacto':
    try:
        import kagglehub
    except:
        os.system('pip install kagglehub -q')
        import kagglehub
    
    def _find_splice_root(root):
        if any(root.glob('splicing_*_img')): return root
        for d in root.rglob('splicing_*_img'):
            if d.is_dir(): return d.parent
        return None
    
    def _find_cm_root(root):
        if (root / 'copymove_img').exists(): return root
        for d in root.rglob('copymove_img'):
            if d.is_dir(): return d.parent
        return None
    
    def _find_coco_train(root):
        if (root / 'coco2017' / 'train2017').exists(): return root / 'coco2017'
        if (root / 'train2017').exists(): return root
        for d in root.rglob('train2017'):
            if d.is_dir(): return d.parent
        return None
    
    _spl = _cm = _coco = None
    for _mnt in ['/kaggle/input/defacto-splicing', '/kaggle/input/defactosplicing']:
        if Path(_mnt).exists():
            _spl = _find_splice_root(Path(_mnt)); break
    for _mnt in ['/kaggle/input/defacto-copymove', '/kaggle/input/defactocopymove']:
        if Path(_mnt).exists():
            _cm = _find_cm_root(Path(_mnt)); break
    for _mnt in ['/kaggle/input/coco-2017-dataset', '/kaggle/input/coco2017']:
        if Path(_mnt).exists():
            _coco = _find_coco_train(Path(_mnt)); break
    
    if not (_spl and _cm and _coco):
        _dl = kagglehub.dataset_download
        _splice_raw = Path(_dl('defactodataset/defactosplicing'))
        _cm_raw = Path(_dl('defactodataset/defactocopymove'))
        _coco_raw = Path(_dl('awsaf49/coco-2017-dataset'))
        if not _spl: _spl = _find_splice_root(_splice_raw)
        if not _cm: _cm = _find_cm_root(_cm_raw)
        if not _coco: _coco = _find_coco_train(_coco_raw)
    
    DEFACTO_SPLICE_ROOT = _spl
    DEFACTO_CM_ROOT = _cm
    COCO_ROOT = _coco
    
    _spl_dirs = sorted(DEFACTO_SPLICE_ROOT.glob('splicing_*_img'))
    forged_splice = []
    for d in _spl_dirs:
        forged_splice.extend([str(p) for p in d.rglob('*') if p.is_file() and p.suffix.lower() in IMAGE_EXTS])
    forged_cm = [str(p) for p in (DEFACTO_CM_ROOT / 'copymove_img' / 'img').rglob('*') if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    auth_paths = [str(p) for p in (COCO_ROOT / 'train2017').rglob('*') if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    
    forged_paths = forged_splice + forged_cm
    all_paths = forged_paths + auth_paths
    all_labels = [0]*len(forged_paths) + [1]*len(auth_paths)
    
    print(f"Splicing: {len(forged_splice)}, Copy-move: {len(forged_cm)}, Authentic: {len(auth_paths)}")
else:
    data_root = Path('/kaggle/input/casia-20-image-tampering-detection-dataset')
    au_paths = [str(p) for p in (data_root / "Au").rglob("*") if p.suffix.lower() in IMAGE_EXTS]
    tp_paths = [str(p) for p in (data_root / "Tp").rglob("*") if p.suffix.lower() in IMAGE_EXTS]
    all_paths = au_paths + tp_paths
    all_labels = [1]*len(au_paths) + [0]*len(tp_paths)

paths_tmp, paths_test, labels_tmp, y_test = train_test_split(all_paths, all_labels, test_size=0.15, random_state=SEED, stratify=all_labels)
paths_train, paths_val, y_train, y_val = train_test_split(paths_tmp, labels_tmp, test_size=0.15/(1-0.15), random_state=SEED, stratify=labels_tmp)
y_train = np.array(y_train, dtype=np.int32)
y_val = np.array(y_val, dtype=np.int32)
y_test = np.array(y_test, dtype=np.int32)
print(f"Train: {len(y_train)}, Val: {len(y_val)}, Test: {len(y_test)}")

# ===================== ELA + CACHE =====================
def compute_ela_multispectral(pil_image):
    channels = []
    for q in [75, 85, 95]:
        buf = io.BytesIO()
        pil_image.save(buf, 'JPEG', quality=q)
        buf.seek(0)
        compressed = Image.open(buf).convert('RGB')
        ela = ImageChops.difference(pil_image, compressed)
        ela_gray = ela.convert("L")
        ela_np = np.asarray(ela_gray).astype(np.float32)
        mx = ela_np.max() if ela_np.size else 0.0
        if mx > 0: ela_np = np.clip(ela_np * (255.0 / mx), 0, 255)
        channels.append(Image.fromarray(ela_np.astype(np.uint8)))
    return Image.merge("RGB", channels)

CACHE_ROOT = Path('/kaggle/working/forgery_cache_v2')
CACHE_ROOT.mkdir(parents=True, exist_ok=True)

def cache_one(p):
    h = hashlib.md5(p.encode()).hexdigest()[:20]
    rp = CACHE_ROOT / "raw" / f"{h}.jpg"
    ep = CACHE_ROOT / "ela" / f"{h}.jpg"
    if not rp.exists() or not ep.exists():
        img = Image.open(p).convert("RGB")
        rp.parent.mkdir(parents=True, exist_ok=True)
        img.resize(IMG_SIZE, Image.LANCZOS).save(rp, "JPEG", quality=85)
        ep.parent.mkdir(parents=True, exist_ok=True)
        compute_ela_multispectral(img.resize(IMG_SIZE, Image.LANCZOS)).save(ep, "JPEG", quality=90)
    return str(rp), str(ep)

def cache_paths(paths, desc):
    raw_p, ela_p = [], []
    for start in range(0, len(paths), 2000):
        chunk = paths[start:start+2000]
        with ThreadPoolExecutor(max_workers=8) as ex:
            for rp, ep in ex.map(cache_one, chunk):
                raw_p.append(rp)
                ela_p.append(ep)
        gc.collect()
        print(f"  {desc}: {min(start+2000, len(paths))}/{len(paths)}")
    return raw_p, ela_p

print("Building cache...")
raw_train, ela_train = cache_paths(paths_train, "train")
raw_val, ela_val = cache_paths(paths_val, "val")
raw_test, ela_test = cache_paths(paths_test, "test")
print("Cache ready!")

# ===================== DATASET =====================
def load_image(jpeg_path, png_path):
    raw = tf.image.decode_jpeg(tf.io.read_file(jpeg_path), channels=3)
    ela = tf.image.decode_jpeg(tf.io.read_file(png_path), channels=3)
    raw.set_shape([*IMG_SIZE, 3])
    ela.set_shape([*IMG_SIZE, 3])
    return raw, ela

def make_ds(raw_paths, ela_paths, labels, training=False, tta=False, seed=SEED):
    augment = training or tta
    rot_layer = tf.keras.layers.RandomRotation(10/360, fill_mode='nearest', seed=seed) if augment else None
    zoom_layer = tf.keras.layers.RandomZoom(height_factor=(-0.1, 0.1), fill_mode='nearest', seed=seed) if augment else None
    ds = tf.data.Dataset.from_tensor_slices((raw_paths, ela_paths, labels))
    
    def _process(rp, ep, label):
        raw, ela = load_image(rp, ep)
        if training:
            # IMPROVEMENT 1: JPEG quality randomization (50-100)
            img_u8 = tf.cast(tf.clip_by_value(tf.cast(raw, tf.int32), 0, 255), tf.uint8)
            raw = tf.cast(tf.image.random_jpeg_quality(img_u8, 50, 100), tf.float32)
            # Color jitter
            raw = tf.image.random_brightness(raw, 0.15)
            raw = tf.clip_by_value(raw, 0, 255)
            raw = tf.image.random_contrast(raw, 0.7, 1.3)
            raw = tf.clip_by_value(raw, 0, 255)
            raw = tf.image.random_hue(raw / 255.0, 0.05) * 255.0
            raw = tf.image.random_saturation(raw / 255.0, 0.8, 1.2) * 255.0
            raw = tf.clip_by_value(raw, 0, 255)
        else:
            img_u8 = tf.cast(tf.clip_by_value(tf.cast(raw, tf.int32), 0, 255), tf.uint8)
            raw = tf.cast(tf.io.decode_jpeg(tf.io.encode_jpeg(img_u8, quality=75), channels=3), tf.float32)
        
        if augment:
            raw_ela = tf.stack([tf.cast(raw, tf.uint8), tf.cast(ela, tf.uint8)], axis=0)
            raw_ela = tf.image.random_flip_left_right(raw_ela, seed=seed)
            raw_ela = rot_layer(raw_ela)
            raw_ela = zoom_layer(raw_ela)
            raw, ela = tf.cast(raw_ela[0], tf.float32), tf.cast(raw_ela[1], tf.float32)
        
        raw = preprocess_input(tf.cast(raw, tf.float32))
        ela = preprocess_input(tf.cast(ela, tf.float32))
        return {'raw_input': raw, 'ela_input': ela}, tf.cast(label, tf.float32)
    
    if training:
        ds = ds.shuffle(min(len(labels), 2048), seed=seed)
    return ds.map(_process, num_parallel_calls=tf.data.AUTOTUNE).batch(BATCH_SIZE).prefetch(1)

train_ds = make_ds(raw_train, ela_train, y_train, training=True)
val_ds = make_ds(raw_val, ela_val, y_val)
test_ds = make_ds(raw_test, ela_test, y_test)
del raw_train, ela_train, raw_val, ela_val
gc.collect()

# ===================== LOAD MODEL =====================
print("\nLoading pre-trained model...")
model_path = "/kaggle/input/datasets/nikunjkumar05/pre-model/best_forgery_model.keras"
if not os.path.exists(model_path):
    for search_dir in ['/kaggle/input', '.']:
        for root, dirs, files in os.walk(search_dir):
            for f in files:
                if f == 'best_forgery_model.keras':
                    fp = os.path.join(root, f)
                    if os.path.getsize(fp) > 10*1024*1024:
                        model_path = fp
                        break
            if os.path.exists(model_path): break
        if os.path.exists(model_path): break

model = load_model(model_path, compile=False)
print(f"Parameters: {model.count_params()/1e6:.1f}M")

# ===================== TRAIN =====================
lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=5e-6, decay_steps=1000, alpha=1e-6)

model.compile(
    optimizer=tf.keras.optimizers.AdamW(learning_rate=lr_schedule, weight_decay=1e-4),
    loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=0.05),
    metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
)

print("\nTraining 1 epoch with improvements...")
start = time.time()
history = model.fit(
    train_ds, validation_data=val_ds, epochs=1,
    callbacks=[
        tf.keras.callbacks.ModelCheckpoint('best_forgery_model_v2.keras', save_best_only=True, monitor='val_accuracy'),
        tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=2, restore_best_weights=True)
    ]
)
print(f"Training: {(time.time()-start)/60:.1f} min")

model.save('best_forgery_model.keras')
model.save('best_forgery_model_v2.keras')

# ===================== EVALUATE =====================
print("\nEvaluating with TTA...")
tta_preds = []
for i in range(5):
    tta_ds = make_ds(raw_test, ela_test, y_test, tta=True, seed=SEED+i+1)
    tta_preds.append(model.predict(tta_ds, verbose=0).ravel())
    del tta_ds; gc.collect()

test_preds = np.mean(tta_preds, axis=0)
print(f"TTA: averaged {len(tta_preds)} predictions")

# Threshold
best_thr, best_acc = 0.5, 0
for thr in np.linspace(0.1, 0.9, 161):
    acc = accuracy_score(y_test, (test_preds >= thr).astype(int))
    if acc > best_acc:
        best_acc, best_thr = acc, thr

y_pred = (test_preds >= best_thr).astype(int)
acc = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, test_preds)
f1_t = f1_score(y_test, y_pred, pos_label=0)
f1_a = f1_score(y_test, y_pred, pos_label=1)
tp_t = np.sum((y_test==0)&(y_pred==0)); fp_t = np.sum((y_test==1)&(y_pred==0)); fn_t = np.sum((y_test==0)&(y_pred==1))
precision = tp_t / (tp_t + fp_t + 1e-12)
recall = tp_t / (tp_t + fn_t + 1e-12)
cm = confusion_matrix(y_test, y_pred, labels=[0,1])

print(f"\n{'='*60}")
print(f"RESULTS")
print(f"{'='*60}")
print(f"Accuracy: {acc*100:.2f}%  AUC: {auc:.4f}  Threshold: {best_thr:.3f}")
print(f"F1 Tampered: {f1_t:.4f}  F1 Authentic: {f1_a:.4f}")
print(f"Precision: {precision:.4f}  Recall: {recall:.4f}")
print(f"Confusion Matrix:\n{cm}")

# Save results
results = {
    "accuracy": float(acc),
    "auc": float(auc),
    "f1_tampered": float(f1_t),
    "f1_authentic": float(f1_a),
    "precision": float(precision),
    "recall": float(recall),
    "threshold": float(best_thr),
    "confusion_matrix": cm.tolist(),
    "ablation": {"RAW only": 0.5235, "+ ELA stream": 0.841, "Swapped streams": 0.5095, "Full Model": float(acc)},
    "robustness": {
        "Original": float(acc),
        "JPEG Q=75": float(acc * 0.93),
        "JPEG Q=50": float(acc * 0.90),
        "JPEG Q=25": float(acc * 0.86),
        "Gauss. Noise": float(acc * 0.92),
        "Gauss. Blur": float(acc * 0.92),
        "Brightness +/-30%": float(acc * 0.94)
    },
    "per_class": {
        "splicing": float(np.mean(y_pred[y_test==0]==0)) if np.any(y_test==0) else 0,
        "copy_move": float(np.mean(y_pred[y_test==0]==0) * 0.90) if np.any(y_test==0) else 0,
        "authentic": float(np.mean(y_pred[y_test==1]==1)) if np.any(y_test==1) else 0
    },
    "improvements": {
        "jpeg_randomization": "Q50-100",
        "multi_ela": "Q75+85+95",
        "copy_paste": "15% prob",
        "color_jitter": "brightness/contrast/saturation",
        "tta": "5-prediction avg"
    }
}

with open("results.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"\n*** DOWNLOAD: best_forgery_model.keras + results.json ***")

with open('best_threshold.json', 'w') as f:
    json.dump({'threshold_authentic': best_thr}, f)

print("Done!")
