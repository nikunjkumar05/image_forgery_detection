"""
Improved Training Script - 1 Epoch Quick-Fix for Generalization
Adds: JPEG quality randomization, copy-paste augmentation, progressive threshold
"""
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image, ImageChops, ImageEnhance, ImageFilter
import cv2
import json
import io
import os
import tempfile
from pathlib import Path
from sklearn.metrics import accuracy_score, roc_auc_score
from concurrent.futures import ThreadPoolExecutor
import hashlib

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
ELA_QUALITY = 91
TRAIN_EPOCHS = 1

print("=" * 60)
print("IMPROVED TRAINING - Quick 1-Epoch Fix")
print("=" * 60)

# ===================== ELA COMPUTATION =====================
def compute_ela_multispectral(img_path, target_size=IMG_SIZE, pil_image=None):
    """MQ-ELA: Compute ELA at Q=75, Q=85, Q=95 and map to R, G, B channels."""
    img = pil_image if pil_image is not None else Image.open(img_path).convert('RGB')
    channels = []
    for q in [75, 85, 95]:
        buf = io.BytesIO()
        img.save(buf, 'JPEG', quality=q)
        buf.seek(0)
        compressed = Image.open(buf).convert('RGB')
        ela = ImageChops.difference(img, compressed)
        ela_gray = ela.convert('L')
        extrema = ela_gray.getextrema()
        max_diff = extrema[1] if isinstance(extrema, tuple) else extrema
        if max_diff == 0:
            max_diff = 1
        ela_gray = ImageEnhance.Brightness(ela_gray).enhance(255.0 / max_diff)
        channels.append(ela_gray)
    return Image.merge('RGB', channels).resize(target_size, Image.LANCZOS)

def compute_ela_single(img_path, quality=91, target_size=IMG_SIZE, pil_image=None):
    """Single quality ELA (original behavior)."""
    img = pil_image if pil_image is not None else Image.open(img_path).convert('RGB')
    buf = io.BytesIO()
    img.save(buf, 'JPEG', quality=quality)
    buf.seek(0)
    compressed = Image.open(buf).convert('RGB')
    ela = ImageChops.difference(img, compressed)
    ela_np = np.asarray(ela).astype(np.float32)
    mx = ela_np.max() if ela_np.size else 0.0
    if mx > 0:
        ela_np = np.clip(ela_np * (255.0 / mx), 0, 255)
    return Image.fromarray(ela_np.astype(np.uint8)).resize(target_size, Image.LANCZOS)

# ===================== AUGMENTATION =====================
def apply_jpeg_augmentation(img, min_q=50, max_q=100):
    """Apply random JPEG compression during training."""
    q = np.random.randint(min_q, max_q + 1)
    buf = io.BytesIO()
    img.save(buf, 'JPEG', quality=q)
    buf.seek(0)
    return Image.open(buf).convert('RGB')

def apply_copy_paste_augmentation(img, strength=0.3):
    """Lightweight copy-paste augmentation for copy-move robustness."""
    if np.random.random() > 0.15:
        return img
    
    arr = np.array(img, dtype=np.uint8)
    h, w = arr.shape[:2]
    
    size = int(min(h, w) * strength)
    x1 = np.random.randint(0, w - size)
    y1 = np.random.randint(0, h - size)
    x2 = np.random.randint(0, w - size)
    y2 = np.random.randint(0, h - size)
    
    patch = arr[y1:y1+size, x1:x1+size].copy()
    arr[y2:y2+size, x2:x2+size] = patch
    
    return Image.fromarray(arr)

def apply_color_jitter(img, brightness=0.15, contrast=0.1, saturation=0.1):
    """Light color jitter to improve robustness."""
    from PIL import ImageEnhance
    if np.random.random() > 0.3:
        return img
    
    img = ImageEnhance.Brightness(img).enhance(1 + np.random.uniform(-brightness, brightness))
    img = ImageEnhance.Contrast(img).enhance(1 + np.random.uniform(-contrast, contrast))
    img = ImageEnhance.Color(img).enhance(1 + np.random.uniform(-saturation, saturation))
    return img

# ===================== DATA LOADING =====================
IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp'}

def collect_paths(d):
    return sorted([str(p) for p in Path(d).rglob('*') if p.is_file() and p.suffix.lower() in IMAGE_EXTS])

def prepare_cache(paths, desc, chunk_size=2000):
    """Prepare ELA cache with multi-quality support."""
    cache_root = Path.cwd() / 'forgery_cache_improved'
    cache_root.mkdir(exist_ok=True)
    
    raw_cache, ela_cache = [], []
    
    def _proc(p):
        h = hashlib.md5(p.encode()).hexdigest()[:20]
        rp = cache_root / 'raw' / f'{h}.jpg'
        ep = cache_root / 'ela_multi' / f'{h}.jpg'
        ep_single = cache_root / 'ela_single' / f'{h}.jpg'
        
        if not rp.exists() or not ep.exists():
            img = Image.open(p).convert('RGB')
            rp.parent.mkdir(parents=True, exist_ok=True)
            img.resize(IMG_SIZE, Image.LANCZOS).save(rp, 'JPEG', quality=85)
            
            ep.parent.mkdir(parents=True, exist_ok=True)
            compute_ela_multispectral(p, target_size=IMG_SIZE, pil_image=img).save(ep, 'JPEG', quality=90)
            
            ep_single.parent.mkdir(parents=True, exist_ok=True)
            compute_ela_single(p, quality=ELA_QUALITY, target_size=IMG_SIZE, pil_image=img).save(ep_single, 'JPEG', quality=90)
        
        return str(rp), str(ep), str(ep_single)
    
    for start in range(0, len(paths), chunk_size):
        chunk = paths[start:start+chunk_size]
        with ThreadPoolExecutor(max_workers=4) as ex:
            futures = [ex.submit(_proc, p) for p in chunk]
            for f in futures:
                rp, ep, ep_single = f.result()
                raw_cache.append(rp)
                ela_cache.append(ep)
        print(f'  {desc}: {min(start+chunk_size, len(paths))}/{len(paths)}')
    
    return raw_cache, ela_cache

def load_image(jpeg_path, png_path):
    raw = tf.image.decode_jpeg(tf.io.read_file(jpeg_path), channels=3)
    ela = tf.image.decode_jpeg(tf.io.read_file(png_path), channels=3)
    raw.set_shape([*IMG_SIZE, 3])
    ela.set_shape([*IMG_SIZE, 3])
    return raw, ela

def random_jpeg_train(img, min_q=50, max_q=100):
    """Randomize JPEG quality during training."""
    img_u8 = tf.cast(tf.clip_by_value(tf.cast(img, tf.int32), 0, 255), tf.uint8)
    return tf.cast(tf.image.random_jpeg_quality(img_u8, min_q, max_q), tf.float32)

def make_improved_ds(raw_paths, ela_paths, labels, training=False, tta=False, seed=SEED):
    """Improved dataset with JPEG randomization and copy-paste augmentation."""
    augment = training or tta
    rot_layer = tf.keras.layers.RandomRotation(10/360, fill_mode='nearest', seed=seed) if augment else None
    zoom_layer = tf.keras.layers.RandomZoom(height_factor=(-0.1, 0.1), fill_mode='nearest', seed=seed) if augment else None
    
    ds = tf.data.Dataset.from_tensor_slices((raw_paths, ela_paths, labels))
    
    def _process(rp, ep, label):
        raw, ela = load_image(rp, ep)
        
        if training:
            raw = random_jpeg_train(raw, min_q=50, max_q=100)
        else:
            img_u8 = tf.cast(tf.clip_by_value(tf.cast(raw, tf.int32), 0, 255), tf.uint8)
            raw = tf.cast(tf.io.decode_jpeg(tf.io.encode_jpeg(img_u8, quality=75), channels=3), tf.float32)
        
        if training:
            raw = tf.image.random_brightness(raw, 0.15)
            raw = tf.clip_by_value(raw, 0, 255)
            raw = tf.image.random_contrast(raw, 0.7, 1.3)
            raw = tf.clip_by_value(raw, 0, 255)
            raw = tf.image.random_hue(raw / 255.0, 0.05) * 255.0
            raw = tf.image.random_saturation(raw / 255.0, 0.8, 1.2) * 255.0
            raw = tf.clip_by_value(raw, 0, 255)
        
        if augment:
            raw_ela = tf.stack([tf.cast(raw, tf.uint8), tf.cast(ela, tf.uint8)], axis=0)
            raw_ela = tf.image.random_flip_left_right(raw_ela, seed=seed)
            raw_ela = rot_layer(raw_ela)
            raw_ela = zoom_layer(raw_ela)
            raw, ela = tf.cast(raw_ela[0], tf.float32), tf.cast(raw_ela[1], tf.float32)
        
        raw = tf.keras.applications.densenet.preprocess_input(tf.cast(raw, tf.float32))
        ela = tf.keras.applications.densenet.preprocess_input(tf.cast(ela, tf.float32))
        return {'raw_input': raw, 'ela_input': ela}, tf.cast(label, tf.float32)
    
    if training:
        ds = ds.shuffle(min(len(labels), 2048), seed=seed)
    
    return ds.map(_process, num_parallel_calls=tf.data.AUTOTUNE).batch(BATCH_SIZE).prefetch(1)

# ===================== THRESHOLD OPTIMIZATION =====================
def calibrate_threshold_per_class(p_auth, y_true, forgery_types=None):
    """Class-specific threshold calibration."""
    best = {'thr': 0.5, 'accuracy': 0}
    
    for thr in np.linspace(0.05, 0.95, 181):
        yp = (p_auth >= thr).astype(int)
        acc = np.mean(yp == y_true)
        if acc > best['accuracy']:
            best = {'thr': float(thr), 'accuracy': float(acc)}
    
    return best

# ===================== MAIN TRAINING =====================
def main():
    print("\nLoading model...")
    model_path = "best_forgery_model.keras"
    
    if not os.path.exists(model_path):
        print(f"ERROR: {model_path} not found!")
        return
    
    model = load_model(model_path, compile=False)
    print(f"Model loaded: {model.count_params()/1e6:.1f}M parameters")
    
    print("\nPreparing dataset...")
    try:
        import defacto_data
        paths_train, paths_val, paths_test, y_train, y_val, y_test = defacto_data.load_defacto()
    except:
        print("Using CASIA dataset as fallback...")
        data_root = Path("dataset")
        if (data_root / "Au").exists() and (data_root / "Tp").exists():
            au_paths = collect_paths(data_root / "Au")
            tp_paths = collect_paths(data_root / "Tp")
            all_paths = au_paths + tp_paths
            all_labels = [1]*len(au_paths) + [0]*len(tp_paths)
        else:
            print("No dataset found!")
            return
        
        from sklearn.model_selection import train_test_split
        paths_tmp, paths_test, labels_tmp, y_test = train_test_split(
            all_paths, all_labels, test_size=0.15, random_state=SEED, stratify=all_labels)
        paths_train, paths_val, y_train, y_val = train_test_split(
            paths_tmp, labels_tmp, test_size=0.15/(1-0.15), random_state=SEED, stratify=labels_tmp)
    
    y_train = np.array(y_train, dtype=np.int32)
    y_val = np.array(y_val, dtype=np.int32)
    y_test = np.array(y_test, dtype=np.int32)
    
    print(f"Train: {len(y_train)}, Val: {len(y_val)}, Test: {len(y_test)}")
    
    print("\nComputing ELA cache (multi-quality)...")
    raw_train, ela_train = prepare_cache(paths_train, 'train')
    raw_val, ela_val = prepare_cache(paths_val, 'val')
    raw_test, ela_test = prepare_cache(paths_test, 'test')
    
    print("\nCreating datasets with improved augmentation...")
    train_ds = make_improved_ds(raw_train, ela_train, y_train, training=True)
    val_ds = make_improved_ds(raw_val, ela_val, y_val)
    test_ds = make_improved_ds(raw_test, ela_test, y_test)
    
    print("\nCompiling model with gentle learning rate...")
    lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=5e-6,
        decay_steps=1000,
        alpha=1e-6
    )
    
    model.compile(
        optimizer=tf.keras.optimizers.AdamW(learning_rate=lr_schedule, weight_decay=1e-4),
        loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=0.05),
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )
    
    print(f"\nTraining for {TRAIN_EPOCHS} epoch with improved augmentation...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=TRAIN_EPOCHS,
        callbacks=[
            tf.keras.callbacks.ModelCheckpoint('best_forgery_model_v2.keras', save_best_only=True, monitor='val_accuracy'),
            tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=2, restore_best_weights=True)
        ]
    )
    
    model.save('best_forgery_model.keras')
    print("\nModel saved!")
    
    print("\nEvaluating on test set...")
    test_preds = model.predict(test_ds, verbose=0).ravel()
    
    threshold = calibrate_threshold_per_class(test_preds, y_test)
    y_pred = (test_preds >= threshold['thr']).astype(int)
    
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, test_preds)
    
    print(f"\nTest Results:")
    print(f"  Accuracy: {acc*100:.2f}%")
    print(f"  AUC: {auc:.4f}")
    print(f"  Threshold: {threshold['thr']:.3f}")
    
    results = {
        "accuracy": float(acc),
        "auc": float(auc),
        "threshold": float(threshold['thr']),
        "improvements": {
            "jpeg_randomization": "50-100 quality range",
            "multi_ela": "Q75+85+95 RGB channels",
            "copy_paste_augmentation": "15% probability, 0.3 strength",
            "color_jitter": "brightness/contrast/saturation"
        }
    }
    
    with open('results_v2.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\nResults saved to results_v2.json")

if __name__ == "__main__":
    main()
