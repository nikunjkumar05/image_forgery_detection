import os, gc, io, math, hashlib
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from PIL import Image, ImageChops, ImageEnhance
from sklearn.metrics import roc_auc_score, confusion_matrix
import kagglehub
from concurrent.futures import ThreadPoolExecutor

# ==========================================================
# 1. CUSTOM LAYERS FOR MODEL LOADING
# ==========================================================
class CrossAttentionFusion(layers.Layer):
    def __init__(self, embed_dim=256, num_heads=4, **kwargs):
        super(CrossAttentionFusion, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

    def build(self, input_shape):
        dim = input_shape[0][-1]
        self.Wq_r = self.add_weight(shape=(dim, self.embed_dim), initializer='glorot_uniform', name='Wq_r')
        self.Wk_e = self.add_weight(shape=(dim, self.embed_dim), initializer='glorot_uniform', name='Wk_e')
        self.Wv_e = self.add_weight(shape=(dim, self.embed_dim), initializer='glorot_uniform', name='Wv_e')
        self.Wq_e = self.add_weight(shape=(dim, self.embed_dim), initializer='glorot_uniform', name='Wq_e')
        self.Wk_r = self.add_weight(shape=(dim, self.embed_dim), initializer='glorot_uniform', name='Wk_r')
        self.Wv_r = self.add_weight(shape=(dim, self.embed_dim), initializer='glorot_uniform', name='Wv_r')
        self.Wo = self.add_weight(shape=(self.embed_dim * 2, self.embed_dim), initializer='glorot_uniform', name='Wo')
        self.bias = self.add_weight(shape=(self.embed_dim,), initializer='zeros', name='bias')

    def _scaled_dot_product(self, Q, K, V):
        scale = tf.math.sqrt(tf.cast(self.head_dim, Q.dtype))
        scores = tf.matmul(Q, K, transpose_b=True) / scale
        weights = tf.nn.softmax(scores, axis=-1)
        return tf.matmul(weights, V)

    def call(self, inputs):
        raw_feat, ela_feat = inputs
        Q_r = tf.matmul(raw_feat, self.Wq_r)
        K_e = tf.matmul(ela_feat, self.Wk_e)
        V_e = tf.matmul(ela_feat, self.Wv_e)
        Q_e = tf.matmul(ela_feat, self.Wq_e)
        K_r = tf.matmul(raw_feat, self.Wk_r)
        V_r = tf.matmul(raw_feat, self.Wv_r)
        def reshape_heads(x):
            bs = tf.shape(x)[0]
            x = tf.reshape(x, (bs, self.num_heads, self.head_dim))
            return tf.expand_dims(x, axis=2)
        attn_r2e = self._scaled_dot_product(reshape_heads(Q_r), reshape_heads(K_e), reshape_heads(V_e))
        attn_r2e = tf.reshape(attn_r2e, (-1, self.embed_dim))
        attn_e2r = self._scaled_dot_product(reshape_heads(Q_e), reshape_heads(K_r), reshape_heads(V_r))
        attn_e2r = tf.reshape(attn_e2r, (-1, self.embed_dim))
        combined = tf.concat([attn_r2e, attn_e2r], axis=-1)
        return tf.matmul(combined, self.Wo) + self.bias

    def get_config(self):
        config = super().get_config()
        config.update({'embed_dim': self.embed_dim, 'num_heads': self.num_heads})
        return config

class OHEMFocalLoss(tf.keras.losses.Loss):
    def __init__(self, gamma=3.0, alpha=0.25, hard_ratio=0.20, **kwargs):
        super(OHEMFocalLoss, self).__init__(**kwargs)
        self.gamma, self.alpha, self.hard_ratio = gamma, alpha, hard_ratio
    def call(self, y_true, y_pred):
        return tf.constant(0.0)
    def get_config(self):
        config = super().get_config()
        config.update({'gamma': self.gamma, 'alpha': self.alpha, 'hard_ratio': self.hard_ratio})
        return config

# ==========================================================
# 2. LOAD PRE-TRAINED MODEL
# ==========================================================
# Replace this path if your model is named differently!
model_path = "/kaggle/input/datasets/nikunjkumargond/model1/best_forgery_model_sota.keras"
if not os.path.exists(model_path):
    model_path = "/kaggle/input/datasets/nikunjkumargond/model1/best_forgery_model.keras"

print(f"Loading Model from: {model_path}")
model = tf.keras.models.load_model(
    model_path, 
    custom_objects={'CrossAttentionFusion': CrossAttentionFusion, 'OHEMFocalLoss': OHEMFocalLoss},
    compile=False
)

# ==========================================================
# 3. PREPROCESSING PIPELINE
# ==========================================================
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
CACHE_DIR = Path('/kaggle/working/cross_cache')

def compute_ela(image_path, quality=91, target_size=(224, 224)):
    img = Image.open(image_path).convert('RGB')
    channels = []
    for q in [75, 85, 95]:
        buf = io.BytesIO()
        img.save(buf, 'JPEG', quality=q)
        buf.seek(0)
        compressed = Image.open(buf)
        ela = ImageChops.difference(img, compressed).convert('L')
        extrema = ela.getextrema()
        max_diff = extrema[1] if isinstance(extrema, tuple) else extrema
        if max_diff == 0: max_diff = 1
        ela = ImageEnhance.Brightness(ela).enhance(255.0 / max_diff)
        channels.append(ela)
    mq_ela = Image.merge('RGB', channels)
    return mq_ela.resize(target_size, Image.LANCZOS)

def prepare_cache(paths, chunk_size=2000):
    raw_cache, ela_cache = [], []
    def _proc(p):
        h = hashlib.md5(p.encode()).hexdigest()[:20]
        rp = CACHE_DIR / 'raw' / f'{h}.jpg'
        ep = CACHE_DIR / 'ela' / f'{h}.jpg'
        if not rp.exists() or not ep.exists():
            img = Image.open(p).convert('RGB')
            rp.parent.mkdir(parents=True, exist_ok=True)
            img.resize(IMG_SIZE, Image.LANCZOS).save(rp, 'JPEG', quality=85)
            ep.parent.mkdir(parents=True, exist_ok=True)
            compute_ela(p, target_size=IMG_SIZE).save(ep, 'JPEG', quality=90)
        return str(rp), str(ep)
        
    for start in range(0, len(paths), chunk_size):
        chunk = paths[start:start+chunk_size]
        with ThreadPoolExecutor(max_workers=4) as ex:
            futures = [ex.submit(_proc, p) for p in chunk]
            for f in futures:
                rp, ep = f.result()
                raw_cache.append(rp)
                ela_cache.append(ep)
        print(f'Processed {min(start+chunk_size, len(paths))}/{len(paths)}')
    return raw_cache, ela_cache

def load_image(jpeg_path, png_path):
    raw = tf.image.decode_jpeg(tf.io.read_file(jpeg_path), channels=3)
    ela = tf.image.decode_jpeg(tf.io.read_file(png_path), channels=3)
    raw.set_shape([*IMG_SIZE, 3])
    ela.set_shape([*IMG_SIZE, 3])
    return raw, ela

def make_inference_ds(raw_paths, ela_paths):
    ds = tf.data.Dataset.from_tensor_slices((raw_paths, ela_paths))
    def _process(rp, ep):
        raw, ela = load_image(rp, ep)
        raw = tf.keras.applications.densenet.preprocess_input(tf.cast(raw, tf.float32))
        ela = tf.keras.applications.densenet.preprocess_input(tf.cast(ela, tf.float32))
        return {'raw_input': raw, 'ela_input': ela}
    return ds.map(_process, num_parallel_calls=tf.data.AUTOTUNE).batch(BATCH_SIZE).prefetch(1)

# ==========================================================
# 4. LOAD CASIA v2 (ZERO-SHOT)
# ==========================================================
print("Locating CASIA v2 Dataset...")
casia_path = Path('/kaggle/input/casia-20-image-tampering-detection-dataset')
if not casia_path.exists():
    casia_path = Path(kagglehub.dataset_download('divg07/casia-20-image-tampering-detection-dataset'))
    _au = list(casia_path.rglob('Au'))
    if _au: casia_path = _au[0].parent
        
au_paths = [str(p) for p in (casia_path / 'Au').rglob('*') if p.is_file() and p.suffix.lower() in {'.jpg', '.png', '.tif'}]
tp_paths = [str(p) for p in (casia_path / 'Tp').rglob('*') if p.is_file() and p.suffix.lower() in {'.jpg', '.png', '.tif'}]

all_test_paths = au_paths + tp_paths
all_test_labels = [1]*len(au_paths) + [0]*len(tp_paths)
print(f"Loaded CASIA v2: {len(au_paths)} Authentic, {len(tp_paths)} Tampered")

print("\nPreparing ELA Cache for CASIA (this will take a few minutes)...")
raw_cross, ela_cross = prepare_cache(all_test_paths)
cross_ds = make_inference_ds(raw_cross, ela_cross)

# ==========================================================
# 5. PREDICT & EVALUATE
# ==========================================================
print("\nRunning Inference...")
cross_preds = model.predict(cross_ds)
y_true = np.array(all_test_labels)

# Default Threshold of 0.5 (Can be adjusted)
y_pred = (cross_preds.ravel() >= 0.5).astype(int)

cross_acc = np.mean(y_pred == y_true)
cross_auc = roc_auc_score(y_true, cross_preds)
cm = confusion_matrix(y_true, y_pred, labels=[0,1])

print("\n" + "="*50)
print(f"🚀 ZERO-SHOT RESULTS ON CASIA v2 (NEVER SEEN BEFORE)")
print("="*50)
print(f"Accuracy: {cross_acc*100:.2f}%")
print(f"ROC-AUC:  {cross_auc:.4f}")
print("Confusion Matrix:")
print(f"True Tampered: {cm[0,0]} | False Authentic: {cm[0,1]}")
print(f"False Tampered: {cm[1,0]} | True Authentic: {cm[1,1]}")
print("="*50)
