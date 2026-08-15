import os
import io
import shutil
from pathlib import Path
import numpy as np
from PIL import Image, ImageChops
import tensorflow as tf
from tensorflow.keras import layers
import concurrent.futures
import time

print("Starting Evaluation Script...")

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
        
        attn_r2e = self._scaled_dot_product(Q_r, K_e, V_e)
        attn_e2r = self._scaled_dot_product(Q_e, K_r, V_r)
        
        combined = tf.concat([attn_r2e, attn_e2r], axis=-1)
        output = tf.matmul(combined, self.Wo) + self.bias
        return output

class OHEMFocalLoss(tf.keras.losses.Loss):
    def __init__(self, gamma=3.0, alpha=0.25, hard_ratio=0.20, **kwargs):
        super(OHEMFocalLoss, self).__init__(**kwargs)
        self.gamma = gamma
        self.alpha = alpha
        self.hard_ratio = hard_ratio

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1.0 - tf.keras.backend.epsilon())
        bce = tf.keras.backend.binary_crossentropy(y_true, y_pred)
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        alpha_factor = y_true * self.alpha + (1 - y_true) * (1 - self.alpha)
        focal_loss = alpha_factor * tf.pow(1.0 - p_t, self.gamma) * bce
        
        bs = tf.shape(focal_loss)[0]
        k = tf.cast(tf.math.ceil(tf.cast(bs, tf.float32) * self.hard_ratio), tf.int32)
        top_k_loss, _ = tf.math.top_k(focal_loss, k=k)
        return tf.reduce_mean(top_k_loss)

def compute_ela_and_raw(img_path, target_size=(224, 224)):
    try:
        img = Image.open(img_path).convert('RGB')
        channels = []
        for q in [75, 85, 95]:
            buf = io.BytesIO()
            img.save(buf, 'JPEG', quality=q)
            buf.seek(0)
            compressed = Image.open(buf)
            ela = ImageChops.difference(img, compressed).convert('L')
            extrema = ela.getextrema()
            if extrema[1] != 0:
                scale = 255.0 / extrema[1]
                ela = ela.point(lambda p: p * scale)
            channels.append(ela)
            
        mq_ela = Image.merge('RGB', channels)
        mq_ela_resized = mq_ela.resize(target_size, Image.LANCZOS)
        raw_resized = img.resize(target_size, Image.LANCZOS)
        
        raw_arr = np.array(raw_resized)
        ela_arr = np.array(mq_ela_resized)
        
        raw_arr = tf.keras.applications.densenet.preprocess_input(tf.cast(raw_arr, tf.float32))
        ela_arr = tf.keras.applications.densenet.preprocess_input(tf.cast(ela_arr, tf.float32))
        
        return raw_arr, ela_arr
    except Exception as e:
        return None, None

def find_desktop():
    onedrive_desktop = Path(os.path.expanduser("~")) / "OneDrive" / "Desktop"
    if onedrive_desktop.exists():
        return onedrive_desktop
    return Path(os.path.expanduser("~")) / "Desktop"

def run_evaluation():
    desktop = find_desktop()
    target_dir = desktop / "Model_Evaluation_Results"
    
    auth_dir = target_dir / "auth"
    tamp_dir = target_dir / "tamp"
    
    auth_dir.mkdir(parents=True, exist_ok=True)
    tamp_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Created output directories on Desktop:\n- {auth_dir}\n- {tamp_dir}")
    
    print("Loading model...")
    model = tf.keras.models.load_model(
        "best_forgery_model_sota.keras", 
        custom_objects={'CrossAttentionFusion': CrossAttentionFusion, 'OHEMFocalLoss': OHEMFocalLoss},
        compile=False
    )
    
    dataset_dir = Path("dataset")
    auth_paths = list((dataset_dir / "authentic").glob("*.*"))
    tamp_paths = list((dataset_dir / "tampered").glob("*.*"))
    
    # We will just process up to a large number of images but maybe not all 12,000 if it takes too long
    # Let's process batches
    
    all_files = [(p, 1) for p in auth_paths] + [(p, 0) for p in tamp_paths] # 1=Authentic, 0=Tampered
    import random
    random.shuffle(all_files)
    
    BATCH_SIZE = 32
    print(f"Found {len(all_files)} total images to process. Starting batch inference...")
    
    tp_count = 0
    tn_count = 0
    total_processed = 0
    
    start_time = time.time()
    
    for i in range(0, len(all_files), BATCH_SIZE):
        batch = all_files[i:i+BATCH_SIZE]
        raw_list = []
        ela_list = []
        valid_meta = []
        
        for p, label in batch:
            raw_arr, ela_arr = compute_ela_and_raw(str(p))
            if raw_arr is not None:
                raw_list.append(raw_arr)
                ela_list.append(ela_arr)
                valid_meta.append((p, label))
                
        if not raw_list:
            continue
            
        raw_tensor = np.stack(raw_list)
        ela_tensor = np.stack(ela_list)
        
        preds = model.predict({'raw_input': raw_tensor, 'ela_input': ela_tensor}, verbose=0)
        
        for j, pred in enumerate(preds):
            score = float(pred[0])
            is_authentic_pred = score > 0.5
            path, true_label = valid_meta[j]
            is_authentic_true = (true_label == 1)
            
            # True Negative (Authentic correctly predicted as Authentic)
            if is_authentic_true and is_authentic_pred:
                shutil.copy2(path, auth_dir / path.name)
                tn_count += 1
                
            # True Positive (Tampered correctly predicted as Tampered)
            elif not is_authentic_true and not is_authentic_pred:
                shutil.copy2(path, tamp_dir / path.name)
                tp_count += 1
                
        total_processed += len(valid_meta)
        
        if total_processed % 320 == 0:
            elapsed = time.time() - start_time
            print(f"Processed {total_processed}/{len(all_files)} | TP: {tp_count}, TN: {tn_count} | Time: {elapsed:.1f}s")
            
    print(f"Finished! Total TP: {tp_count}, Total TN: {tn_count}")
    print(f"Results saved to: {target_dir}")

if __name__ == "__main__":
    run_evaluation()
