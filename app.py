import os
import io
import base64
import numpy as np
from PIL import Image, ImageChops
import tensorflow as tf
from tensorflow.keras import layers
from typing import List
from fastapi import FastAPI, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse
import uvicorn

app = FastAPI()

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def read_root():
    return FileResponse("static/index.html")

# Custom Classes required by the model
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

# Load the model globally
MODEL_PATH = "best_forgery_model_sota.keras"
print("Loading model...")
try:
    model = tf.keras.models.load_model(
        MODEL_PATH, 
        custom_objects={'CrossAttentionFusion': CrossAttentionFusion, 'OHEMFocalLoss': OHEMFocalLoss},
        compile=False
    )
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

def compute_ela_and_raw(img, target_size=(224, 224)):
    img = img.convert('RGB')
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
    
    # Preprocess
    raw_arr = np.array(raw_resized)
    ela_arr = np.array(mq_ela_resized)
    
    raw_arr = tf.keras.applications.densenet.preprocess_input(tf.cast(raw_arr, tf.float32))
    ela_arr = tf.keras.applications.densenet.preprocess_input(tf.cast(ela_arr, tf.float32))
    
    # Return flat arrays for stacking
    return raw_arr, ela_arr, mq_ela_resized

@app.post("/predict")
async def predict_images(files: List[UploadFile] = File(...)):
    if model is None:
        return JSONResponse(status_code=500, content={"error": "Model not loaded."})
        
    if len(files) > 10:
        return JSONResponse(status_code=400, content={"error": "Maximum 10 images allowed per batch."})
        
    try:
        raw_batch = []
        ela_batch = []
        meta_data = []
        
        for file in files:
            contents = await file.read()
            img = Image.open(io.BytesIO(contents))
            
            # Prepare inputs
            raw_arr, ela_arr, mq_ela_img = compute_ela_and_raw(img)
            raw_batch.append(raw_arr)
            ela_batch.append(ela_arr)
            
            # Encode ELA image to base64 for frontend display
            buffered = io.BytesIO()
            mq_ela_img.save(buffered, format="JPEG")
            ela_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
            
            # Encode original image
            orig_buffered = io.BytesIO()
            img.copy().convert("RGB").save(orig_buffered, format="JPEG")
            orig_b64 = base64.b64encode(orig_buffered.getvalue()).decode("utf-8")
            
            meta_data.append({
                "ela_image": f"data:image/jpeg;base64,{ela_b64}",
                "original_image": f"data:image/jpeg;base64,{orig_b64}"
            })
            
        # Stack into batch tensors
        raw_tensor = np.stack(raw_batch)
        ela_tensor = np.stack(ela_batch)
        
        # Predict all at once!
        preds = model.predict({'raw_input': raw_tensor, 'ela_input': ela_tensor})
        
        results = []
        for i, pred in enumerate(preds):
            score = float(pred[0])
            is_authentic = bool(score > 0.5)
            confidence = score if is_authentic else (1 - score)
            
            results.append({
                "is_authentic": is_authentic,
                "confidence": f"{confidence * 100:.2f}%",
                "score": score,
                "ela_image": meta_data[i]["ela_image"],
                "original_image": meta_data[i]["original_image"]
            })
            
        return results
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
