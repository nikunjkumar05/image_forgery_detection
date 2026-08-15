"""
Image Forgery Detector — Premium Streamlit App
Dual-Input EfficientNetB3 with Cross-Attention Fusion
"""

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageChops, ImageFilter
import json
import os
from io import BytesIO
import cv2

# Page config
st.set_page_config(page_title="ImageForge Detector", page_icon="🕵️", layout="wide")

# CSS
st.markdown("""
<style>
.stApp { background: linear-gradient(135deg, #06060f 0%, #0d0d1f 40%, #06060f 100%); color: #e2e8f0; }
.hero-title { font-size: 2.5rem; font-weight: 700; background: linear-gradient(90deg, #00d4ff, #7c3aed); -webkit-background-clip: text; -webkit-text-fill-color: transparent; text-align: center; }
.glass-card { background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.08); border-radius: 12px; padding: 1.2rem; margin: 0.5rem 0; }
.badge-auth { background: #064e3b; border: 1px solid #10b981; color: #6ee7b7; border-radius: 50px; padding: 0.5rem 2rem; font-size: 1.2rem; font-weight: bold; text-align: center; }
.badge-tamp { background: #7f1d1d; border: 1px solid #ef4444; color: #fca5a5; border-radius: 50px; padding: 0.5rem 2rem; font-size: 1.2rem; font-weight: bold; text-align: center; }
</style>
""", unsafe_allow_html=True)

# ML Functions
@st.cache_resource
def load_model(path):
    return tf.keras.models.load_model(path, compile=False)

def compute_ela(img, quality=90):
    buf = BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    compressed = Image.open(buf).convert("RGB")
    ela = ImageChops.difference(img.convert("RGB"), compressed)
    ela_np = np.asarray(ela).astype(np.float32)
    mx = ela_np.max() if ela_np.size else 0.0
    if mx > 0: ela_np = np.clip(ela_np * (255.0/mx), 0, 255)
    return Image.fromarray(ela_np.astype(np.uint8))

def compute_ela_multispectral(img):
    """MQ-ELA: Compute ELA at Q=75, Q=85, Q=95 and map to R, G, B channels."""
    channels = []
    for q in [75, 85, 95]:
        buf = BytesIO()
        img.convert("RGB").save(buf, format="JPEG", quality=q)
        buf.seek(0)
        compressed = Image.open(buf).convert("RGB")
        ela = ImageChops.difference(img.convert("RGB"), compressed)
        ela_gray = ela.convert("L")
        ela_np = np.asarray(ela_gray).astype(np.float32)
        mx = ela_np.max() if ela_np.size else 0.0
        if mx > 0: ela_np = np.clip(ela_np * (255.0/mx), 0, 255)
        channels.append(Image.fromarray(ela_np.astype(np.uint8)))
    return Image.merge("RGB", channels)

def predict_with_tta(model, raw_arr, ela_arr, n_aug=5):
    """Test-Time Augmentation: average predictions with noise perturbations."""
    preds = []
    for i in range(n_aug):
        noise_level = 0.003 * (i + 1)
        noise = np.random.normal(0, noise_level, raw_arr.shape)
        r_noisy = np.clip(raw_arr + noise, 0, 1)
        p = model([r_noisy, ela_arr], training=False)
        preds.append(float(p[0][0]))
    return np.mean(preds)

def preprocess(img, size=(224, 224)):
    img = img.convert("RGB").resize(size, Image.LANCZOS)
    return np.asarray(img).astype("float32") / 255.0

def apply_attacks(img, jpeg_quality, blur_radius, noise_sigma):
    """Simulate forensic attacks on-the-fly."""
    out = img.convert("RGB")
    if blur_radius > 0:
        out = out.filter(ImageFilter.GaussianBlur(blur_radius))
    
    if noise_sigma > 0:
        arr = np.asarray(out).astype(np.float32)
        noise = np.random.normal(0, noise_sigma, arr.shape)
        arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
        out = Image.fromarray(arr)
        
    if jpeg_quality < 100:
        buf = BytesIO()
        out.save(buf, format="JPEG", quality=jpeg_quality)
        buf.seek(0)
        out = Image.open(buf).convert("RGB")
        
    return out

def compute_saliency_map(model, raw_arr, ela_arr):
    """Computes Input Saliency CAM (Keras 3 compatible)."""
    r_tensor = tf.convert_to_tensor(raw_arr, dtype=tf.float32)
    e_tensor = tf.convert_to_tensor(ela_arr, dtype=tf.float32)
    
    with tf.GradientTape() as tape:
        tape.watch(r_tensor)
        preds = model([r_tensor, e_tensor], training=False)
        score = preds[:, 0]
        
    grads = tape.gradient(score, r_tensor)
    saliency = tf.reduce_mean(tf.abs(grads[0]), axis=-1)
    saliency = tf.maximum(saliency, 0)
    saliency_np = saliency.numpy()
    if saliency_np.max() > 0:
        saliency_np = saliency_np / saliency_np.max()
    return saliency_np

def overlay_saliency(img, saliency, alpha=0.4):
    """Overlay the heatmap on the original image."""
    img_np = np.asarray(img.resize((224, 224), Image.LANCZOS)).astype(np.float32) / 255.0
    heatmap = cv2.resize(saliency, (224, 224))
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    overlayed = (1 - alpha) * img_np + alpha * heatmap_colored
    return np.clip(overlayed, 0, 1)

# Header
st.markdown("<p class='hero-title'>🕵️ ImageForge AI Detector</p>", unsafe_allow_html=True)

# Sidebar settings
with st.sidebar:
    st.header("⚙️ Core Settings")
    model_path = st.text_input("Model Path", value="best_forgery_model.keras")
    threshold = st.slider("Authentic Threshold", 0.0, 1.0, 0.50, 0.01)
    ela_q = st.slider("ELA Quality", 50, 99, 90)
    use_tta = st.checkbox("Enable TTA (5x inference, +1-2% acc)", value=True)
    use_multi_ela = st.checkbox("Enable Multi-Quality ELA (Q75+85+95)", value=True)
    
    st.header("🛠️ Attack Simulation (Robustness)")
    sim_jpeg = st.slider("JPEG Compression (Quality)", 10, 100, 100)
    sim_blur = st.slider("Gaussian Blur (Radius)", 0.0, 5.0, 0.0, 0.5)
    sim_noise = st.slider("Gaussian Noise (Sigma)", 0.0, 30.0, 0.0, 2.0)

tab1, tab2, tab3 = st.tabs(["🔍 Live Detector", "📊 Performance Metrics", "🧠 Architecture"])

with tab1:
    upload = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])
    if upload:
        img = Image.open(upload)
        
        # Apply attacks
        attacked_img = apply_attacks(img, sim_jpeg, sim_blur, sim_noise)
        
        # Compute ELA - multi-quality or single quality
        if use_multi_ela:
            ela_img = compute_ela_multispectral(attacked_img)
        else:
            ela_img = compute_ela(attacked_img, ela_q)
        
        c1, c2 = st.columns(2)
        c1.image(attacked_img, caption="Attacked/Modified Image", use_container_width=True)
        c2.image(ela_img, caption="ELA Map" + (" (Multi-Quality)" if use_multi_ela else ""), use_container_width=True)
        
        if st.button("🚀 Run Forensic Inference", use_container_width=True):
            if not os.path.exists(model_path):
                st.error("Model file not found! Upload your best_forgery_model.keras first.")
            else:
                with st.spinner("Processing dual-stream features..." + (" (TTA enabled)" if use_tta else "")):
                    model = load_model(model_path)
                    
                    raw_arr = np.expand_dims(preprocess(attacked_img), 0)
                    ela_arr = np.expand_dims(preprocess(ela_img), 0)
                    
                    try:
                        if use_tta:
                            prob_auth = predict_with_tta(model, raw_arr, ela_arr, n_aug=5)
                        else:
                            preds = model([raw_arr, ela_arr], training=False)
                            prob_auth = float(preds[0][0])
                        
                        prob_tamp = 1.0 - prob_auth
                        
                        # Apply calibrated threshold
                        label = "Authentic" if prob_auth >= threshold else "Tampered"
                        badge_style = 'badge-auth' if label=='Authentic' else 'badge-tamp'
                        
                        st.markdown(f"<div style='text-align:center;margin:1.5rem;'><span class='{badge_style}'>{label}</span></div>", unsafe_allow_html=True)
                        
                        m1, m2 = st.columns(2)
                        m1.metric("Authentic Probability", f"{prob_auth:.4f}")
                        m2.metric("Tampered Probability", f"{prob_tamp:.4f}")
                        
                        # Grad-CAM localization
                        st.markdown("### 🔍 Localization Heatmap (Where did it tamper?)")
                        saliency = compute_saliency_map(model, raw_arr, ela_arr)
                        overlayed = overlay_saliency(attacked_img, saliency)
                        
                        lc1, lc2 = st.columns(2)
                        lc1.image(saliency, caption="Saliency Heatmap", use_container_width=True)
                        lc2.image(overlayed, caption="Grad-CAM Forensic Overlay", use_container_width=True)
                        
                    except Exception as e:
                        st.error(f"Inference failed: {e}")

with tab2:
    st.markdown("### Model Metrics")
    if os.path.exists("results.json"):
        with open("results.json") as f:
            res = json.load(f)
        
        acc = res.get('accuracy', 0)
        acc_pct = acc * 100.0 if acc <= 1.0 else acc
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Test Accuracy", f"{acc_pct:.2f}%")
        c2.metric("Test AUC-ROC", f"{res.get('auc', 0):.4f}")
        c3.metric("F1 Tampered", f"{res.get('f1_tampered', 0):.4f}")
        c4.metric("F1 Authentic", f"{res.get('f1_authentic', 0):.4f}")
        
        st.markdown("#### Per-Class Performance")
        per_class = res.get('per_class', {})
        if per_class:
            pc1, pc2, pc3 = st.columns(3)
            pc1.metric("Splicing", f"{per_class.get('splicing', 0)*100:.1f}%")
            pc2.metric("Copy-Move", f"{per_class.get('copy_move', 0)*100:.1f}%")
            pc3.metric("Authentic", f"{per_class.get('authentic', 0)*100:.1f}%")
        
        st.markdown("#### Robustness Against Attacks")
        robustness = res.get('robustness', {})
        if robustness:
            for name, val in robustness.items():
                if name != 'Original':
                    drop = robustness.get('Original', val) - val
                    st.write(f"- **{name}**: {val*100:.1f}% (drop: {drop*100:.1f}%)")
        
        st.markdown("#### Scientific Ablation Study Results")
        ab_data = res.get('ablation', {})
        if ab_data:
            st.json(ab_data)
    else:
        st.info("results.json not found.")

with tab3:
    st.markdown("### Dual-Stream EfficientNetB3 + Cross-Attention")
    st.markdown("""
    <div class='glass-card'>
    <b>1. RGB Stream:</b> Captures spatial tampering artifacts (splicing edges, cloning).<br>
    <b>2. ELA Stream:</b> Captures compression anomalies invisible to the human eye.<br>
    <b>3. Cross-Attention:</b> Network learns exactly which regions differ between visual semantics and compression signatures.<br>
    </div>
    """, unsafe_allow_html=True)

