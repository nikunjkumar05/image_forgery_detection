import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageChops, ExifTags
import json
import tempfile
import os
from io import BytesIO
import matplotlib.cm as cm

# optional EfficientNet preprocess
try:
    from tensorflow.keras.applications.efficientnet import preprocess_input as ef_preprocess
except Exception:
    ef_preprocess = None

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Forgery Detector", layout="centered")

# ---------------- STYLING ----------------
st.markdown("""
<style>
h1, h2, h3 {
    color: white;
}
.stButton>button {
    background: linear-gradient(90deg, #4CAF50, #2ecc71);
    color: white;
    border-radius: 10px;
    height: 3em;
    width: 100%;
    font-size: 16px;
}
.stFileUploader {
    border: 2px dashed #444;
    padding: 10px;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown("""
<h1 style='text-align: center;'>🕵️ Image Forgery Detector</h1>
<p style='text-align: center; color: gray;'>
Detect whether an image is <b>Authentic</b> or <b>Tampered</b>
</p>
""", unsafe_allow_html=True)


# ---------------- MODEL LOADER ----------------
def _load_model(path):
    """Load a Keras model from disk."""
    return tf.keras.models.load_model(path, compile=False)

if hasattr(st, "cache_resource"):
    load_model_cached = st.cache_resource(_load_model)
else:
    load_model_cached = st.cache(allow_output_mutation=True)(_load_model)


# ---------------- ELA ----------------
def compute_ela(pil_img, quality=70, scale=1.0, resize_to=None):
    """Compute Error Level Analysis image from a PIL image."""
    buf = BytesIO()
    pil_img.convert("RGB").save(buf, format="JPEG", quality=int(quality))
    buf.seek(0)
    compressed = Image.open(buf).convert("RGB")

    ela = ImageChops.difference(pil_img.convert("RGB"), compressed)
    ela_np = np.asarray(ela).astype(np.float32)

    mx = ela_np.max() if ela_np.size else 0.0
    if mx > 0:
        ela_np = (ela_np * (255.0 / mx)) * float(scale)

    ela_np = np.clip(ela_np, 0, 255).astype(np.uint8)
    ela_img = Image.fromarray(ela_np)

    if resize_to:
        ela_img = ela_img.resize(resize_to, Image.LANCZOS)

    return ela_img


# ---------------- PREPROCESS ----------------
def preprocess_pil(img_pil, target_size=(224, 224), use_efficientnet=False):
    """Resize and normalize a PIL image for model input."""
    img = img_pil.convert("RGB").resize(target_size, Image.LANCZOS)
    arr = np.asarray(img).astype("float32")

    if use_efficientnet and ef_preprocess is not None:
        arr = ef_preprocess(arr)
    else:
        arr = arr / 255.0

    return arr


def infer_input_shapes(model):
    """Infer expected input shapes from a loaded Keras model."""
    try:
        inputs = model.inputs
        if not isinstance(inputs, (list, tuple)):
            inputs = [inputs]

        shapes = []
        for inp in inputs:
            s = inp.shape.as_list()
            h = s[1] if s[1] else 224
            w = s[2] if s[2] else 224
            shapes.append((int(h), int(w)))

        return shapes
    except Exception:
        return [(224, 224)]


def load_threshold(uploaded, path):
    """Load decision threshold from uploaded JSON or file path.

    Returns the threshold value or None if not found.
    """
    try:
        if uploaded:
            data = json.loads(uploaded.getvalue())
        elif path and os.path.isfile(path):
            with open(path) as f:
                data = json.load(f)
        else:
            return None

        for k in ("threshold_authentic", "thr", "threshold", "best_threshold"):
            if k in data:
                return float(data[k])

    except Exception:
        return None


def extract_prob(preds):
    """Extract the authentic-class probability from model predictions.

    For 2-output models: index 1 = authentic (authentic=1, tampered=0).
    For 1-output sigmoid models: the value is p_authentic directly.
    """
    preds = np.asarray(preds)

    if preds.ndim == 2 and preds.shape[1] == 2:
        return float(preds[0, 1])
    elif preds.ndim == 2 and preds.shape[1] == 1:
        return float(preds[0, 0])
    else:
        return float(preds.flatten()[0])


def generate_heatmap_overlay(pil_img, ela_img):
    """Generate a heatmap overlay from the ELA image to highlight tampered regions."""
    ela_np = np.asarray(ela_img).astype(np.float32)
    gray = np.mean(ela_np, axis=2)
    norm = gray / (gray.max() + 1e-8)
    heatmap = cm.jet(norm)[:, :, :3]
    heatmap = (heatmap * 255).astype(np.uint8)
    heatmap_img = Image.fromarray(heatmap).resize(pil_img.size, Image.LANCZOS)

    blended = Image.blend(pil_img.convert("RGB"), heatmap_img, alpha=0.45)
    return blended


def get_exif_tags(pil_img):
    """Extract EXIF metadata from a PIL image as a dict."""
    info = {}
    try:
        exif_data = pil_img.getexif()
        if exif_data:
            for tag_id, value in exif_data.items():
                tag_name = ExifTags.TAGS.get(tag_id, str(tag_id))
                info[tag_name] = str(value)[:200]  # truncate long values
    except Exception:
        pass
    return info


# ---------------- UI ----------------
st.subheader("⚙️ Model Configuration")

col1, col2 = st.columns(2)

with col1:
    model_file = st.file_uploader("Upload Model (.keras/.h5)", type=["keras", "h5"])

with col2:
    model_path = st.text_input("Model Path", value="best_forgery_model.keras")

threshold_file = st.file_uploader("Threshold JSON", type=["json"])
threshold_path = st.text_input("Threshold Path", value="best_threshold.json")

st.subheader("🧠 Prediction Settings")

col1, col2 = st.columns(2)

with col1:
    input_hint = st.selectbox("Input Mode",
        ["Auto detect (recommended)", "RAW only", "ELA only", "Dual (RAW+ELA)"])

with col2:
    preproc_choice = st.selectbox("Preprocessing",
        ["Auto (EfficientNet if available)", "Normalize 0..1"])

ela_quality = st.slider("ELA JPEG Quality", 50, 99, 91,
    help="Must match the quality used during training (default: 91).")

uploaded_image = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

# Load threshold early so user can see/adjust it before clicking Run
_threshold_json = load_threshold(threshold_file, threshold_path)
_default_threshold = _threshold_json if _threshold_json is not None else 0.5
threshold = st.slider("Decision Threshold", 0.0, 1.0, float(_default_threshold),
                       help="Lower = more sensitive to tampering. Higher = fewer false positives.")

col_btn = st.columns([1, 2, 1])
with col_btn[1]:
    run = st.button("🚀 Run Prediction")


# ---------------- MAIN LOGIC ----------------
if run:
    if not uploaded_image:
        st.error("Upload an image first.")
        st.stop()

    # Load model — validate path is a real file to prevent arbitrary path access
    if model_file:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".keras")
        try:
            tmp.write(model_file.getvalue())
            tmp.close()
            model_path_to_load = tmp.name
        except Exception as e:
            st.error(f"Failed to save uploaded model: {e}")
            st.stop()
    else:
        if not os.path.isfile(model_path):
            st.error(f"Model file not found: `{model_path}`")
            st.stop()
        model_path_to_load = model_path

    # Load model with error handling
    try:
        with st.spinner("Loading model..."):
            model = load_model_cached(model_path_to_load)
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        # Clean up temp file if it was an upload
        if model_file and os.path.exists(model_path_to_load):
            os.unlink(model_path_to_load)
        st.stop()

    shapes = infer_input_shapes(model)
    is_dual = len(shapes) >= 2

    # Respect the user's input mode selection
    if input_hint.startswith("Auto"):
        input_mode = "dual" if is_dual else "raw"
    elif "ELA" in input_hint and "RAW" in input_hint:
        input_mode = "dual"
    elif "ELA" in input_hint:
        input_mode = "ela"
    else:
        input_mode = "raw"

    use_effnet = (preproc_choice.startswith("Auto") and ef_preprocess is not None)

    # Open and preprocess image with error handling
    try:
        with st.spinner("Preprocessing image..."):
            img = Image.open(uploaded_image).convert("RGB")
    except Exception as e:
        st.error(f"Failed to open image: {e}")
        if model_file and os.path.exists(model_path_to_load):
            os.unlink(model_path_to_load)
        st.stop()

    with st.spinner("Preprocessing image..."):
        raw_arr = preprocess_pil(img, shapes[0], use_effnet)

        # Only compute ELA if needed for model input or display
        ela_img = compute_ela(img, quality=ela_quality, resize_to=shapes[0])
        if input_mode in ("dual", "ela"):
            ela_arr = preprocess_pil(ela_img, shapes[0], use_effnet)
        else:
            ela_arr = None

        raw_batch = np.expand_dims(raw_arr, 0)

        if input_mode == "dual" and ela_arr is not None:
            ela_batch = np.expand_dims(ela_arr, 0)
            inputs = [raw_batch, ela_batch]
        elif input_mode == "ela" and ela_arr is not None:
            inputs = np.expand_dims(ela_arr, 0)
        else:
            inputs = raw_batch

    with st.spinner("Running inference..."):
        preds = model(inputs, training=False)
    prob_auth = extract_prob(preds)

    label = "Authentic" if prob_auth >= threshold else "Tampered"

    # Clean up temp file after inference is complete
    if model_file and os.path.exists(model_path_to_load):
        os.unlink(model_path_to_load)

    # ---------------- RESULT UI ----------------
    st.markdown("---")
    st.subheader("📊 Prediction Result")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Prediction", label)
        st.metric("Confidence", f"{max(prob_auth, 1 - prob_auth) * 100:.1f}%")

    with col2:
        st.metric("Authentic Prob", f"{prob_auth:.4f}")
        st.metric("Tampered Prob", f"{1 - prob_auth:.4f}")

    # Color-coded result bar
    if label == "Authentic":
        st.success(f"✅ Image classified as **{label}** (threshold: {threshold:.2f})")
    else:
        st.error(f"🚨 Image classified as **{label}** (threshold: {threshold:.2f})")

    # ---------------- EXIF METADATA ----------------
    exif = get_exif_tags(img)
    if exif:
        with st.expander("📋 EXIF Metadata (may reveal editing software)"):
            for k, v in exif.items():
                st.text(f"{k}: {v}")

    # ---------------- HEATMAP OVERLAY ----------------
    overlay = generate_heatmap_overlay(img, ela_img)

    # ---------------- IMAGE DISPLAY ----------------
    st.subheader("🖼 Image Analysis")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.image(img, caption="Original", use_container_width=True)

    with col2:
        st.image(ela_img, caption="ELA", use_container_width=True)

    with col3:
        st.image(overlay, caption="Heatmap Overlay", use_container_width=True)

    # ---------------- DOWNLOAD ----------------
    st.subheader("⬇️ Download Results")

    buf_out = BytesIO()
    overlay.save(buf_out, format="PNG")
    st.download_button(
        label="Download Heatmap Overlay",
        data=buf_out.getvalue(),
        file_name="heatmap_overlay.png",
        mime="image/png",
    )
