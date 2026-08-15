# Image Forgery Detection: Lightweight Multi-Spectral Forgery Network

![Image Forgery Detection Banner](https://img.shields.io/badge/Architecture-Dual%_DenseNet121-blue) ![Parameters](https://img.shields.io/badge/Parameters-16.5M-success) ![Accuracy](https://img.shields.io/badge/CASIA_v2_Accuracy-88.97%25-brightgreen)

Image Forgery Detection is a state-of-the-art, physics-informed neural network designed to detect image manipulations (splicing, copy-move, and removal). By combining raw spatial semantics with Multi-Spectral Error Level Analysis (MQ-ELA) frequency data, it mathematically exposes invisible compression boundaries left behind by forgers.

---

## 🟢 Beginner Guide: Getting Started

If you just want to run the model and test images, this section is for you.

### 1. Installation
Ensure you have Python 3.10+ installed. Open your terminal in this project folder and run:
```bash
pip install -r requirements.txt
# If you don't have the requirements file yet, run:
pip install tensorflow Pillow fastapi uvicorn python-multipart numpy
```

### 2. Running the Web Application
We have built a beautiful, dark-mode web application that allows you to drag-and-drop up to 10 images at once to instantly scan them for forgeries.
```bash
uvicorn app:app --reload
```
Open your web browser and navigate to: **http://localhost:8000**

### 3. Understanding the Results
When you upload an image, the model returns two things:
- **Verdict & Confidence:** E.g., `Authentic (98.4%)` or `Forged (99.1%)`.
- **MQ-ELA Frequency Map:** A side-by-side visualization showing the "physical heat map" of the image. Forgers often leave behind sharp, jagged compression artifacts around spliced objects. The model uses this map to mathematically prove the forgery.

---

## 🟡 Intermediate Guide: Datasets & Scripts

### Project Structure
- `app.py`: The FastAPI backend serving the model and UI.
- `static/`: Contains the HTML, CSS, and JS for the web interface.
- `best_forgery_model_sota.keras`: The primary trained neural network weights.
- `evaluate_and_sort.py`: A high-throughput script that runs inference on massive datasets (e.g., CASIA v2) and automatically sorts True Positives (TP) and True Negatives (TN) into separate folders on your Desktop for presentation purposes.

### Using the Batch Sorting Script
If you have a large dataset folder containing `authentic` and `tampered` subfolders, you can use the evaluation script to process the entire dataset in the background:
```bash
python evaluate_and_sort.py
```
This will automatically read the dataset, predict every image, and copy the correctly classified images directly to your Desktop under `Model_Evaluation_Results/auth` and `Model_Evaluation_Results/tamp`.

### Evaluation Metrics
Our model was rigorously tested across multiple benchmark datasets, demonstrating state-of-the-art generalization and parameter efficiency.

| Dataset | Type | Accuracy | ROC-AUC | F1-Score |
|---|---|---|---|---|
| **DEFACTO** | Splicing & Copy-Move | **89.66%** | **0.9279** | 0.9012 |
| **CASIA v2** | JPEG Compression/Splicing | **88.97%** | **0.9656** | 0.8930 |
| **Columbia** | Uncompressed Splicing | **90.41%** | **0.9814** | 0.9150 |

*(Note: Image Forgery Detection achieves these metrics using only ~16.5 Million parameters, representing a 55% footprint reduction compared to recent Vision Mamba and Transformer baselines).*

---

## 🔴 Advanced Guide: Architecture & Custom Implementation

Image Forgery Detection achieves SOTA precision by using a highly customized architecture. 

### 1. Dual DenseNet121 Backbone
Unlike standard CNNs or Vision Transformers, we utilize two parallel `DenseNet121` feature extractors. The dense connections prevent the "vanishing gradient" problem, ensuring that microscopic, high-frequency ELA artifacts are preserved all the way to the final classification head. This allows us to achieve high accuracy with a fraction of the parameters (~16.5M).

### 2. Bidirectional Cross-Attention
We do not simply concatenate the spatial and frequency streams. We implemented a custom `CrossAttentionFusion` layer. 
- The **Spatial stream** queries the **Frequency stream** (`Q_r * K_e`).
- The **Frequency stream** queries the **Spatial stream** (`Q_e * K_r`).
This mathematically forces the network to correlate spatial semantics (e.g., "this is a person") with physical anomalies (e.g., "there is a compression error exactly along this person's outline"), drastically reducing false positives.

### 3. OHEM Focal Loss
The model utilizes a custom `OHEMFocalLoss` ($\gamma=3.0$). It applies Online Hard Example Mining (OHEM) to discard the bottom 80% of easy predictions in every batch, strictly updating weights based on the top 20% hardest forged images. 

### 4. Loading the Model in Custom Python Scripts
If you want to use the `.keras` model in your own custom pipeline, you MUST instantiate the custom objects during load:

```python
import tensorflow as tf
from app import CrossAttentionFusion, OHEMFocalLoss

model = tf.keras.models.load_model(
    "best_forgery_model_sota.keras", 
    custom_objects={
        'CrossAttentionFusion': CrossAttentionFusion, 
        'OHEMFocalLoss': OHEMFocalLoss
    },
    compile=False # Load in inference mode
)

# Remember to calculate the MQ-ELA tensor before passing it to predict()!
```
