# 🫁 Multimodal Pulmonary Disease Diagnosis using Deep Learning & Explainable AI

A multimodal deep learning system for pulmonary disease classification using **respiratory audio (spectrograms)** and **chest X-ray imaging**, enhanced with **ensemble learning** and **Explainable AI (XAI)** for clinically interpretable predictions.

---

## 🚀 Key Highlights

* 🔊 **Audio + Imaging (Multimodal)**: Combines ICBHI respiratory sounds and VinBigData chest X-rays
* 🧠 **Deep CNN Architectures**: ResNet50, VGG19, Xception, Inception variants
* 🔗 **Ensemble Learning**: Averaging + **stacking (Logistic Regression / MLP meta-models)**
* 🔍 **Explainability (XAI)**: Grad-CAM, Grad-CAM++, LIME, Score-CAM, Layer-CAM, Occlusion
* ⚙️ **Reproducible Pipeline**: Preprocessing → Training → Evaluation → Stacking → XAI

---

## 📊 Results

* **ICBHI (Audio)**: **97.08% Accuracy (Macro F1: 0.97)** *(4-class)*
* **VinBigData (X-ray)**: **91.28% Accuracy (Macro F1: 0.874)** *(3-class)*
* **~30% absolute improvement** over baseline ICBHI models

---

## 📊 Datasets

### 🔊 ICBHI 2017 Respiratory Sound Dataset

| Attribute              | Description                                            |
| ---------------------- | ------------------------------------------------------ |
| Domain                 | Biomedical Audio                                       |
| Recordings             | 920                                                    |
| Patients               | 126                                                    |
| Respiratory Cycles     | ~6,898                                                 |
| Generated Spectrograms | 4,787                                                  |
| Sampling Rate          | 4 kHz                                                  |
| Labels                 | COPD, URTI, Pneumonia, Bronchiectasis, Asthma, Healthy |

🔗 https://bhichallenge.med.auth.gr/

---

### 🩻 VinBigData Chest X-ray Dataset

| Attribute         | Description               |
| ----------------- | ------------------------- |
| Domain            | Medical Imaging           |
| Images            | ~18,000+                  |
| Type              | Frontal Chest X-rays      |
| Preprocessed Size | 224 × 224                 |
| Classes           | 14 Thoracic Abnormalities |
| Annotations       | Bounding boxes + labels   |

🔗 https://www.kaggle.com/c/vinbigdata-chest-xray-abnormalities-detection

---

## 🧠 Label Engineering

### 🔊 ICBHI → 4 Classes

* **COPD**
* **Healthy**
* **Infection** *(Pneumonia, URTI, LRTI)*
* **OtherRD** *(Bronchiolitis, Bronchiectasis, Asthma)*

---

### 🩻 VinBigData → 3 Classes

* **no_finding**
* **vascular_abnormality**
* **nonvascular_abnormality** *(aggregates multiple thoracic conditions)*

> ⚠️ The nonvascular class includes diverse pathologies, increasing intra-class variability and making the task more challenging.

---

## 🔊 Audio Pipeline (ICBHI)

* Converted respiratory sounds → **Mel spectrograms**
* Generated labeled spectrogram samples per patient
* Applied **stratified sampling (~300/class)**
* Trained models:

  * ResNet50, VGG19, Xception
  * InceptionV3, InceptionV4
  * Inception-ResNetV1, V2
* Ensemble:

  * Averaging + **stacking (Logistic Regression, MLP)**

---

## 🩻 Imaging Pipeline (VinBigData)

* Custom PyTorch dataset for PNG X-rays
* Data augmentation:

  * RandomResizedCrop, flips, rotation
  * Color jitter
  * **CLAHE (contrast enhancement)**
  * Random Erasing
* Trained:

  * ResNet50, VGG19, Xception
* Used **stratified cross-validation**
* Ensemble stacking with Logistic Regression

---

### 🔍 Explainability (XAI)

Applied multiple XAI techniques to interpret model predictions:

- **Grad-CAM / Grad-CAM++** → highlights important regions influencing predictions  
- **Score-CAM / Layer-CAM** → improves localization of salient features  
- **Occlusion** → evaluates prediction sensitivity by masking input regions  
- **LIME** → provides local, model-agnostic explanations  

Used to verify model focus on clinically relevant regions in spectrograms and X-ray images

### For Ensemble Models:

* Grad-CAM, Grad-CAM++

Ensures model focuses on **clinically relevant regions**

---

## 📁 Project Structure

```bash
multimodal-pulmonary-disease-xai/
│
├── data/
├── notebooks/
├── src/
├── models/
├── results/
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

```bash
git clone https://github.com/your-username/multimodal-pulmonary-disease-xai.git
cd multimodal-pulmonary-disease-xai
pip install -r requirements.txt
```

---

## 📸 Sample Outputs

* Spectrogram samples
* <img width="670" height="242" alt="image" src="https://github.com/user-attachments/assets/65c8530a-dbf2-4fd3-bc3f-51913b9c69bd" />
* X-Ai visualizations
  <Figure size 1600x1200 with 9 Axes><img width="1532" height="1180" alt="image" src="https://github.com/user-attachments/assets/27143d78-b7ad-4489-be5a-168ebafcef44" />
  <Figure size 1600x1200 with 9 Axes><img width="1589" height="1181" alt="image" src="https://github.com/user-attachments/assets/4e732be4-9147-4d66-9128-4c2beb6e51b4" />
  <Figure size 600x600 with 1 Axes><img width="567" height="640" alt="image" src="https://github.com/user-attachments/assets/01a57c66-cc33-4d4c-9274-1ed6292108f2" />
  <img width="1387" height="712" alt="image" src="https://github.com/user-attachments/assets/20c9a713-db6d-48ed-bb24-bb86be6faa4c" />
* Results
  *On VIGBIG19 stacked model
  <img width="713" height="337" alt="image" src="https://github.com/user-attachments/assets/81574d0c-0112-41b4-a899-0833de3fdee2" />
  *On ICHBI data stacked ensembled model
  <img width="537" height="375" alt="image" src="https://github.com/user-attachments/assets/224895a8-a015-45d0-951b-6fad4a17dffd" />

## 🔮 Future Work

* Real-time deployment (Flask / FastAPI)
* Transformer-based multimodal fusion
* Clinical validation

---

## 👤 Author

**Jyotiraditya Biswas**
**M.Tech Data Science, DTU**
