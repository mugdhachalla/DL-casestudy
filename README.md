# Explainable AI for Diabetic Retinopathy Screening

**A comparative study of EfficientNetB0, ResNet50 and DenseNet121 with Grad-CAM**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mugdhachalla/DL-casestudy/blob/main/Batch5_Code_Updated.ipynb)
![Python](https://img.shields.io/badge/Python-3-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Keras](https://img.shields.io/badge/Keras-3-red)

Deep learning case study from the Department of CSE (AI & ML), Chaitanya Bharathi Institute of Technology, Hyderabad.

This project grades retinal fundus images into the five stages of diabetic retinopathy (DR) and flags **referable DR** (Moderate or worse) for specialist review. It is aimed at screening in rural Primary Health Centres where ophthalmologists are scarce. Three ImageNet-pretrained CNNs are trained under an identical protocol, combined in an ensemble, and explained with Grad-CAM heatmaps.

---

## Highlights

- **3 backbones + ensemble:** EfficientNetB0, ResNet50 and DenseNet121 share the same classification head, so the comparison isolates the backbone
- **Clinically meaningful evaluation:** Quadratic Weighted Kappa (the official APTOS metric) and referable-DR ROC-AUC, with the referral threshold tuned on validation data for **≥ 90% sensitivity**
- **Rigorous split:** stratified 70 / 15 / 15 train / validation / test; the test set is used only once
- **Class-imbalance handling:** class-weighted loss (the largest class is 9.4× the smallest)
- **Explainability:** side-by-side Grad-CAM for all three models, exposing both lesion-focused attention and shortcut learning
- **Screening demo:** a single image goes in, and a grade, referral probability, refer / don't-refer decision and heatmap come out

---

## Results (held-out test set, 550 images)

| Model | Accuracy | Macro-F1 | QWK | Referable AUC | Sensitivity | Specificity | Params |
|---|---|---|---|---|---|---|---|
| EfficientNetB0 | **72.36%** | **0.5345** | 0.7762 | 0.9582 | 89.24% | 88.99% | **4.4 M** |
| ResNet50 | 69.27% | 0.4862 | **0.7994** | 0.9651 | **91.93%** | 89.60% | 24.1 M |
| DenseNet121 | 64.55% | 0.4583 | 0.7518 | 0.9501 | 90.58% | 85.93% | 7.3 M |
| **Ensemble** | 70.18% | 0.5231 | 0.7906 | **0.9660** | 91.03% | **89.91%** | 35.8 M |

- **ResNet50** has the best grading agreement (QWK 0.799), so it makes the fewest large grading errors.
- **EfficientNetB0** has the best accuracy and macro-F1 with 5.5× fewer parameters, the best fit for edge devices.
- **The ensemble** gives the best referral performance (AUC 0.966).
- **Moderate DR is the hardest grade** for every model, often confused with Mild or Severe. Most of these errors stay within the referable group, so referral performance remains high.

---

## Grad-CAM: where each model looks

ResNet50 and DenseNet121 focus on the exudate clusters in the Moderate and Severe examples. In some cases, though, models attend to image borders or corners. These are correct answers for the wrong reason, which accuracy alone would never reveal.

---

## Dataset

[APTOS 2019 Blindness Detection](https://www.kaggle.com/competitions/aptos2019-blindness-detection), in the pre-processed [224×224 Gaussian-filtered release](https://www.kaggle.com/datasets/sovitrath/diabetic-retinopathy-224x224-gaussian-filtered). The notebook downloads it automatically via `kagglehub`.

| Grade | Class | Images | Train | Val | Test |
|---|---|---|---|---|---|
| 0 | No DR | 1,805 | 1,263 | 271 | 271 |
| 1 | Mild | 370 | 259 | 55 | 56 |
| 2 | Moderate | 999 | 699 | 150 | 150 |
| 3 | Severe | 193 | 135 | 29 | 29 |
| 4 | Proliferative | 295 | 207 | 44 | 44 |
| | **Total** | **3,662** | **2,563** | **549** | **550** |

Grades 0–1 are **non-referable** (routine monitoring). Grades 2–4 are **referable** (specialist review).

---

## Method

```
Fundus image (224×224×3)
   └─ model-specific preprocess_input
       └─ Backbone (ImageNet weights): EfficientNetB0 | ResNet50 | DenseNet121
           └─ GlobalAvgPool → Dropout(0.3) → Dense(256, ReLU) → Dropout(0.3) → Dense(5, Softmax)
```

**Two-stage training (identical for all three backbones)**
1. **Warm-up:** backbone frozen, head trained (Adam, LR 1e-3, up to 8 epochs)
2. **Fine-tuning:** top backbone layers unfrozen with BatchNorm kept frozen (LR 1e-4, up to 15 epochs)

Both stages use class-weighted cross-entropy, early stopping (best weights restored) and `ReduceLROnPlateau`. Augmentation: flips, ±54° rotation, zoom, brightness and contrast.

**Referral decision:** P(referable) = P(Moderate) + P(Severe) + P(Proliferative). A patient is referred when this exceeds a threshold chosen on the validation set to reach 90% sensitivity.

**Ensemble:** average of the three models' softmax probabilities.

---

## How to run

1. Click **Open in Colab** above.
2. Select **Runtime → Change runtime type → T4 GPU**.
3. Select **Runtime → Run all**. It takes about 25–35 minutes.

Outputs are saved to `DR_Project_MultiModel/`:
```
DR_Project_MultiModel/
├── models/    # trained .keras models (one per backbone)
├── figures/   # learning curves, confusion matrices, ROC, Grad-CAM (300 dpi)
└── metrics/   # classification reports, model_comparison.csv, efficiency.csv, summary.json
```

Colab clears local files when the session ends. To keep the outputs, set `USE_DRIVE = True` in the configuration cell to save them to Google Drive, or zip and download the folder at the end.

---

## Notebook structure

| Section | Contents |
|---|---|
| 1. Setup | Imports, configuration, output folders |
| 2. Data | Download, stratified split, class weights, sample images |
| 3. Input pipeline | Cached `tf.data`, augmentation, per-model preprocessing |
| 4. Model factory | One function builds any of the three backbones |
| 5. Training | Two-stage training for each model |
| 6. Evaluation | Accuracy, Macro-F1, QWK, referable AUC / sensitivity / specificity |
| 7. Ensemble | Soft voting and model agreement |
| 8. Comparison | Tables, charts, confusion matrices, ROC, efficiency |
| 9. Grad-CAM | Side-by-side heatmaps for all models |
| 10. Screening demo | One image to triage decision |

---

## Limitations and future work

- Single dataset; external validation on EyePACS, Messidor-2 and the Indian IDRiD dataset is needed
- Few Severe and Proliferative images; 224×224 resolution hides the smallest lesions
- Select models on validation QWK instead of loss, and use softer class weights to protect the Moderate grade
- Higher resolution (380–512 px) and ordinal loss functions
- TensorFlow Lite deployment on a tablet for offline screening at PHCs

---

## Tech stack

Python · TensorFlow / Keras · scikit-learn · NumPy · Pandas · Matplotlib · Seaborn · Google Colab

---

## References

1. V. Gulshan et al., "Development and validation of a deep learning algorithm for detection of diabetic retinopathy in retinal fundus photographs," *JAMA*, 2016.
2. M. Tan and Q. V. Le, "EfficientNet: Rethinking model scaling for convolutional neural networks," *ICML*, 2019.
3. K. He et al., "Deep residual learning for image recognition," *CVPR*, 2016.
4. G. Huang et al., "Densely connected convolutional networks," *CVPR*, 2017.
5. R. R. Selvaraju et al., "Grad-CAM: Visual explanations from deep networks via gradient-based localization," *ICCV*, 2017.
6. APTOS 2019 Blindness Detection, Kaggle, 2019.

> **Disclaimer:** This is an academic project and not a medical device. 
