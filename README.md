# 🩺 Hybrid Computer Vision Pipeline for Breast Oncology  

**Detection, Segmentation, and Classification of Breast Cancer Tumors**

📚 **Final Project**  
**TI0176 – Introduction to Digital Image Processing**  
Course: **Computer Engineering – Federal University of Ceará (UFC)**

---

## 📌 Overview

This project presents the development of an **automated clinical decision support pipeline** for breast imaging analysis, integrating **lesion segmentation** and **pathological classification** (*benign vs. malignant*).

The proposed approach is **modular**, enabling independent evaluation of segmentation and classification architectures, as well as their integration into a unified inference workflow, with emphasis on **high diagnostic sensitivity**, a critical requirement in oncological applications.

---

## 🎯 Main Objective

To develop and validate a **deep learning–based hybrid pipeline** capable of:

- Accurately segmenting breast lesions in medical images;
- Automatically classifying segmented regions as **benign or malignant**;
- Supporting the clinical diagnostic process by minimizing false negatives.

---

## 🎯 Specific Objectives

- **Compare Segmentation Architectures**  
  Evaluate different segmentation models in terms of spatial accuracy and ability to delineate irregular lesions.

- **Optimize Feature Extraction**  
  Identify which classification model best interprets morphological and textural patterns of segmented tumors.

- **Integrate Pipeline Modules**  
  Build a data flow where the output of the best segmentation model feeds the input of the best classification model.

- **Assess Clinical Impact**  
  Prioritize metrics such as **sensitivity (recall)** to reduce false negatives, which are critical in cancer diagnosis.

---

## 🧠 Pipeline Architecture

Breast Image
      ↓
[ Lesion Segmentation ]
      ↓
ROI (Region of Interest)
      ↓
[ Pathological Classification ]
      ↓
Benign | Malignant

The pipeline is designed to be **flexible**, allowing model substitution at each stage without disrupting the overall workflow.

---

## 🧩 Evaluated Models

### 🔍 Segmentation

* **U-Net (Baseline)**
  Classical encoder–decoder architecture widely used in medical imaging tasks.

* **Attention U-Net**
  Extension of U-Net that incorporates attention mechanisms to focus on relevant regions and suppress background noise.

---

### 🧪 Classification

* **EfficientNetV2-S**
  An optimized convolutional neural network with high performance and parameter efficiency.

* **VGG16**
  A deep, sequential architecture effective for extracting basic patterns and textures.

* **ResNet50**
  Introduces residual connections (*skip connections*), facilitating training of deep networks and improving convergence stability.

* **Vision Transformer (ViT)**
  A self-attention–based approach capable of modeling global relationships between image regions, suitable for complex malignancy patterns.

---

## 📊 Evaluation Metrics

* **Segmentation**

  * Dice Coefficient
  * Intersection over Union (IoU)

* **Classification**

  * Accuracy
  * Sensitivity (Recall) ⭐
  * Specificity
  * F1-score
  * Confusion Matrix

> ⚠️ **Sensitivity** is treated as a priority metric due to the critical importance of minimizing false negatives in breast cancer diagnosis.

---

## 🛠️ Technologies Used

* Python
* PyTorch / TensorFlow
* OpenCV
* NumPy / Pandas
* Matplotlib / Seaborn
* Scikit-learn

---

## 📁 Project Structure

> ⚠️ Work in progress

---

## 👥 Team

This project was developed by **four Computer Engineering students**:

### 👤 Gabriela Bezerra Pereira

* Student ID: 554663
* E-mail: [gabrielapereira@alu.ufc.br](mailto:gabrielapereira@alu.ufc.br)
* GitHub: [@gabriwrld](https://github.com/gabriwrld)

### 👤 Kauã Ribeiro de Sousa

* Student ID: 548213
* E-mail: [kauarb.2@gmail.com](mailto:kauarb.2@gmail.com)
* GitHub: [@Kaua-Rbs](https://github.com/Kaua-Rbs)

### 👤 Larissa Vitória Santos Menezes

* Student ID: 553875
* E-mail: [larissa.vitoria@alu.ufc.br](mailto:larissa.vitoria@alu.ufc.br)
* GitHub: [@larisvitoria](https://github.com/larisvitoria)

### 👤 Maria Cecília Alves Castro

* Student ID: 553577
* E-mail: [cecialves@alu.ufc.br](mailto:cecialves@alu.ufc.br)
* GitHub: [@CeciAlves](https://github.com/CeciAlves)

---

## 📌 Notes

This project is intended **for academic and research purposes only**, developed as the **final project for the course TI0176 – Introduction to Digital Image Processing**, and must not be used for clinical diagnosis without proper regulatory validation.

---