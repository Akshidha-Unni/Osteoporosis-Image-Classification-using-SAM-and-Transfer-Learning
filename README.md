# 🦴 Osteoporosis Image Classification using SAM & Transfer Learning

## 📌 Overview
This project focuses on classifying knee X-ray images into “Normal” and “Osteoporosis” categories using state-of-the-art transfer learning models and the Segment Anything Model (SAM) for region-of-interest (ROI) extraction.
By combining DenseNet201, VGG19, and ResNet50 architectures with SAM-based segmentation, the study evaluates classification performance under three scenarios:

**1.Combined Dataset** – Large dataset from multiple sources.  

**2.Sampled Dataset** – Small dataset without segmentation.  

**3.ROI Dataset** – SAM-segmented images to enhance feature localization.  

The results highlight that ROI extraction significantly improves accuracy in small datasets, with DenseNet201 achieving up to **93% accuracy** on the combined dataset and VGG19 achieving **78% accuracy** on ROI-based datasets.

## 🧠 Motivation
Osteoporosis is a silent bone disease that increases fracture risk, especially in the elderly.
While** X-ray imaging** is a cost-effective diagnostic tool, its **manual interpretation is subjective and time-consuming.**
This project demonstrates how **Deep Learning -powered classification with segmentation** can support radiologists by improving diagnostic accuracy.

## 📂 Repository Structure
The repository contains two main Python scripts:  
- **Osteoporosis.py** – Full pipeline: data upload, preprocessing, transfer learning, prediction, and evaluation.
- **SAM.py** – Python script to perform segmentation to extract ROI from the images.

## 📊 Datasets
We used three publicly available datasets:
| Dataset Name                                            | Link                                                                                                    |
| ------------------------------------------------------- | ------------------------------------------------------------------------------------------------------- |
| Knee Osteoporosis Dataset – Mendeley                    | [🔗 Link](https://data.mendeley.com/datasets/fxjm8fb6mw/2)                                              |
| Osteoporosis Knee Dataset – Kaggle (Sachin Kumar)       | [🔗 Link](https://www.kaggle.com/datasets/sachinkumar413/osteoporosis-knee-dataset-preprocessed128x256) |
| Osteoporosis Knee X-ray Dataset – Kaggle (Steve Python) | [🔗 Link](https://www.kaggle.com/datasets/stevepython/osteoporosis-knee-xray-dataset)                   |

## ⚙️ Methodology
**1. Data Preprocessing**
- Resized images to 224x224
- RGB conversion
- Batch processing and normalization using tf.keras.applications.mobilenet_v2.preprocess_input

**2. Segmentation with SAM**
- Used Meta’s Segment Anything Model (SAM) to extract ROIs (femur and tibia regions).
- Generated binary masks, converted them to NumPy arrays, and extracted relevant regions.
- Created a balanced ROI dataset from ~60 original images (118 osteoporosis ROIs, 119 normal ROIs).

**3. Model Training**
- Fine-tuned DenseNet201, VGG19, and ResNet50 pre-trained on ImageNet.
- Frozen initial layers, unfreezed last 50 layers, and added custom fully-connected layers with dropout and L2 regularization.

**4. Evaluation Metrics**
- Accuracy, Precision, Recall, F1-score
- Confusion matrices for classification visualization.

## 🚀 Results Summary
| Dataset Type      | Best Model  | Accuracy | Epochs |
| ----------------- | ----------- | -------- | ------ |
| Combined Dataset  | DenseNet201 | **93%**  | 60     |
| Sampled Dataset   | DenseNet201 | 54%      | 10     |
| ROI Dataset (SAM) | VGG19       | **78%**  | 10     |  

**Key Findings:**
- Large datasets with transfer learning yield the best performance.
- SAM-based ROI extraction improves classification accuracy on small datasets.
- ROI datasets reduce overfitting and increase recall for osteoporosis detection.

## 💻 Installation & Usage
1️⃣ Clone the Repository
```bash
git clone https://github.com/Akshidha-Unni/Osteoporosis-Image-Classification-using-SAM-and-Transfer-Learning.git
cd Osteoporosis-Image-Classification-using-SAM-and-Transfer-Learning
```

2️⃣Run SAM.ipynb to extract ROI.  

3️⃣Run Osteoporosis.ipynb for image Classification.

## ⚠️ Limitations
- **Segmentation Time & Resources** – SAM segmentation is computationally expensive and time-consuming, especially for large datasets.
- **Inconsistent ROI Performance** – While ROI generally improved accuracy for small datasets, performance fluctuated in some runs.
- **Dataset Size** – Small dataset experiments still suffer from overfitting and lack of generalizability.
- **Manual ROI Validation** – Selecting optimal segmented regions sometimes required manual inspection.

## 🔮 Future Scope
- **Increase ROI Count** – Extract more ROIs per image to boost training data.
- **Automated ROI Filtering** – Develop methods to automatically select the most informative regions.
- **Data Augmentation** – Apply advanced augmentation (rotation, brightness, elastic transformations) to improve generalization.
- **Test Other Segmentation Models** – Compare SAM with U-Net, Mask R-CNN, and other medical segmentation architectures.

## 👩‍💻 Authors
- Akshidha Unni
- Diana Rogachova
- Adnan Alfarra
