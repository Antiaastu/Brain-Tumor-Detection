# Advance_Brain_Tumor_Classification

## Overview

This project is a **machine learning application** that detects the presence of brain tumors in MRI scans using **deep learning techniques**. The model achieves **85% accuracy** on test data and helps in early detection for medical diagnostics.

## Tools and Technologies:
* Python
* Jupyter Notebooks
* TensorFlow
* Keras
* Opencv
* Scikit-Learn
* Pandas
* Numpy
* Matplotlib

## Features

* Detects whether a brain tumor exists in an MRI scan.
* Preprocesses MRI images with **normalization, augmentation, and resizing** to improve model performance.
* Trained using **TensorFlow/Keras** for high accuracy and generalization.
* Evaluated with **cross-validation, confusion matrices, and accuracy metrics**.

## Tech Stack

* **Programming Language:** Python
* **Frameworks/Libraries:** TensorFlow, Keras, NumPy, Pandas, OpenCV
* **Tools:** Jupyter Notebook, Matplotlib, Seaborn

## Dataset

* MRI brain scan images (provide dataset source or link if public).
* Preprocessing steps include **resizing, normalization, and augmentation**.

## Installation

1. Clone the repository:

   ```bash
   git clone <repository-url>
   ```
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Load the trained model:

   ```python
   from tensorflow.keras.models import load_model
   model = load_model("brain_tumor_model.h5")
   ```
2. Preprocess MRI images and predict tumor presence:

   ```python
   result = model.predict(preprocessed_image)
   ```

## Evaluation

* **Accuracy:** 85%
* Evaluation metrics include **confusion matrix**, **precision**, **recall**, and **F1-score**.

## Future Work

* Improve model accuracy with **larger datasets** and **transfer learning**.
* Extend to classify **types of brain tumors**.
* Deploy as a **web application** for easier access by medical professionals.


---
