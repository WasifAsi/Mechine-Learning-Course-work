# Bank Marketing Campaign Prediction

This repository contains the coursework project for predicting client subscription to a term deposit based on the **Bank Marketing Dataset** from the UCI Machine Learning Repository. The project involves implementing and comparing two machine learning models: **Random Forest Classifier** and **Neural Networks**.

---

## Project Overview

### Objective:
To build and evaluate machine learning models to predict whether a client will subscribe to a term deposit based on the features of a bank marketing campaign.

### Key Features:
1. **Binary Classification Problem**: Target variable `y` (yes/no).
2. **Dataset**: Bank marketing dataset with 17 features.
3. **Models Used**: 
   - Random Forest Classifier
   - Neural Networks
4. **Evaluation**: Comparison of models using key metrics like accuracy, precision, recall, F1-score, and ROC-AUC.

---

## Getting Started

### Prerequisites
- Python 3.8 or above
- Libraries: 
  - pandas
  - numpy
  - scikit-learn
  - matplotlib
  - seaborn
  - tensorflow/keras

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/<your-username>/bank-marketing-prediction.git
   cd bank-marketing-prediction
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. (Optional) Create a virtual environment:
   ```bash
   python -m venv env
   source env/bin/activate  # For Windows: .\env\Scripts\activate
   ```

---

## Workflow

### 1. Data Preparation
- Data cleaning: Handling missing and duplicate values.
- Feature engineering: Handling outliers, transforming features, and encoding categorical variables.
- Feature selection: Using Random Forest importance scores and correlation analysis.
- Scaling: Normalizing numerical features.
- Dimensionality reduction: Applying PCA.

### 2. Model Training
#### Random Forest Classifier:
- `n_estimators=100, max_depth=15, min_samples_split=10, min_samples_leaf=2, class_weight='balanced'`

#### Neural Network:
- Architecture with hidden layers, ReLU activation, dropout, and Adam optimizer.

### 3. Evaluation Metrics:
- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC

---

## Results
- **Random Forest Classifier**: Achieved accuracy of ~X%, with precision and recall tailored for imbalanced data.
- **Neural Network**: Achieved accuracy of ~Y%, with fine-tuned hyperparameters.
- ROC-AUC curves indicate model performance under imbalanced conditions.

---

## Report
Detailed documentation is available in the `CW report.pdf`. It includes:
- Introduction
- Data Preprocessing
- Model Implementation
- Experimental Results
- Limitations and Future Enhancements

---

## Acknowledgements
- Dataset: UCI Machine Learning Repository
- Libraries: scikit-learn, TensorFlow, Matplotlib, Seaborn

---

## License
This project is licensed under the MIT License - see the LICENSE file for details.
