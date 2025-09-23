# Maize Yield Prediction Model 

This folder contains the final implementation of a machine learning model for predicting maize yield using demographic and environmental factors.  
The project was developed in Google Colab and saved here as a Jupyter Notebook.

---

## 📂 Contents
- `Maize_Yield_Predicting_Model.ipynb` – main Colab notebook containing the full pipeline
- `tests/` – folder for test scripts
- `access.py`, `address.py`, `adress.py`, `assess.py`, `config.py`, `defaults.yml` – supporting files for the project

---

## ⚙️ Workflow Overview
The notebook follows these main steps:
1. **Data Loading & Exploration** – load maize production datasets and inspect distributions.
2. **Data Cleaning** – handle missing values, fix column names, and align county-level data.
3. **Feature Engineering** – construct input features from demographic/environmental datasets.
4. **Train/Test Split** – split the dataset into training and validation subsets.
5. **Model Training** – build machine learning models for yield prediction.
6. **Model Evaluation** – use metrics such as:
   - R² (Coefficient of Determination)
   - Mean Squared Error (MSE)
   - Mean Absolute Error (MAE)
7. **Cross Validation** – validate stability of the model across multiple folds.
8. **Model Interpretation** – apply SHAP values to understand feature importance.
9. **Model Saving** – export trained model using `joblib` for reuse.

---

## ▶️ How to Run the Notebook

You can run the notebook in two main ways:

### 🔹 Option 1: Google Colab (recommended)
1. Go to [Google Colab](https://colab.research.google.com/).  
2. Click **File → Upload notebook**.  
3. Select `Maize_Yield_Predicting_Model.ipynb` from this folder.  
4. Run each cell using **Shift + Enter**.  

Most required libraries are pre-installed in Colab.

### 🔹 Option 2: Run Locally with Jupyter
1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/fynesse_mlfc.git
   cd fynesse_mlfc/fynesse
