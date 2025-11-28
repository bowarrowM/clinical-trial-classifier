# Clinical Trial Eligibility Classifier

**An end-to-end ML/NLP system for predicting clinical trial eligibility using fine-tuned transformers and LLM-enhanced reasoning.**

**NOTE: THIS PROJECT WILL BE UPDATED TO A FULLSTACK PROJECT**

##  Project Overview

This project demonstrates a production-ready pipeline for automating clinical trial patient screening. It combines:

- **Fine-tuned transformer models** (DistilBERT) for binary classification
- **Structured + unstructured data** (demographics, labs, clinical notes)
- **LLM-enhanced reasoning** for interpretable decisions
- **REST API** for real-time inference
- **Synthetic but realistic** oncology dataset

##  Dataset

- **500 synthetic patient records** with:
  - Demographics (age, cancer type, stage)
  - Lab values (hemoglobin, creatinine, neutrophils, platelets)
  - Performance status (ECOG score)
  - Biomarkers (HER2, ER, PD-L1, EGFR)
  - Clinical notes (unstructured text)
  
- **Eligibility criteria** based on real oncology trials:
  - Age: 18-75 years
  - Stage: I-III (Stage IV excluded)
  - ECOG: 0-2
  - Lab thresholds for safety

##  Architecture

```
Patient Data → Feature Engineering → Transformer Model → Prediction
                                           ↓
                                    LLM Reasoning → Interpretable Output
```


### 1. Installation

```bash
# Clone or create project directory
mkdir clinical-trial-classifier
cd clinical-trial-classifier

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Generate Data

```bash
python generate_synthetic_data.py
```

Output: `clinical_trial_data.csv` (500 patients)

### 3. Preprocess Data

```bash
python data_preprocess.py
```

Output: `train_data.csv`, `val_data.csv`, `test_data.csv`

### 4. Train Model

```bash
python train_model.py
```

Training takes ~5-10 minutes on CPU, ~2 minutes on GPU.

Output: `./clinical_trial_model/` directory

### 5. Evaluate Model

```bash
python model_evaluate.py
```

Shows classification report, confusion matrix, and sample predictions.

### 6. Test LLM Reasoning

```bash
python llm_reasonings.py
```

Demonstrates interpretable eligibility explanations.

### 7. Start API Server

```bash
python app.py
```

Server runs on `http://localhost:8000`

### 8. Test API

In a new terminal:

```bash
python api_testing.py
```

##  API Endpoints

### Health Check
```bash
GET /health
```

### Single Patient Prediction
```bash
POST /predict
Content-Type: application/json

{
  "patient_id": "PT001",
  "age": 55,
  "cancer_type": "Breast",
  "stage": "II",
  "biomarker": "HER2+",
  "ecog_score": 1,
  "hemoglobin": 12.5,
  "creatinine": 1.1,
  "neutrophil_count": 3.8,
  "platelet_count": 245.0,
  "clinical_notes": "Patient with stage II breast cancer..."
}
```

### Batch Prediction
```bash
POST /predict/batch
Content-Type: application/json

{
  "patients": [...]
}
```

## Model Performance

Expected metrics on test set:

- **Accuracy**: ~85-90%
- **F1 Score**: ~0.85-0.90
- **AUC-ROC**: ~0.90-0.95

*Note: Performance varies based on synthetic data generation seed*

## Key Features

### 1. **Transformer Fine-tuning**
- Uses DistilBERT (lightweight, fast)
- Custom tokenization of clinical text
- Binary classification with softmax probabilities

### 2. **Multi-modal Input**
- Combines structured (age, labs) and unstructured (notes) data
- Feature engineering creates rich input representation

### 3. **LLM Reasoning**
- Rule-based reasoning engine (can be replaced with GPT/Claude)
- Provides criterion-by-criterion analysis
- Explains why patients are eligible/ineligible

### 4. **Production-Ready API**
- FastAPI with automatic documentation
- Batch processing support
- Error handling and validation
- Health check endpoints

## Project Structure

```
root/
├── synthetic_data.py            # Dataset creation
├── data_preprocess.py           # Feature engineering
├── model.py                     # Model training
├── model_evaluate.py            # Evaluation & inference
├── llm_reasonings.py            # Interpretable reasoning
├── app.py                       # FastAPI server
├── api_testing.py               # API testing
├── requirements.txt             # Dependencies
├── clinical_trial_data.csv      # Generated dataset
├── train_data.csv               # Training split / will be created after running scripts
├── val_data.csv                 # Validation split / will be created after running scripts
├── test_data.csv                # Test split /  will be created after running scripts
├── label_encoders.json          # Categorical encoders / will be created after running scripts
└── clinical_trial_model/        # Trained model /will be created after running scripts
```

## Potential Extensions

1. **Real LLM Integration**: Replace rule-based reasoning with GPT-4/Claude API
2. **Multi-trial Matching**: Extend to match patients with multiple trials
3. **Document Processing**: Add PDF parsing for actual trial protocols
4. **Active Learning**: Flag uncertain cases for human review
5. **Explainability**: Add SHAP/LIME for model interpretability
6. **Deployment**: Containerize with Docker, deploy to cloud


## Acknowledgments

- Inspired by real clinical trial matching systems
- Uses synthetic data to protect patient privacy
- Built with open-source tools (Hugging Face, FastAPI, scikit-learn, pytorch)
---
**Note**: This is a demonstration project with synthetic data. Real clinical trial matching requires IRB approval, HIPAA compliance, and integration with EHR systems.
