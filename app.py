from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, List
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from llm_reasonings import ClinicalReasoningEngine
import uvicorn
from contextlib import asynccontextmanager



class ModelLoader:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = None
        self.model = None
        self.reasoning_engine = None
    
    def load(self):
        print("...Loading model....")
        self.tokenizer = AutoTokenizer.from_pretrained('./clinical_trial_model')
        self.model = AutoModelForSequenceClassification.from_pretrained('./clinical_trial_model')
        self.model.to(self.device)
        self.model.eval()
        self.reasoning_engine = ClinicalReasoningEngine()
        print(f"Model loaded successfully on {self.device}")

model_loader = ModelLoader()

#(deprecated)
# @app.on_event("startup")
# async def startup_event():
#     model_loader.load()

@asynccontextmanager
async def lifespan(app: FastAPI):
    #startup
    model_loader.load()
    #app runs
    yield

# FastAPI app
app = FastAPI(
    title="Clinical Trial Eligibility API",
    description="AI-powered clinical trial eligibility screening system",
    version="1.0.0",
    lifespan=lifespan
)

# Request/Response models
class PatientData(BaseModel):
    patient_id: str = Field(..., example="PT0001")
    age: int = Field(..., ge=0, le=120, example=65)
    cancer_type: str = Field(..., example="Breast")
    stage: str = Field(..., example="II")
    biomarker: str = Field(..., example="HER2+")
    ecog_score: int = Field(..., ge=0, le=4, example=1)
    hemoglobin: float = Field(..., ge=0, example=12.5)
    creatinine: float = Field(..., ge=0, example=1.1)
    neutrophil_count: float = Field(..., ge=0, example=3.5)
    platelet_count: float = Field(..., ge=0, example=250.0)
    clinical_notes: Optional[str] = Field(None, example=" Stage II breast cancer")

class EligibilityResponse(BaseModel):
    patient_id: str
    eligible: bool
    confidence: float
    probability_eligible: float
    probability_ineligible: float
    reasoning: Optional[dict] = None

class BatchRequest(BaseModel):
    patients: List[PatientData]

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str

def create_combined_text(patient: PatientData) -> str:
    return (
        f"Age: {patient.age} years. Cancer: {patient.cancer_type} Stage {patient.stage}. "
        f"Biomarker: {patient.biomarker}. ECOG: {patient.ecog_score}. "
        f"Labs: Hgb {patient.hemoglobin}, Cr {patient.creatinine}, "
        f"Neut {patient.neutrophil_count}, Plt {patient.platelet_count}. "
        f"Notes: {patient.clinical_notes or 'No additional notes.'}"
    )

def predict_eligibility(text: str) -> dict:
    encoding = model_loader.tokenizer(
        text,
        add_special_tokens=True,
        max_length=512,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )
    
    input_ids = encoding['input_ids'].to(model_loader.device)
    attention_mask = encoding['attention_mask'].to(model_loader.device)
    
    with torch.no_grad():
        outputs = model_loader.model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1)
        prediction = torch.argmax(probs, dim=-1).item()
        confidence = probs[0][prediction].item()
    
    return {
        'eligible': bool(prediction),
        'confidence': confidence,
        'probability_eligible': probs[0][1].item(),
        'probability_ineligible': probs[0][0].item()
    }

# API Endpoints
@app.get("/", response_model=HealthResponse)
async def root():
    return {
        "status": "running",
        "model_loaded": model_loader.model is not None,
        "device": str(model_loader.device)
    }

@app.post("/predict", response_model=EligibilityResponse)
async def predict(patient: PatientData):
    """
    Predicting eligibility for: single patient
    """
    try:
        text = create_combined_text(patient)
        prediction = predict_eligibility(text)

        reasoning = model_loader.reasoning_engine.analyze_patient(patient.dict())
        
        return EligibilityResponse(
            patient_id=patient.patient_id,
            eligible=prediction['eligible'],
            confidence=prediction['confidence'],
            probability_eligible=prediction['probability_eligible'],
            probability_ineligible=prediction['probability_ineligible'],
            reasoning=reasoning
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/batch")
async def predict_batch(request: BatchRequest):
    """
    Predicting eligibility for: multiple patients
    """
    try:
        results = []
        
        for patient in request.patients:
            text = create_combined_text(patient)
            prediction = predict_eligibility(text)
            reasoning = model_loader.reasoning_engine.analyze_patient(patient.model_dump())
            
            results.append({
                "patient_id": patient.patient_id,
                "eligible": prediction['eligible'],
                "confidence": prediction['confidence'],
                "probability_eligible": prediction['probability_eligible'],
                "reasoning": reasoning
            })
        
        return {"results": results, "total_patients": len(results)}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": model_loader.model is not None,
        "device": str(model_loader.device),
        "model_type": "DistilBERT",
        "version": "1.0.0"
    }

if __name__ == "__main__":
    print("Starting API")
    uvicorn.run(app, host="0.0.0.0", port=8000)