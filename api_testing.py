import requests
import json

# API base URL
BASE_URL = "http://localhost:8000"

def test_health_check():
    """healthcheck endpoint"""
    print("Testing health check...")
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print()

def test_single_prediction():
    """ single patient """
    print("Testing single patient prediction...")
    
    patient_data = {
        "patient_id": "TEST001",
        "age": 55,
        "cancer_type": "Breast",
        "stage": "II",
        "biomarker": "HER2+",
        "ecog_score": 1,
        "hemoglobin": 12.5,
        "creatinine": 1.1,
        "neutrophil_count": 3.8,
        "platelet_count": 245.0,
        "clinical_notes": "Patient with stage II breast cancer, HER2 positive, good performance status"
    }
    
    response = requests.post(f"{BASE_URL}/predict", json=patient_data)
    print(f"Status: {response.status_code}")
    result = response.json()
    
    print(f"\nPatient: {result['patient_id']}")
    print(f"Eligible: {result['eligible']}")
    print(f"Confidence: {result['confidence']:.2%}")
    print(f"P(Eligible): {result['probability_eligible']:.2%}")
    print(f"\nReasoning Summary:")
    print(f"  Criteria Met: {result['reasoning']['num_criteria_met']}")
    print(f"  Criteria Failed: {result['reasoning']['num_criteria_failed']}")
    print(f"\nDetailed Reasoning:")
    print(result['reasoning']['reasoning_text'])
    print()

def test_batch_prediction():
    print("Testing batch prediction...")
    
    batch_data = {
        "patients": [
            {
                "patient_id": "BATCH001",
                "age": 45,
                "cancer_type": "Lung",
                "stage": "III",
                "biomarker": "EGFR+",
                "ecog_score": 1,
                "hemoglobin": 11.5,
                "creatinine": 1.0,
                "neutrophil_count": 4.2,
                "platelet_count": 280.0,
                "clinical_notes": "Stage III lung cancer, EGFR positive"
            },
            {
                "patient_id": "BATCH002",
                "age": 82,
                "cancer_type": "Colon",
                "stage": "IV",
                "biomarker": "ER-",
                "ecog_score": 3,
                "hemoglobin": 8.5,
                "creatinine": 2.3,
                "neutrophil_count": 1.1,
                "platelet_count": 180.0,
                "clinical_notes": "Stage IV colon cancer with poor performance status"
            }
        ]
    }
    
    response = requests.post(f"{BASE_URL}/predict/batch", json=batch_data)
    print(f"Status: {response.status_code}")
    result = response.json()
    
    print(f"\nTotal patients processed: {result['total_patients']}")
    print("\nResults:")
    for patient_result in result['results']:
        print(f"\n  Patient: {patient_result['patient_id']}")
        print(f"  Eligible: {'Yes' if patient_result['eligible'] else 'No'}")
        print(f"  Confidence: {patient_result['confidence']:.2%}")
    print()

def test_ineligible_patient():
    print("Testing ineligible patient case scenario")
    
    #dummy patient data
    patient_data = {
        "patient_id": "INELIG001",
        "age": 85,  # Too old
        "cancer_type": "Melanoma",
        "stage": "IV",  # Advanced stage
        "biomarker": "PD-L1-",
        "ecog_score": 4,  # Poor performance status
        "hemoglobin": 7.5,  # Low
        "creatinine": 3.2,  # High
        "neutrophil_count": 0.8,  # Low
        "platelet_count": 120.0,
        "clinical_notes": "Stage IV melanoma with multiple comorbidities"
    }
    
    response = requests.post(f"{BASE_URL}/predict", json=patient_data)
    result = response.json()
    
    print(f"Patient: {result['patient_id']}")
    print(f"Eligible: {result['eligible']}")
    print(f"Confidence: {result['confidence']:.2%}")
    print(f"\nExclusion reasons:")
    for reason in result['reasoning']['reasons_ineligible']:
        print(f"  • {reason}")
    print()

if __name__ == "__main__":
    print("="*70)
    print("CLINICAL TRIAL ELIGIBILITY API -- TEST")
    print("="*70)
    
    import time
    time.sleep(3)
    
    try:
        test_health_check()
        test_single_prediction()
        test_batch_prediction()
        test_ineligible_patient()
        
        print("="*70)
        print("ALL TESTS COMPLETED SUCCESSFULLY")
        print("="*70)
        
    except requests.exceptions.ConnectionError:
        print("ERROR: Could not connect to API. Make sure it's running on port 8000")
    except Exception as e:
        print(f"ERROR: {e}")