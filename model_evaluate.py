import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

class EligibilityPredictor:
    
    def __init__(self, model_path='./clinical_trial_model'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
    
    def predict(self, text):
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=512,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
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
    print(" control log1")
    def predict_batch(self, texts):
        results = []
        for text in texts:
            results.append(self.predict(text))
        return results

def evaluate_test_set():
    test_df = pd.read_csv('test_data.csv')
    
    print("model loading ")
    
    predictor = EligibilityPredictor()
    
    predictions = []
    probabilities = []
    
    for text in test_df['combined_text']:
        result = predictor.predict(text)
        predictions.append(1 if result['eligible'] else 0)
        probabilities.append(result['probability_eligible'])
    
    # Evaluation metrics
    print("\n" + "="*60)
    print("CLASSIFICATION REPORT")
    print("="*60)
    print(classification_report(
        test_df['eligible'], 
        predictions,
        target_names=['Ineligible', 'Eligible']
    ))
    
    print("\n" + "="*60)
    print("CONFUSION MATRIX")
    print("="*60)
    cm = confusion_matrix(test_df['eligible'], predictions)
    print(f"                 Predicted")
    print(f"                 Inelig  Elig")
    print(f"Actual Inelig    {cm[0][0]:5d}  {cm[0][1]:5d}")
    print(f"       Elig      {cm[1][0]:5d}  {cm[1][1]:5d}")
    
    print("\n" + "="*60)
    print("SAMPLE PREDICTIONS")
    print("="*60)
    
    sample_indices = np.random.choice(len(test_df), 5, replace=False)
    for idx in sample_indices:
        row = test_df.iloc[idx]
        pred = predictions[idx]
        prob = probabilities[idx]
        
        print(f"\nPatient: {row['patient_id']}")
        print(f"Age: {row['age']}, Cancer: {row['cancer_type']} {row['stage']}")
        print(f"ECOG: {row['ecog_score']}, Biomarker: {row['biomarker']}")
        print(f"Actual: {'Eligible' if row['eligible'] else 'Ineligible'}")
        print(f"Predicted: {'Eligible' if pred else 'Ineligible'} (confidence: {prob:.2%})")
        print(f"Match: {'MATCH' if row['eligible'] == pred else ' MISMATCH'}")
    
    return predictor, predictions, probabilities

if __name__ == "__main__":
    predictor, predictions, probabilities = evaluate_test_set()
    print("\n Completed evaluation")