"""
LLM-Enhanced Reasoning for Clinical Trial Eligibility. This script demonstrates how to add interpretable reasoning to predictions.

For a real trial implementation -> an actual LLM API lik OpenAI(chatgbt's), Anthropic (claude's) etc
this one simulates similar logic with rule-based reasoning.
"""

class ClinicalReasoningEngine:
    """script here provides interpretable reasoning for eligibility decisions"""
    
    def __init__(self):
        self.criteria = {
            'age': (18, 75),
            'ecog_max': 2,
            'hemoglobin_min': 9.0,
            'creatinine_max': 2.0,
            'neutrophil_min': 1.5,
            'excluded_stages': ['IV']
        }
    
    def analyze_patient(self, patient_data):
        
        reasons_eligible = []
        reasons_ineligible = []
        
        age = patient_data.get('age', 0)
        if self.criteria['age'][0] <= age <= self.criteria['age'][1]:
            reasons_eligible.append(
                f"Age {age} is within range, ({self.criteria['age'][0]}-{self.criteria['age'][1]} years)"
            )
        else:
            reasons_ineligible.append(
                f"Age {age} is outside range, ({self.criteria['age'][0]}-{self.criteria['age'][1]} years)"
            )
        
        # Stage check
        stage = patient_data.get('stage', '')
        if stage not in self.criteria['excluded_stages']:
            reasons_eligible.append(f"Stage {stage} is acceptable (Stage IV excluded)")
        else:
            reasons_ineligible.append(f"Stage {stage} patients are excluded from this trial")
        
        # ECOG check
        ecog = patient_data.get('ecog_score', 5)
        if ecog <= self.criteria['ecog_max']:
            reasons_eligible.append(
                f"ECOG score {ecog} indicates adequate performance, (≤{self.criteria['ecog_max']} required)"
            )
        else:
            reasons_ineligible.append(
                f"ECOG score {ecog} is too high, (≤{self.criteria['ecog_max']} required)"
            )
        
        # Lab vals
        hgb = patient_data.get('hemoglobin', 0)
        if hgb >= self.criteria['hemoglobin_min']:
            reasons_eligible.append(
                f"Hemoglobin {hgb} g/dL is adequate, (≥{self.criteria['hemoglobin_min']} required)"
            )
        else:
            reasons_ineligible.append(
                f"Hemoglobin {hgb} g/dL is too low, (≥{self.criteria['hemoglobin_min']} required)"
            )
        
        cr = patient_data.get('creatinine', 10)
        if cr <= self.criteria['creatinine_max']:
            reasons_eligible.append(
                f"Creatinine {cr} mg/dL is within limits (≤{self.criteria['creatinine_max']} required)"
            )
        else:
            reasons_ineligible.append(
                f"Creatinine {cr} mg/dL is elevated, (≤{self.criteria['creatinine_max']} required)"
            )
        
        neut = patient_data.get('neutrophil_count', 0)
        if neut >= self.criteria['neutrophil_min']:
            reasons_eligible.append(
                f"Neutrophil count {neut} K/μL is adequate, (≥{self.criteria['neutrophil_min']} required)"
            )
        else:
            reasons_ineligible.append(
                f"Neutrophil count {neut} K/μL is too low (≥{self.criteria['neutrophil_min']} required)"
            )
        
        # add more criteria as needed / wanted
        
        # Generating the reasoning text
        final_decision = len(reasons_ineligible) == 0
        
        reasoning = self._format_reasoning(
            patient_data, 
            reasons_eligible, 
            reasons_ineligible, 
            final_decision
        )
        
        return {
            'eligible': final_decision,
            'reasons_eligible': reasons_eligible,
            'reasons_ineligible': reasons_ineligible,
            'reasoning_text': reasoning,
            'num_criteria_met': len(reasons_eligible),
            'num_criteria_failed': len(reasons_ineligible)
        }
    
    def _format_reasoning(self, patient_data, eligible, ineligible, decision):
        """Formatting in a more readable way"""
        
        reasoning = f"""
CLINICAL TRIAL ELIGIBILITY ANALYSIS
=====================================

Patient Profile:
- ID: {patient_data.get('patient_id', 'N/A')}
- Age: {patient_data.get('age')} years
- Cancer Type: {patient_data.get('cancer_type')} Stage {patient_data.get('stage')}
- Biomarker: {patient_data.get('biomarker')}
- ECOG Score: {patient_data.get('ecog_score')}

Laboratory Values:
- Hemoglobin: {patient_data.get('hemoglobin')} g/dL
- Creatinine: {patient_data.get('creatinine')} mg/dL
- Neutrophils: {patient_data.get('neutrophil_count')} K/μL
- Platelets: {patient_data.get('platelet_count')} K/μL

Eligibility Assessment:
"""
        
        if eligible:
            reasoning += "\n CRITERIA MET:\n"
            for reason in eligible:
                reasoning += f"  • {reason}\n"
        
        if ineligible:
            reasoning += "\n EXCLUSION CRITERIA:\n"
            for reason in ineligible:
                reasoning += f"  • {reason}\n"
        
        reasoning += f"\n{'='*50}\n"
        reasoning += f"FINAL DECISION: {'ELIGIBLE' if decision else 'INELIGIBLE'}\n"
        reasoning += f"{'='*50}\n"
        
        return reasoning

# Demo usage
if __name__ == "__main__":
    import pandas as pd
    
    test_df = pd.read_csv('test_data.csv')
    
    engine = ClinicalReasoningEngine()
    
    # few example analyses
    print("LLM-ENHANCED CLINICAL REASONING DEMO")
    print("=" * 70)
    
    for i in range(3):
        patient = test_df.iloc[i].to_dict()
        analysis = engine.analyze_patient(patient)
        
        print(analysis['reasoning_text'])
        print()
    
    print("\n Demo working. Process completed.")