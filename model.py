import pandas as pd
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, EarlyStoppingCallback
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score

class ClinicalTrialDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    probs = torch.softmax(torch.tensor(pred.predictions), dim=-1)[:, 1].numpy()
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average='binary'
    )
    acc = accuracy_score(labels, preds)
    auc = roc_auc_score(labels, probs)
    
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'auc': auc
    }

def train_model():
    """Finetuning a transformerfor eligibility classification"""
    
    train_df = pd.read_csv('train_data.csv')
    val_df = pd.read_csv('val_data.csv')
    
    # clinical BERT model or distilbert for speed
    model_name = 'distilbert-base-uncased'
    print(f"Loading model {model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2
    )
    
    # Create datasets
    train_dataset = ClinicalTrialDataset(
        train_df['combined_text'].values,
        train_df['eligible'].values,
        tokenizer
    )
    
    val_dataset = ClinicalTrialDataset(
        val_df['combined_text'].values,
        val_df['eligible'].values,
        tokenizer
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir='./model_output',
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_steps=100,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model='f1',
        greater_is_better=True,
        report_to='none'
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )
    
    # Train
    print("\n Training start")
    trainer.train()
    
    # Save model and tokenizer
    model.save_pretrained('./clinical_trial_model')
    tokenizer.save_pretrained('./clinical_trial_model')
    print("\n model saved")

    
    # Evaluate on validation set
    print("\n Evaluation on val set:")
    results = trainer.evaluate()
    for key, value in results.items():
        print(f"  {key}: {value:.4f}")
    
    return model, tokenizer, results

if __name__ == "__main__":
    model, tokenizer, results = train_model()
    print("Model saved to './clinical_trial_model'")