import os
import numpy as np
import pandas as pd 
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME = 'bert-base-uncased'

class TextClassificationDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = list(texts)
        self.labels = list(labels)
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    def __len__(self):
        return len(self.texts)
        
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def prepare_data(df, tokenizer, batch_size=16, max_len=128):
    # Split: Train (80%), Temp (20%)
    train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42)
    # Split Temp: Val (50% of temp = 10% total), Test (50% of temp = 10% total)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)
    
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    
    train_dataset = TextClassificationDataset(train_df['text'], train_df['label'], tokenizer, max_len)
    val_dataset = TextClassificationDataset(val_df['text'], val_df['label'], tokenizer, max_len)
    test_dataset = TextClassificationDataset(test_df['text'], test_df['label'], tokenizer, max_len)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader

def train_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    
    for batch in dataloader:
        optimizer.zero_grad()
        
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        
        loss = outputs.loss
        total_loss += loss.item()
        
        loss.backward()
        optimizer.step()
        
    return total_loss / len(dataloader)

def evaluate(model, dataloader, device):
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    model.eval()
    predictions = []
    actual_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            
            preds = torch.argmax(logits, dim=1)
            
            predictions.extend(preds.cpu().numpy())
            actual_labels.extend(labels.cpu().numpy())
            
    acc = accuracy_score(actual_labels, predictions)
    report = classification_report(actual_labels, predictions, zero_division=0)
    matrix = confusion_matrix(actual_labels, predictions)
    
    return acc, report, matrix

if __name__ == "__main__":
    # Mock data for sentiment classification (1 = Positive, 0 = Negative)
    mock_data = {
        "text": [
            "I love this product, it works perfectly!",
            "This is the worst experience of my life.",
            "Absolutely fantastic results, highly recommend.",
            "Terrible service, it broke on the first day.",
            "Great value for money and fast shipping.",
            "Complete waste of time, do not buy.",
            "Very satisfied with the quality of this item.",
            "Horrible product, it does not work at all.",
            "Amazing customer support, very friendly.",
            "Worst purchase ever made, extremely disappointed.",
            "Super happy with the purchase!",
            "Broken and useless, going to return it.",
            "Excellent build quality and design.",
            "Awful smell and poor durability.",
            "Highly recommended, would buy again.",
            "Very bad experience, avoid this seller.",
            "This is a great tool for daily work.",
            "Really poor performance and battery life.",
            "Wonderful customer service and product.",
            "Defective item, received it damaged."
        ],
        "label": [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
    }
    
    print(f"Using device: {DEVICE}")
    print("Initializing tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2).to(DEVICE)
    
    df = pd.DataFrame(mock_data)
    train_loader, val_loader, test_loader = prepare_data(df, tokenizer, batch_size=4)
    
    optimizer = AdamW(model.parameters(), lr=5e-5)
    
    print("\n--- Starting Training ---")
    epochs = 3
    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, optimizer, DEVICE)
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f}")
        
    print("\n--- Evaluating on Validation Set ---")
    val_acc, val_report, val_matrix = evaluate(model, val_loader, DEVICE)
    print(f"Validation Accuracy: {val_acc:.4f}")
    print("\nClassification Report:\n", val_report)
    print("Confusion Matrix:\n", val_matrix)