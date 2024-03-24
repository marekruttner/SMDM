import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import time

torch.cuda.empty_cache()

class NumericalPredictionDataset(Dataset):
    def __init__(self, sentences, numerical_values, tokenizer):
        self.sentences = sentences
        self.numerical_values = numerical_values
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        num_value = self.numerical_values[idx]
        if not isinstance(sentence, str):
            sentence = ""
        inputs = self.tokenizer(sentence, return_tensors="pt", max_length=128, truncation=True, padding="max_length", add_special_tokens=True)
        target = torch.tensor([num_value], dtype=torch.float)
        input_ids = inputs['input_ids'].squeeze()
        attention_mask = inputs['attention_mask'].squeeze()
        return input_ids, attention_mask, target

class DirectNumericalPredictionModel(nn.Module):
    def __init__(self, dropout_rate=0.1):
        super(DirectNumericalPredictionModel, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        hidden_size = self.bert.config.hidden_size
        self.dropout = nn.Dropout(dropout_rate)
        self.regressor = nn.Linear(hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = self.dropout(outputs.pooler_output)
        predicted_value = self.regressor(sequence_output)
        return predicted_value

# Load data and initialize model components
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
df = pd.read_csv('dataset/torchtest.csv')
sentences = df['text'].tolist()
numerical_values = df['number'].astype(float).tolist()
dataset = NumericalPredictionDataset(sentences, numerical_values, tokenizer)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

model = DirectNumericalPredictionModel()
model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
loss_function = nn.MSELoss()

epochs = 20
loss_values = []

for epoch in range(epochs):
    model.train()
    total_loss = 0
    for input_ids, attention_mask, targets in dataloader:
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        predictions = model(input_ids=input_ids, attention_mask=attention_mask).squeeze()
        loss = loss_function(predictions, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    scheduler.step()
    avg_loss = total_loss / len(dataloader)
    loss_values.append(avg_loss)
    print(f"Epoch {epoch + 1}, Loss: {avg_loss}")

# Save the model and plot training loss
model_save_path = "model_with_improvements.pth"
torch.save(model.state_dict(), model_save_path)

plt.plot(range(1, epochs+1), loss_values, marker='o', label='Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss Over Epochs')
plt.legend()
plt.show()
