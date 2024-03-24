import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
from transformers import RobertaTokenizer, RobertaModel
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import time

torch.cuda.empty_cache()

# Load dataset (Ensure 'torchtest.csv' with 'text' and 'number' columns is available)

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

        # Check if sentence is a string
        if not isinstance(sentence, str):
            print(f"Skipping idx {idx}, not a string: {sentence}")
            sentence = ""  # Use an empty string or some placeholder text

        inputs = self.tokenizer(sentence, return_tensors="pt", max_length=128, truncation=True, padding="max_length",
                                add_special_tokens=True)
        target = torch.tensor([num_value], dtype=torch.float)
        input_ids = inputs['input_ids'].squeeze()
        attention_mask = inputs['attention_mask'].squeeze()
        return input_ids, attention_mask, target


class GaussianMixtureLayer(nn.Module):
    def __init__(self, input_dim, num_components):
        super(GaussianMixtureLayer, self).__init__()
        self.num_components = num_components
        self.means = nn.Parameter(torch.randn(num_components, input_dim))
        self.log_vars = nn.Parameter(torch.zeros(num_components, input_dim))  # Log variance for stability
        self.weights = nn.Parameter(torch.zeros(num_components))  # Initial weights are equal

    def forward(self, x):
        # Ensure positive weights that sum to 1
        weights = F.softmax(self.weights, dim=0)

        # Compute Gaussian densities
        x_expand = x.unsqueeze(1).expand(-1, self.num_components, -1)
        means_expand = self.means.unsqueeze(0)
        std_devs = torch.exp(self.log_vars / 2)
        std_devs_expand = std_devs.unsqueeze(0)
        exponent = torch.sum(((x_expand - means_expand) / std_devs_expand) ** 2, dim=2)
        prefactor = torch.prod(std_devs_expand, dim=2) * (2 * np.pi) ** (std_devs_expand.size(2) / 2)
        densities = torch.exp(-0.5 * exponent) / prefactor

        # Weighted sum of densities
        weighted_densities = densities * weights
        pdf = torch.sum(weighted_densities, dim=1)

        return pdf
"""
class EnhancedNumericalPredictionModel(nn.Module):
    def __init__(self, num_components=3):
        super(EnhancedNumericalPredictionModel, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        hidden_size = self.bert.config.hidden_size
        self.output_layer = GaussianMixtureLayer(hidden_size, num_components)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.pooler_output
        pdf = self.output_layer(sequence_output)
        return pdf

"""
"""
class DirectNumericalPredictionModel(nn.Module):
    def __init__(self):
        super(DirectNumericalPredictionModel, self).__init__()
        self.roberta = RobertaModel.from_pretrained('roberta-base')
        hidden_size = self.roberta.config.hidden_size
        self.regressor = nn.Linear(hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state[:, 0, :]
        predicted_value = self.regressor(sequence_output)
        return predicted_value

tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
"""
class DirectNumericalPredictionModel(nn.Module):
    def __init__(self):
        super(DirectNumericalPredictionModel, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        #self.bert = RobertaModel.from_pretrained('roberta-base')
        hidden_size = self.bert.config.hidden_size
        self.regressor = nn.Linear(hidden_size, 1)  # Predict a single continuous value
        #self.regressor = nn.Linear(hidden_size)  # Predict a single continuous value

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.pooler_output
        predicted_value = self.regressor(sequence_output)
        return predicted_value


# Training Preparation
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
#tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
df = pd.read_csv('dataset/torchtest.csv')
sentences = df['text'].tolist()
numerical_values = df['number'].astype(float).tolist()
dataset = NumericalPredictionDataset(sentences, numerical_values, tokenizer)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
# Initialize the model, optimizer, etc. as before
# 1. Check for CUDA Availability
if torch.cuda.is_available():
    device = "cuda"
    print("CUDA Available")
else:
    device = "cpu"
    print("CUDA NOT Available")

"""
model = EnhancedNumericalPredictionModel()
model.train()

optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

epochs = 10
loss_values = []  # Initialize an empty list to store the loss values

for epoch in range(epochs):
    total_loss = 0
    for input_ids, attention_mask, targets in dataloader:
        optimizer.zero_grad()
        pdf = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = -torch.log(pdf).mean()  # NLL loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(dataloader)
    loss_values.append(avg_loss)  # Store the average loss for this epoch
    print(f"Epoch {epoch+1}, Loss: {avg_loss}")

"""

# Initialize the model
model = DirectNumericalPredictionModel()  # or EnhancedNumericalPredictionModel()

# 2. Move the Model to GPU
model.to(device)

# Initialize the optimizer, loss function, etc.
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
loss_function = nn.MSELoss()

# Start the training loop
epochs = 20 # Adjust epochs according to your need
loss_values = []

training_start_time = time.time()

for epoch in range(epochs):
    total_loss = 0
    for input_ids, attention_mask, targets in dataloader:
        # 3. Move Data to GPU
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        predictions = model(input_ids=input_ids, attention_mask=attention_mask).squeeze()
        loss = loss_function(predictions, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    loss_values.append(avg_loss)
    epoch_time = time.time() - training_start_time
    print(f"Epoch {epoch + 1}, Loss: {avg_loss}, Time: {epoch_time}")

# Save the model's state dictionary
model_save_path = "2K20B64lre-5_enhanced_numerical_prediction_model.pth"
torch.save(model.state_dict(), model_save_path)

print(f"Model saved to {model_save_path}")

# Plot Training Loss
plt.plot(range(1, epochs+1), loss_values, marker='o', label='Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title(f'Training Loss Over Epochs: {model_save_path}')
plt.legend()
plt.show()