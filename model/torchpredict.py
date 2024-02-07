import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertTokenizer, BertModel
import numpy as np

"""
class GaussianMixtureLayer(nn.Module):
    def __init__(self, input_dim, num_components):
        super(GaussianMixtureLayer, self).__init__()
        self.num_components = num_components
        self.means = nn.Parameter(torch.randn(num_components, input_dim))
        self.log_vars = nn.Parameter(torch.zeros(num_components, input_dim))  # Log variance for stability
        self.weights = nn.Parameter(torch.zeros(num_components))  # Initial weights are equal

    def forward(self, x):
        weights = F.softmax(self.weights, dim=0)
        x_expand = x.unsqueeze(1).expand(-1, self.num_components, -1)
        means_expand = self.means.unsqueeze(0)
        std_devs = torch.exp(self.log_vars / 2)
        std_devs_expand = std_devs.unsqueeze(0)
        exponent = torch.sum(((x_expand - means_expand) / std_devs_expand) ** 2, dim=2)
        prefactor = torch.prod(std_devs_expand, dim=2) * (2 * np.pi) ** (std_devs_expand.size(2) / 2)
        densities = torch.exp(-0.5 * exponent) / prefactor
        weighted_densities = densities * weights
        pdf = torch.sum(weighted_densities, dim=1)
        return pdf


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


def load_model(model_path):
    model = EnhancedNumericalPredictionModel()
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()  # Set the model to evaluation mode
    return model


def predict(text, model, tokenizer):
    inputs = tokenizer(text, return_tensors="pt", max_length=128, truncation=True, padding="max_length",
                       add_special_tokens=True)
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']

    with torch.no_grad():
        pdf = model(input_ids=input_ids, attention_mask=attention_mask)

    # Assuming we can access the means and weights of the Gaussian Mixture directly
    # This part of the code might need to be adjusted based on the actual model architecture and output
    means = model.output_layer.means.detach().cpu().numpy()
    weights = torch.softmax(model.output_layer.weights, dim=0).detach().cpu().numpy()

    # Calculate the weighted average of the means
    weighted_average = np.dot(weights, means)

    # For simplicity, let's take the mean of the weighted averages across all dimensions
    predicted_value = np.mean(weighted_average)

    return predicted_value

# Load the tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model_path = "enhanced_numerical_prediction_model.pth"  # Update this path as needed
model = load_model(model_path)

# Example prediction
text = "What's next?"
predicted_value = predict(text, model, tokenizer)
print(f"Predicted value: {predicted_value}")

"""
class DirectNumericalPredictionModel(nn.Module):
    def __init__(self):
        super(DirectNumericalPredictionModel, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        hidden_size = self.bert.config.hidden_size
        self.regressor = nn.Linear(hidden_size, 1)  # Predict a single continuous value

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.pooler_output
        predicted_value = self.regressor(sequence_output)
        return predicted_value

def predict(text, model, tokenizer):
    inputs = tokenizer(text, return_tensors="pt", max_length=128, truncation=True, padding="max_length",
                       add_special_tokens=True)
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']

    with torch.no_grad():
        predicted_value = model(input_ids=input_ids, attention_mask=attention_mask).squeeze().item()

    return predicted_value


# Example prediction usage
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model_path = "path_to_your_trained_model.pth"  # Ensure this path points to your trained model
model = DirectNumericalPredictionModel()
model.load_state_dict(torch.load(model_path))
model.eval()

text = "Example text input for prediction."
predicted_value = predict(text, model, tokenizer)
print(f"Predicted value: {predicted_value}")
