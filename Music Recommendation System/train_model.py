import numpy as np
from pymongo import MongoClient
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import os
from mutagen.mp3 import MP3

# Function to extract title from file path using Mutagen library
def extract_title(file_path):
    audio = MP3(file_path)
    title = audio.get('TIT2')
    return title.text[0] if title else 'Unknown Title'

# Connect to MongoDB
client = MongoClient('localhost', 27017)
db = client['Audio1']
collection = db['Features1']

# Load data from MongoDB collection
data_from_mongodb = list(collection.find())

# Extract mfcc_features and titles
mfcc_features = [doc["mfcc_features"] for doc in data_from_mongodb]
file_paths = [doc["file_path"] for doc in data_from_mongodb]
titles = [extract_title(file_path) for file_path in file_paths]

# Pad sequences to ensure uniform length
max_length = max(len(features) for features in mfcc_features)
mfcc_features_padded = pad_sequences(mfcc_features, maxlen=max_length, padding='post', dtype='float32')

# Convert padded sequences to NumPy array
mfcc_features_array = np.array(mfcc_features_padded)

# Convert NumPy arrays to PyTorch tensors
X = torch.tensor(mfcc_features_array, dtype=torch.float32)

# Split the data into training and testing sets
X_train, X_test, titles_train, titles_test = train_test_split(X, titles, test_size=0.2, random_state=1234)

# Reshape the input data from 3 dimensions to 2 dimensions
X_train_reshaped = X_train.reshape(X_train.shape[0], -1)
X_test_reshaped = X_test.reshape(X_test.shape[0], -1)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_reshaped)
X_test_scaled = scaler.transform(X_test_reshaped)

# Convert scaled arrays back to PyTorch tensors
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)

# Create DataLoader for training and testing sets
train_dataset = TensorDataset(X_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_dataset = TensorDataset(X_test_tensor)
test_loader = DataLoader(test_dataset, batch_size=64)

# Define the neural network model
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Initialize model, loss function, and optimizer
model = MLP(input_size=X_train_tensor.shape[1], hidden_size1=64, hidden_size2=32, output_size=X_train_tensor.shape[1])
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
for epoch in range(100):
    for batch_X in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X[0])
        loss = criterion(outputs, batch_X[0])
        loss.backward()
        optimizer.step()

# Save the trained model using pickle
with open('trained_model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Evaluate the model
with torch.no_grad():
    model.eval()
    predictions = model(X_test_tensor)
    rmse = torch.sqrt(criterion(predictions, X_test_tensor))
    print("Root Mean Squared Error (RMSE):", rmse.item())

