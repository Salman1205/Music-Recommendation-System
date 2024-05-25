import numpy as np
from pymongo import MongoClient
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
from train_model import MLP  # Assuming you have the MLP class defined in a file named features2.py

# Connect to MongoDB
client = MongoClient('localhost', 27017)
db = client['Audio1']
collection = db['Features1']

# Retrieve all documents (songs) from the collection
data_from_mongodb = list(collection.find())

# Extract mfcc_features
mfcc_features = [doc["mfcc_features"] for doc in data_from_mongodb]

# Pad sequences to ensure uniform length
max_length = max(len(features) for features in mfcc_features)
mfcc_features_padded = pad_sequences(mfcc_features, maxlen=max_length, padding='post', dtype='float32')

# Convert padded sequences to NumPy array
features_array = np.array(mfcc_features_padded)

# Reshape the array to have two dimensions
num_samples, _, num_features = features_array.shape
features_reshaped = features_array.reshape(num_samples, -1)

# Standardize features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features_reshaped)

# Convert scaled arrays back to PyTorch tensors
features_tensor = torch.tensor(features_scaled, dtype=torch.float32)

# Load the trained model using pickle
with open('trained_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Define function to find similar songs
# Define function to find similar songs
# Define function to find similar songs
def find_similar_songs(song_title, num_results=10):
    # Find index of the song in the dataset
    song_index = next((i for i, doc in enumerate(data_from_mongodb) if doc['title'] == song_title), None)
    if song_index is None:
        print("Song not found in the dataset.")
        return None
    
    # Get features of the input song
    input_features = features_tensor[song_index].unsqueeze(0)  # Add batch dimension
    
    # Pass input features through the model to get predictions
    with torch.no_grad():
        model.eval()
        predicted_features = model(input_features)
    
    # Calculate cosine similarity between predicted features and all other features
    similarities = nn.functional.cosine_similarity(predicted_features, features_tensor)
    
    # Find indices of most similar songs
    similar_indices = torch.argsort(similarities, descending=True)[:num_results + 1]  # Add 1 to include input song
    
    # Remove the index of the input song from the list
    similar_indices = [idx for idx in similar_indices if idx != song_index]
    
    # Get titles of similar songs
    similar_song_titles = [data_from_mongodb[idx]["title"] for idx in similar_indices]
    
    return similar_song_titles[:num_results]  # Return only the specified number of similar songs


# Example usage:
input_song_title = "Outskirts"  # Change this to the desired song title
similar_songs = find_similar_songs(input_song_title)

# Check if similar songs were found
if similar_songs is not None:
    # Display similar song titles
    print(f"Top 10 most similar songs to '{input_song_title}':")
    for idx, title in enumerate(similar_songs, 1):
        print(f"{idx}. {title}")
else:
    print("No similar songs found.")


