import os
from pymongo import MongoClient
import numpy as np
import librosa
from tqdm import tqdm
from mutagen.mp3 import MP3

def extract_mfcc(file_path, num_mfcc=20, max_length=500):
    try:
        # Load audio file
        audio, sr = librosa.load(file_path, sr=None)
        # Extract MFCC features
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=num_mfcc)
        # Pad or truncate MFCC features to max_length
        if mfccs.shape[1] < max_length:
            mfccs = np.pad(mfccs, ((0, 0), (0, max_length - mfccs.shape[1])), mode='constant')
        else:
            mfccs = mfccs[:, :10000]
        # Return MFCC features
        return mfccs.T
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def extract_title(file_path):
    audio = MP3(file_path)
    title = audio.get('TIT2')
    return title.text[0] if title else 'Unknown Title'

# Function to process audio files and extract features
def process_files(input_folder):
    features = []
    for root, dirs, files in os.walk(input_folder):
        for file in tqdm(files):
            if file.endswith('.mp3'):  # Assuming input files are in MP3 format
                file_path = os.path.join(root, file)
                try:
                    mfcc_features = extract_mfcc(file_path)
                    if mfcc_features is not None:
                        title = extract_title(file_path)
                        features.append({
                            "title": title,
                            "file_path": file_path,  # Include file path for reference
                            "mfcc_features": mfcc_features.tolist()  # Convert to list for MongoDB
                        })
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")
                    continue  # Move to the next file if an error occurs
    return features

# Function to save features to MongoDB
def save_to_mongodb(features):
    client = MongoClient('mongodb://localhost:27017/')
    db = client['Audio1']
    collection = db['Features1']
    collection.insert_many(features)
    print("Data saved to MongoDB successfully.")

# Main function
def main():
    # Input folder containing audio files (MP3 format)
    input_folder = "/home/salman/Downloads/data"

    # Process MP3 files and extract features
    features = process_files(input_folder)

    # Save features to MongoDB
    save_to_mongodb(features)

if __name__ == "__main__":
    main()

