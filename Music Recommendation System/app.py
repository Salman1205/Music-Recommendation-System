import os
import tempfile
import shutil
import speech_recognition as sr
import eyed3
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
from pymongo import MongoClient
from flask import Flask, render_template, request, send_file, jsonify
from train_model import MLP
# Initialize Flask app
app = Flask(__name__, template_folder='templates')

# Connect to MongoDB
client = MongoClient('localhost', 27017)
db = client['Audio1']
collection = db['Features1']

# Load the trained model
with open('trained_model.pkl', 'rb') as f:
    model = pickle.load(f)

def convert_to_wav(audio_file_path):
    # Get the directory path where the script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Generate a unique temporary filename within the script directory
    temp_wav_file = os.path.join(script_dir, "temp.wav")

    # Convert audio file to WAV format
    os.system(f"ffmpeg -i {audio_file_path} -acodec pcm_s16le -ar 16000 {temp_wav_file}")

    return temp_wav_file

# Function to extract lyrics
def extract_lyrics(file_path):
    recognizer = sr.Recognizer()
    with sr.AudioFile(file_path) as source:
        failure_count = 0
        lyrics = ""
        while True:
            audio_data = recognizer.record(source, duration=10)
            try:
                lyrics += recognizer.recognize_google(audio_data) + "\n"
                failure_count = 0
            except sr.UnknownValueError:
                failure_count += 1
                if failure_count >= 3:
                    break
            except sr.RequestError as e:
                failure_count += 1
                if failure_count >= 3:
                    break
    return lyrics

# Function to find similar songs
def find_similar_songs(song_title, num_results=5, exclude=[]):
    data_from_mongodb = list(collection.find())
    mfcc_features = [doc["mfcc_features"] for doc in data_from_mongodb]
    max_length = max(len(features) for features in mfcc_features)
    mfcc_features_padded = pad_sequences(mfcc_features, maxlen=max_length, padding='post', dtype='float32')
    features_array = np.array(mfcc_features_padded)
    num_samples, _, num_features = features_array.shape
    features_reshaped = features_array.reshape(num_samples, -1)
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features_reshaped)
    features_tensor = torch.tensor(features_scaled, dtype=torch.float32)
    song_index = next((i for i, doc in enumerate(data_from_mongodb) if doc['title'] == song_title), None)
    if song_index is None:
        return None, None, None  # Return None for both similar songs, metadata, and lyrics if song not found
    
    # Extract metadata using eyed3
    audio_file_path = data_from_mongodb[song_index]["file_path"]
    audio_file = eyed3.load(audio_file_path)
    if audio_file is None or audio_file.tag is None:
        song_metadata = {
            "title": song_title,
            "artist": "Not found",
            "album": "Not found",
            "year": "Not found",
            "genre": "Not found",
            "file_path": audio_file_path
        }
    else:
        song_metadata = {
            "title": audio_file.tag.title,
            "artist": audio_file.tag.artist if audio_file.tag.artist else "Not found",
            "album": audio_file.tag.album if audio_file.tag.album else "Not found",
            "year": audio_file.tag.release_date.year if audio_file.tag.release_date else "Not found",
            "genre": audio_file.tag.genre if audio_file.tag.genre else "Not found",
            "file_path": audio_file_path
        }
    
    # Extract lyrics
    temp_wav_file = convert_to_wav(audio_file_path)
    lyrics = extract_lyrics(temp_wav_file)

    input_features = features_tensor[song_index].unsqueeze(0)
    with torch.no_grad():
        model.eval()
        predicted_features = model(input_features)
    similarities = nn.functional.cosine_similarity(predicted_features, features_tensor)
    similar_indices = torch.argsort(similarities, descending=True)[:num_results + 1]
    similar_indices = [idx for idx in similar_indices if idx != song_index and idx not in exclude]  # Exclude already selected similar songs
    similar_song_titles = [data_from_mongodb[idx]["title"] for idx in similar_indices]

    return similar_song_titles[:num_results], song_metadata, lyrics

# Route to render the index.html template
@app.route('/')
def index():
    return render_template('index.html')

# Route to find similar songs
@app.route('/find_similar', methods=['GET', 'POST'])
def find_similar():
    if request.method == 'POST':
        song_title = request.form.get('song_title')
        exclude = request.form.getlist('exclude')
    elif request.method == 'GET':
        song_title = request.args.get('song_title')
        exclude = request.args.getlist('exclude')
    else:
        return render_template('error.html', message="Invalid request method.")
    
    if song_title:
        similar_songs, song_metadata, lyrics = find_similar_songs(song_title, exclude=exclude)
        if similar_songs:
            return render_template('songs.html', song_title=song_title, similar_songs=similar_songs, song_metadata=song_metadata, lyrics=lyrics)
        else:
            return render_template('error.html', message="Similar songs not found.")
    else:
        return render_template('error.html', message="Song title parameter missing from request.")

# Route to play a song
@app.route('/play_song/<song_title>')
def play_song(song_title):
    song = collection.find_one({'title': song_title})
    if song:
        audio_file_path = song.get('file_path')
        if audio_file_path:
            return send_file(audio_file_path, as_attachment=True)
    return render_template('error.html', message="Audio file not found.")

# Route to load lyrics
@app.route('/load_lyrics')
def load_lyrics():
    # Get the song title from the request parameters
    song_title = request.args.get('song_title')

    # Call the find_similar_songs function to retrieve the lyrics for the specified song title
    _, _, lyrics = find_similar_songs(song_title)

    # Return the lyrics as a response
    return lyrics


# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
