# Music Recommendation System

This project is a music recommendation system that finds similar songs based on extracted features from audio files. The system also provides lyrics extraction for songs using speech recognition.

## Features

- **Find Similar Songs**: Given a song title, the system finds similar songs based on their audio features.
- **Lyrics Extraction**: Extracts and displays lyrics from the provided audio files.
- **Play Songs**: Allows users to play songs directly from the web interface.

## Requirements

- Python 3.x
- MongoDB
- Flask
- ffmpeg

### Python Libraries

- `os`
- `tempfile`
- `shutil`
- `speech_recognition`
- `eyed3`
- `numpy`
- `torch`
- `scikit-learn`
- `tensorflow`
- `pymongo`
- `flask`

## Installation

1. **Clone the repository**:

   ```bash
   git clone https://github.com/your-username/music-recommendation-system.git
   cd music-recommendation-system
   ```

2. **Set up a virtual environment**:

   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows use `venv\Scripts\activate`
   ```

3. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

4. **Install `ffmpeg`**:

   Follow the instructions on the [FFmpeg website](https://ffmpeg.org/download.html) to install `ffmpeg` on your system.

5. **Set up MongoDB**:

   Ensure MongoDB is installed and running on your local machine. Create a database named `Audio1` and a collection named `Features1`.

6. **Download the FMA Dataset**:

   You can download the `fma_large` dataset which contains songs and `fma_metadata` which contains metadata about songs from this link: [FMA: A Dataset For Music Analysis](https://github.com/mdeff/fma).

7. **Extract Features and Save to MongoDB**:

   After downloading the dataset, you can save the data in MongoDB by running the `features.py` file. This script will extract features from the audio files and store them in the MongoDB collection.

   ```bash
   python features.py
   ```

## Usage

1. **Start the Flask app**:

   ```bash
   python app.py
   ```

2. **Open your browser** and navigate to `http://127.0.0.1:5000` to access the web interface.

## Directory Structure

```
music-recommendation-system/
│
├── app.py                   # Main application script
├── train_model.py           # Contains the MLP class definition and training script
├── features.py              # Script to extract features and save to MongoDB
├── requirements.txt         # Python dependencies
├── templates/
│   ├── index.html           # Home page template
│   ├── songs.html           # Songs display template
│   ├── error.html           # Error display template
├── static/                  # Static files (CSS, JS, images)
└── README.md                # This README file
```

## Model Training

### MLP Model

The recommendation system uses a Multi-Layer Perceptron (MLP) model to learn features from the audio data. The MLP model is implemented in `train_model.py` and consists of the following:

- **Input Layer**: Takes the flattened MFCC features as input.
- **Hidden Layer**: A fully connected layer followed by a ReLU activation function.
- **Output Layer**: A fully connected layer that outputs the feature representation for similarity comparison.

### Training the Model

1. **Feature Extraction**: Extract MFCC features from audio files and store them in MongoDB.
2. **Data Preprocessing**: Scale the features using `StandardScaler` and convert them into PyTorch tensors.
3. **Model Training**: Train the MLP model using the preprocessed features. The model learns to generate a feature representation for each audio file.
4. **Model Saving**: Save the trained model as a pickle file (`trained_model.pkl`) for later use in the recommendation system.

### Example Workflow

1. **Extract Features**: Extract MFCC features from your audio files.
2. **Store Features**: Store the extracted features in MongoDB.
3. **Train Model**: Train the MLP model using the stored features.
4. **Save Model**: Save the trained model to a file.

## Troubleshooting

### File Not Found Error

If you encounter a `FileNotFoundError` related to `temp.wav`, ensure that `ffmpeg` is correctly installed and accessible. Check that the input file path provided to `ffmpeg` is correct and that the script has permission to write to the directory.

### Import Errors

Ensure all necessary Python libraries are installed. You can install missing libraries using `pip install library_name`.

## Contributors

- [Stradok](https://github.com/Stradok)
