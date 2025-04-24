# SHL Grammar Scoring Engine

### Dataset :- [https://www.kaggle.com/competitions/shl-intern-hiring-assessment/data]

## Project Overview
  This project is a Grammar Scoring Engine developed for the SHL Intern Hiring Assessment. It processes and evaluates spoken grammar proficiency using audio files as input. The primary goal is to train a regression model that predicts grammar scores on a scale based on speech recordings provided in .wav format.

## Dataset

- CSV File: Contains the metadata and grammar labels (label column).

- Audio Files: Stored in Dataset/audios/train/ directory with .wav format.

## Steps Followed

## 1. Data Loading

- Read the CSV from the dataset folder.

- Sanitize and standardize the filenames.

- Validate file existence in the directory.

## 2. Feature Extraction

- Used librosa to extract MFCC (Mel Frequency Cepstral Coefficients) features from each audio file.

- Computed the mean and standard deviation of the MFCCs as input features.

## 3. Preprocessing

- Scaled the features using StandardScaler.

- Split the dataset into training and validation sets (80:20 ratio).

## 4. Model Building

- Built a Sequential neural network using TensorFlow/Keras.

- Architecture:

  - Dense(128, ReLU) + Dropout(0.3)

  - Dense(64, ReLU)

  - Output: Dense(1, Linear)

## 5. Model Training

- Trained for 50 epochs with batch size 16.

- Used MSE as loss and MAE as additional metric.

## 6. Evaluation Metrics

- Validation RMSE: 0.9416

- Validation R² Score: 0.3450

- Validation MAE: 0.7126

- Training RMSE: 0.3104

## 7. Visualization

- Plotted training and validation loss over epochs.

- Plotted True vs Predicted scores for validation set.

## 8. Final Submission CSV

- Generated a CSV file with 204 rows containing predicted grammar label values.
