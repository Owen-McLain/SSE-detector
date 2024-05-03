# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 07:41:40 2024

@author: owenm
"""

import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import io
import pandas as pd
from tensorflow.keras.models import load_model
import os

# %%
import subprocess

# List of required libraries
required_libraries = ['librosa', 'numpy', 'matplotlib', 'opencv-python', 'Pillow', 'pandas', 'tensorflow']

# Check and install required libraries
for library in required_libraries:
    try:
        __import__(library)
        print(f"{library} is already installed.")
    except ImportError:
        print(f"{library} is not installed. Installing...")
        try:
            subprocess.check_call(['pip', 'install', library])
            print(f"{library} installed successfully.")
        except Exception as e:
            print(f"Error installing {library}: {e}")

# %%


def Spectrogram_to_Array(audio, 
                         fs):
    """
    computes mel spectrogram of audio then saves as array of size
    (1, 217, 217)
    
    Parameters
    ----------
    Parameters:
    - audio: array of float32 
        Audio data.
    - fs: int
        The sampling rate of the audio file.

    Returns
    -------
    Array of spectrogram size (1, 217, 217)
    """
    #compute spectrogram
    mel_spectrogram = librosa.feature.melspectrogram(y = audio,
                                                     n_mels=128,
                                                     sr=fs,
                                                     n_fft=2048,
                                                     hop_length=1024,
                                                     center=False)
    #convert to dB
    dB_mel_spectrogram = 10*np.log10(mel_spectrogram/1.0)
    #reshape
    resize_mel = cv2.resize(dB_mel_spectrogram, (256, 256))
    #add axis
    resized_im_of_mel_spec = resize_mel[:, :, np.newaxis]
    
    #plot spec
    plt.figure()
    plt.imshow(np.flipud(resized_im_of_mel_spec), cmap='gray')
    plt.axis('off')
    
    #convert to numpy array
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', bbox_inches="tight", pad_inches=0, transparent=True)
    buffer.seek(0)

    image = Image.open(buffer)
    im_array = np.array(image)
    im_array = im_array[:, :, :1]
    im_array = np.reshape(im_array, (217, 217))
    
    #close figure
    plt.close('all')
    
    return(im_array)
    
# is_mono = librosa.get_samplerate(r'C:\Users\owenm\Downloads\spawning_instance.wav') == 1
# audio, fs = librosa.load(r'C:\Users\owenm\Downloads\spawning_instance.wav', sr=44100, mono=is_mono)
# audio = audio[0, :]

# array_of_im = Spectrogram_to_Array(audio, fs)


# %%
def Apply_CNN(audio, 
              fs, 
              model_path, 
              spec_array=None):
    """
    Loads a Keras CNN model and applies it to the spectrogram of the audio.
    
    Parameters:
    - audio: numpy.ndarray
        Audio data.
    - fs: int
        The sampling rate of the audio file.
    - model_path: str
        Path to the Keras CNN model file.
    - spec_array: array
        array containing many spectrograms
    Returns:
    - numpy.ndarray
        Output of the CNN model applied to the spectrogram.
    """
    # Load the Keras model
    model = load_model(model_path)
    
    if audio is not None:
        # Get the spectrogram of the audio
        spectrogram = Spectrogram_to_Array(audio, 
                                       fs)
        # Reshape the spectrogram for the model input
        input_data = spectrogram[np.newaxis, :, :, np.newaxis]
        
    if spec_array is not None:
        spectrogram = spec_array
        # Reshape the spectrogram for the model input
        input_data = spectrogram[..., np.newaxis]
    
    # Apply the model to the input data
    output = model.predict(input_data)
    
    return output, input_data, spectrogram

# model_path = r'C:\Users\owenm\OneDrive - University of Southampton\CNN\CNN6\ML6_round_6'

# predict, inp, spec=Apply_CNN(audio, fs, model_path)

# %%
def Wav_to_Predictions(wav_path, 
                       overlap_len,
                       model_path):
    """
    takes wav file, convert to mono if needed, then produced spectrogram image 
    applies CNN rpediction to it

    Parameters
    ----------
    wav_path : str
        Path to wav file
    model_path : TYPE
        Path to CNN model
    overlap_len : int
        number of seconds overlap from 0-5

    Returns
    -------
    Array of positive predictions and time stamps

    """
            
    
    #read audio, checking if mono
    is_mono = librosa.get_samplerate(wav_path) == 1
    audio, fs = librosa.load(wav_path, sr=44100, mono=is_mono)
    if is_mono == False:
        audio = audio[0, :]
    # audio = audio[:int(16*fs)] # for fast debugging, only takes first 16 seconds
    
    #find total lenght of recording and segement length
    total_time = len(audio) / fs 
    seg_len = 5
    
    #initialise values
    # overlap_len = int(entry_overlap_len.get())
    start_time = 0
    end_time = int(seg_len)    

    # Initialize an empty array to store concatenated spectrograms
    concatenated_spectrograms = np.empty((0, 217, 217))
    timestamps = []
    print('creating spectrogram array')
    while end_time <= total_time:
        segment = audio[start_time * fs:end_time * fs]

        # Compute the spectrogram for the current segment
        spectrogram = Spectrogram_to_Array(segment, fs)

        # Concatenate the spectrogram to the array
        concatenated_spectrograms = np.concatenate(
            (concatenated_spectrograms, spectrogram[np.newaxis, :, :]))
        
        timestamps.append(start_time)
        # Update start and end time
        overlap = int(overlap_len)
        start_time = end_time - overlap
        end_time = start_time + seg_len
        
        
    print('Finding predictions')
    # Apply the CNN model to the concatenated spectrograms
    predictions = Apply_CNN(audio=None, 
                            fs=fs, 
                            model_path=model_path, 
                            spec_array = concatenated_spectrograms)
    
    predictions = predictions[0]
    
    
    return predictions, np.array(timestamps)


# overlap_len = 0
# wav_path = r"D:\IP_data\data\Scheldt_27_April\01_20230428_012257.wav"

# predictions, timestamps = Wav_to_Predictions(wav_path, model_path, overlap_len)        

# %%
def seconds_to_hhmmss(seconds):
    """
    Convert seconds to hh:mm:ss format.

    Parameters:
        seconds (float): Time in seconds.

    Returns:
        str: Time in hh:mm:ss format.
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

# %%

def Save_Positives(wav_path, 
                   overlap_len, 
                   ):
    """
    Identify positive predictions from a model on a WAV file and save them to an Excel file.

    Parameters:
        wav_path (str): Path to the WAV file.
        model_path (str): Path to the machine learning model.
        overlap_len (int): Length of overlap for processing the audio file.
        save_dir (str, optional): Directory to save the Excel file. If None, results won't be saved.

    Returns:
        np.ndarray: NumPy array containing filtered positive predictions.
    
    """
    # Get the current directory of the script
    script_dir = os.path.dirname(os.path.realpath(__file__))

    # Construct the path to the CNN folder
    
    model_type = 'CB7'
    model_path = os.path.join(script_dir, model_type)
    
    if model_type == 'CB7':
        threshold = 0.5
    
    if model_type == 'CB5':
        threshold = 0.45    

    if model_type == 'CB7':
        threshold = 0.55

    save_name = os.path.splitext(os.path.basename(wav_path))[0]
    save_dir = os.path.join(script_dir, save_name + ".xlsx")
    
    # overlap_len = int(entry_overlap_len.get())
    predictions, timestamps = Wav_to_Predictions(wav_path, overlap_len, model_path)
    filtered_predictions = []
    
    # Filter positive predictions
    for i in range(len(predictions)):
        if predictions[i, 1] > threshold and predictions[i, 0] < 0.3:
            filtered_predictions.append([timestamps[i], predictions[i, 0], predictions[i, 1]])
    
    # Convert filtered_predictions to a DataFrame
    df = pd.DataFrame(filtered_predictions, columns=['Timestamp', 'No SSE Confidence', 'SSE Confidence'])
    
    # Convert timestamp to hh:mm:ss format
    df['Time (hh:mm:ss)'] = df['Timestamp'].apply(seconds_to_hhmmss)
    
    df.to_excel(save_dir, index=False)
        
    return np.array(filtered_predictions)

# save_dir = r"C:\Users\owenm\OneDrive - University of Southampton\positive_preds.xlsx"
# filtered = Save_Positives(wav_path, overlap_len)

# %% for GUI
import tkinter as tk
from tkinter import filedialog, messagebox
from functools import partial

def browse_file(entry):
    filename = filedialog.askopenfilename(filetypes=[("WAV files", "*.wav")])
    entry.delete(0, tk.END)
    entry.insert(0, filename)


def save_predictions():
    wav_path = entry_wav_path.get()
    if not wav_path:
        messagebox.showerror("Error", "Please select a WAV file.")
        return
    
    
    overlap_len = int(entry_overlap_len.get())
    
    predictions = Save_Positives(wav_path, overlap_len)
    if predictions is not None:
        messagebox.showinfo("Success", "Filtered predictions saved in same folder as this script.")

root = tk.Tk()
root.title("SSE CCN Runner")

# Labels
label_wav_path = tk.Label(root, text="WAV File:")
label_overlap_len = tk.Label(root, text="Overlap Length (0-4 seconds):")

# Entry fields
entry_wav_path = tk.Entry(root, width=50)
entry_overlap_len = tk.Entry(root, width=10)
entry_overlap_len.insert(0, "0")
entry_save_dir = tk.Entry(root, width=50)

# Buttons
button_browse_wav = tk.Button(root, text="Browse", command=partial(browse_file, entry_wav_path))
button_save = tk.Button(root, text="OK", command=save_predictions)

# Layout
label_wav_path.grid(row=0, column=0, padx=5, pady=5, sticky="e")
entry_wav_path.grid(row=0, column=1, padx=5, pady=5)
button_browse_wav.grid(row=0, column=2, padx=5, pady=5)


label_overlap_len.grid(row=2, column=0, padx=5, pady=5, sticky="e")
entry_overlap_len.grid(row=2, column=1, padx=5, pady=5)

button_save.grid(row=4, column=1, padx=5, pady=5)

root.mainloop()
