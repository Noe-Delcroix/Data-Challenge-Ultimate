import librosa
import numpy as np
import h5py
import os

DATA_PATH = "../data/ESC_train"
AUDIO_FEATURES_PATH = "../data/audio_features_fft.hdf5"

SAMPLE_RATE = 44100
HOP_LENGTH = 1024
FRAME_LENGTH = 2048
SILENCE_THRESHOLD = 0.01


def extract_audio_features_fft(data_folder_path,hdf5_file_path):
    """
    Fonction pour extraire les caractéristiques audio (FFT) des fichiers audio
    :param data_folder_path: dossier contenant les fichiers audio
    :param hdf5_file_path: fichier HDF5 pour sauvegarder les caractéristiques audio
    """
    count = 0
    with h5py.File(hdf5_file_path, 'w') as hdf5_file:
        folder_list = os.listdir(data_folder_path)
        for folder in folder_list:
            folder_path = os.path.join(data_folder_path, folder)
            count += 1
            if os.path.isdir(folder_path):
                print("=======================================\n"
                      "Processing folder (class): ",
                      folder,
                      " ("+str(count)+"/"+str(folder_list.__len__())+") ",
                      "\n=======================================")
                for filename in os.listdir(folder_path):
                    if filename.endswith('.wav'):
                        print("Processing file: ", filename)
                        file_path = os.path.join(folder_path, filename)
                        audio, sr = librosa.load(file_path, sr=SAMPLE_RATE, mono=True)
                        # Normalisation par la valeur maximale
                        audio = audio / np.max(np.abs(audio))
                        # Découpage en segments
                        segments = librosa.util.frame(audio, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH)
                        for i, segment in enumerate(segments.T):
                            # Vérifier si le segment est silencieux
                            segment_energy = np.sum(segment**2)
                            if segment_energy > SILENCE_THRESHOLD:
                                # Calcul de la FFT pour chaque segment
                                fft_segment = librosa.stft(segment)
                                fft_magnitude_segment = np.abs(fft_segment)
                                # Sauvegarde dans le fichier HDF5
                                dataset = hdf5_file.create_dataset(name=f'{folder}/{filename}_seg{i}', data=fft_magnitude_segment)
                                dataset.attrs['label'] = folder  # Ajout de l'attribut de label


extract_audio_features_fft(DATA_PATH,AUDIO_FEATURES_PATH)