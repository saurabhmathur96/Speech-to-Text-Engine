from os import path
import numpy as np
import librosa
import tqdm
import argparse
import os
import zipfile
from tqdm.auto import tqdm

parser = argparse.ArgmentParser()
parser.add_argument('zip_path')
parser.add_argument('target_path')
args = parser.parse_args()

zip_path = args.zip_path # '../gdrive/My Drive/en/train.zip'
target_path = args.target_path # '../gdrive/My Drive/en/train'
os.makedirs(target_path, exist_ok=True)
archive = zipfile.ZipFile(zip_path)
for name in tqdm(archive.namelist()):
  if name.endswith('.mp3'):
    archive.extract(name)
    signal, rate = librosa.load(name)
    encoded = np.abs(librosa.stft(signal, n_fft=1024, hop_length=512)).T
    os.remove(name)
    name = path.split(name)[-1]
    np.save(f'{name}.npy', encoded)
