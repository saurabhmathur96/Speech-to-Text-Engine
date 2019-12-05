''' encode.py: takes the absolute value of STFT of mp3 files

  Usage:
    $ python3 encode.py mp3_dir/ output_dir/
'''
import numpy as np
import librosa
import tqdm
import argparse
import os


parser = argparse.ArgumentParser()
parser.add_argument('mp3_dir')
parser.add_argument('output_dir')
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)
for mp3 in tqdm.tqdm(os.listdir(args.mp3_dir)):
  signal, rate = librosa.load(os.path.join(args.mp3_dir, mp3))
  encoded = np.abs(librosa.stft(signal, n_fft=1024, hop_length=512)).T
  np.save(os.path.join(args.output_dir, f'{mp3}.npy'), encoded)
