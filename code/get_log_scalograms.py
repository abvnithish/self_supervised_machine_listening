### Importing Libraries
import os
import time
import random
from glob import glob
import numpy as np
import pandas as pd
import librosa
from librosa.display import specshow
import multiprocessing as mp

ROOT_DIR = '/beegfs/bva212/fma_small/'
PATH_FILE = '/checksums'
SR = 22050


filenames = []
dump = []
if os.path.exists(ROOT_DIR + PATH_FILE):
    with open(ROOT_DIR + PATH_FILE) as fp:
        lines = fp.readlines()
        for line in lines:
            track_id, file_name = line.split()
            if os.path.exists(ROOT_DIR + file_name):
                filenames.append(file_name)
print(f'There are a total of {len(filenames)} music files in the root directory')

def save_waveforms(file):

    ROOT_DIR = '/beegfs/bva212/fma_small/'
    SAVE_DIR = '/beegfs/bva212/fma_small_cqt/'
    waveform, fs_ = librosa.load((ROOT_DIR + file), sr = SR)
    #waveform_flipped = np.ascontiguousarray(np.flip(waveform), dtype=np.float32)

    cqt = librosa.cqt(waveform)
    #cqt_flipped = librosa.cqt(waveform_flipped)

    logscalogram = librosa.amplitude_to_db(np.abs(cqt))
    #logscalogram_flipped = librosa.amplitude_to_db(np.abs(cqt_flipped))

    os.makedirs(os.path.dirname(SAVE_DIR + file[:4]), exist_ok=True)

    np.save((SAVE_DIR + file + '_cqt.npy'), logscalogram)
    #np.save((SAVE_DIR + file + '_cqt_flipped.npy'), logscalogram_flipped)

    return True


print(f'Number of workers available are {os.cpu_count()}')
start = time.time()
with mp.Pool(os.cpu_count()) as p:
    results = p.map_async(save_waveforms, filenames)
    output = results.get()
print(f'\nTime for loading files is {time.time() - start}')









