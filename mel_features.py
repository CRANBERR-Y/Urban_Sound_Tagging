import librosa
import pandas as pd
import numpy as np
import os
import csv

header = 'filename,'
for i in range(1,21):
    header += f'mfcc_{i},'
for i in range(1,18):
    header += f'mfcc_delta_{i},'
for i in range(1,18):
    header += f'mfcc_double_delta_{i},'
header = header.split(',')

file = open('data.csv', 'w', newline='')
with file:
    writer = csv.writer(file)
    writer.writerow(header)

path = f"F:\ADocs\Year_3_Sem_2\Project\DCASE_UST\Dataset\Audio_dev"
dirs_1 = os.listdir(path)

for filename in dirs_1:
    path_1 = f"F:\ADocs\Year_3_Sem_2\Project\DCASE_UST\Dataset\Audio_dev\{filename}"
    dirs_2 = os.listdir(path_1)
    for audiofile in dirs_2:
        audio_snip = f'F:\ADocs\Year_3_Sem_2\Project\DCASE_UST\Dataset\Audio_dev\{filename}\{audiofile}'
        y = librosa.load(audio_snip, sr = 44100)[0]
        mfccs = librosa.feature.mfcc(y=y, sr=44100, n_mels=128, fmin=20, fmax = 44100//2 )
        mfcc_delta = librosa.feature.delta(mfccs)[:,:18]
        mfcc_double_delta = librosa.feature.delta(mfcc_delta)
        mfccs = mfccs[:, 2:22]
        to_append = f'{audiofile},'
        for e in mfccs:
            to_append += f'{np.mean(e)},'
        for j in mfcc_delta:
            to_append += f'{np.mean(j)},'
        for k in mfcc_double_delta:
            to_append += f'{np.mean(k)},'
        file = open('data.csv', 'a', newline='')
        with file:
            writer = csv.writer(file)
            writer.writerow(to_append.split(','))
    
feature_file = pd.read_csv('data.csv')
excel_writer = pd.ExcelWriter('F:\ADocs\Year_3_Sem_2\Project\DCASE_UST\Codes\mfcc_dataset.xlsx', engine='xlsxwriter')
feature_file.to_excel(excel_writer, 'mfcc_dataset')
excel_writer.save()

#Make sure you give absoulte path address and the path doesn't contain any \n, \t, \(numeric), \a etc.
#Avoid Spaces in your path name 