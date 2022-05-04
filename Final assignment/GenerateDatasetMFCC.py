# -*- coding: utf-8 -*-
"""
Created on Wed May  4 14:29:40 2022

@author: JK-WORK
"""

from scipy.io import wavfile
from python_speech_features import mfcc,delta
from scipy.stats import zscore
from scipy import signal
import numpy as np
import os
from pathlib import Path


def read_wav(path):
    fs, x = wavfile.read(path)
    x=x/(2**15) #Assuming 16bit wav
    return fs,x

def get_features(x,fs,N=2):
    features_mfcc = mfcc(x, fs);
    features_mfcc_norm = zscore(features_mfcc, axis=0, ddof=1) #Normalize
    return features_mfcc_norm;

def get_features_with_dynamics(x,fs,N=2):
    features_mfcc = mfcc(x, fs);
    features_mfcc_d1 = delta(features_mfcc, N)
    features_mfcc_d2 = delta(features_mfcc_d1, N)
    features_mfcc_norm = zscore(features_mfcc, axis=0, ddof=1) #Normalize
    features_mfcc_norm_d1 = zscore(features_mfcc_d1, axis=0, ddof=1) #Normalize
    features_mfcc_norm_d2 = zscore(features_mfcc_d2, axis=0, ddof=1) #Normalize
    final_feature_vector = np.concatenate((features_mfcc_norm,features_mfcc_norm_d1,features_mfcc_norm_d2),axis=1)
    #Each row contains all the features
    return final_feature_vector;

Dataset_base_folder="Dataset/train/audio/"
Dataset_MFCC_folder="DatasetMFCC/"
INCLUDE_DYNAMICS=False;

ftrain = open('training.txt', 'r')
lines = ftrain.readlines()
for line in lines:
    data_name=line[:-1]; #Remove eol
    fs,x = read_wav(Dataset_base_folder+data_name)
    features = get_features_with_dynamics(x,fs) if INCLUDE_DYNAMICS else get_features(x,fs);
    Path(Dataset_MFCC_folder+"train/"+os.path.dirname(data_name)).mkdir(parents=True, exist_ok=True)
    np.save(Dataset_MFCC_folder+"train/"+data_name[:-4],features)
    
ftrain = open('test.txt', 'r')
lines = ftrain.readlines()
for line in lines:
    data_name=line[:-1]; #Remove eol
    fs,x = read_wav(Dataset_base_folder+data_name)
    features = get_features_with_dynamics(x,fs) if INCLUDE_DYNAMICS else get_features(x,fs);
    Path(Dataset_MFCC_folder+"test/"+os.path.dirname(data_name)).mkdir(parents=True, exist_ok=True)
    np.save(Dataset_MFCC_folder+"test/"+data_name[:-4],features)