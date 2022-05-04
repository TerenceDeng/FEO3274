# -*- coding: utf-8 -*-
"""
Created on Wed May  4 14:56:02 2022

@author: JK-WORK
"""

from scipy.io import wavfile
import numpy as np

Dataset_base_folder="Dataset/train/audio/"
Dataset_MFCC_folder="DatasetMFCC/"

def read_elem(element, MFCC=True,train=True): #To read data using the lines from the train and  test .txt
    if MFCC:
        return np.load(Dataset_MFCC_folder+"train/"+element) if train else np.load(Dataset_MFCC_folder+"test/"+element);
    else:
        fs, x = wavfile.read(Dataset_base_folder+element)
        x=x/(2**15) #Assuming 16bit wav
        return fs, x;