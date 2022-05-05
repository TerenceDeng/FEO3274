# -*- coding: utf-8 -*-
"""
Created on Wed May  4 14:56:02 2022

@author: JK-WORK
"""

from scipy.io import wavfile
import numpy as np
import regex as re
import os
import hashlib as hashlib
from sklearn.metrics import confusion_matrix,accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

Dataset_base_folder="Dataset/train/audio/"
Dataset_MFCC_folder="DatasetMFCC/"

def read_elem_single(element, MFCC=True,train=True): #To read data using the lines from the train and  test .txt
    if MFCC:
        return np.load(Dataset_MFCC_folder+"train/"+element) if train else np.load(Dataset_MFCC_folder+"test/"+element);
    else:
        fs, x = wavfile.read(Dataset_base_folder+element)
        x=x/(2**15) #Assuming 16bit wav
        return fs, x;
    
def get_list_of_ds_elem_names(txt_path):
    ftrain = open('training.txt', 'r')
    lines = ftrain.readlines()
    ds_elem_names=[];
    for line in lines:
        ds_elem_names.append(line[:-1])
    return ds_elem_names;

def read_elem(element, MFCC=True,train=True):
    if isinstance(element,list):
        res=[];
        for i in element:
            res.append(read_elem_single(i, MFCC=MFCC,train=train))
    else:
        res=read_elem_single(i, MFCC=MFCC,train=train);
    return res;

def get_class_of_meas(elem):
    return elem;

def select_class_of_meas(ds_elem_names,class_wanted):
    ds_elem_names_class = [];
    return ds_elem_names_class;

MAX_NUM_WAVS_PER_CLASS = 2**27 - 1  # ~134M

def which_set(filename, testing_percentage, seed_str):
  """Determines which data partition the file should belong to.

  We want to keep files in the same training, validation, or testing sets even
  if new ones are added over time. This makes it less likely that testing
  samples will accidentally be reused in training when long runs are restarted
  for example. To keep this stability, a hash of the filename is taken and used
  to determine which set it should belong to. This determination only depends on
  the name and the set proportions, so it won't change as other files are added.

  It's also useful to associate particular files as related (for example words
  spoken by the same person), so anything after '_nohash_' in a filename is
  ignored for set determination. This ensures that 'bobby_nohash_0.wav' and
  'bobby_nohash_1.wav' are always in the same set, for example.

  Args:
    filename: File path of the data sample.
    validation_percentage: How much of the data set to use for validation.
    testing_percentage: How much of the data set to use for testing.

  Returns:
    String, one of 'training', 'validation', or 'testing'.
  """
  base_name = os.path.basename(filename)
  hash_name = re.sub(r'_nohash_.*$', '', base_name)
  hash_name=hash_name+seed_str;
  hash_name_hashed = hashlib.sha1(hash_name.encode('utf-8')).hexdigest()
  percentage_hash = ((int(hash_name_hashed, 16) %
                      (MAX_NUM_WAVS_PER_CLASS + 1)) *
                     (100.0 / MAX_NUM_WAVS_PER_CLASS))
  if percentage_hash < testing_percentage:
    result = 'testing'
  else:
    result = 'training'
  return result

def classifier_single(scores,labels): #there must be a final function that given the scores selects the class
    return labels[np.argmax(scores)]

def classify(scores,labels):
    if isinstance(scores,list):
        ret=[]
        for elem in scores:
            ret.append(classifier_single(elem,labels))
    else:
        ret=classifier_single(scores,labels);
    return ret;

def plot_results(y_true,y_pred,labels): #When evaluating in one dataset, generates confusion matrix to plot results.
    cm=confusion_matrix(y_true, y_pred, labels=labels)
    #https://stackoverflow.com/questions/65618137/confusion-matrix-for-multiple-classes-in-python
    fig = plt.figure(figsize=(16, 14))
    ax= plt.subplot()
    sns.heatmap(cm, annot=True, ax = ax, fmt = 'g'); #annot=True to annotate cells
    # labels, title and ticks
    ax.set_xlabel('Predicted', fontsize=20)
    ax.xaxis.set_label_position('bottom')
    plt.xticks(rotation=90)
    ax.xaxis.set_ticklabels(labels, fontsize = 10)
    ax.xaxis.tick_bottom()
    
    ax.set_ylabel('True', fontsize=20)
    ax.yaxis.set_ticklabels(labels, fontsize = 10)
    plt.yticks(rotation=0)
    
    plt.title('Classifier performance', fontsize=20)
    
    #plt.savefig('ConMat24.png')
    plt.show()
    
def get_accuracy(y_true,y_pred):
    return accuracy_score(y_true, y_pred);