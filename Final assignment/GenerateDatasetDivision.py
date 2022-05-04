# -*- coding: utf-8 -*-
"""
Created on Wed May  4 14:00:19 2022

@author: JK-WORK
"""
import regex as re
import os
import hashlib as hashlib

MAX_NUM_WAVS_PER_CLASS = 2**27 - 1  # ~134M

def which_set(filename, testing_percentage):
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
  # We want to ignore anything after '_nohash_' in the file name when
  # deciding which set to put a wav in, so the data set creator has a way of
  # grouping wavs that are close variations of each other.
  hash_name = re.sub(r'_nohash_.*$', '', base_name)
  # This looks a bit magical, but we need to decide whether this file should
  # go into the training, testing, or validation sets, and we want to keep
  # existing files in the same set even if more files are subsequently
  # added.
  # To do that, we need a stable way of deciding based on just the file name
  # itself, so we do a hash of that and then use that to generate a
  # probability value that we use to assign it.
  hash_name_hashed = hashlib.sha1(hash_name.encode('utf-8')).hexdigest()
  percentage_hash = ((int(hash_name_hashed, 16) %
                      (MAX_NUM_WAVS_PER_CLASS + 1)) *
                     (100.0 / MAX_NUM_WAVS_PER_CLASS))
  if percentage_hash < testing_percentage:
    result = 'testing'
  else:
    result = 'training'
  return result


origFolder="C:/Users/JK-WORK/Documents/FEO3274/Final assignment/Dataset/train/audio"

classes_to_use=["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go", "_background_noise_"]

training=[];
test=[];
testing_percentage=10;

for root, dirs, files in os.walk(origFolder, topdown=False): ##Searchs all files inside of it and its subfolders
    for file in files:
        if file.endswith(".wav") and os.path.basename(root) in classes_to_use:
            if which_set(file, testing_percentage) == 'testing':
                test.append(os.path.basename(root) + "/" + file)
            else:
                training.append(os.path.basename(root) + "/" + file)


with open('training.txt', 'w') as f:
    for item in training:
        f.write("%s\n" % item)
with open('test.txt', 'w') as f:
    for item in test:
        f.write("%s\n" % item)