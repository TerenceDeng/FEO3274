# -*- coding: utf-8 -*-
"""
Created on Thu May  5 13:35:25 2022

@author: JK-WORK
"""

from SingleWordRecognizer import SingleWordRecognizer
from generalFuncs import classify,which_set,get_list_of_ds_elem_names,read_elem,select_class_of_meas,get_class_of_meas,plot_results,get_accuracy
import numpy as np
from sklearn.metrics import confusion_matrix,accuracy_score

classes_to_use=["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"]
word_recognizers={}
for word in classes_to_use:
    word_recognizers[word]=SingleWordRecognizer(word);

train_ds_names=get_list_of_ds_elem_names('train.txt');
test_ds_names=get_list_of_ds_elem_names('test.txt');
all_names=train_ds_names + test_ds_names;
testing_percentage=0.1;
N_cross_val=10
cm=np.zeros((len(classes_to_use),len(classes_to_use)))
accuracy=0;

for n_run in range(N_cross_val):
    new_cat=[which_set(i, testing_percentage, str(n_run)) for i in all_names]
    train_ds_names= list(np.asarray(all_names)[np.where(np.asarray(new_cat)=="train")[0]])
    test_ds_names= list(np.asarray(all_names)[np.where(np.asarray(new_cat)=="test")[0]])
    #train
    for word in classes_to_use:
        res=word_recognizers[word].train(read_elem(select_class_of_meas(train_ds_names,word)))
    #evaluation
    scores=np.zeros((len(test_ds_names),len(classes_to_use)))
    y_true=["" for i in range(len(test_ds_names))];y_pred=["" for i in range(len(test_ds_names))]
    for elem_idx in range(len(test_ds_names)):
        elem=test_ds_names[elem_idx]
        y_true[elem_idx]=get_class_of_meas(elem)
        features=read_elem(elem)
        for word_idx in range(len(classes_to_use)):
            scores[elem_idx,word_idx]=word_recognizers[classes_to_use[word_idx]].evaluate(elem);
    y_pred=[classify(scores[elem_idx,:],classes_to_use) for elem_idx in  range(len(test_ds_names))];
    confusion_matrix(y_true, y_pred, labels=classes_to_use)
    accuracy+=accuracy_score(y_true, y_pred);
    
accuracy=accuracy/N_cross_val;
cm=cm/N_cross_val;
