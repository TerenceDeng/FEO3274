# -*- coding: utf-8 -*-
"""
Created on Thu May  5 11:13:26 2022

@author: JK-WORK
"""
from SingleWordRecognizer import SingleWordRecognizer
from generalFuncs import classify,get_list_of_ds_elem_names,read_elem,select_class_of_meas,get_class_of_meas,plot_results,get_accuracy
import numpy as np

classes_to_use=["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"]
word_recognizers={}
for word in classes_to_use:
    word_recognizers[word]=SingleWordRecognizer(word);

train_ds_names=get_list_of_ds_elem_names('train.txt');

res=[];
for word in classes_to_use:
    res=word_recognizers[word].train(read_elem(select_class_of_meas(train_ds_names,word)[:10]))
    word_recognizers[word].save_model("Models/");
#Here maybe some plots?

#
for word in classes_to_use:
    word_recognizers[word].load_model("Models/");
test_ds_names=get_list_of_ds_elem_names('test.txt');
scores=np.zeros((len(test_ds_names),len(classes_to_use)))
labels=["" for i in range(len(test_ds_names))]

for elem_idx in range(len(test_ds_names)):
    elem=test_ds_names[elem_idx]
    labels[elem_idx]=get_class_of_meas(elem)
    features=read_elem(elem)
    for word_idx in range(len(classes_to_use)):
        scores[elem_idx,word_idx]=word_recognizers[classes_to_use[word_idx]].evaluate(elem);
        
np.save("Outputs/scores", scores)
np.save('Outputs/labels.npy', labels, allow_pickle=True)

#Here a section to use the features to guess and minimize misclassifications

#Then call the classifier and plot the confusion matrix to show the results
y_pred=[classify(scores[elem_idx,:],classes_to_use) for elem_idx in  range(len(test_ds_names))];
plot_results(labels,y_pred,classes_to_use)
print("Accuracy: " + str(get_accuracy(labels,y_pred)));
