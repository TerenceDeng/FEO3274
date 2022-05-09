from SingleWordRecognizer import SingleWordRecognizer
from generalFuncs import classify,get_list_of_ds_elem_names,read_elem,select_class_of_meas,get_class_of_meas,plot_results,get_accuracy
import numpy as np

classes_to_use=["yes"]
word_recognizers={}
for word in classes_to_use:
    word_recognizers[word]=SingleWordRecognizer(word);
    
train_ds_names=get_list_of_ds_elem_names('train.txt');

res=[];
for word in classes_to_use:
    res=word_recognizers[word].train(read_elem(select_class_of_meas(train_ds_names,word)[:10]))
    word_recognizers[word].save_model("Models/");