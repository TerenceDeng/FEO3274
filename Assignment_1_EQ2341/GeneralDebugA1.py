# -*- coding: utf-8 -*-
"""
Created on Fri Apr  8 11:41:52 2022

@author: JK-WORK
"""
# import sys
# import os
# import numpy as np
# func_path="C:/Users/JK-WORK/Documents/FEO3274/Assignment_1_EQ2341/PattRecClasses"
# if not any(os.path.normcase(sp) == os.path.normcase(func_path) for sp in sys.path):
#     sys.path.append(func_path) ##Adds path to the python path to fi
from PattRecClasses import PattRecClasses
    
mc=MarkovChain(np.asarray([0.5,0.5]),np.asarray([[0.1,0.9],[0.1,0.9]]))
mc.rand(5)