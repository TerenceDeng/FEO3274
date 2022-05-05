# -*- coding: utf-8 -*-
"""
Code to test forward algorithm
"""

from HMM import HMM
from MarkovChain import MarkovChain
from DiscreteD import DiscreteD
from GaussD import GaussD
import numpy as np
import matplotlib.pyplot as plt


##To test that it works with discrete distributions we use excercise 5.2 from book. Also checking infinite HMM


mc1 = MarkovChain( np.array( [ 0.5, 0.5 ] ), np.array( [ [ 0.2, 0.8 ], [ 0.8, 0.2 ] ] ) ) 
mc2 = MarkovChain( np.array( [ 0.5, 0.5 ] ), np.array( [ [ 0.8, 0.2 ], [ 0.2, 0.8 ] ] ) ) 

D1 = DiscreteD( [0.1,0.2,0.3,0.4] )   # Distribution for state = 1
D2 = DiscreteD( [0.4,0.3,0.2,0.1] )   # Distribution for state = 2

h1  = HMM( mc1, [D1, D2])
h2  = HMM( mc2, [D1, D2])

z=np.array([[3,1,4,2]])

#In order for this one to work I had to:
    #Create prob function in discrete D.
    #Solved dimentions notations, sometimes it was called row when in our code was a column
alphas_1,cs_1=h1.forward(z,norm=False);
p1=np.prod(cs_1);
alphas_2,cs_2=h2.forward(z,norm=False);
p2=np.prod(cs_2);


print("For lambda1 the book says that P=" + "0.005704" + " and the obtained probability is " + str(p1))
print("For lambda2 the book says that P=" + "0.002824" + " and the obtained probability is " + str(p2))

#Since the logic is fine now, I will check that there are no problems when the distributions are others:
    
u1=0;u2=3;std1=1;std2=2;
mc = MarkovChain( np.array( [ 0.75, 0.25 ] ), np.array( [ [ 0.99, 0.01 ], [ 0.03, 0.97 ] ] ) ) 
g1 = GaussD( means=[u1], stdevs=[std1] )   # Distribution for state = 1
g2 = GaussD( means=[u2], stdevs=[std2] )   # Distribution for state = 2
h  = HMM( mc, [g1, g2])                # The HMM
x,s = h.rand(10)

alphas,cs=h.forward(x);
p1=np.prod(cs);

#This checked that simple gaussian as emissor works too.

##Test of multiple gaussians:
u1=0;u2=1;std1=1;std2=2;
mc = MarkovChain( np.array( [ 0.75, 0.25 ] ), np.array( [ [ 0.99, 0.01 ], [ 0.03, 0.97 ] ] ) ) 
g1 = GaussD( means=[u1, u1], cov=np.asarray([[2,1],[1,4]]) )   # Distribution for state = 1
g2 = GaussD( means=[u2, u2], stdevs=np.asarray([[1,0],[0,1]]) )   # Distribution for state = 2
h  = HMM( mc, [g1, g2])                # The HMM
x,s = h.rand(5)

alphas,cs=h.forward(x);
p1=np.prod(cs);

##Now the test of the book 

u1=0;u2=3;std1=1;std2=2;
mc = MarkovChain( np.array( [ 1, 0 ] ), np.array( [ [ 0.9, 0.1,0 ], [ 0, 0.9, 0.1 ] ] ) ) 
g1 = GaussD( means=[u1], stdevs=[std1] )   # Distribution for state = 1
g2 = GaussD( means=[u2], stdevs=[std2] )   # Distribution for state = 2
h  = HMM( mc, [g1, g2])                # The HMM

x=np.array([[-0.2,2.6,1.3]])

alphas,cs=h.forward(x);

print(h.logprob(x,norm=False)) #Shows that it gets the value mentioned in the book
