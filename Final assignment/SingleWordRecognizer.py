import numpy as np
from HMM import HMM
from MarkovChain import MarkovChain
from DiscreteD import DiscreteD
from GaussD import GaussD

class SingleWordRecognizer:
    
    def __init__(self, word):
        self.word=word;
        self.is_noise= (word=="_background_noise_");
        self.n_phonemes=len(word); #Maybe other word has more sounds?
        self.HMM = None;
        
    def initialize(self,mean_length,feature_size,prob_emission_type="Gaussian"):
        n_states=self.n_phonemes;
        A=np.zeros((n_states,n_states+1))
        mean_time_per_phoneme=mean_length/n_states;
        a_ii=(mean_time_per_phoneme-1) / mean_time_per_phoneme; #aii in the diagonal so in the init each phoneme has the same given duration
        for i in range(n_states):
            A[i,i]=a_ii;A[i,i+1]=1-a_ii;
        q=np.zeros((n_states,))
        q[0]=1;
        mc = MarkovChain( q, A ) 
        if prob_emission_type=="Gaussian":
            B = [GaussD( np.repeat([0],5), stdevs=1 ) for i in range(n_states)]  # Distribution for all states
            
        self.HMM = HMM(mc, B)
    def train(self,ds): #ds should be a list of runs. 
        mean_length=np.mean([i.shape[0] for i in ds]);
        feature_size=ds[0].shape[1];
        self.initialize(mean_length,feature_size);
        #history=self.HMM.BaumWelch(eval_p=True,tol=1e-6)
        #return history;
        
    def evaluate(self,data):
        res=None;
        if self.HMM ==None:
            print("No HMM created")
        else:   
            alpha, c = self.HMM.forward(data);
            res=np.sum(np.log10(c));
        return res