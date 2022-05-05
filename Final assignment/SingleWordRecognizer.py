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
        
    def initialize(self,mean_length,feature_size,prob_emission_type="Gaussian",add_states_for_silence=True):
        n_states= self.n_phonemes+2 if add_states_for_silence else self.n_phonemes; #Silence at the beggining and the end.
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
    def evaluate(self,data,norm=False):
        if isinstance(data,list):
            res=[];
            for i in data:
                res.append(self.evaluate_simple_record(i,norm=False));
        else:
            res=self.evaluate_simple_record(data,norm=False);
        return res;
        
    def evaluate_simple_record(self,data,norm=False):
        res=None;
        if self.HMM ==None:
            print("No HMM created")
        else:   
            res = self.HMM.logprob(data,norm=norm);
        return res
    
    def save_model(self,dir_to_save):
        np.save(dir_to_save+self.word+"_A",self.HMM.stateGen.A)
        np.save(dir_to_save+self.word+"_q",self.HMM.stateGen.q)
        means= [i.means for i in self.HMM.outputDistr]
        variances= [i.cov.diagonal() for i in self.HMM.outputDistr] #Assuming diagonal cov matrix
        np.save(dir_to_save+self.word+"_B_means",means)
        np.save(dir_to_save+self.word+"_B_variances",variances)
        
    def load_model(self,dir_to_load):
        A=np.load(dir_to_load+self.word+"_A.npy")
        q=np.load(dir_to_load+self.word+"_q.npy")
        means=np.load(dir_to_load+self.word+"_B_means.npy")
        variances=np.load(dir_to_load+self.word+"_B_variances.npy")
        mc = MarkovChain( q, A ) 
        B = [GaussD( means[i], cov=np.diag(variances[i]) ) for i in range(len(means))]  # Distribution for all states
        self.HMM = HMM(mc, B)
        
        
if __name__ == "__main__":
    #test load and save of the model.
    word="yes";
    a=SingleWordRecognizer(word);
    mc = MarkovChain( np.array( [ 0.5, 0.5 ] ), np.array( [ [ 0.2, 0.8 ], [ 0.8, 0.2 ] ] ) ) 
    g1 = GaussD( means=[1,2], stdevs=[3,4] )   # Distribution for state = 1
    g2 = GaussD( means=[-1,-2], stdevs=[5,6] )   # Distribution for state = 1
    a.HMM = HMM(mc, [g1,g2])
    a.save_model("Models/")
    a.HMM = None;
    a.load_model("Models/")