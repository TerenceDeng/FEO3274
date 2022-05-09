import numpy as np
from DiscreteD import DiscreteD
from GaussD import GaussD
from MarkovChain import MarkovChain


class HMM:
    """
    HMM - class for Hidden Markov Models, representing
    statistical properties of random sequences.
    Each sample in the sequence is a scalar or vector, with fixed DataSize.
    
    Several HMM objects may be collected in a single multidimensional array.
    
    A HMM represents a random sequence(X1,X2,....Xt,...),
    where each element Xt can be a scalar or column vector.
    The statistical dependence along the (time) sequence is described
    entirely by a discrete Markov chain.
    
    A HMM consists of two sub-objects:
    1: a State Sequence Generator of type MarkovChain
    2: an array of output probability distributions, one for each state
    
    All states must have the same class of output distribution,
    such as GaussD, GaussMixD, or DiscreteD, etc.,
    and the set of distributions is represented by an object array of that class,
    although this is NOT required by general HMM theory.
    
    All output distributions must have identical DataSize property values.
    
    Any HMM output sequence X(t) is determined by a hidden state sequence S(t)
    generated by an internal Markov chain.
    
    The array of output probability distributions, with one element for each state,
    determines the conditional probability (density) P[X(t) | S(t)].
    Given S(t), each X(t) is independent of all other X(:).
    
    
    References:
    Leijon, A. (20xx) Pattern Recognition. KTH, Stockholm.
    Rabiner, L. R. (1989) A tutorial on hidden Markov models
    	and selected applications in speech recognition.
    	Proc IEEE 77, 257-286.
    
    """
    def __init__(self, mc, distributions):

        self.stateGen = mc
        self.outputDistr = distributions

        self.nStates = mc.nStates
        self.dataSize = distributions[0].dataSize
    
    def rand(self, nSamples):
        """
        [X,S]=rand(self,nSamples); generates a random sequence of data
        from a given Hidden Markov Model.
        
        Input:
        nSamples=  maximum no of output samples (scalars or column vectors)
        
        Result:
        X= matrix or row vector with output data samples
        S= row vector with corresponding integer state values
          obtained from the self.StateGen component.
          nS= length(S) == size(X,2)= number of output samples.
          If the StateGen can generate infinite-duration sequences,
              nS == nSamples
          If the StateGen is a finite-duration MarkovChain,
              nS <= nSamples
        """
        
        #*** Insert your own code here and remove the following error message 
        S=self.stateGen.rand(nSamples);
        n_features=len(self.outputDistr[0].rand(1))
        nSamples_final=len(S[0,:]);
        X=np.empty((n_features,nSamples_final));
        for i in range(nSamples_final):
            s=S[0,i];
            if s<self.stateGen.nStates:
                X[:,i]=self.outputDistr[s].rand(1).ravel()
            else:
                X=X[:,:i]
        return X,S
        
    
    def forward(self,obs,norm=True):
        res, scaled = self.prob(obs);
        if not norm:
            scaled = res;
        self.res=res;self.scaled=scaled; #To avoid recalculations of this, which is expensive!
        return self.stateGen.forward(scaled)
    
    def backward(self,obs,alphas=None,cs=None,norm=True):
        if cs is None: #When cs are not given, it needs to calculate them
            alphas,cs=self.forward(obs,norm=norm);
        betas=alphas; #Just to check the other algorithms
        return alphas,betas,cs
    
    def viterbi(self):
        pass
    
    def calcabc(self, obs):
        alphahats_list = []
        betahats_list = []
        cs_list = []
        self.res_list=[];self.scaled_list=[]
        for i in range(len(obs)):
            alphahats,betahats,cs = self.backward(obs[i])
            self.res_list.append(self.res);self.scaled_list.append(self.scaled);
            alphahats_list += [alphahats]
            betahats_list += [betahats]
            cs_list += [cs]
        return alphahats_list, betahats_list, cs_list
    
    def calcgammas(self, alphahats_list, betahats_list, cs_list, obs, uselog=True):
        gammas = []
        for i in range(len(obs)):
            temp = []
            for t in range(obs[i].shape[1]):
                if uselog:
                    temp += [np.log(alphahats_list[i][:,t])+np.log(betahats_list[i][:,t])+np.log(cs_list[i][t])]
                else:
                    temp += [alphahats_list[i][:,t]*betahats_list[i][:,t]*cs_list[i][t]]
            gammas += [np.array(temp)]
        #gammas = np.array(gammas)
        return gammas
    def calcinit(self, gammas, uselog=False):
        init_gamma=np.asarray([i[0,:] for i in gammas]);
        if uselog:
            return np.sum(np.exp(init_gamma), axis = 0)/np.sum(np.exp(init_gamma))
        else: 
            return np.sum(init_gamma, axis = 0)/np.sum(init_gamma)
        
    def calcxi(self, alphahats_list, betahats_list, cs_list, obs, uselog=False):
        xirbars = []
        xirs = []
        for i in range(len(obs)): 
            if self.stateGen.is_finite:
                xi = np.zeros((obs[i].shape[1], self.stateGen.A.shape[0], self.stateGen.A.shape[1]))
            else:
                xi = np.zeros((obs[i].shape[1]-1, self.stateGen.A.shape[0], self.stateGen.A.shape[1]))
            
            if uselog: 
                xi = np.log(xi)
                p=np.log(self.res_list[i]);scaled=np.log(self.scaled_list[i])
                #p, scaled = np.log(self.prob(obs[i])) #Saved in the other variables because it was calculated before.
            else:
                p=self.res_list[i];scaled=self.scaled_list[i]
            for t in range(obs[i].shape[1]-1):
                for j in range(self.stateGen.A.shape[0]):
                    for k in range(self.stateGen.A.shape[0]):
                        if uselog:
                            xi[t, j, k] = np.log(alphahats_list[i][j][t])+np.log(self.stateGen.A[j,k])+p[k][t+1]+np.log(betahats_list[i][k][t+1])
                        else:
                            xi[t, j, k] = alphahats_list[i][j][t]*self.stateGen.A[j,k]*p[k][t+1]*betahats_list[i][k][t+1]
            if self.stateGen.is_finite:
                for j in range(self.stateGen.A.shape[0]):
                    if uselog:
                        xi[-1][j][-1] = np.log(alphahats_list[i][-1][j])+np.log(betahats_list[i][-1][j])+np.log(cs_list[i][-1])
                    else:
                        xi[-1][j][-1] = alphahats_list[i][-1][j]*betahats_list[i][-1][j]*cs_list[i][-1]
                
            if uselog:
                xi = np.exp(xi)
            xirs += [xi]
            xirbars += [np.sum(xi, axis = 0)]
            
        xibar = np.sum(xirbars, axis = 0)
        return xibar
    
    def calcmeansandcov(self,obs,gammas):
        summ = np.zeros((len(self.outputDistr), obs[0].shape[0]))
        sumc = np.zeros((len(self.outputDistr), obs[0].shape[0], obs[0].shape[0]))
        sumg = np.zeros((len(self.outputDistr)))

        for i in range(len(obs)):
            for t in range(obs[i].shape[0]): 
                for j in range(len(self.outputDistr)):
                    summ[j] += obs[i][:,t]*gammas[i][t][j]
                    sumg[j] += gammas[i][t][j]
                    temp = obs[i][:,t] - np.atleast_2d(self.outputDistr[j].means);
                    sumc[j] += gammas[i][t][j]*(temp.T.dot(temp))
                    
        newmean = np.zeros(summ.shape)
        newcov = np.zeros(sumc.shape)
        for i in range(newmean.shape[0]):
            if sumg[i] > 0:
                newmean[i] = summ[i]/sumg[i]
                newcov[i] = sumc[i]/sumg[i]
            else:
                newmean[i] = 0
                newcov[i]  = 0
        return newmean,newcov
    def baum_welch(self,obs,niter,uselog=True,history=True):

        history=[];
        for it in range(niter):
            alphahats_list, betahats_list, cs_list = self.calcabc(obs) #from Assignment 3 and 4
            gammas = self.calcgammas(alphahats_list, betahats_list, cs_list, obs, uselog) #alpha*beta*c
            newpi = self.calcinit(gammas, uselog) #average of gammas[:,0]
            xibar = self.calcxi(alphahats_list, betahats_list, cs_list, obs, uselog) #page 132
            if uselog: 
                xibar = np.exp(xibar)
                
            newA = np.array([i/np.sum(i) for i in xibar]) #xibar/sum_k(xibar); page 130
            
            if uselog: 
                gammas = [np.exp(i) for i in gammas]

            newmean,newcov = self.calcmeansandcov(obs,gammas)
            
            #update all variables
            self.stateGen.q = newpi
            self.stateGen.A = newA
            newoutputDistr = [GaussD( newmean[i], cov=newcov[i]) for i in range(len(self.outputDistr))]
            self.outputDistr = newoutputDistr
            if history:
                history.append(np.mean([np.exp(self.logprob(obs[i])) for i in range(obs.shape[0])]))
        return history;

    def stateEntropyRate(self):
        pass

    def setStationary(self):
        pass

    def logprob(self,obs,norm=True):
        alphas, cs =self.forward(obs, norm=norm)
        return np.sum(np.log(cs))

    def adaptStart(self):
        pass

    def adaptSet(self):
        pass

    def adaptAccum(self):
        pass
    def prob(self, x):
        T = x.shape[1]
        N = len(self.outputDistr)
        res = np.zeros((N, T))
        for i in range(N):
            res[i,:] = self.outputDistr[i].prob(x[:,:].T)
        scaled = np.zeros(res.shape)
        for i in range(scaled.shape[0]):
            for j in range(scaled.shape[1]):
                scaled[i, j] = res[i,j]/np.amax(res[:,j])
        return res, scaled
    
    
    
def test_learning():
        # Define a HMM
    q = np.array([0.8, 0.2])
    A = np.array([[0.95, 0.05],
                  [0.30, 0.70]])
    
    means = np.array( [[0, 0], [2, 2]] )
    covs  = np.array( [[[1, 2],[2, 4]], 
                       [[1, 0],[0, 3]]] )
    mc = MarkovChain( q, A ) 
    g1 = GaussD( means=means[0], cov=covs[0] )   # Distribution for state = 1
    g2 = GaussD(means=means[1], cov=covs[1] )   # Distribution for state = 1
    
    hm  = HMM(mc,[g1,g2])
    obs = np.array([ hm.rand(100)[0] for _ in range(10) ])
    
    print('True HMM parameters:')
    print('q:')
    print(q)
    print('A:')
    print(A)
    print('B: means, covariances')
    print(means)
    print(covs)
    
    # Estimate the HMM parameters from the obseved samples
    # Start by. assigning initial HMM parameter values,
    # then refine these iteratively
    qstar = np.array([0.8, 0.2])
    Astar = np.array([[0.5, 0.5], [0.5, 0.5]])
    mc2 = MarkovChain( qstar, Astar ) 
    meansstar = np.array( [[0, 0], [0, 0]] )
    
    covsstar  = np.array( [[[1, 0],[0, 1]], 
                           [[1, 0],[0,1]]] )
    
    g1_2 = GaussD( means=meansstar[0], cov=covsstar[0] )   # Distribution for state = 1
    g2_2 = GaussD(means=meansstar[1], cov=covsstar[1] )   # Distribution for state = 1
    
    
    hm_learn  = HMM(mc2,[g1_2,g2_2])
    
    print("Running the Baum Welch Algorithm...")
    hist=hm_learn.baum_welch(obs, 20, uselog=False)
    
    print('True HMM parameters:')
    print('q:')
    print(hm_learn.stateGen.q)
    print('A:')
    print(hm_learn.stateGen.A)
    print('B: means, covariances of 1')
    print(hm_learn.outputDistr[0].means)
    print(hm_learn.outputDistr[0].means)
    
    hm_learn2 = HMM(qstar, Astar, Bstar)
    
    print("Running the Baum Welch Algorithm with logs...")
    hist=hm_learn2.baum_welch(obs, 20, prin=1, uselog=True)
    print('True HMM parameters:')
    print('q:')
    print(hm_learn2.stateGen.q)
    print('A:')
    print(hm_learn2.stateGen.A)
    print('B: means, covariances of 1')
    print(hm_learn2.outputDistr[0].means)
    print(hm_learn2.outputDistr[0].means)
if __name__ == "__main__":
    #test load and save of the model.
    test_learning();
    