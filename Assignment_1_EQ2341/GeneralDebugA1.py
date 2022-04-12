# -*- coding: utf-8 -*-
"""
Created on Fri Apr  8 11:41:52 2022

@author: JK-WORK
"""
import numpy as np
import numpy.matlib


class GaussD:
    """
    GaussD - Probability distribution class, representing
    Gaussian random vector
    EITHER with statistically independent components,
               i.e. diagonal covariance matrix, with zero correlations,
    OR with a full covariance matrix, including correlations
    -----------------------------------------------------------------------
    
    Several GaussD objects may be collected in a multidimensional array,
               even if they do not have the same DataSize.
    """
    def __init__(self, means, stdevs=None, cov=None):

        self.means = np.array(means)
        self.stdevs = np.array(stdevs)
        self.dataSize = len(self.means)

        if cov is None:
            self.variance = self.stdevs**2
            self.cov = np.eye(self.dataSize)*self.variance
            self.covEigen = 1
        else:
            self.cov = cov
            v, self.covEigen = np.linalg.eig(0.5*(cov + cov.T))
            self.stdevs = np.sqrt(np.abs(v))
            self.variance = self.stdevs**2
    
   
    def rand(self, nData):
        """
        R=rand(pD,nData) returns random vectors drawn from a single GaussD object.
        
        Input:
        pD=    the GaussD object
        nData= scalar defining number of wanted random data vectors
        
        Result:
        R= matrix with data vectors drawn from object pD
           size(R)== [length(pD.Mean), nData]
        """
        R = np.random.randn(self.dataSize, nData)
        R = np.diag(self.stdevs)@R
        
        if not isinstance(self.covEigen, int):
            R = self.covEigen@R

        R = R + np.matlib.repmat(self.means.reshape(-1, 1), 1, nData)

        return R
    
    def init(self):
        pass
    
    def logprob(self):
        pass
    
    def plotCross(self):
        pass

    def adaptStart(self):
        pass
    
    def adaptSet(self):
        pass

    def adaptAccum(self):
        pass

class DiscreteD:
    """
    DiscreteD - class representing random discrete integer.
    
    A Random Variable with this distribution is an integer Z
    with possible values 1,...,length(ProbMass).
    
    Several DiscreteD objects may be collected in an array
    """
    def __init__(self, x):
        self.pseudoCount = 0
        self.probMass = x/np.sum(x)
        
    def rand(self, nData):
        """
        R=rand(nData) returns random scalars drawn from given Discrete Distribution.
        
        Input:
        nData= scalar defining number of wanted random data elements
        
        Result:
        R= row vector with integer random data drawn from the DiscreteD object
           (size(R)= [1, nData]
        """
        
        #*** Insert your own code here and remove the following error message 
        
        print('Not yet implemented')
        
        
    def init(self, x):
        """
        initializes DiscreteD object or array of such objects
        to conform with a set of given observed data values.
        The agreement is crude, and should be further refined by training,
        using methods adaptStart, adaptAccum, and adaptSet.
        
        Input:
        x=     row vector with observed data samples
        
        Method:
        For a single DiscreteD object: Set ProbMass using all observations.
        For a DiscreteD array: Use all observations for each object,
               and increase probability P[X=i] in pD(i),
        This is crude, but there is no general way to determine
               how "close" observations X=m and X=n are,
               so we cannot define "clusters" in the observed data.
        """
        if len(np.shape(x))>1: 
            print('DiscreteD object can have only scalar data')
            
        x = np.round(x)
        maxObs = int(np.max(x))
        # collect observation frequencies
        fObs = np.zeros(maxObs) # observation frequencies

        for i in range(maxObs):
            fObs[i] = 1 + np.sum(x==i)
        
        self.probMass = fObs/np.sum(fObs)

        return self


    def entropy(self):
        pass

    def prob(self):
        pass
    
    def double(self):
        pass

    def adaptStart(self):
        pass

    def adaptSet(self):
        pass

    def adaptAccum(self):
        pass


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

        return X,S
        
    def viterbi(self):
        pass

    def train(self):
        pass

    def stateEntropyRate(self):
        pass

    def setStationary(self):
        pass

    def logprob(self):
        pass

    def adaptStart(self):
        pass

    def adaptSet(self):
        pass

    def adaptAccum(self):
        pass
    
    

class MarkovChain:
    """
    MarkovChain - class for first-order discrete Markov chain,
    representing discrete random sequence of integer "state" numbers.
    
    A Markov state sequence S(t), t=1..T
    is determined by fixed initial probabilities P[S(1)=j], and
    fixed transition probabilities P[S(t) | S(t-1)]
    
    A Markov chain with FINITE duration has a special END state,
    coded as nStates+1.
    The sequence generation stops at S(T), if S(T+1)=(nStates+1)
    """
    def __init__(self, initial_prob, transition_prob):

        self.q = initial_prob  #InitialProb(i)= P[S(1) = i]
        self.A = transition_prob #TransitionProb(i,j)= P[S(t)=j | S(t-1)=i]
        
        self.nStates = transition_prob.shape[0]
        

        self.is_finite = False
        if self.A.shape[0] != self.A.shape[1]:
            self.is_finite = True
            self.end_state = self.nStates;


    def probDuration(self, tmax):
        """
        Probability mass of durations t=1...tMax, for a Markov Chain.
        Meaningful result only for finite-duration Markov Chain,
        as pD(:)== 0 for infinite-duration Markov Chain.
        
        Ref: Arne Leijon (201x) Pattern Recognition, KTH-SIP, Problem 4.8.
        """
        pD = np.zeros(tmax)

        if self.is_finite:
            pSt = (np.eye(self.nStates)-self.A.T)@self.q

            for t in range(tmax):
                pD[t] = np.sum(pSt)
                pSt = self.A.T@pSt

        return pD

    def probStateDuration(self, tmax):
        """
        Probability mass of state durations P[D=t], for t=1...tMax
        Ref: Arne Leijon (201x) Pattern Recognition, KTH-SIP, Problem 4.7.
        """
        t = np.arange(tmax).reshape(1, -1)
        aii = np.diag(self.A).reshape(-1, 1)
        
        logpD = np.log(aii)*t+ np.log(1-aii)
        pD = np.exp(logpD)

        return pD

    def meanStateDuration(self):
        """
        Expected value of number of time samples spent in each state
        """
        return 1/(1-np.diag(self.A))
    
    def rand(self, tmax):
        """
        S=rand(self, tmax) returns a random state sequence from given MarkovChain object.
        
        Input:
        tmax= scalar defining maximum length of desired state sequence.
           An infinite-duration MarkovChain always generates sequence of length=tmax
           A finite-duration MarkovChain may return shorter sequence,
           if END state was reached before tmax samples.
        
        Result:
        S= integer row vector with random state sequence,
           NOT INCLUDING the END state,
           even if encountered within tmax samples
        If mc has INFINITE duration,
           length(S) == tmax
        If mc has FINITE duration,
           length(S) <= tmaxs
        """
        
        #*** Insert your own code here and remove the following error message 
        S=np.empty([1, tmax], dtype=int);
        S[0,0]=np.random.choice(self.nStates , 1, p=self.q)[0];

        for i in range(1,tmax):
            S[0,i]=np.random.choice(self.A.shape[1], 1, p=self.A[S[0,i-1],:])[0];
            if self.is_finite and S[0,i]==self.end_state:
                return S[:, :(i-1)]
        return S;

    def viterbi(self):
        pass
    
    def stationaryProb(self):
        pass
    
    def stateEntropyRate(self):
        pass
    
    def setStationary(self):
        pass

    def logprob(self):
        pass

    def join(self):
        pass

    def initLeftRight(self):
        pass
    
    def initErgodic(self):
        pass

    def forward(self):
        pass

    def finiteDuration(self):
        pass
    
    def backward(self):
        pass

    def adaptStart(self):
        pass

    def adaptSet(self):
        pass

    def adaptAccum(self):
        pass


# u1=0;u2=3;std1=1;std2=2;
# mc = MarkovChain( np.array( [ 0.75, 0.25 ] ), np.array( [ [ 0.99, 0.01], [ 0.03, 0.97 ] ] ) ) 

# g1 = GaussD( means=[u1, u1], cov=np.asarray([[2,1],[1,4]]) )   # Distribution for state = 1
# g2 = GaussD( means=[u2, u2], stdevs=np.asarray([[0.0001,0],[0,0.0001]]) )   # Distribution for state = 2
# h  = HMM( mc, [g1, g2])                # The HMM
# x,s = h.rand(100)