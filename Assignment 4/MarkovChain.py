import numpy as np
from DiscreteD import DiscreteD

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

    def forward(self, pX):
        if self.is_finite:
            A = self.A[:,:-1]
        else:
            A = self.A
        c = np.zeros(pX.shape[1])
        alpha = np.zeros((A.shape[0],pX.shape[1]))
        temp = np.zeros(A.shape[0])
        c[0] = np.sum(self.q*pX[:,0])
        alpha[:,0] = (self.q*pX[:,0])/c[0]

        for t in range(1, pX.shape[1]):
            for j in range(A.shape[0]):
                temp[j] = alpha[:,t-1].dot(A[:,j]) * pX[j,t]
            c[t] = np.sum(temp)
            alpha[:,t] = temp/c[t]

        if self.is_finite:
            variable = alpha[:,-1].dot(self.A[:, -1])
            c = np.append(c, np.array([variable]))
            
        return alpha, c

    def backward(self,scaled):
        if self.is_finite:
            A = self.A[:,:-1]
        else:
            A = self.A
        beta = np.zeros((self.A.shape[0],scaled.shape[1]))
        temp = np.zeros(self.A.shape[0])
        alphas,cs = self.forward(scaled)
        
        if self.is_finite:
            temp = self.A[:,-1]
            temp = temp/(cs[-1]*cs[-2])
        else:
            temp = np.ones((self.A.shape[0]))
            temp = temp/cs[-1]
        beta[:,-1] = temp
        
        
        for t in range(scaled.shape[1]-2, -1, -1):
            temp = np.zeros(A.shape[0])
            for i in range(A.shape[0]):                
                for j in range(A.shape[0]):
                    temp[i] += A[i,j]*scaled[j,t+1]*beta[j,t+1]
            beta[:,t] = temp/cs[t]

        if self.is_finite:
            temp = self.A[:,-1]
            temp = temp/(cs[-1]*cs[-2])
        else:
            temp = np.ones((self.A.shape[0]))
            temp = temp/cs[-1]
        beta[:,-1] = temp
        return alphas,beta, cs



    def finiteDuration(self):
        pass    

    def adaptStart(self):
        pass

    def adaptSet(self):
        pass

    def adaptAccum(self):
        pass
