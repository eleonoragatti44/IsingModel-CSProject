import numpy as np
import matplotlib.pyplot as plt
import scipy

class Ising():
  
    def __init__(self,n1,n2,T,B=1,J=1):
        self.n1 = n1
        self.n2 = n2
        self.N = n1 * n2
        self.matrix = np.random.choice([-1, 1], size=(self.n1, self.n2))
        self.ham = [] 
        self.mag = []
        self.B = B
        self.J = J
        self.T = T

    # Compute the interaction between one spin and first neighbors
    def get_onepoint_interaction(self,i,j):
        interaction = - self.J * self.matrix[i,j] * (self.matrix[(i+1)%self.n1,j]+self.matrix[i-1,j]
                                                     +self.matrix[i,(j+1)%self.n2]+self.matrix[i,j-1])
        return(interaction)
    
    def get_mf_deltaE(self,i,j):
        self.update_mag()
        s = self.matrix[i,j]
        J = self.J
        B = self.B
        N = self.N
        mf = self.mag[-1] 
        deltaE = s * (8*J*mf-2*B) - 8*J/N
        return deltaE
    
    def get_total_interaction(self):
        # list of pairs of adjacent sites as four-element tuples:
        # (i1, j1, i2, j2) represents two adjacent sites located
        # at (i1, j1) and (i2, j2)
        horizontal_edges = [
            (i, j-1, i, j)
            for i in range(self.n1) for j in range(self.n2)
        ]
        vertical_edges = [
            (i-1, j, i, j)
            for i in range(self.n1) for j in range(self.n2)
        ]
        # summing lists in python concatenates them
        # [1, 2, 3] + [4, 5] = [1, 2, 3, 4, 5]
        edges = horizontal_edges + vertical_edges
        E = 0
        for i1, j1, i2, j2 in edges:
            E -= self.matrix[i1,j1]*self.matrix[i2,j2]
        return E                                               


    def update_ham(self):
        # Compute the energy of interactions
        h = self.get_total_interaction()
        # Compute the energy of external field
        self.update_mag()
        h += self.B * self.mag[-1] * self.N
        self.ham.append(h)

    def update_mag(self):
        self.mag.append(np.sum(self.matrix) / self.N)

    def flip_one_point(self, i, j):
        self.matrix[i,j] = self.matrix[i,j] * (-1)
              
    def heatmap(self):
        fig, ax = plt.subplots()
        im = ax.imshow(self.matrix, cmap = 'binary')
        plt.show()

    # deltaE -> difference btw final energy and initial energy of the single point (before and after the eventual swap)
    def metropolis(self, n_updates):
        for _ in range(n_updates):
            i = np.random.randint(self.n1)
            j = np.random.randint(self.n2)
            E_i = self.get_onepoint_interaction(i, j)
            deltaE = - 2 * self.matrix[i,j] * self.B - 2 * E_i
            if deltaE < 0:
                self.flip_one_point(i, j)
            else:
                # what is k_B???
                if np.random.uniform() < np.exp(-deltaE/self.T):
                    self.flip_one_point(i, j)
              
    def metropolis_mf(self, n_updates):
        for _ in range(n_updates):
            i = np.random.randint(self.n1)
            j = np.random.randint(self.n2)
            deltaE = self.get_mf_deltaE(i, j)
            if deltaE < 0:
                self.flip_one_point(i, j)
            else:
                # what is k_B???
                if np.random.uniform() < np.exp(-deltaE/self.T):
                    self.flip_one_point(i, j)
                    
                    
                    
                    
                    
                    

    def WolffMove(self):
        """
        Faster, list-based Wolff move.
        #
        Pick a random spin; remember its direction as oldSpin
        Push it onto a list "toFlip" of spins to flip
        Set spinsFlipped = 0
        While there are spins left in toFlip
           Remove the first spin
           If it has not been flipped in between
              Flip it
              Add one to spinsFlipped
              For each of its neighbors
                  if the neighbor is in the oldSpin direction
                  with probability p, put it on the stack
        Return spinsFlipped
        """ 
        i = np.random.randint(0, self.n1)
        j = np.random.randint(0, self.n2)
        oldSpin = self.matrix[i, j]
        toFlip = [(i, j)]
        spinsFlipped = 0
        while len(toFlip) > 0:
            i, j = toFlip.pop(0)
            # Check if flipped in between
            if self.matrix[i, j] == oldSpin:
                self.matrix[i, j] = self.matrix[i, j]*(-1)
                spinsFlipped += 1
                ip1 = (i + 1) % self.n1
                im1 = (i - 1) % self.n1
                jp1 = (j + 1) % self.n2
                jm1 = (j - 1) % self.n2
                neighbors = [(ip1, j), (im1, j), (i, jp1), (i, jm1)]
                for m, n in neighbors:
                    if self.matrix[m, n] == oldSpin:
                        if scipy.random.random() < (1.0 - np.exp(-2. * self.J / self.T)): 
                            toFlip.append((m, n))
        return spinsFlipped

    def SweepWolff(self, nTimes=1, partialSweep=0):
        """
        Do nTimes sweeps of the Wolff algorithm, returning partialSweep
        (1) The variable partialSweep is the number of `extra' spins flipped
        in the previous Wolff cluster moved that belong to the current sweep.
        (2) A sweep is comprised of Wolff cluster moves until at least
        N*N-partialSweep spins have flipped. (Just add the spinsFlipped
        from WolffMove to partialSweep, while partialSweep < N*N, the
        new partialSweep is the current one minus N*N.)
        (3) Return the new value of partialSweep after nTimes sweeps.
        (4) You might print an error message if the field is not zero
        
        if self.B != 0.:
            print("Field will be ignored by Wolff algorithm")
        """
        for time in range(nTimes):
            while partialSweep < self.N:
                partialSweep += self.WolffMove()
            partialSweep = partialSweep - self.N
        return partialSweep

"""
Taken from
http://pages.physics.cornell.edu/~myers/teaching/ComputationalMethods/ComputerExercises/PythonSoftware/Ising.py

and slightly modified by
Francesc Font-Clos
May 2018
"""