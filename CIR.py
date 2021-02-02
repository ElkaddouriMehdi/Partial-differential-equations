import numpy as np
import matplotlib.pyplot as plt
import math
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
class my_CIR:
    def __init__(self,rmin,rmax,T,M,N,k,theta,sigma,Lambda):
        self.rmin=rmin
        self.rmax=rmax
        self.T=T
        self.M=M
        self.N=N
        self.theta=theta
        self.sigma=sigma
        self.Lambda=Lambda
        self.k=k  #kappa
    def CIR_A_matrix(self):
        dt = self.T / self.N
        dr = ((self.rmax - self.rmin) / self.M)
        # Create Matrices
        A = np.zeros((self.M, self.M))
        # Create factor vectors for A
        PAu = np.zeros((self.M, 1))
        PAm = np.zeros((self.M))
        PAd = np.zeros((self.M, 1))
        # Define factors for A I+dt/2 *A(t_i)
        # boundary conditions for A , let the second derivative with respect to r be 0
        PAu[0] = - dt / 4 * (self.k * (self.theta - self.rmin) - self.Lambda * self.rmin) / dr
        PAm[0] = 1 + self.rmin * dt / 2
        PAd[0] = - dt / 4 * (-self.k * (self.theta - self.rmin) + self.Lambda * self.rmin) / dr
        for i in range(1, self.M):
            PAu[i] = -dt / 4 * ((math.pow(self.sigma, 2) * PAu[i - 1]) / (math.pow(dr, 2)) - (
                        (self.k * (self.theta - PAu[i - 1]) - self.Lambda * PAu[i - 1]) / dr))
            PAd[i] = -dt / 4 * ((math.pow(self.sigma, 2) * PAd[i - 1]) / (math.pow(dr, 2)) + (
                        (self.k * (self.theta - PAd[i - 1]) - self.Lambda * PAd[i - 1]) / dr))
            for j in range(1, self.M):
                PAm[j] = 1 + dt / 2 * ((math.pow(self.sigma, 2) * PAm[j - 1]) / math.pow(dr, 2) + PAm[j - 1])
        A = np.diagflat([PAu[i] for i in range(self.M - 1)], -1) + \
            np.diagflat([PAm[i] for i in range(self.M)]) + \
            np.diagflat([PAd[i] for i in range(self.M - 1)], +1)
        return(A)
    def CIR_B_matrix(self):
        dt = self.T / self.N
        dr = ((self.rmax - self.rmin) / self.M)
        # Create Matrices
        B = np.zeros((self.M, self.M))
        # Create factor vectors for B
        PBu = np.zeros((self.M, 1))
        PBm = np.zeros((self.M, 1))
        PBd = np.zeros((self.M, 1))
        # Define factors for B . Idt/2 *A(t_i+1)
        # boundary conditions for B , let the second derivative with respect to r be 0
        PBu[0] = dt / 4 * (self.k * (self.theta - self.rmin) - self.Lambda * self.rmin) / dr
        PBm[0] = 1 - self.rmin * dt / 2
        PBd[0] = dt / 4 * (-self.k * (self.theta - self.rmin) + self.Lambda * self.rmin) / dr
        for i in range(1, self.M):
            PBu[i] = dt / 4 * ((math.pow(self.sigma, 2) * PBu[i - 1]) / (math.pow(dr, 2)) - (
                        (self.k A* (self.theta - PBu[i - 1]) - self.Lambda * PBu[i - 1]) / dr))
            PBd[i] = dt / 4 * ((math.pow(self.sigma, 2) * PBd[i - 1]) / (math.pow(dr, 2)) + (
                        (self.k * (self.theta - PBd[i - 1]) - self.Lambda * PBd[i - 1]) / dr))
            for j in range(1, self.M):
                PBm[j] = 1 - dt / 2 * ((math.pow(self.sigma, 2) * PBm[j - 1]) / math.pow(dr, 2) + PBm[j - 1])
        B = np.diagflat([PBu[i] for i in range(self.M - 1)], -1) + \
            np.diagflat([PBm[i] for i in range(self.M)]) + \
            np.diagflat([PBd[i] for i in range(self.M - 1)], +1)
        return(B)
    def solution_CIR(self,A,B):
        U = np.zeros((self.M, self.N))
        for i in range(0,self. M):
            U[i, 0] = 1
        for i in range(0, self.N - 1):
            U[:, i + 1] = np.dot(np.linalg.inv(A), np.dot(B, U[:, i]))
        return(U)






