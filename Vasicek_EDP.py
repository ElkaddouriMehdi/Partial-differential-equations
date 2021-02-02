import numpy as np
import matplotlib.pyplot as plt
import math
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
class my_vasicek:
    def __init__(self,rmin,rmax,T,M,N,a,b,sigma,Lambda):
        self.rmin=rmin
        self.rmax=rmax
        self.T=T
        self.M=M
        self.N=N
        self.a=a
        self.b=b
        self.sigma=sigma
        self.Lambda=Lambda
        self.a=a
        self.b=b

    def create_A_matrix(self):
        dt=self.T/self.N
        dr=((self.rmax-self.rmin)/self.M)
        theta=self.a*self.b-self.Lambda*self.sigma #theta is a*b-lambda*sigma
        #Creat matrix :
        A = np.zeros((self.M, self.M))
        # Create factor vectors for A
        PAu = np.zeros((self.M, 1))
        PAm = np.zeros((self.M))
        PAd = np.zeros((self.M, 1))
        # Define factors for A . I+dt/2 *A(t_i)
        # boundary conditions for A , let the second derivative with respect to r be 0
        PAu[0] = - dt / 4 * (theta - self.a * self.rmin) / dr
        PAm[0] = 1 + self.rmin * dt / 2
        PAd[0] = - dt / 4 * (-theta + self.a * self.rmin) / dr
        for i in range(1,self.M):
            PAu[i] = -dt / 4 * (math.pow(self.sigma, 2) / (math.pow(dr, 2)) - (theta - self.a * PAu[i - 1]) / dr)
            PAd[i] = -dt / 4 * (math.pow(self.sigma, 2) / (math.pow(dr, 2)) + (theta - self.a * PAd[i - 1]) / dr)
            for j in range(1, self.M):
                PAm[j] = 1 + dt / 2 * (math.pow(self.sigma, 2) / math.pow(dr, 2) + PAm[j - 1])
        A = np.diagflat([PAu[i] for i in range(self.M - 1)], -1) + \
            np.diagflat([PAm[i] for i in range(self.M)]) + \
            np.diagflat([PAd[i] for i in range(self.M - 1)], +1)
        return(A)
    def create_B_matrix(self):
        dt = self.T / self.N
        dr = ((self.rmax - self.rmin) / self.M)
        theta = self.a * self.b - self.Lambda * self.sigma
        # Create factor vectors for A
        PBu = np.zeros((self.M, 1))
        PBm = np.zeros((self.M, 1))
        PBd = np.zeros((self.M, 1))
        # Define factors for B . Idt/2 *A(t_i+1)
        # boundary conditions for B , let the second derivative with respect to r be 0
        PBu[0] = dt / 4 * (theta - self.a * self.rmin) / dr
        PBm[0] = 1 - self.rmin * dt / 2
        PBd[0] = dt / 4 * (-theta + self.a * self.rmin) / dr
        for i in range(1, self.M):
            PBu[i] = dt / 4 * (math.pow(self.sigma, 2) / (math.pow(dr, 2)) - (theta - self.a * PBu[i - 1]) / dr)
            PBd[i] = dt / 4 * (math.pow(self.sigma, 2) / (math.pow(dr, 2)) + (theta - self.a * PBd[i - 1]) / dr)
            for j in range(1, self.M):
                PBm[j] = 1 - dt / 2 * (math.pow(self.sigma, 2) / math.pow(dr, 2) + PBm[j - 1])
        B = np.diagflat([PBu[i] for i in range(self.M - 1)], -1) + \
            np.diagflat([PBm[i] for i in range(self.M)]) + \
            np.diagflat([PBd[i] for i in range(self.M - 1)], +1)
        return B

    def solution_crank(self,A,B):
        U = np.zeros((self.M, self.N))
        for i in range(0,self. M):
            U[i, 0] = 1
        for i in range(0, self.N - 1):
            U[:, i + 1] = np.dot(np.linalg.inv(A), np.dot(B, U[:, i]))
        return (U)

    def create_A_theta(self,theta):
        dt = self.T / self.N
        dr = ((self.rmax - self.rmin) / self.M)
        theta2 = self.a * self.b - self.Lambda * self.sigma
        A = np.zeros((self.M, self.M))
        # Create factor vectors for A
        PAu = np.zeros((self.M, 1))
        PAm = np.zeros((self.M))
        PAd = np.zeros((self.M, 1))
        # Define factors for A . I+dt/2 *A(t_i)
        # boundary conditions for A , let the second derivative with respect to r be 0
        PAu[0] = - (1 - theta) * dt / 2 * (theta2 - self.a * self.rmin) / dr
        PAm[0] = 1 + self.rmin * (1 - theta) * dt
        PAd[0] = - (1 - theta) * dt / 2 * (-theta2 + self.a * self.rmin) / dr
        for i in range(1, self.M):
            PAu[i] = - (1 - theta) * dt * 1 / 2 * (
                        math.pow(self.sigma, 2) / (math.pow(dr, 2)) - (theta2 - self.a * PAu[i - 1]) / dr)
            PAd[i] = - (1 - theta) * dt * 1 / 2 * (
                        math.pow(self.sigma, 2) / (math.pow(dr, 2)) + (theta2 - self.a * PAd[i - 1]) / dr)
            for j in range(1, self.M):
                PAm[j] = 1 + (1 - theta) * dt * (math.pow(self.sigma, 2) / math.pow(dr, 2) + PAm[j - 1])
        A = np.diagflat([PAu[i] for i in range(self.M - 1)], -1) + \
            np.diagflat([PAm[i] for i in range(self.M)]) + \
            np.diagflat([PAd[i] for i in range(self.M - 1)], +1)
        return(A)
    def create_B_theta(self,theta):
        dt = self.T / self.N
        dr = ((self.rmax - self.rmin) / self.M)
        theta2 = self.a * self.b - self.Lambda * self.sigma
        B = np.zeros((self.M, self.M))
        # Create factor vectors for A
        PBu = np.zeros((self.M, 1))
        PBm = np.zeros((self.M))
        PBd = np.zeros((self.M, 1))
        # Define factors for B . Idt/2 *A(t_i+1)
        # boundary conditions for B , let the second derivative with respect to r be 0
        PBu[0] = theta * dt / 2 * (theta2 - self.a * self.rmin) / dr
        PBm[0] = 1 - self.rmin * theta * dt
        PBd[0] = theta * dt / 2 * (-theta2 + self.a * self.rmin) / dr
        for i in range(1, self.M):
            PBu[i] = theta * dt * 1 / 2 * (
                    math.pow(self.sigma, 2) / (math.pow(dr, 2)) - (theta2 - self.a * PBu[i - 1]) / dr)
            PBd[i] = theta * dt * 1 / 2 * (
                    math.pow(self.sigma, 2) / (math.pow(dr, 2)) + (theta2 - self.a * PBd[i - 1]) / dr)
            for j in range(1, self.M):
                PBm[j] = 1 - theta * dt * (math.pow(self.sigma, 2) / math.pow(dr, 2) + PBm[j - 1])
        B = np.diagflat([PBu[i] for i in range(self.M - 1)], -1) + \
            np.diagflat([PBm[i] for i in range(self.M)]) + \
            np.diagflat([PBd[i] for i in range(self.M - 1)], +1)
        return (B)


