import numpy as np
import matplotlib.pyplot as plt

class my_bs_edp:

    def __init__(self):

        pass
    def explicit_euler(self,sigma,Smin,Smax,K,tau,r,b,Nx,Nt):
        """

         sigma: implied volalatility 0.2
            K:  the strike 100
          tau: maturity 0.25
            r:
           b: rS the cost of carry parameter
          Nx: space grid
         Nt : time grid
         this method is based on euler explicit method
         the jupyter notebook contains an explication about stability (dt/(dS*dS) * ((sigma * Smax)**2)/2 < 1/2)
        """
        dt = tau/Nt

        dS = (Smax-Smin)/Nx

        "Grid initialization"

        S_grid = np.linspace(Smin, Smax, Nx)

        c=np.zeros((Nt,Nx))

        " Condition aux limites"

        c[-1] = np.maximum(S_grid - K, 0)

        "Dirichlet condition  " # same as the course i just capitalized with np.exp

        c[:, -1] = Smax - K * np.exp(-r * (tau - np.linspace(0, tau, Nt)))

        # stability of exlicit method

        print('Stability condition:', dt / (dS * dS) * ((sigma * Smax) ** 2) / 2, ' < 1/2 ? \t',
            dt / (dS * dS) * ((sigma * Smax) ** 2) / 2 < 1 / 2)

        # Calculation Loop for Explicit Scheme
        # the cost of carry cofficient depend on Si but i used it as a constant value as the given data

        for t in range (Nt-2,-1,-1):  #inversed loop
            c[t, 0:-1] = c[t + 1, 0:-1]  # we exclude the boundary condition from calculation
            c[t, 1:-1] += dt * (b * (c[t + 1, 2:] - c[t + 1, 0:-2])/(2*dS)
            + sigma*sigma * np.square(S_grid[1:-1])/2 * (c[t + 1, 2:] - 2*c[t + 1, 1:-1] + c[t + 1, 0:-2])/(dS * dS)
            - r * c[t + 1, 1:-1]) #instead of using the loop of  i we explore the numpy
        return c

    def implicite_euler(self,sigma,Smin,Smax,K,tau,r,b,Nx,Nt):
        """

                 sigma: implied volalatility 0.2
                    K:  the strike 100
                  tau: maturity 0.25
                    r:
                   b: rS the cost of carry parameter
                  Nx: space grid
                 Nt : time grid
                 this method is based on euler explicit method
                 the jupyter notebook contains an explication about stability (dt/(dS*dS) * ((sigma * Smax)**2)/2 < 1/2)
                """
        dt = tau / Nt

        dS = (Smax - Smin) / Nx

        "Grid initialization"

        S_grid = np.linspace(Smin, Smax, Nx)

        c = np.zeros((Nt, Nx))

        " Dirichlet condition c(ST)"

        c[-1] = np.maximum(S_grid - K, 0)

        "Dirichlet condition  "  # same as the course i just capitalized with np.exp

        c[:, -1] = Smax - K * np.exp(-r * (tau - np.linspace(0, tau, Nt)))
        B = 1 + r * dt + sigma * sigma * np.square(S_grid) / (dS * dS) * dt
        A = - dt * (- b / (2 * dS) + sigma * sigma * np.square(S_grid) / (2 * dS * dS))
        C = - dt * (b / (2 * dS) + sigma * sigma * np.square(S_grid) / (2 * dS * dS))
        md = np.arange(Nx)
        sd = np.arange(Nx - 1)
        Matr = np.zeros((Nx, Nx))
        Matr[md, md] = B
        Matr[sd + 1, sd] = A[1:]
        Matr[sd, sd + 1] = C[:-1]
        # Calculation Loop for Implicit Scheme
        for t in range(Nt - 2, -1, -1):
            p_eq = c[t + 1].copy()
            p_eq[0] -= A[0] * c[t, 0]
            p_eq[-1] -= C[-1] * c[t, -1]
            c[t, 1:-1] = np.linalg.solve(Matr, p_eq)[1:-1]
        return c

# test
Sigma = 0.2
Smin = 50
Smax = 150
K = 100
tau = 0.25
r = 0.08
b = -0.04
Nx = 100
Nt = 1000

prices= my_bs_edp()

prices = prices.implicite_euler(Sigma, Smin, Smax, K, r, tau,b, Nx, Nt)



plt.plot(np.linspace(Smin, Smax, Nx), prices[0])
plt.show()




